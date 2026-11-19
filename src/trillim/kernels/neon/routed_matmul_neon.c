#include "common.h"
#include <arm_neon.h>

typedef struct {
    size_t tokens, routes, experts, n, k;
    size_t a_token_stride, a_route_stride, index_token_stride, index_route_stride;
    uint32_t a_dtype;
} Parameters;

enum { ROUTES_PER_TILE = 4 };

typedef struct {
    void *output;
    const uint8_t *activation, *codes, *scales;
    const Parameters *p;
    const TrillimKernelContext *context;
    const size_t *offsets, *tiles, *slots;
    size_t columns, column_tiles;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!p->experts || !p->n || !p->k || p->k % 32 || !trillim_float(p->a_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    if (p->n > SIZE_MAX / p->k || p->tokens > SIZE_MAX / p->routes) {
        return TRILLIM_ERROR;
    }
    size_t slots = p->tokens * p->routes;
    size_t index_shape[] = {p->tokens, p->routes};
    size_t index_strides[] = {p->index_token_stride, p->index_route_stride};
    size_t a_shape[] = {p->tokens, p->routes, p->k};
    size_t a_strides[] = {p->a_token_stride, p->a_route_stride, 1};
    size_t a_span, index_span;
    size_t a_size = trillim_itemsize(p->a_dtype);
    size_t scratch_limit = SIZE_MAX / sizeof(size_t);
    if (!trillim_span(3, a_shape, a_strides, &a_span) || a_span > SIZE_MAX / a_size ||
        !trillim_span(2, index_shape, index_strides, &index_span) || index_span > SIZE_MAX / sizeof(int64_t) ||
        slots > SIZE_MAX / sizeof(float) / p->n ||
        slots > scratch_limit - 2 || p->experts > (scratch_limit - 2 - slots) / 2 ||
        !trillim_packed_payload_size(p->experts, p->n * p->k, TRILLIM_MXFP4, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a_span * a_size;
    inputs[2] = index_span * sizeof(int64_t);
    outputs[0] = slots * p->n * sizeof(float);
    *scratch = (2 * (p->experts + 1) + slots) * sizeof(size_t);
    return TRILLIM_OK;
}

static inline bool cancelled(const TrillimKernelContext *context) {
    return context->is_cancelled(context->cancellation_context);
}

static inline uint64_t expert_at(const Parameters *p, const void *indices, size_t slot) {
    size_t index = slot / p->routes * p->index_token_stride + slot % p->routes * p->index_route_stride;
    return (uint64_t)trillim_load(indices, index, TRILLIM_INT64).i;
}

static TrillimStatus group_routes(
    const Parameters *p, const void *indices, size_t *offsets, size_t *tiles, size_t *slots,
    const TrillimKernelContext *context
) {
    size_t count = p->tokens * p->routes;
    for (size_t expert = 0; expert < p->experts; ++expert) {
        if ((expert & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        offsets[expert] = 0;
    }
    for (size_t slot = 0; slot < count; ++slot) {
        if ((slot & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        uint64_t expert = expert_at(p, indices, slot);
        if (expert >= p->experts) {
            return TRILLIM_ERROR;
        }
        ++offsets[expert];
    }
    size_t next = 0;
    tiles[0] = 0;
    for (size_t expert = 0; expert < p->experts; ++expert) {
        if ((expert & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        size_t routes = offsets[expert];
        tiles[expert + 1] = tiles[expert] + routes / ROUTES_PER_TILE + (routes % ROUTES_PER_TILE != 0);
        next += routes;
        offsets[expert] = next;
    }
    offsets[p->experts] = count;
    /* Inserting backward turns each end offset into the start of its expert's routes. */
    for (size_t slot = count; slot != 0; --slot) {
        if ((slot & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        uint64_t expert = expert_at(p, indices, slot - 1);
        slots[--offsets[expert]] = slot - 1;
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) float32x4x4_t load16(const uint8_t *data, size_t index, uint32_t dtype) {
    float32x4x4_t result;
    if (dtype == TRILLIM_FP32) {
        memcpy(&result, data + index * sizeof(float), sizeof(result));
        return result;
    }
    uint16x8x2_t bits;
    memcpy(&bits, data + index * sizeof(uint16_t), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        result.val[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits.val[0]), 16));
        result.val[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits.val[0]), 16));
        result.val[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits.val[1]), 16));
        result.val[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits.val[1]), 16));
    } else {
        result.val[0] = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(bits.val[0])));
        result.val[1] = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(bits.val[0])));
        result.val[2] = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(bits.val[1])));
        result.val[3] = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(bits.val[1])));
    }
    return result;
}

/* Finite MXFP4 weights fit exactly in BF16. Scales 0 and 1 need subnormal encodings. */
#define MXFP4_BITS(E, BASE, MIN_SCALE) \
    ((E) == 0 ? (MIN_SCALE) : ((E) == 1 && (MIN_SCALE) == 0x0020 ? 0x0040 : (BASE) + ((E) - 127) * 128))
#define MXFP4_BYTE(E, BASE, MIN_SCALE, SHIFT) ((MXFP4_BITS(E, BASE, MIN_SCALE) >> (SHIFT)) & 255)
#define MXFP4_PALETTE(E, SHIFT, SIGN) \
    (SIGN), \
    MXFP4_BYTE(E, 0x3f00, 0x0020, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x3f80, 0x0040, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x3fc0, 0x0060, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4000, 0x0080, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4040, 0x00c0, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4080, 0x0100, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x40c0, 0x0140, SHIFT) | (SIGN)
#define MXFP4_ENTRY(E) {MXFP4_PALETTE(E, 0, 0), MXFP4_PALETTE(E, 0, 0), \
                        MXFP4_PALETTE(E, 8, 0), MXFP4_PALETTE(E, 8, 128)}
#define MXFP4_ENTRIES4(E) MXFP4_ENTRY(E), MXFP4_ENTRY((E) + 1), MXFP4_ENTRY((E) + 2), MXFP4_ENTRY((E) + 3)
#define MXFP4_ENTRIES16(E) MXFP4_ENTRIES4(E), MXFP4_ENTRIES4((E) + 4), MXFP4_ENTRIES4((E) + 8), MXFP4_ENTRIES4((E) + 12)
static const uint8_t mxfp4_bf16_bits[256][32] = {
    MXFP4_ENTRIES16(0), MXFP4_ENTRIES16(16), MXFP4_ENTRIES16(32), MXFP4_ENTRIES16(48),
    MXFP4_ENTRIES16(64), MXFP4_ENTRIES16(80), MXFP4_ENTRIES16(96), MXFP4_ENTRIES16(112),
    MXFP4_ENTRIES16(128), MXFP4_ENTRIES16(144), MXFP4_ENTRIES16(160), MXFP4_ENTRIES16(176),
    MXFP4_ENTRIES16(192), MXFP4_ENTRIES16(208), MXFP4_ENTRIES16(224), MXFP4_ENTRIES16(240),
};
#undef MXFP4_ENTRIES16
#undef MXFP4_ENTRIES4
#undef MXFP4_ENTRY
#undef MXFP4_PALETTE
#undef MXFP4_BYTE
#undef MXFP4_BITS

static inline __attribute__((always_inline)) void decode16(const Task *task, size_t block, size_t k, float32x4_t weights[4]) {
    uint8_t exponent = task->scales[block + k / 32];
    uint8x8_t packed = vld1_u8(task->codes + block * 16 + k / 2);
    uint8x8x2_t split = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
    uint8x16_t codes = vcombine_u8(split.val[0], split.val[1]);
    uint8x16x2_t tables = vld1q_u8_x2(mxfp4_bf16_bits[exponent]);
    uint8x16_t low = vqtbl1q_u8(tables.val[0], codes);
    uint8x16_t high = vqtbl1q_u8(tables.val[1], codes);
    uint16x8_t a = vreinterpretq_u16_u8(vzip1q_u8(low, high));
    uint16x8_t b = vreinterpretq_u16_u8(vzip2q_u8(low, high));
    weights[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16));
    weights[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a), 16));
    weights[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
    weights[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16));
}

static TrillimStatus validate_weights(const Task *task, size_t expert, size_t first, size_t end) {
    size_t groups = task->p->k / 32;
    size_t block = (expert * task->p->n + first) * groups;
    size_t count = (end - first) * groups;
    const uint8_t *scales = task->scales + block;
    uint8x16_t maximum = vdupq_n_u8(0);
    size_t i = 0;
    for (; count - i >= 16; i += 16) {
        if ((i & 8191) == 0 && cancelled(task->context)) {
            return TRILLIM_CANCELLED;
        }
        maximum = vmaxq_u8(maximum, vld1q_u8(scales + i));
    }
    uint8_t bound = vmaxvq_u8(maximum);
    for (; i < count; ++i) {
        if (scales[i] > bound) {
            bound = scales[i];
        }
    }
    if (bound < 253) {
        return TRILLIM_OK;
    }

    /* Packed gather rejects NaN scales and weights that overflow FP32. */
    const uint8_t *codes = task->codes + block * 16;
    for (i = 0; i < count; ++i) {
        if ((i & 255) == 0 && cancelled(task->context)) {
            return TRILLIM_CANCELLED;
        }
        uint8_t exponent = scales[i];
        if (exponent < 253) {
            continue;
        }
        uint8x16_t packed = vld1q_u8(codes + i * 16);
        uint8x16_t mask = vdupq_n_u8(7);
        uint8x16_t low = vandq_u8(packed, mask);
        uint8x16_t high = vandq_u8(vshrq_n_u8(packed, 4), mask);
        if (exponent == 255 || vmaxvq_u8(vmaxq_u8(low, high)) >= (exponent == 253 ? 6 : 4)) {
            return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_fp32(
    const Task *task, size_t expert, size_t first, size_t end, const size_t *slots,
    const uint8_t *const *activation, size_t count, size_t rows, size_t columns
) {
    const Parameters *p = task->p;
    size_t groups = p->k / 32;
    for (size_t column = first; column < end; column += columns) {
        size_t width = end - column < columns ? end - column : columns;
        size_t blocks[4];
        float32x4_t sums[ROUTES_PER_TILE][4];
        for (size_t c = 0; c < columns; ++c) {
            blocks[c] = (expert * p->n + column + (c < width ? c : 0)) * groups;
            for (size_t r = 0; r < rows; ++r) {
                sums[r][c] = vdupq_n_f32(0);
            }
        }
        for (size_t k = 0; k < p->k; k += 16) {
            if ((k & 1023) == 0 && cancelled(task->context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t weights[4][4];
            for (size_t c = 0; c < columns; ++c) {
                decode16(task, blocks[c], k, weights[c]);
            }
            /* Match packed matmul's four FP32 reduction lanes per output. */
            for (size_t r = 0; r < rows; ++r) {
                float32x4x4_t x = load16(activation[r], k, p->a_dtype);
                for (size_t c = 0; c < columns; ++c) {
                    float32x4_t sum = sums[r][c];
                    sum = vfmaq_f32(sum, x.val[0], weights[c][0]);
                    sum = vfmaq_f32(sum, x.val[1], weights[c][1]);
                    sum = vfmaq_f32(sum, x.val[2], weights[c][2]);
                    sums[r][c] = vfmaq_f32(sum, x.val[3], weights[c][3]);
                }
            }
        }
        for (size_t r = 0; r < count; ++r) {
            for (size_t c = 0; c < width; ++c) {
                TrillimValue value = {.f = vaddvq_f32(sums[r][c])};
                trillim_store(task->output, slots[r] * p->n + column + c, TRILLIM_FP32, value);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t route_tile = tile / task->column_tiles;
    size_t low = 0, high = p->experts;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (task->tiles[mid + 1] <= route_tile) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    size_t expert = low;
    size_t offset = task->offsets[expert] + (route_tile - task->tiles[expert]) * ROUTES_PER_TILE;
    size_t count = task->offsets[expert + 1] - offset;
    if (count > ROUTES_PER_TILE) {
        count = ROUTES_PER_TILE;
    }
    const size_t *slots = task->slots + offset;
    const uint8_t *activation[ROUTES_PER_TILE];
    size_t a_size = trillim_itemsize(p->a_dtype);
    for (size_t r = 0; r < count; ++r) {
        size_t index = slots[r] / p->routes * p->a_token_stride + slots[r] % p->routes * p->a_route_stride;
        activation[r] = task->activation + index * a_size;
    }
    for (size_t r = count; r < ROUTES_PER_TILE; ++r) {
        activation[r] = activation[0];
    }
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->n - first < task->columns ? p->n : first + task->columns;
    TrillimStatus status = validate_weights(task, expert, first, end);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (count == 1) {
        return project_fp32(task, expert, first, end, slots, activation, 1, 1, 4);
    }
    if (count == 2) {
        return project_fp32(task, expert, first, end, slots, activation, 2, 2, 2);
    }
    return project_fp32(task, expert, first, end, slots, activation, count, ROUTES_PER_TILE, 2);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_routed_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (cancelled(context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers || scratch == NULL) {
        return TRILLIM_ERROR;
    }
    size_t *offsets = scratch;
    size_t *tiles = offsets + p->experts + 1;
    size_t *slots = tiles + p->experts + 1;
    TrillimStatus status = group_routes(p, inputs[2], offsets, tiles, slots, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t columns = p->n / workers / 2 / 4 * 4;
    if (columns < 16) {
        columns = 16;
    }
    size_t column_tiles = p->n / columns + (p->n % columns != 0);
    const uint8_t *codes = inputs[1];
    const uint8_t *scales = codes + p->experts * p->n * (p->k / 32) * 16;
    Task task = {outputs[0], inputs[0], codes, scales, p, context, offsets, tiles, slots, columns, column_tiles};
    return context->thread_pool->run(context->thread_pool->context, tiles[p->experts] * column_tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_routed_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('a', 'weight', 'indices')\n"
           "def bind(c):\n"
           "    a, weight, indices = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    if a.dtype not in {'fp16', 'bf16', 'fp32'} or out.dtype != 'fp32' or weight.dtype != 'mxfp4' or indices.dtype != 'int64':\n"
           "        raise ValueError('routed matmul requires floating activations, MXFP4 weights and int64 indices')\n"
           "    if (len(a.shape), len(weight.shape), len(indices.shape), len(out.shape)) != (4, 3, 3, 4):\n"
           "        raise ValueError('routed matmul requires rank-four activations, a rank-three bank and rank-three indices')\n"
           "    experts, n, k = weight.shape\n"
           "    if any(dim.value is None or dim.value <= 0 for dim in weight.shape) or k.value % 32:\n"
           "        raise ValueError('routed matmul requires a fixed bank with aligned reduction rows')\n"
           "    if a.shape[2].value != 1 or a.shape[3] != k or indices.shape[2].value != 1 or out.shape[2].value != 1 or out.shape[3] != n:\n"
           "        raise ValueError('routed matmul requires one activation row and one index per route')\n"
           "    if _storage_shape(weight) != (experts, _product(weight.shape[1:])) or (weight.strides and weight.strides != _contiguous_strides(weight.shape)):\n"
           "        raise ValueError('routed matmul requires a canonical contiguous expert bank')\n"
           "    index_strides = _contiguous_strides(indices.shape)\n"
           "    if indices.strides and indices.strides != index_strides:\n"
           "        raise ValueError('routed matmul requires contiguous indices')\n"
           "    a_strides = a.strides or _contiguous_strides(a.shape)\n"
           "    if a_strides[-1].value != 1:\n"
           "        raise ValueError('routed matmul requires contiguous activation rows')\n"
           "    token_stride, route_stride = _broadcast_strides(a.shape[:2], out.shape[:2], a_strides[:2])\n"
           "    id_token_stride, id_route_stride = _broadcast_strides(indices.shape[:2], out.shape[:2], index_strides[:2])\n"
           "    return dict(tokens=out.shape[0], routes=out.shape[1], experts=experts, n=n, k=k,\n"
           "                a_token_stride=token_stride, a_route_stride=route_stride,\n"
           "                index_token_stride=id_token_stride, index_route_stride=id_route_stride,\n"
           "                a_dtype=_DTYPE_IDS[a.dtype])\n"
           "def pattern(a: Tensor, weight: Tensor, indices: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, Permute(Gather(weight, indices=indices), (0, 1, 3, 2))),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_routed_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"routes", "size_t", offsetof(Parameters, routes), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
        {"n", "size_t", offsetof(Parameters, n), NULL},
        {"k", "size_t", offsetof(Parameters, k), NULL},
        {"a_token_stride", "size_t", offsetof(Parameters, a_token_stride), NULL},
        {"a_route_stride", "size_t", offsetof(Parameters, a_route_stride), NULL},
        {"index_token_stride", "size_t", offsetof(Parameters, index_token_stride), NULL},
        {"index_route_stride", "size_t", offsetof(Parameters, index_route_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(size_t)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_routed_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
