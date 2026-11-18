#include "common.h"
#include <arm_neon.h>

enum { PACKED_ROWS = 4, PACKED_COLUMNS = 16 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype;
} Parameters;

typedef struct {
    size_t width, values, scales, high, subscales;
} Layout;

typedef struct {
    const uint8_t *values, *high;
    const int8_t *subscales;
    uint32_t dtype;
} Block;

typedef struct {
    void *output;
    const int8_t *input;
    const uint8_t *values, *scales, *high;
    const int8_t *subscales;
    const float *activation_scales;
    const Parameters *p;
    const Layout *layout;
    size_t groups, columns, column_tiles;
    const TrillimKernelContext *context;
} Task;

static const Layout *packed_layout(uint32_t dtype) {
    static const Layout layouts[] = {
        [TRILLIM_GRP_INT4 - TRILLIM_GRP_INT4] = {32, 16, 2, 0, 0},
        [TRILLIM_GRP_INT5 - TRILLIM_GRP_INT4] = {32, 16, 2, 4, 0},
        [TRILLIM_DUBGRP_INT6 - TRILLIM_GRP_INT4] = {256, 128, 2, 64, 16},
        [TRILLIM_GRP_BINARY - TRILLIM_GRP_INT4] = {128, 16, 2, 0, 0},
        [TRILLIM_GRP_TERNARY - TRILLIM_GRP_INT4] = {128, 32, 2, 0, 0},
        [TRILLIM_MXFP4 - TRILLIM_GRP_INT4] = {32, 16, 1, 0, 0},
    };
    return dtype >= TRILLIM_GRP_INT4 && dtype <= TRILLIM_MXFP4 ? &layouts[dtype - TRILLIM_GRP_INT4] : NULL;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t record = layout->values + layout->scales + layout->high + layout->subscales;
    if (groups > SIZE_MAX / record || p->outputs > SIZE_MAX / (groups * record) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * record;
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static float weight_scale(const Task *task, size_t block) {
    if (task->p->weight_dtype == TRILLIM_MXFP4) {
        // The integer E2M1 palette contains twice each decoded value.
        return trillim_mxfp4_scale(task->scales[block]) * 0.5f;
    }
    uint16_t bits;
    memcpy(&bits, task->scales + block * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline uint8x16_t spread_bits(const uint8_t *packed, uint8x16_t mask) {
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(packed[0]), vdup_n_u8(packed[1]));
    return vceqq_u8(vandq_u8(bytes, mask), mask);
}

static inline int8x16_t decode(const Block *block, size_t k) {
    const uint8_t *values = block->values;
    const uint8x16_t one = vdupq_n_u8(1);
    const uint8x16_t nibbles = vdupq_n_u8(15);
    switch (block->dtype) {
    case TRILLIM_GRP_BINARY: {
        const uint8x16_t mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
        uint8x16_t bits = spread_bits(values + k / 8, mask);
        return vreinterpretq_s8_u8(vsubq_u8(vdupq_n_u8(0), vorrq_u8(bits, one)));
    }
    case TRILLIM_GRP_TERNARY: {
        int8x16_t shift = vdupq_n_s8((int)(k / 32) * 2 - 6);
        uint8x16_t packed = vld1q_u8(values + k % 32);
        uint8x16_t codes = vandq_u8(vshlq_u8(packed, shift), vdupq_n_u8(3));
        return vreinterpretq_s8_u8(vsubq_u8(codes, one));
    }
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5: {
        uint8x16_t packed = vld1q_u8(values);
        uint8x16_t codes = k == 0 ? vandq_u8(packed, nibbles) : vshrq_n_u8(packed, 4);
        if (block->dtype == TRILLIM_GRP_INT5) {
            const uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
            uint8x16_t high = spread_bits(block->high + k / 8, mask);
            codes = vorrq_u8(codes, vandq_u8(high, vdupq_n_u8(16)));
            return vreinterpretq_s8_u8(vsubq_u8(codes, vdupq_n_u8(16)));
        }
        return vreinterpretq_s8_u8(vsubq_u8(codes, vdupq_n_u8(8)));
    }
    case TRILLIM_DUBGRP_INT6: {
        size_t half = k / 128;
        size_t lane = k % 128 / 32;
        uint8x16_t low = vld1q_u8(values + half * 64 + lane / 2 * 32 + k % 32);
        uint8x16_t high = vld1q_u8(block->high + half * 32 + k % 32);
        low = lane % 2 == 0 ? vandq_u8(low, nibbles) : vshrq_n_u8(low, 4);
        high = vshlq_u8(high, vdupq_n_s8(-(int)lane * 2));
        high = vshlq_n_u8(vandq_u8(high, vdupq_n_u8(3)), 4);
        return vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(low, high), vdupq_n_u8(32)));
    }
    case TRILLIM_MXFP4: {
        const int8x16_t palette = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        uint8x8_t packed = vld1_u8(values + k / 2);
        uint8x8_t low = vand_u8(packed, vdup_n_u8(15));
        uint8x8_t high = vshr_n_u8(packed, 4);
        uint8x8x2_t codes = vzip_u8(low, high);
        return vqtbl1q_s8(palette, vcombine_u8(codes.val[0], codes.val[1]));
    }
    default:
        return vdupq_n_s8(0);
    }
}

static inline int32x4_t products(int8x16_t x, int8x16_t w) {
    // Weight codes have magnitude at most 32, so paired products fit in int16.
    int16x8_t low = vmull_s8(vget_low_s8(x), vget_low_s8(w));
    int16x8_t pairs = vmlal_high_s8(low, x, w);
    return vpaddlq_s16(pairs);
}

static inline bool packed_dot(const Block *block, const int8_t *input, size_t stride, size_t rows,
                              size_t count, int32_t *sums) {
    int32x4_t acc[PACKED_ROWS] = {0};
    for (size_t k = 0; k < count; k += 16) {
        int8x16_t weights = decode(block, k);
        size_t lanes = count - k < 16 ? count - k : 16;
        if (block->dtype == TRILLIM_GRP_TERNARY) {
            int8x16_t valid = weights;
            if (lanes < 16) {
                const uint8x16_t indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
                uint8x16_t mask = vcltq_u8(indices, vdupq_n_u8((uint8_t)lanes));
                valid = vandq_s8(weights, vreinterpretq_s8_u8(mask));
            }
            if (vmaxvq_s8(valid) == 2) {
                return false;
            }
        }
        for (size_t row = 0; row < PACKED_ROWS; ++row) {
            if (row == rows) {
                break;
            }
            const int8_t *src = input + row * stride + k;
            int8x16_t x;
            if (lanes < 16) {
                int8_t tail[16] = {0};
                memcpy(tail, src, lanes);
                x = vld1q_s8(tail);
            } else {
                x = vld1q_s8(src);
            }
            int32x4_t dot = products(x, weights);
            if (block->dtype == TRILLIM_DUBGRP_INT6) {
                acc[row] = vmlaq_n_s32(acc[row], dot, block->subscales[k / 16]);
            } else {
                acc[row] = vaddq_s32(acc[row], dot);
            }
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = vaddvq_s32(acc[row]);
    }
    return true;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    const Layout *layout = task->layout;
    size_t row = tile / task->column_tiles * PACKED_ROWS;
    size_t rows = p->tokens - row < PACKED_ROWS ? p->tokens - row : PACKED_ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    for (size_t column = first; column < end; ++column) {
        float totals[PACKED_ROWS] = {0};
        for (size_t g = 0; g < task->groups; ++g) {
            size_t k = g * layout->width;
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->width - k < layout->width ? p->width - k : layout->width;
            size_t index = column * task->groups + g;
            Block block = {
                task->values + index * layout->values,
                task->high + index * layout->high,
                task->subscales + index * layout->subscales,
                p->weight_dtype,
            };
            int32_t sums[PACKED_ROWS] = {0};
            if (!packed_dot(&block, task->input + row * p->width + k, p->width, rows, count, sums)) {
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = NAN;
                }
                break;
            }
            float scale = weight_scale(task, index);
            for (size_t r = 0; r < rows; ++r) {
                totals[r] += (float)sums[r] * scale;
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue result = {.f = totals[r] / task->activation_scales[row + r]};
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_quantized_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t blocks = p->outputs * groups;
    const uint8_t *values = inputs[1];
    const uint8_t *scales = values + blocks * layout->values;
    const uint8_t *high = scales + blocks * layout->scales;
    const int8_t *subscales = (const int8_t *)(high + blocks * layout->high);
    size_t columns = p->outputs / workers / (4 * PACKED_COLUMNS) * PACKED_COLUMNS;
    if (columns < PACKED_COLUMNS) {
        columns = PACKED_COLUMNS;
    }
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    size_t rows = p->tokens / PACKED_ROWS + (p->tokens % PACKED_ROWS != 0);
    Task task = {
        outputs[0], inputs[0], values, scales, high, subscales, inputs[2],
        p, layout, groups, columns, tiles, context,
    };
    return rows * tiles == 1 ? project(&task, 0, 0)
                            : context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_quantized_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _DTYPE_IDS, _quantized_linear, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if weight.dtype not in {'grp_binary', 'grp_ternary', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'mxfp4'}:\n"
           "        raise ValueError('projection requires grouped low-bit or MXFP4 weights')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return dict(_quantized_linear(context, weight_dtype=weight.dtype), weight_dtype=_DTYPE_IDS[weight.dtype])\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_quantized_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 3, 1, requirements, NULL, inputs, outputs, 1,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_quantized_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON};
    return &capabilities;
}
