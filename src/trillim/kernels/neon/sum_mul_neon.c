#include "common.h"
#include <arm_neon.h>

/* Mul rounds before Sum; contraction would change the primitive result. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t heads, reduce, columns, rank;
    const size_t *a_strides, *b_strides;
    size_t a_rows, a_columns, b_rows, b_columns;
    uint32_t a_dtype, b_dtype, dtype;
} Parameters;

typedef struct {
    void *output;
    const void *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    uint32_t promoted = p->a_dtype == p->b_dtype ? p->a_dtype : TRILLIM_FP32;
    if (p->rank != 3 || !p->a_strides || !p->b_strides ||
        !trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        !trillim_float(p->dtype) || p->dtype != promoted) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->columns) {
        return TRILLIM_OK;
    }
    if (p->heads > SIZE_MAX / trillim_itemsize(p->dtype) / p->columns) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->columns * trillim_itemsize(p->dtype);
    if (!p->reduce) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->reduce, p->columns};
    const size_t *strides[] = {p->a_strides, p->b_strides};
    size_t rows[] = {p->a_rows, p->b_rows};
    size_t columns[] = {p->a_columns, p->b_columns};
    uint32_t dtypes[] = {p->a_dtype, p->b_dtype};
    for (size_t i = 0; i < 2; ++i) {
        size_t span;
        if ((columns[i] && rows[i] > SIZE_MAX / columns[i]) ||
            !trillim_span(3, shape, strides[i], &span) ||
            span > rows[i] * columns[i] || span > SIZE_MAX / trillim_itemsize(dtypes[i])) {
            return TRILLIM_ERROR;
        }
        inputs[i] = span * trillim_itemsize(dtypes[i]);
    }
    return TRILLIM_OK;
}

static float native_float(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(float), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)data + index * sizeof(bits), sizeof(bits));
    return dtype == TRILLIM_FP16 ? trillim_fp16(bits) : trillim_bf16(bits);
}

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
    } else if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index,
                 vreinterpret_u16_f16(vcvt_f16_f32(value)));
    } else {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
    }
}

static float32x4_t round4(float32x4_t value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vcvt_f16_f32(value));
    }
    if (dtype == TRILLIM_BF16) {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        return vreinterpretq_f32_u32(vshlq_n_u32(rounded, 16));
    }
    return value;
}


static float32x4_t load_strided4(const void *data, size_t base, size_t stride, uint32_t dtype) {
    if (stride == 1) {
        return load4(data, base, dtype);
    }
    if (stride == 0) {
        return vdupq_n_f32(native_float(data, base, dtype));
    }
    float values[4];
    for (size_t lane = 0; lane < 4; ++lane) {
        values[lane] = native_float(data, base + lane * stride, dtype);
    }
    return vld1q_f32(values);
}

static TrillimStatus reduce_columns(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->columns / 64 + (p->columns % 64 != 0);
    size_t head = tile / tiles, first = tile % tiles * 64;
    size_t count = p->columns - first < 64 ? p->columns - first : 64;
    size_t abase = head * p->a_strides[0] + first * p->a_strides[2];
    size_t bbase = head * p->b_strides[0] + first * p->b_strides[2];
    size_t vectors = count / 4;
    float32x4_t sums[16];
    float tail[3] = {0};
    for (size_t lane = 0; lane < vectors; ++lane) {
        sums[lane] = vdupq_n_f32(0.0f);
    }
    for (size_t k = 0; k < p->reduce; ++k) {
        if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t arow = abase + k * p->a_strides[1];
        size_t brow = bbase + k * p->b_strides[1];
        if (p->a_strides[2] == 1 && p->b_strides[2] == 0) {
            float32x4_t b = vdupq_n_f32(native_float(task->b, brow, p->b_dtype));
            for (size_t lane = 0; lane < vectors; ++lane) {
                float32x4_t product = round4(vmulq_f32(load4(task->a, arow + lane * 4, p->a_dtype), b), p->dtype);
                sums[lane] = vaddq_f32(sums[lane], product);
            }
        } else {
            for (size_t lane = 0; lane < vectors; ++lane) {
                float32x4_t a = load_strided4(task->a, arow + lane * 4 * p->a_strides[2], p->a_strides[2], p->a_dtype);
                float32x4_t b = load_strided4(task->b, brow + lane * 4 * p->b_strides[2], p->b_strides[2], p->b_dtype);
                float32x4_t product = round4(vmulq_f32(a, b), p->dtype);
                sums[lane] = vaddq_f32(sums[lane], product);
            }
        }
        for (size_t lane = vectors * 4; lane < count; ++lane) {
            float a = native_float(task->a, arow + lane * p->a_strides[2], p->a_dtype);
            float b = native_float(task->b, brow + lane * p->b_strides[2], p->b_dtype);
            float product = round_dtype(a * b, p->dtype);
            tail[lane - vectors * 4] += product;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t output = head * p->columns + first;
    for (size_t lane = 0; lane < vectors; ++lane) {
        store4(task->output, output + lane * 4, p->dtype, sums[lane]);
    }
    for (size_t lane = vectors * 4; lane < count; ++lane) {
        TrillimValue value = {.f = tail[lane - vectors * 4]};
        trillim_store(task->output, output + lane, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_reduce_sum_mul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->heads || !p->columns) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->columns / 64 + (p->columns % 64 != 0);
    return p->heads * tiles == 1 ? reduce_columns(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->heads * tiles, reduce_columns, &task);
}

TRILLIM_KERNEL const char *trillim_neon_reduce_sum_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _native_dtype, _storage_shape\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression, _promote\n"
           "\n"
           "def bind(context):\n"
           "    a, b, output = context.inputs['a'], context.inputs['b'], context.outputs[0]\n"
           "    rank = max(len(a.shape), len(b.shape))\n"
           "    if rank not in {2, 3} or a.dtype not in {'fp16', 'bf16', 'fp32'} or b.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('fused multiply-sum requires rank-two or rank-three native floating tensors')\n"
           "    if output.dtype != _promote(a.dtype, b.dtype):\n"
           "        raise ValueError('fused multiply-sum requires the promoted product dtype')\n"
           "    one = _constant_expression(1)\n"
           "    aa = (one,) * (rank - len(a.shape)) + a.shape\n"
           "    bb = (one,) * (rank - len(b.shape)) + b.shape\n"
           "    shape = tuple(_expression('broadcast', x, y) for x, y in zip(aa, bb, strict=True))\n"
           "    if output.shape != shape[:-2] + (shape[-1],):\n"
           "        raise ValueError('fused multiply-sum requires reduction of the penultimate axis')\n"
           "    physical_shape = (one,) * (3 - rank) + shape\n"
           "    ar, ac = _storage_shape(a)\n"
           "    br, bc = _storage_shape(b)\n"
           "    return {'heads': physical_shape[0], 'reduce': physical_shape[1], 'columns': physical_shape[2], 'rank': 3, 'a_strides': _broadcast_strides(a.shape, physical_shape, a.strides), 'b_strides': _broadcast_strides(b.shape, physical_shape, b.strides), 'a_rows': ar, 'a_columns': ac, 'b_rows': br, 'b_columns': bc, 'a_dtype': _native_dtype(a.dtype), 'b_dtype': _native_dtype(b.dtype), 'dtype': _native_dtype(output.dtype)}\n"
           "\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Sum(Mul(a, b), axes=(-2,)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_reduce_sum_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"reduce", "size_t", offsetof(Parameters, reduce), NULL},
        {"columns", "size_t", offsetof(Parameters, columns), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"},
        {"a_rows", "size_t", offsetof(Parameters, a_rows), NULL},
        {"a_columns", "size_t", offsetof(Parameters, a_columns), NULL},
        {"b_rows", "size_t", offsetof(Parameters, b_rows), NULL},
        {"b_columns", "size_t", offsetof(Parameters, b_columns), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, b_dtype), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 13, fields, 2, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_reduce_sum_mul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        /* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
        0
#else
        TRILLIM_CAP_NEON
#endif
    };
    return &capabilities;
}
