#include "common.h"
#include <math.h>
#include <arm_neon.h>

typedef struct {
    size_t rows, width;
    uint32_t dtype;
    float eps;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || !p->width ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        p->rows > SIZE_MAX / trillim_itemsize(p->dtype) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    return TRILLIM_OK;
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

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
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

static TrillimStatus normalize_precise(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width, column = 0;
    float sum = 0.0f;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t x = load4(task->input, start + column, p->dtype);
        float32x4_t squared = round4(vmulq_f32(x, x), p->dtype);
        sum += vgetq_lane_f32(squared, 0);
        sum += vgetq_lane_f32(squared, 1);
        sum += vgetq_lane_f32(squared, 2);
        sum += vgetq_lane_f32(squared, 3);
    }
    for (; column < p->width; ++column) {
        float x = trillim_load(task->input, start + column, p->dtype).f;
        sum += round_dtype(x * x, p->dtype);
    }
    sum = round_dtype(sum, p->dtype);
    float added = round_dtype(sum + round_dtype(p->eps, p->dtype), p->dtype);
    float denominator = round_dtype(sqrtf(added), p->dtype);
    float32x4_t denominator4 = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, start + column, p->dtype,
               vdivq_f32(load4(task->input, start + column, p->dtype), denominator4));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {
            .f = trillim_load(task->input, start + column, p->dtype).f / denominator
        };
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve each primitive dtype boundary and its reduction order. */
        return normalize_precise(opaque, row, worker);
    }
    size_t start = row * p->width, column = 0;
    float32x4_t sums0 = vdupq_n_f32(0.0f), sums1 = sums0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t a = load4(task->input, start + column, p->dtype);
        float32x4_t b = load4(task->input, start + column + 4, p->dtype);
        sums0 = vfmaq_f32(sums0, a, a);
        sums1 = vfmaq_f32(sums1, b, b);
    }
    float sum = vaddvq_f32(vaddq_f32(sums0, sums1));
    for (; column < p->width; ++column) {
        float x = trillim_load(task->input, start + column, p->dtype).f;
        sum += x * x;
    }
    float denominator = sqrtf(sum + p->eps);
    float32x4_t denominator4 = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, start + column, p->dtype,
               vdivq_f32(load4(task->input, start + column, p->dtype), denominator4));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {
            .f = trillim_load(task->input, start + column, p->dtype).f / denominator
        };
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_l2norm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows == 1) {
        return normalize(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     normalize, &task);
}

TRILLIM_KERNEL const char *trillim_neon_l2norm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('L2Norm requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('L2Norm requires matching floating input and output')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('L2Norm requires contiguous input')\n"
           "    eps = context.attrs.get('eps', 1e-6)\n"
           "    if not isinstance(eps, (int, float)):\n"
           "        raise ValueError('invalid L2Norm epsilon')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'dtype': _native_dtype(x.dtype), 'eps': float(eps)}\n"
           "def pattern(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (L2Norm(x, axis=axis, eps=eps),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_l2norm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_l2norm_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        /* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
        TRILLIM_CAP_LIBM
#else
        TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM
#endif
    };
    return &capabilities;
}
