#include "common.h"
#include <math.h>
#include <arm_neon.h>

typedef struct {
    size_t rows, width;
    uint32_t dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    float *scratch;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || !p->width ||
        p->rows > SIZE_MAX / sizeof(float) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    *scratch = p->rows * p->width * sizeof(float);
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


static float32x4_t exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
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

static TrillimStatus softmax_precise(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float maximum = -INFINITY;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->dtype).f;
        maximum = maximum >= value ? maximum : value;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float difference = round_dtype(trillim_load(task->input, start + column, p->dtype).f - maximum, p->dtype);
        float numerator = round_dtype(expf(difference), p->dtype);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    sum = round_dtype(sum, p->dtype);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}


static TrillimStatus softmax(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve low-precision rounding at scalar-reference boundaries. */
        return softmax_precise(opaque, row, worker);
    }
    size_t start = row * p->width, column = 0;
    float32x4_t maximum4 = vdupq_n_f32(-INFINITY);
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = load4(task->input, start + column, p->dtype);
        if (vminvq_u32(vceqq_f32(value, value)) != UINT32_MAX) {
            return softmax_precise(opaque, row, worker);
        }
        maximum4 = vmaxnmq_f32(maximum4, value);
    }
    float maximum = vmaxnmvq_f32(maximum4);
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->dtype).f;
        if (isnan(value)) {
            return softmax_precise(opaque, row, worker);
        }
        maximum = maximum >= value ? maximum : value;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float32x4_t sum4 = vdupq_n_f32(0.0f), maximum_vector = vdupq_n_f32(maximum);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t numerator = exp4(vsubq_f32(load4(task->input, start + column, p->dtype),
                                              maximum_vector));
        vst1q_f32(task->scratch + start + column, numerator);
        sum4 = vaddq_f32(sum4, numerator);
    }
    float sum = vaddvq_f32(sum4);
    for (; column < p->width; ++column) {
        float numerator = expf(trillim_load(task->input, start + column, p->dtype).f - maximum);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    float denominator = sum > 0.0f ? sum : 1.0f;
    float32x4_t denominator_vector = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, start + column, p->dtype,
               vdivq_f32(vld1q_f32(task->scratch + start + column), denominator_vector));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_normalize_softmax(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], scratch, p, context};
    if (p->rows == 1) {
        return softmax(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     softmax, &task);
}

TRILLIM_KERNEL const char *trillim_neon_normalize_softmax_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('Softmax requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('Softmax requires matching floating input and output')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Softmax requires contiguous input')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'dtype': _native_dtype(x.dtype)}\n"
           "def pattern(x: Tensor, *, axis: int = -1) -> tuple[Tensor, ...]:\n"
           "    return (Softmax(x, axis=axis),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_normalize_softmax_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_normalize_softmax_capabilities(void) {
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
