#include "common.h"
#include <math.h>
#include <arm_neon.h>

typedef struct {
    size_t count;
    uint32_t dtype;
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
    if (!trillim_float(p->dtype) || p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->dtype);
    outputs[0] = p->count * trillim_itemsize(TRILLIM_FP32);
    return TRILLIM_OK;
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

static float sigmoid(float value, uint32_t dtype) {
    float e = round_dtype(expf(-fabsf(value)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    return value >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
}

static float apply(float value, uint32_t dtype) {
    return sigmoid(value, dtype);
}

static float32x4_t exp_reduced4(float32x4_t x) {
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
    return exp_reduced4(x);
}

static bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return (fpcr & UINT64_C(0x07c80003)) == 0;
}

static float32x4_t round_bf16_4(float32x4_t value) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
        vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
    return vreinterpretq_f32_u32(vandq_u32(vaddq_u32(bits, bias), vdupq_n_u32(0xffff0000)));
}

static float32x4_t sigmoid_bf16_4(uint16x4_t bits) {
    float32x4_t value = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
    uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(value), vdupq_n_u32(0x7fffffff));
    if (vminvq_u32(vcltq_u32(magnitude, vdupq_n_u32(0x7f800000))) != UINT32_MAX) {
        float lanes[4];
        vst1q_f32(lanes, value);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = sigmoid(lanes[lane], TRILLIM_BF16);
        }
        return vld1q_f32(lanes);
    }
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(value)));
    uint32x4_t low = vandq_u32(vreinterpretq_u32_f32(e), vdupq_n_u32(0xffff));
    uint32x4_t near = vandq_u32(vcgeq_u32(low, vdupq_n_u32(0x7fc0)),
        vcleq_u32(low, vdupq_n_u32(0x8040)));
    if (vmaxvq_u32(near) != 0) {
        /* Refine exponentials close to a BF16 rounding midpoint. */
        float lanes[4], inputs[4];
        uint32_t mask[4];
        vst1q_f32(lanes, e);
        vst1q_f32(inputs, value);
        vst1q_u32(mask, near);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (mask[lane] != 0) {
                lanes[lane] = expf(-fabsf(inputs[lane]));
            }
        }
        e = vld1q_f32(lanes);
    }
    e = round_bf16_4(e);
    float32x4_t denominator = round_bf16_4(vaddq_f32(e, vdupq_n_f32(1.0f)));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = round_bf16_4(vdivq_f32(e, denominator));
    return vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0f)), positive, negative);
}

static float32x4_t sigmoid_fp16_4(uint16x4_t bits) {
    float32x4_t value = vcvt_f32_f16(vreinterpret_f16_u16(bits));
    float32x4_t x = vnegq_f32(vabsq_f32(value));
    /* Below -18, exp(x) rounds to zero in FP16. */
    float32x4_t e = exp_reduced4(vmaxnmq_f32(x, vdupq_n_f32(-18.0f)));

    /* Check both sides of the approximation, including half subnormal midpoints. */
    uint32x4_t approximation = vreinterpretq_u32_f32(e);
    uint32x4_t margin = vdupq_n_u32(8);
    uint16x4_t lo = vreinterpret_u16_f16(vcvt_f16_f32(
        vreinterpretq_f32_u32(vsubq_u32(approximation, margin))));
    uint16x4_t hi = vreinterpret_u16_f16(vcvt_f16_f32(
        vreinterpretq_f32_u32(vaddq_u32(approximation, margin))));
    uint16x4_t special = vcgt_u16(vand_u16(bits, vdup_n_u16(0x7fff)), vdup_n_u16(0x7c00));
    uint16x4_t refine = vorr_u16(vmvn_u16(vceq_u16(lo, hi)), special);
    e = vcvt_f32_f16(vcvt_f16_f32(e));
    float32x4_t denominator = vcvt_f32_f16(vcvt_f16_f32(vaddq_f32(e, vdupq_n_f32(1.0f))));
    uint32x4_t positive = vcgeq_f32(value, vdupq_n_f32(0.0f));
    float32x4_t quotient = vdivq_f32(vbslq_f32(positive, vdupq_n_f32(1.0f), e), denominator);
    float32x4_t result = vbslq_f32(positive, quotient, vcvt_f32_f16(vcvt_f16_f32(quotient)));
    if (vmaxv_u16(refine)) {
        float lanes[4];
        uint16_t input[4], mask[4];
        vst1q_f32(lanes, result);
        vst1_u16(input, bits);
        vst1_u16(mask, refine);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (mask[lane]) {
                lanes[lane] = sigmoid(trillim_fp16(input[lane]), TRILLIM_FP16);
            }
        }
        result = vld1q_f32(lanes);
    }
    return result;
}

static float32x4_t sigmoid4(float32x4_t value) {
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(value)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    return vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0f)), positive, negative);
}

static float32x4_t apply4(float32x4_t value) {
    return sigmoid4(value);
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;
    if (p->dtype == TRILLIM_FP32) {
        for (; end - i >= 4; i += 4) {
            vst1q_f32((float *)task->output + i,
                       apply4(vld1q_f32((const float *)task->input + i)));
        }
    }
    if (p->dtype == TRILLIM_BF16 && default_fpcr()) {
        for (; end - i >= 4; i += 4) {
            vst1q_f32((float *)task->output + i,
                sigmoid_bf16_4(vld1_u16((const uint16_t *)task->input + i)));
        }
    }
    if (p->dtype == TRILLIM_FP16 && default_fpcr()) {
        for (; end - i >= 4; i += 4) {
            vst1q_f32((float *)task->output + i,
                sigmoid_fp16_4(vld1_u16((const uint16_t *)task->input + i)));
        }
    }
    for (; i < end; ++i) {
        TrillimValue value = {.f = apply(trillim_load(task->input, i, p->dtype).f, p->dtype)};
        trillim_store(task->output, i, TRILLIM_FP32, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_activation_sigmoid(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_activation_sigmoid_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != 'fp32' or output.shape != x.shape:\n"
           "        raise ValueError('Sigmoid requires native floating tensors')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Sigmoid requires contiguous input')\n"
           "    return {'count': _product(x.shape), 'dtype': _native_dtype(x.dtype)}\n"
           "\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Sigmoid(x),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_activation_sigmoid_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_activation_sigmoid_capabilities(void) {
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
