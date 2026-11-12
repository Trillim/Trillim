#include "common.h"
#include <math.h>
#include <float.h>
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


typedef struct {
    size_t tokens;
    size_t width;
    uint32_t input_dtype;
} QuantizeParameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input;
    const QuantizeParameters *parameters;
    const TrillimKernelContext *context;
} QuantizeTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const QuantizeParameters *p = parameters;
    inputs[0] = outputs[0] = outputs[1] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->input_dtype);
    if (!trillim_float(p->input_dtype) || (p->tokens && !p->width) ||
        p->tokens > SIZE_MAX / sizeof(float) ||
        (p->width && p->tokens > SIZE_MAX / bytes / p->width)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * bytes;
    outputs[0] = p->tokens * p->width;
    outputs[1] = p->tokens * sizeof(float);
    return TRILLIM_OK;
}

static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline int32x4_t round4(float32x4_t values, float32x4_t scale) {
    float32x4_t scaled = vmulq_f32(values, scale);
    scaled = vminq_f32(vmaxq_f32(scaled, vdupq_n_f32(-127.0f)), vdupq_n_f32(127.0f));
    return vcvtnq_s32_f32(scaled);
}

static TrillimStatus quantize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const QuantizeTask *task = opaque;
    const QuantizeParameters *p = task->parameters;
    size_t start = row * p->width;
    size_t column = 0;
    float32x4_t maximum4 = vdupq_n_f32(1e-5f);
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t magnitude = vabsq_f32(load4(task->input, start + column, p->input_dtype));
        if (vminvq_u32(vcleq_f32(magnitude, vdupq_n_f32(FLT_MAX))) != UINT32_MAX) {
            return TRILLIM_ERROR;
        }
        maximum4 = vmaxq_f32(maximum4, magnitude);
    }
    float maximum = vmaxvq_f32(maximum4);
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        maximum = fmaxf(maximum, fabsf(value));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    float32x4_t multiplier = vdupq_n_f32(scale);
    column = 0;
    for (; p->width - column >= 16; column += 16) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int32x4_t a = round4(load4(task->input, start + column, p->input_dtype), multiplier);
        int32x4_t b = round4(load4(task->input, start + column + 4, p->input_dtype), multiplier);
        int32x4_t c = round4(load4(task->input, start + column + 8, p->input_dtype), multiplier);
        int32x4_t d = round4(load4(task->input, start + column + 12, p->input_dtype), multiplier);
        int16x8_t low = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        int16x8_t high = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        vst1q_s8(task->codes + start + column, vcombine_s8(vmovn_s16(low), vmovn_s16(high)));
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f * scale;
        float magnitude = fminf(fabsf(value), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[start + column] = (int8_t)(value < 0 ? -code : code);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_quantize_int8(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const QuantizeParameters *p = parameters;
    QuantizeTask task = {outputs[0], outputs[1], inputs[0], p, context};
    return p->tokens == 1 ? quantize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, quantize, &task);
}

TRILLIM_KERNEL const char *trillim_neon_quantize_int8_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantize_int8 as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(x)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_quantize_int8_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(QuantizeParameters, tokens), NULL},
        {"width", "size_t", offsetof(QuantizeParameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(QuantizeParameters, input_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(QuantizeParameters), TRILLIM_ALIGNOF(QuantizeParameters),
        3, fields, 1, 2, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_quantize_int8_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
