#include "common.h"
#include <arm_neon.h>
#include <float.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


typedef struct {
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, output_dtype;
    float eps, weight_offset;
} Parameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input, *weight;
    void *normalized;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

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

static inline void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index, vreinterpret_u16_f16(vcvt_f16_f32(value)));
        return;
    }
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = outputs[1] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP32) ||
        (p->weight_dtype != TRILLIM_BF16 && p->weight_dtype != TRILLIM_FP32) ||
        (p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32) ||
        !p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width) return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    outputs[0] = p->rows * p->width;
    outputs[1] = p->rows * sizeof(float);
    *scratch = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float32x4_t sum0 = vdupq_n_f32(0), sum1 = vdupq_n_f32(0);
    size_t column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float32x4_t a = load4(task->input, start + column, p->input_dtype);
        float32x4_t b = load4(task->input, start + column + 4, p->input_dtype);
        sum0 = vfmaq_f32(sum0, a, a);
        sum1 = vfmaq_f32(sum1, b, b);
    }
    /* Match the scalar fused kernel's eight accumulators before rounding codes. */
    float lanes[8];
    vst1q_f32(lanes, sum0);
    vst1q_f32(lanes + 4, sum1);
    float total = 0;
    for (size_t lane = 0; lane < 8; ++lane) total += lanes[lane];
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        total += value * value;
    }
    float inverse = 1.0f / sqrtf(total / (float)p->width + p->eps);
    float32x4_t multiplier = vdupq_n_f32(inverse), offset = vdupq_n_f32(p->weight_offset);
    float32x4_t maximum = vdupq_n_f32(1e-5f);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float32x4_t value = vmulq_f32(vmulq_f32(load4(task->input, start + column, p->input_dtype), multiplier),
            vaddq_f32(load4(task->weight, column, p->weight_dtype), offset));
        store4(task->normalized, start + column, p->output_dtype, value);
        value = load4(task->normalized, start + column, p->output_dtype);
        float32x4_t magnitude = vabsq_f32(value);
        if (vminvq_u32(vcleq_f32(magnitude, vdupq_n_f32(FLT_MAX))) != UINT32_MAX) return TRILLIM_ERROR;
        maximum = vmaxq_f32(maximum, magnitude);
    }
    float absmax = vmaxvq_f32(maximum);
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = trillim_load(task->input, start + column, p->input_dtype).f * inverse *
            (trillim_load(task->weight, column, p->weight_dtype).f + p->weight_offset)};
        trillim_store(task->normalized, start + column, p->output_dtype, value);
        float rounded = trillim_load(task->normalized, start + column, p->output_dtype).f;
        if (!isfinite(rounded)) return TRILLIM_ERROR;
        absmax = fmaxf(absmax, fabsf(rounded));
    }
    float scale = 127.0f / absmax;
    task->scales[row] = scale;
    multiplier = vdupq_n_f32(scale);
    column = 0;
    for (; p->width - column >= 16; column += 16) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        int32x4_t codes[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            float32x4_t scaled = vmulq_f32(load4(task->normalized, start + column + lane * 4, p->output_dtype), multiplier);
            scaled = vminq_f32(vmaxq_f32(scaled, vdupq_n_f32(-127)), vdupq_n_f32(127));
            codes[lane] = vcvtnq_s32_f32(scaled);
        }
        int16x8_t low = vcombine_s16(vmovn_s32(codes[0]), vmovn_s32(codes[1]));
        int16x8_t high = vcombine_s16(vmovn_s32(codes[2]), vmovn_s32(codes[3]));
        vst1q_s8(task->codes + start + column, vcombine_s8(vmovn_s16(low), vmovn_s16(high)));
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->normalized, start + column, p->output_dtype).f * scale;
        float magnitude = fminf(fabsf(value), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) ++code;
        task->codes[start + column] = (int8_t)(value < 0 ? -code : code);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_rmsnorm_quantize(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], outputs[1], inputs[0], inputs[1], scratch, p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_neon_rmsnorm_quantize_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm_quantize as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_rmsnorm_quantize_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 2, 2,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_rmsnorm_quantize_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
