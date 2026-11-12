#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t count;
    uint32_t input_dtype, output_dtype;
} TrillimActivationParameters;

typedef struct {
    void *output;
    const void *gate;
    const void *up;
    const TrillimActivationParameters *parameters;
    const TrillimKernelContext *context;
} TrillimActivationTask;

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

static inline float32x4_t exp_negative(float32x4_t x) {
    float32x4_t original = x;
    x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(x, vdupq_n_f32(0)));
    float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(0.00021219444005469058f));
    /* Range reduction bounds |r| by ln(2)/2 for the degree-seven polynomial. */
    float32x4_t value = vdupq_n_f32(1.0f / 5040.0f);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(0.5f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    int32x4_t exponent = vcvtq_s32_f32(n);
    int32x4_t low = vminq_s32(vaddq_s32(exponent, vdupq_n_s32(126)), vdupq_n_s32(0));
    int32x4_t high = vsubq_s32(exponent, low);
    float32x4_t main_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(high, vdupq_n_s32(127)), 23));
    float32x4_t tail_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(low, vdupq_n_s32(127)), 23));
    value = vmulq_f32(vmulq_f32(value, main_scale), tail_scale);
    return vbslq_f32(vceqq_f32(original, original), value, original);
}

static inline uint32x4_t rounding_sensitive(float32x4_t value, uint32_t dtype) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    if (dtype == TRILLIM_BF16) {
        uint32x4_t fraction = vandq_u32(bits, vdupq_n_u32(0xffff));
        return vcleq_u32(vsubq_u32(fraction, vdupq_n_u32(0x8000 - 16)), vdupq_n_u32(32));
    }
    uint32x4_t fraction = vandq_u32(bits, vdupq_n_u32(0x1fff));
    uint32x4_t tie = vcleq_u32(vsubq_u32(fraction, vdupq_n_u32(0x1000 - 16)), vdupq_n_u32(32));
    uint32x4_t subnormal = vcltq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x38800000));
    return vorrq_u32(tie, subnormal);
}

static inline TrillimStatus trillim_activation_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                            size_t *scratch) {
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->input_dtype), output_bytes = trillim_itemsize(p->output_dtype);
    if ((p->input_dtype != TRILLIM_FP32 && p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP16) ||
        (p->output_dtype != TRILLIM_FP32 && p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP16) ||
        p->count > SIZE_MAX / bytes || p->count > SIZE_MAX / output_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * bytes;
    inputs[1] = inputs[0];
    outputs[0] = p->count * output_bytes;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_activation_task_neon(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimActivationTask *task = (const TrillimActivationTask *)opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 4096;
    size_t end = task->parameters->count - begin < 4096 ? task->parameters->count : begin + 4096;
    size_t i = begin;
    for (; end - i >= 4; i += 4) {
        float32x4_t gate = load4(task->gate, i, task->parameters->input_dtype);
        float32x4_t e = exp_negative(vnegq_f32(vabsq_f32(gate)));
        float32x4_t denominator = vaddq_f32(vdupq_n_f32(1.0f), e);
        float32x4_t numerator = vbslq_f32(vcgeq_f32(gate, vdupq_n_f32(0)), vdupq_n_f32(1.0f), e);
        float32x4_t value = vmulq_f32(gate, vdivq_f32(numerator, denominator));
        value = vmulq_f32(value, load4(task->up, i, task->parameters->input_dtype));
        if (task->parameters->output_dtype != TRILLIM_FP32) {
            uint32x4_t sensitive = rounding_sensitive(value, task->parameters->output_dtype);
            if (vmaxvq_u32(sensitive) != 0) {
                /* Resolve near-ties with libm before rounding to a lower precision. */
                uint32_t mask[4];
                float lanes[4];
                vst1q_u32(mask, sensitive);
                vst1q_f32(lanes, value);
                for (size_t lane = 0; lane < 4; ++lane) {
                    if (mask[lane]) {
                        float x = trillim_load(task->gate, i + lane, task->parameters->input_dtype).f;
                        float e = expf(-fabsf(x));
                        float sigmoid = x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
                        lanes[lane] = (x * sigmoid) * trillim_load(task->up, i + lane, task->parameters->input_dtype).f;
                    }
                }
                value = vld1q_f32(lanes);
            }
        }
        store4(task->output, i, task->parameters->output_dtype, value);
    }
    for (; i < end; ++i) {
        float gate = trillim_load(task->gate, i, task->parameters->input_dtype).f;
        float e = expf(-fabsf(gate));
        float value = gate * (gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
        TrillimValue output = {.f = value * trillim_load(task->up, i, task->parameters->input_dtype).f};
        trillim_store(task->output, i, task->parameters->output_dtype, output);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const TrillimActivationParameters *p = parameters;
    return output == 0 && input < 2 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_silu_mul(void *const *outputs, const void *const *inputs, void *scratch,
                                                       const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    TrillimActivationTask task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = trillim_activation_task_neon(&task, tile, 0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, trillim_activation_task_neon, &task);
}

TRILLIM_KERNEL const char *trillim_neon_silu_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _product, _contiguous_strides, _DTYPE_IDS\n"
           "def floating(*values):\n"
           "    for v in values:\n"
           "        if v.dtype not in ('fp16', 'bf16', 'fp32') or (v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "def bind(c):\n"
           "    a, b, o = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    floating(a,b,o)\n"
           "    if a.shape != b.shape or a.shape != o.shape or a.dtype != b.dtype:\n"
           "        raise ValueError('SiLU gate requires matching floating inputs')\n"
           "    return dict(count=_product(o.shape),input_dtype=_DTYPE_IDS[a.dtype],output_dtype=_DTYPE_IDS[o.dtype])\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (SiluMul(a,b,dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_silu_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimActivationParameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimActivationParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimActivationParameters, output_dtype), NULL}};
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(TrillimActivationParameters),
                                                  TRILLIM_ALIGNOF(TrillimActivationParameters),
                                                  3,
                                                  fields,
                                                  2,
                                                  1,
                                                  trillim_activation_requirements,
                                                  can_alias,
                                                  inputs,
                                                  outputs,
                                                  1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_silu_mul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
