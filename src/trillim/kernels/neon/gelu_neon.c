#include "common.h"
#include <math.h>
#include <arm_neon.h>
#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input;
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

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype))
        return TRILLIM_ERROR;
    if (p->count > SIZE_MAX / 4) return TRILLIM_ERROR;
    inputs[0] = p->count * trillim_itemsize(p->input_dtype);
    outputs[0] = p->count * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t begin = tile * 4096, end = p->count - begin < 4096 ? p->count : begin + 4096;
    size_t i = begin;
    if (p->output_dtype != TRILLIM_FP32) {
        for (; end - i >= 4; i += 4) {
            float lanes[4];
            vst1q_f32(lanes, load4(t->input, i, p->input_dtype));
            for (size_t lane = 0; lane < 4; ++lane) {
                float x = lanes[lane];
                lanes[lane] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
            }
            store4(t->output, i, p->output_dtype, vld1q_f32(lanes));
        }
        for (; i < end; ++i) {
            float x = trillim_load(t->input, i, p->input_dtype).f;
            TrillimValue value = {.f = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f))};
            trillim_store(t->output, i, p->output_dtype, value);
        }
        return TRILLIM_OK;
    }
    for (; end - i >= 4; i += 4) {
        float32x4_t x = load4(t->input, i, p->input_dtype);
        uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000));
        float32x4_t z = vmulq_n_f32(vabsq_f32(x), 0.7071067811865476f);
        float32x4_t u = vdivq_f32(vdupq_n_f32(1), vfmaq_f32(vdupq_n_f32(1), z, vdupq_n_f32(0.3275911f)));
        /* Erf approximation has absolute error below 1.5e-7 before FP32 rounding. */
        float32x4_t value = vdupq_n_f32(1.061405429f);
        value = vfmaq_f32(vdupq_n_f32(-1.453152027f), value, u);
        value = vfmaq_f32(vdupq_n_f32(1.421413741f), value, u);
        value = vfmaq_f32(vdupq_n_f32(-0.284496736f), value, u);
        value = vfmaq_f32(vdupq_n_f32(0.254829592f), value, u);
        float32x4_t exponent = exp_negative(vnegq_f32(vmulq_f32(z, z)));
        value = vfmsq_f32(vdupq_n_f32(1), vmulq_f32(value, u), exponent);
        value = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(value), sign));
        value = vaddq_f32(vdupq_n_f32(1), value);
        store4(t->output, i, p->output_dtype, vmulq_f32(vmulq_n_f32(x, 0.5f), value));
    }
    for (; i < end; ++i) {

        float x = trillim_load(t->input, i, p->input_dtype).f;
        TrillimValue value = {.f = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f))};
        trillim_store(t->output, i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_neon_gelu(void *const *outputs, const void *const *inputs, void *scratch,
                                                   const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    size_t serial_limit = p->output_dtype == TRILLIM_FP32 ? 16384 : 4096;
    if (p->count <= serial_limit) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_neon_gelu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _product, _contiguous_strides, _DTYPE_IDS\n"
           "def floating(*values):\n"
           "    for v in values:\n"
           "        if v.dtype not in ('fp16', 'bf16', 'fp32') or (v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "def bind(c):\n"
           "    x,o=c.inputs['x'],c.outputs[0]\n"
           "    floating(x,o)\n"
           "    if x.shape != o.shape or c.attrs.get('approximate','none') != 'none':\n"
           "        raise ValueError('GELU requires matching shapes and the erf definition')\n"
           "    return dict(count=_product(o.shape),input_dtype=_DTYPE_IDS[x.dtype],output_dtype=_DTYPE_IDS[o.dtype])\n"
           "def pattern(x: Tensor,*,approximate: str = 'none',dtype: DType | None = None) -> tuple[Tensor,...]:\n"
           "    return (Gelu(x,approximate=approximate) if dtype is None else Cast(Gelu(x,approximate=approximate),dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gelu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gelu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION,
                                             TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &caps;
}
