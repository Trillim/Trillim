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
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
    float eps;
} Parameters;
typedef struct {
    void *output;
    const void *input, *weight, *bias;
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
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    uint32_t types[] = {p->input_dtype, p->weight_dtype, p->bias_dtype, p->output_dtype};
    for (size_t i = 0; i < 4; ++i)
        if (!trillim_float(types[i])) return TRILLIM_ERROR;
    if (!p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width || !isfinite(p->eps) || p->eps < 0)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    inputs[2] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t offset = row * p->width;
    float mean = 0.0f, variance = 0.0f;
    float32x4_t center;
    if (p->output_dtype != TRILLIM_FP32) {
        for (size_t i = 0; i < p->width; ++i) {
            if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            mean += trillim_load(t->input, offset + i, p->input_dtype).f;
        }
        mean /= (float)p->width;
        center = vdupq_n_f32(mean);
        for (size_t i = 0; i < p->width; ++i) {
            if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            float value = trillim_load(t->input, offset + i, p->input_dtype).f - mean;
            variance += value * value;
        }
    } else {
        float32x4_t total = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; p->width - i >= 4; i += 4) {
            if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            total = vaddq_f32(total, load4(t->input, offset + i, p->input_dtype));
        }
        mean = vaddvq_f32(total);
        for (; i < p->width; ++i)
            mean += trillim_load(t->input, offset + i, p->input_dtype).f;
        mean /= (float)p->width;
        center = vdupq_n_f32(mean);
        total = vdupq_n_f32(0.0f);
        i = 0;
        for (; p->width - i >= 4; i += 4) {
            if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            float32x4_t v = vsubq_f32(load4(t->input, offset + i, p->input_dtype), center);
            total = vfmaq_f32(total, v, v);
        }
        variance = vaddvq_f32(total);
        for (; i < p->width; ++i) {
            float v = trillim_load(t->input, offset + i, p->input_dtype).f - mean;
            variance += v * v;
        }
    }
    size_t i = 0;
    float inv = 1.0f / sqrtf(variance / (float)p->width + p->eps);
    float32x4_t inverse = vdupq_n_f32(inv);
    i = 0;
    for (; p->width - i >= 4; i += 4) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        float32x4_t v = vmulq_f32(vsubq_f32(load4(t->input, offset + i, p->input_dtype), center), inverse);
        v = vfmaq_f32(load4(t->bias, i, p->bias_dtype), v, load4(t->weight, i, p->weight_dtype));
        store4(t->output, offset + i, p->output_dtype, v);
    }
    for (; i < p->width; ++i) {
        TrillimValue v = {.f = (trillim_load(t->input, offset + i, p->input_dtype).f - mean) * inv *
                                   trillim_load(t->weight, i, p->weight_dtype).f +
                               trillim_load(t->bias, i, p->bias_dtype).f};
        trillim_store(t->output, offset + i, p->output_dtype, v);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_neon_layernorm(void *const *outputs, const void *const *inputs, void *scratch,
                                                        const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0)
                        : context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}
TRILLIM_KERNEL const char *trillim_neon_layernorm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _product, _contiguous_strides, _DTYPE_IDS\n"
           "def floating(*values):\n"
           "    for v in values:\n"
           "        if v.dtype not in ('fp16', 'bf16', 'fp32') or (v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm\n"
           "def bind(c):\n"
           "    x,w,b,o=c.inputs['x'],c.inputs['weight'],c.inputs['bias'],c.outputs[0]\n"
           "    floating(x,w,b,o)\n"
           "    if c.attrs.get('axes',(-1,)) != (-1,) or not x.shape or b.shape != x.shape[-1:] or o.shape != x.shape:\n"
           "        raise ValueError('LayerNorm requires a final channel axis and matching affine vectors')\n"
           "    p=_rmsnorm(c)\n"
           "    return {k:v for k,v in p.items() if k not in ('storage_rows','storage_columns','weight_offset')} | dict(bias_dtype=_DTYPE_IDS[b.dtype])\n"
           "def pattern(x: Tensor,weight: Tensor,bias: Tensor,*,axes: tuple[int,...] = (-1,),eps: float = 1e-5,dtype: DType | None = None) -> tuple[Tensor,...]:\n"
           "    return (LayerNorm(x,weight=weight,bias=bias,axes=axes,eps=eps) if dtype is None else Cast(LayerNorm(x,weight=weight,bias=bias,axes=axes,eps=eps),dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_layernorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 3, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_layernorm_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
