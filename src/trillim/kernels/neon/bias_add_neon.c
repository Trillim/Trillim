#include "common.h"
#include <arm_neon.h>
#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif
typedef struct {
    size_t rows;
    size_t width;
    uint32_t input_dtype;
    uint32_t bias_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input, *bias;
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
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype))
        return TRILLIM_ERROR;
    inputs[1] = 0;
    if (!trillim_float(p->bias_dtype) || !p->width || p->width > SIZE_MAX / 4 ||
        p->rows > SIZE_MAX / 4 / p->width)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t begin = tile * p->width, i = 0;
    for (; p->width - i >= 4; i += 4) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        store4(t->output, begin + i, p->output_dtype,
               vaddq_f32(load4(t->input, begin + i, p->input_dtype), load4(t->bias, i, p->bias_dtype)));
    }
    for (; i < p->width; ++i) {
        TrillimValue value = {.f = trillim_load(t->input, begin + i, p->input_dtype).f +
                                   trillim_load(t->bias, i, p->bias_dtype).f};
        trillim_store(t->output, begin + i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_neon_bias_add(void *const *outputs, const void *const *inputs, void *scratch,
                                                       const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->rows;
    return tiles == 1 ? apply(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_neon_bias_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _product, _contiguous_strides, _DTYPE_IDS\n"
           "def floating(*values):\n"
           "    for v in values:\n"
           "        if v.dtype not in ('fp16', 'bf16', 'fp32') or (v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "def bind(c):\n"
           "    a,b,o=c.inputs['a'],c.inputs['b'],c.outputs[0]\n"
           "    floating(a,b,o)\n"
           "    if not a.shape or b.shape != a.shape[-1:] or o.shape != a.shape:\n"
           "        raise ValueError('bias must match the final channel axis')\n"
           "    return dict(rows=_product(a.shape[:-1]),width=a.shape[-1],input_dtype=_DTYPE_IDS[a.dtype],bias_dtype=_DTYPE_IDS[b.dtype],output_dtype=_DTYPE_IDS[o.dtype])\n"
           "def pattern(a: Tensor,b: Tensor,*,dtype: DType | None = None) -> tuple[Tensor,...]:\n"
           "    return (Add(a,b) if dtype is None else Cast(Add(a,b),dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bias_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bias_add_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED};
    return &caps;
}
