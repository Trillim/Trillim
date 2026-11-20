#include "common.h"
#include <arm_neon.h>

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern COMPONENT_API TrillimStatus trillim_neon_gelu_tanh_mul(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

#if defined(__APPLE__) || defined(__linux__)
extern TrillimStatus trillim_neon_try_packed_ffn_int8(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8")
#endif
;

extern float32x4_t trillim_neon_gelu_tanh4_prefill(float32x4_t)
#if defined(__linux__)
    __asm__(".trillim_neon_gelu_tanh4_prefill")
#endif
;
extern TrillimStatus trillim_neon_try_packed_ffn_int8_prefill(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    float32x4_t (*)(float32x4_t), const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8_prefill")
#endif
;

static float gelu_tanh(float x) {
    const float cube = (x * x) * x;
    const float correction = cube * 0.044715f;
    const float argument = (x + correction) * 0.7978845608028654f;
    const float scaled = argument * 2.0f;
    const float e = expf(-fabsf(scaled));
    const float denominator = e + 1.0f;
    const float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    const float doubled = probability * 2.0f;
    const float cdf = doubled - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}
#endif

enum { MAX_WIDTH = 65536 };

typedef struct {
    size_t tokens, width, hidden, outputs;
    uint32_t up_dtype, down_dtype;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    const size_t group = p->up_dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    if (p->width < 1024 || p->width > MAX_WIDTH || p->width % group != 0 || p->hidden < 16 ||
        (p->up_dtype != TRILLIM_GRP_INT5 && p->up_dtype != TRILLIM_DUBGRP_INT6 && p->up_dtype != TRILLIM_GRP_INT8) ||
        p->down_dtype < TRILLIM_GRP_INT8 || p->down_dtype > TRILLIM_DUBGRP_INT6 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, p->up_dtype, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, p->down_dtype, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch) ||
        *scratch > SIZE_MAX / 2) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch *= 2;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_quantized_gelu_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    float *hidden = scratch;
    const LinearParameters expand = {p->tokens, p->width, p->hidden, TRILLIM_BF16, p->up_dtype, TRILLIM_FP32};
    bool hidden_handled = false;
#if defined(__APPLE__) || defined(__linux__)
    if (p->tokens == 1 || (p->tokens >= 4 && p->up_dtype == TRILLIM_GRP_INT8)) {
        TrillimStatus status = p->tokens == 1
            ? trillim_neon_try_packed_ffn_int8(
                hidden, inputs[0], inputs[1], inputs[2], &expand, gelu_tanh, context, &hidden_handled)
            : trillim_neon_try_packed_ffn_int8_prefill(
                hidden, inputs[0], inputs[1], inputs[2], &expand, gelu_tanh,
                trillim_neon_gelu_tanh4_prefill, context, &hidden_handled);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
#endif
    if (!hidden_handled) {
        const size_t count = p->tokens * p->hidden;
        float *gate = hidden + count;
        void *up_outputs[] = {hidden}, *gate_outputs[] = {gate};
        const void *up_inputs[] = {inputs[0], inputs[1]}, *gate_inputs[] = {inputs[0], inputs[2]};
        TrillimStatus status = trillim_neon_packed_linear(up_outputs, up_inputs, NULL, &expand, context);
        if (status == TRILLIM_OK) {
            status = trillim_neon_packed_linear(gate_outputs, gate_inputs, NULL, &expand, context);
        }
        if (status == TRILLIM_OK) {
            const void *glu_inputs[] = {hidden, gate};
            status = trillim_neon_gelu_tanh_mul(up_outputs, glu_inputs, NULL, &count, context);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    const void *down_inputs[] = {hidden, inputs[3]};
    const LinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return trillim_neon_packed_linear(outputs, down_inputs, NULL, &down, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_quantized_gelu_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(c):\n"
           "    x, up, gate, down = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or x.dtype != 'bf16' or output.dtype != 'fp32':\n"
           "        raise ValueError('packed GELU FFN requires BF16 input and FP32 output')\n"
           "    for tensor in (x, up, gate, down, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('packed GELU FFN requires contiguous operands')\n"
           "    formats = {'grp_int5', 'dubgrp_int6', 'grp_int8'}\n"
           "    if up.dtype not in formats or gate.dtype != up.dtype or down.dtype not in formats | {'grp_int4'}:\n"
           "        raise ValueError('packed GELU FFN requires compatible grouped weights')\n"
           "    for weight in (up, gate, down):\n"
           "        if len(weight.shape) != 2 or _storage_shape(weight) != weight.shape:\n"
           "            raise ValueError('packed GELU FFN requires complete grouped matrices')\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width or output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed GELU FFN shapes')\n"
           "    group = 256 if up.dtype == 'dubgrp_int6' else 32\n"
           "    if width.value is None or width.value < 1024 or width.value > 65536 or width.value % group or hidden.value is None or hidden.value < 16:\n"
           "        raise ValueError('packed GELU FFN requires aligned fixed-width projections')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden, outputs=down.shape[0], up_dtype=_DTYPE_IDS[up.dtype], down_dtype=_DTYPE_IDS[down.dtype])\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is Silu:\n"
           "        raise ValueError('packed GELU FFN requires an unbiased GELU gate')\n"
           "    return (FFN(x, up=up, gate=gate, down=down, activation=lambda value: Gelu(value, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_quantized_gelu_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_quantized_gelu_ffn_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
