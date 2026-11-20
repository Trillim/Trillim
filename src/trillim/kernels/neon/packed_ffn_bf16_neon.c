#include "common.h"

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

extern COMPONENT_API TrillimStatus trillim_neon_packed_ffn(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, hidden;
} Parameters;

typedef struct {
    size_t tokens, width, outputs, hidden_width;
    uint32_t input_dtype, output_dtype, gate_dtype, up_dtype, down_dtype;
} FfnParameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t hidden_bytes, projection_bytes;
    if (p->width == 0 || p->hidden == 0 || p->width % 256 != 0 || p->hidden % 256 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, TRILLIM_DUBGRP_INT6, inputs + 1) ||
        !trillim_packed_payload_size(p->width, p->hidden, TRILLIM_DUBGRP_INT6, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, &hidden_bytes) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_FP32, &projection_bytes) ||
        hidden_bytes > SIZE_MAX - projection_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch = hidden_bytes + projection_bytes;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_ffn_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    const FfnParameters projection = {p->tokens, p->width, p->width, p->hidden, TRILLIM_BF16,
        TRILLIM_FP32, TRILLIM_DUBGRP_INT6, TRILLIM_DUBGRP_INT6, TRILLIM_DUBGRP_INT6};
    const size_t hidden_bytes = p->tokens * p->hidden * sizeof(float);
    void *temporary = (uint8_t *)scratch + hidden_bytes;
    void *projected[] = {temporary};
    const TrillimStatus status = trillim_neon_packed_ffn(projected, inputs, scratch, &projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->width, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {temporary};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_ffn_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(c):\n"
           "    x, up, gate, down = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16' or output.shape != x.shape:\n"
           "        raise ValueError('requires a matching BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if any(len(v.shape) != 2 or v.dtype != 'dubgrp_int6' or _storage_shape(v) != v.shape for v in (up, gate, down)):\n"
           "        raise ValueError('requires complete Q6 matrices')\n"
           "    hidden, width = up.shape\n"
           "    if up.shape != gate.shape or down.shape != (width, hidden) or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or hidden.value is None or not width.value or not hidden.value or width.value % 256 or hidden.value % 256:\n"
           "        raise ValueError('unaligned Q6 matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden)\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, down: Tensor, gate: Tensor, *, activation, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if activation is not Silu or dtype != 'bf16':\n"
           "        raise ValueError('requires a BF16 SwiGLU cast')\n"
           "    return (Cast(FFN(x, up=up, gate=gate, down=down), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_ffn_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 4, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_ffn_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
