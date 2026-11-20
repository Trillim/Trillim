#include "common.h"

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, outputs;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 2 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->width % 256 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_DUBGRP_INT6, inputs + 1) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_BF16, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_linear_bf16(
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
    const LinearParameters projection = {p->tokens, p->width, p->outputs, TRILLIM_BF16, TRILLIM_DUBGRP_INT6, TRILLIM_FP32};
    void *temporary = scratch;
    void *projected[] = {temporary};
    const TrillimStatus status = trillim_neon_packed_linear(projected, inputs, NULL, &projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->outputs, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {temporary};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_linear_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'weight')\n"
           "def bind(c):\n"
           "    x, weight = (c.inputs[name] for name in ('x', 'weight'))\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16':\n"
           "        raise ValueError('requires BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if len(weight.shape) != 2 or weight.dtype != 'dubgrp_int6' or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete Q6 matrix')\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or not width.value or width.value % 256:\n"
           "        raise ValueError('unaligned Q6 matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, outputs=outputs)\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype != 'bf16':\n"
           "        raise ValueError('requires a BF16 projection cast')\n"
           "    return (Cast(Linear(x, weight), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_linear_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_linear_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
