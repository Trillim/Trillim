#include "common.h"

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear_bias(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

#if defined(__APPLE__) || defined(__linux__)
extern TrillimStatus trillim_neon_try_packed_linear_bias_bf16(
    void *, const void *, const void *, const void *, const void *, const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_linear_bias_bf16")
#endif
;
#endif

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} UnbiasedParameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, output_dtype, weight_dtype, bias_dtype;
} LinearParameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 3 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width < 256 || p->width % 32 != 0 ||
        (p->weight_dtype != TRILLIM_GRP_INT4 && p->weight_dtype != TRILLIM_GRP_INT8) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, inputs + 1) ||
        !trillim_packed_payload_size(1, p->outputs, TRILLIM_BF16, inputs + 2) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_BF16, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_grouped_linear_bias_bf16(
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
#if defined(__APPLE__) || defined(__linux__)
    if (p->tokens >= 4) {
        const UnbiasedParameters unbiased = {
            p->tokens, p->width, p->outputs, TRILLIM_BF16, p->weight_dtype, TRILLIM_FP32
        };
        bool handled;
        TrillimStatus status = trillim_neon_try_packed_linear_bias_bf16(
            outputs[0], inputs[0], inputs[1], inputs[2], &unbiased, context, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
#endif

    void *projected[] = {scratch};
    const LinearParameters projection = {p->tokens, p->width, p->outputs, TRILLIM_BF16,
        TRILLIM_FP32, p->weight_dtype, TRILLIM_BF16};
    const TrillimStatus status = trillim_neon_packed_linear_bias(projected, inputs, NULL, &projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->outputs, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {scratch};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_grouped_linear_bias_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'weight', 'bias')\n"
           "def bind(c):\n"
           "    x, weight = (c.inputs[name] for name in ('x', 'weight'))\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16':\n"
           "        raise ValueError('requires BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if len(weight.shape) != 2 or weight.dtype not in {'grp_int4', 'grp_int8'} or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete Q4 or Q8 matrix')\n"
           "    bias = c.inputs['bias']\n"
           "    if bias.dtype != 'bf16' or bias.shape != weight.shape[:1]:\n"
           "        raise ValueError('requires a matching BF16 bias')\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or width.value < 256 or width.value % 32:\n"
           "        raise ValueError('unaligned grouped matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, outputs=outputs, weight_dtype=_DTYPE_IDS[weight.dtype])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if bias is None or dtype != 'bf16':\n"
           "        raise ValueError('requires a BF16 projection cast')\n"
           "    return (Cast(Linear(x, weight, bias=bias), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_grouped_linear_bias_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_grouped_linear_bias_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
