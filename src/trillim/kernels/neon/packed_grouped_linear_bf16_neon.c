#include "common.h"

extern TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_packed_linear_parameters(void);
extern TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype;
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

static bool grouped_weight(uint32_t dtype) {
    return dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT8;
}

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    if (p->width == 0 || p->width % 32 != 0 || !grouped_weight(p->weight_dtype)) {
        return TRILLIM_ERROR;
    }
    const LinearParameters projection = {p->tokens, p->width, p->outputs, TRILLIM_BF16,
        p->weight_dtype, TRILLIM_FP32};
    size_t projected, child_scratch;
    const TrillimStatus status = trillim_neon_packed_linear_parameters()->requirements(
        &projection, inputs, &projected, &child_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (child_scratch != 0) {
        return TRILLIM_ERROR;
    }
    outputs[0] = projected / 2;
    *scratch = projected;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_grouped_linear_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = opaque;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    const LinearParameters projection = {p->tokens, p->width, p->outputs, TRILLIM_BF16,
        p->weight_dtype, TRILLIM_FP32};
    void *projected[] = {scratch};
    const TrillimStatus status = trillim_neon_packed_linear(projected, inputs, NULL, &projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->outputs, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {scratch};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_grouped_linear_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_grouped_linear_bf16_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}


TRILLIM_KERNEL const char *trillim_neon_packed_grouped_linear_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "\n"
           "physical_inputs = LINEAR_INPUTS = (\"x\", \"weight\")\n"
           "LINEAR_FORMATS = {\"grp_int4\", \"grp_int8\"}\n"
           "\n"
           "def contiguous(*bindings) -> None:\n"
           "    for binding in bindings:\n"
           "        stride = _contiguous_strides(binding.shape)\n"
           "        if (binding.strides or stride) != stride:\n"
           "            raise ValueError(\"requires contiguous operands\")\n"
           "\n"
           "def bind(context):\n"
           "    x, weight = (context.inputs[name] for name in LINEAR_INPUTS)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != \"bf16\" or output.dtype != \"bf16\":\n"
           "        raise ValueError(\"requires BF16 input and output\")\n"
           "    if len(weight.shape) != 2 or weight.dtype not in LINEAR_FORMATS or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError(\"requires a complete Q4 or Q8 matrix\")\n"
           "    contiguous(x, weight, output)\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError(\"incompatible dimensions\")\n"
           "    if width.value is None or not width.value or width.value % 32:\n"
           "        raise ValueError(\"requires an aligned quantized input width\")\n"
           "    return {\"tokens\": _product(x.shape[:-1]), \"width\": width, \"outputs\": outputs,\n"
           "            \"weight_dtype\": _DTYPE_IDS[weight.dtype]}\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = \"bf16\") -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype != \"bf16\":\n"
           "        raise ValueError(\"requires unbiased BF16 projection\")\n"
           "    return (Cast(Linear(x, weight), \"bf16\"),)\n";
}
