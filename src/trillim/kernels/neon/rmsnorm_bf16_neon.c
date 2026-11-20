#include "common.h"

extern TrillimStatus trillim_neon_rmsnorm(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimCapabilities *trillim_neon_rmsnorm_capabilities(void);

typedef struct {
    size_t rows, width;
    float eps, weight_offset;
} Parameters;

/* Parameters for trillim_neon_rmsnorm. */
typedef struct {
    size_t rows, width, row_stride, column_stride, storage_rows, storage_columns;
    uint32_t input_dtype, weight_dtype, output_dtype;
    float eps, weight_offset;
} RMSNormParameters;

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / sizeof(uint16_t) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->width * sizeof(uint16_t);
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rmsnorm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    const Parameters *p = opaque;
    const RMSNormParameters parameters = {
        .rows = p->rows, .width = p->width, .row_stride = p->width, .column_stride = 1,
        .storage_rows = p->rows, .storage_columns = p->width,
        .input_dtype = TRILLIM_BF16, .weight_dtype = TRILLIM_BF16, .output_dtype = TRILLIM_BF16,
        .eps = p->eps, .weight_offset = p->weight_offset
    };
    return trillim_neon_rmsnorm(outputs, inputs, scratch, &parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rmsnorm_tdl(void) {
    return "from trillim.tdl import DType, RMSNorm, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or output.shape != x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('BF16 RMSNorm requires a matching channel vector')\n"
           "    for value in (x, weight, output):\n"
           "        if value.dtype != 'bf16' or (value.strides and value.strides != _contiguous_strides(value.shape)):\n"
           "            raise ValueError('BF16 RMSNorm requires contiguous BF16 tensors')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    return dict(rows=_product(x.shape[:-1]), width=x.shape[-1], eps=float(eps), weight_offset=float(offset))\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rmsnorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rmsnorm_capabilities(void) {
    return trillim_neon_rmsnorm_capabilities();
}
