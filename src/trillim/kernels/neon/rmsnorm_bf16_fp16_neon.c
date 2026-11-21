#include "common.h"

extern TrillimStatus trillim_neon_rmsnorm(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimCapabilities *trillim_neon_rmsnorm_capabilities(void);
extern const TrillimParameterSchema *trillim_neon_bf16_rmsnorm_parameters(void);

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

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rmsnorm_fp16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    const Parameters *p = opaque;
    const RMSNormParameters parameters = {
        .rows = p->rows, .width = p->width, .row_stride = p->width, .column_stride = 1,
        .storage_rows = p->rows, .storage_columns = p->width,
        .input_dtype = TRILLIM_BF16, .weight_dtype = TRILLIM_FP16, .output_dtype = TRILLIM_BF16,
        .eps = p->eps, .weight_offset = p->weight_offset
    };
    return trillim_neon_rmsnorm(outputs, inputs, scratch, &parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rmsnorm_fp16_tdl(void) {
    return "from trillim.tdl import DType, RMSNorm, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or output.shape != x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('BF16 RMSNorm requires matching channels')\n"
           "    if x.dtype != 'bf16' or weight.dtype != 'fp16' or output.dtype != 'bf16':\n"
           "        raise ValueError('RMSNorm requires BF16 activations and FP16 weights')\n"
           "    for value in (x, weight, output):\n"
           "        if value.strides and value.strides != _contiguous_strides(value.shape):\n"
           "            raise ValueError('RMSNorm requires contiguous tensors')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    return dict(rows=_product(x.shape[:-1]), width=x.shape[-1], eps=float(eps), weight_offset=float(offset))\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rmsnorm_fp16_parameters(void) {
    return trillim_neon_bf16_rmsnorm_parameters();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rmsnorm_fp16_capabilities(void) {
    return trillim_neon_rmsnorm_capabilities();
}
