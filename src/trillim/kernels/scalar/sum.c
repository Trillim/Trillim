#include "reduction.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_sum(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_reduction_compute(TRILLIM_SUM, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_sum_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import reduction as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> tuple[Tensor, ...]:\n"
           "    return (Sum(x, axes=axes, keepdims=keepdims),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_sum_parameters(void) {
    return trillim_reduction_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_sum_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
