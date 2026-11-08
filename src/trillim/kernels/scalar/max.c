#include "reduction.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_max(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_reduction_compute(TRILLIM_MAX, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_max_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bind_reduction as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> tuple[Tensor, ...]:\n"
           "    return (Max(x, axes=axes, keepdims=keepdims),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_max_parameters(void) {
    return trillim_reduction_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_max_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
