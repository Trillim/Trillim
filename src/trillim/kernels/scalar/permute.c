#include "cast.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_permute(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return trillim_cast_compute(outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_permute_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _permute as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...]) -> tuple[Tensor, ...]:\n"
           "    return (Permute(x, axes),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_permute_parameters(void) {
    return trillim_cast_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_permute_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
