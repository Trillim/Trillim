#include "elementwise.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_sin(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_unary_compute(TRILLIM_SIN, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_sin_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _unary as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Sin(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_sin_parameters(void) {
    return trillim_unary_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_sin_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
