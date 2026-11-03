#include "elementwise.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_div(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_binary_compute(TRILLIM_DIV, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_div_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import binary as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Div(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_div_parameters(void) {
    return trillim_binary_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_div_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
