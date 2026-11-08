#include "elementwise.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_cos(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_unary_compute(TRILLIM_COS, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_cos_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _unary as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Cos(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_cos_parameters(void) {
    return trillim_unary_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_cos_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
