#include "cast.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_reshape(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return trillim_cast_compute(outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_reshape_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import reshape as bind\n"
           "physical_inputs = ('x',)\n"
           "def pattern(x: Tensor, *, shape) -> tuple[Tensor, ...]:\n"
           "    return (Reshape(x, shape),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_reshape_parameters(void) {
    return trillim_cast_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_reshape_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
