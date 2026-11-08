#include "matmul.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return trillim_matmul_compute(outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_linear_parameters(void) {
    return trillim_matmul_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
