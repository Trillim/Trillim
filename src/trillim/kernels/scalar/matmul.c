#include "matmul.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return trillim_matmul_compute(outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _matmul as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_matmul_parameters(void) {
    return trillim_matmul_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
