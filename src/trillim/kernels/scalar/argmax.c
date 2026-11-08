#include "reduction.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_argmax(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_reduction_compute(TRILLIM_ARGMAX, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_argmax_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _argmax as bind\n"
           "def pattern(x: Tensor, *, axis: int = -1, rank: int) -> tuple[Tensor, ...]:\n"
           "    return (Argmax(x, axis=axis, rank=rank),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_argmax_parameters(void) {
    return trillim_reduction_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_argmax_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
