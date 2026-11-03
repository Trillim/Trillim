#include "normalization.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_rmsnorm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_rms_compute(true, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_rmsnorm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import rmsnorm as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, weight_offset=weight_offset, eps=eps),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_rmsnorm_parameters(void) {
    return trillim_rms_schema(true);
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_rmsnorm_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
