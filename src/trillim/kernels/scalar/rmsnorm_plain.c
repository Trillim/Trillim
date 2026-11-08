#include "normalization.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_rmsnorm_plain(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    return trillim_rms_compute(false, outputs, inputs, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_rmsnorm_plain_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm as bind\n"
           "def pattern(x: Tensor, *, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, eps=eps),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_rmsnorm_plain_parameters(void) {
    return trillim_rms_schema(false);
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_rmsnorm_plain_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
