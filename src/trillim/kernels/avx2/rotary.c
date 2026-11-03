#include "rotary_avx2.h"

TRILLIM_KERNEL TrillimStatus trillim_avx2_rotary(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    TrillimRotaryTask task = {(float *)outputs[0], (const float *)inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context, p->heads * p->tokens, trillim_rotary_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_avx2_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import avx2_rotary as bind\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, amplitude: float = 1.0) -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_avx2_rotary_parameters(void) {
    return trillim_rotary_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_avx2_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "avx2", "avx2", "m"};
    return &capabilities;
}
