#include "activation_avx2.h"

TRILLIM_KERNEL TrillimStatus trillim_x86_relu_squared_mul_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    TrillimActivationTask task = {(float *)outputs[0], (const float *)inputs[0],
        (const float *)inputs[1], p, context, false};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_activation_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_relu_squared_mul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import activation as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(ReluSquared(a), b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_relu_squared_mul_avx2_parameters(void) {
    return trillim_activation_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_relu_squared_mul_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "x86", "avx2", "m"};
    return &capabilities;
}
