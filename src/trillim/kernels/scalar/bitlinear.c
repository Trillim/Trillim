#include "bitlinear.h"

TRILLIM_KERNEL TrillimStatus trillim_scalar_bitlinear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimBitLinearParameters *p = (const TrillimBitLinearParameters *)parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    TrillimScalarBitLinearTask task = {(float *)outputs[0], (const float *)inputs[0],
        (const uint8_t *)inputs[1], (uint8_t *)scratch, p, context};
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context,
        p->tokens, trillim_bitlinear_quantize_scalar, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t columns = p->outputs / 32 + (p->outputs % 32 != 0);
    return context->thread_pool->run(context->thread_pool->context,
        p->tokens * columns, trillim_bitlinear_project_scalar, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_bitlinear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import bitlinear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (BitLinear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bitlinear_parameters(void) {
    return trillim_bitlinear_schema_scalar();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bitlinear_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
