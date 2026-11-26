#include "common.h"

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    (void)parameters;
    inputs[0] = inputs[1] = outputs[0] = sizeof(uint64_t);
    *scratch = 0;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_select_negative_add(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    (void)parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue value = trillim_load(inputs[0], 0, TRILLIM_INT64);
    if (value.i < 0) {
        value.u += trillim_load(inputs[1], 0, TRILLIM_INT64).u;
    }
    trillim_store(outputs[0], 0, TRILLIM_INT64, value);
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_select_negative_add_tdl(void) {
    return "from trillim.tdl import Add, Compare, Select, Tensor\n"
           "physical_inputs = ('when_false', 'when_true')\n"
           "def pattern(when_false: Tensor, when_true: Tensor) -> tuple[Tensor, ...]:\n"
           "    x, offset = when_false, when_true\n"
           "    return (Select(Compare(x, 0, predicate='lt'), Add(x, offset), x),)\n"
           "def bind(c):\n"
           "    if any(value.dtype != 'int64' or value.shape != () for value in (*c.inputs.values(), *c.outputs)):\n"
           "        raise ValueError('expected int64 scalars')\n"
           "    return {}\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_select_negative_add_parameters(void) {
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        0, 1, 0, NULL, 2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_select_negative_add_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
