#include "common.h"

typedef struct {
    int64_t value;
} Parameters;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    (void)inputs;
    const Parameters *p = parameters;
    outputs[0] = *scratch = 0;
    if (p->value < 0) {
        return TRILLIM_ERROR;
    }
    outputs[0] = sizeof(int64_t);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_dim(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)inputs;
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    memcpy(outputs[0], &p->value, sizeof(p->value));
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_dim_tdl(void) {
    return "from trillim.tdl import *\n"
           "physical_inputs = ()\n"
           "def bind(c):\n"
           "    shape = c.inputs['x'].shape\n"
           "    axis = c.attrs['axis']\n"
           "    if not isinstance(axis, int) or not -len(shape) <= axis < len(shape) or c.outputs[0].shape != () or c.outputs[0].dtype != 'int64':\n"
           "        raise ValueError('Dim requires a valid axis and a scalar int64 output')\n"
           "    return {'value': shape[axis]}\n"
           "def pattern(x: Tensor, axis: int) -> tuple[Tensor, ...]:\n"
           "    return (Dim(x, axis),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_dim_parameters(void) {
    static const TrillimParameter fields[] = {
        {"value", "int64_t", offsetof(Parameters, value), NULL}
    };
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 0, 1,
        requirements, NULL, NULL, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_dim_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
