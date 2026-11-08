#include "selection.h"

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimSelectParameters *p = (const TrillimSelectParameters *)parameters;
    TrillimBinaryParameters binary = {p->count, p->rank, p->shape, p->a_strides, p->b_strides, p->dtype};
    TrillimStatus status = trillim_binary_requirements(&binary, inputs + 1, outputs, scratch);
    size_t condition_span;
    if (!trillim_span(p->rank, p->shape, p->condition_strides, &condition_span)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count == 0 ? 0 : condition_span;
    return status;
}

static TrillimStatus select_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimSelectionTask *task = (const TrillimSelectionTask *)opaque;
    const TrillimSelectParameters *p = (const TrillimSelectParameters *)task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t i = begin; i < end; ++i) {
        size_t index = trillim_offset(i, p->rank, p->shape, p->condition_strides);
        bool condition = ((const uint8_t *)task->inputs[0])[index] != 0;
        size_t source = condition ? 1 : 2;
        const size_t *strides = condition ? p->a_strides : p->b_strides;
        TrillimValue value = trillim_load(task->inputs[source], trillim_offset(i, p->rank, p->shape, strides), p->dtype);
        trillim_store(task->output, i, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_select(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const TrillimSelectParameters *p = (const TrillimSelectParameters *)parameters;
    TrillimSelectionTask task = {outputs[0], inputs, p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), select_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_select_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _select as bind\n"
           "physical_inputs = ('condition', 'when_true', 'when_false')\n"
           "def pattern(condition: Tensor, when_true: Tensor, when_false: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Select(condition, when_true, when_false),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_select_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimSelectParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimSelectParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimSelectParameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(TrillimSelectParameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(TrillimSelectParameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimSelectParameters, dtype), NULL},
        {"condition_strides", "size_t", offsetof(TrillimSelectParameters, condition_strides), "rank"}
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimSelectParameters), TRILLIM_ALIGNOF(TrillimSelectParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_select_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
