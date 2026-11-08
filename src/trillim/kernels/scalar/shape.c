#include "generation.h"

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    (void)inputs;
    const TrillimShapeParameters *p = (const TrillimShapeParameters *)parameters;
    outputs[0] = 0;
    *scratch = 0;
    if (p->rank > SIZE_MAX / sizeof(int64_t) || (p->rank != 0 && p->shape == NULL)) {
        return TRILLIM_ERROR;
    }
    for (size_t i = 0; i < p->rank; ++i) {
        if (p->shape[i] > INT64_MAX) {
            return TRILLIM_ERROR;
        }
    }
    outputs[0] = p->rank * sizeof(int64_t);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_shape(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)inputs;
    (void)scratch;
    const TrillimShapeParameters *p = (const TrillimShapeParameters *)parameters;
    for (size_t start = 0; start < p->rank; start += 1024) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->rank - start < 1024 ? p->rank : start + 1024;
        for (size_t i = start; i < end; ++i) {
            ((int64_t *)outputs[0])[i] = (int64_t)p->shape[i];
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_shape_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _shape as bind\n"
           "physical_inputs = ()\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Shape(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_shape_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rank", "size_t", offsetof(TrillimShapeParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimShapeParameters, shape), "rank"},
    };
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimShapeParameters), TRILLIM_ALIGNOF(TrillimShapeParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 0, 1, requirements, NULL, NULL, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_shape_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", ""};
    return &capabilities;
}
