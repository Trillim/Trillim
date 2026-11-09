#include "common.h"
#include <math.h>

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *strides;
    uint32_t dtype;
} TrillimUnaryParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimUnaryParameters *parameters;
    const TrillimKernelContext *context;
} TrillimUnaryTask;

static inline TrillimStatus trillim_unary_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimUnaryParameters *p = (const TrillimUnaryParameters *)parameters;
    size_t bytes = trillim_itemsize(p->dtype);
    size_t span;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (!trillim_float(p->dtype) || !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > SIZE_MAX / bytes || p->count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->count != 0) {
        inputs[0] = span * bytes;
        outputs[0] = p->count * bytes;
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_unary_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimUnaryTask *task = (const TrillimUnaryTask *)opaque;
    const TrillimUnaryParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        TrillimValue value = trillim_load(task->input, trillim_offset(index, p->rank, p->shape, p->strides), p->dtype);
        value.f = expf(value.f);
        trillim_store(task->output, index, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_exp(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimUnaryParameters *p = (const TrillimUnaryParameters *)parameters;
    TrillimUnaryTask task = {outputs[0], inputs[0], p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_unary_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_exp_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _unary as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Exp(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_exp_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimUnaryParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimUnaryParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimUnaryParameters, shape), "rank"},
        {"strides", "size_t", offsetof(TrillimUnaryParameters, strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimUnaryParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimUnaryParameters), TRILLIM_ALIGNOF(TrillimUnaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_unary_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_exp_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
