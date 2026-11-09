#include "common.h"

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *a_strides;
    const size_t *b_strides;
    uint32_t dtype;
} TrillimBinaryParameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const TrillimBinaryParameters *p;
    const TrillimKernelContext *context;
} TrillimBinaryTask;

static inline TrillimStatus trillim_binary_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimBinaryParameters *p = (const TrillimBinaryParameters *)parameters;
    size_t bytes = trillim_itemsize(p->dtype), a, b;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (bytes == 0 || !trillim_span(p->rank, p->shape, p->a_strides, &a) || !trillim_span(p->rank, p->shape, p->b_strides, &b)) {
        return TRILLIM_ERROR;
    }
    size_t elements = 1;
    for (size_t i = 0;i<p->rank;++i) {
        if (p->shape[i] != 0 && elements>SIZE_MAX/p->shape[i]) {
            return TRILLIM_ERROR;
        }
        elements *= p->shape[i];
    }
    if (elements != p->count) {
        return TRILLIM_ERROR;
    }
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    if (a > SIZE_MAX/bytes || b > SIZE_MAX/bytes || p->count > SIZE_MAX/bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a*bytes;
    inputs[1] = b*bytes;
    outputs[0] = p->count*bytes;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_binary_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimBinaryTask *task = (const TrillimBinaryTask *)opaque;
    const TrillimBinaryParameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        TrillimValue a = trillim_load(task->a, trillim_offset(index, p->rank, p->shape, p->a_strides), p->dtype);
        TrillimValue b = trillim_load(task->b, trillim_offset(index, p->rank, p->shape, p->b_strides), p->dtype);
        TrillimValue result;
        if (!trillim_float(p->dtype)) {
            return TRILLIM_ERROR;
        }
        result.f = a.f / b.f;
        trillim_store(task->output, index, p->dtype, result);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input < 2;
    }
    const TrillimBinaryParameters *p = (const TrillimBinaryParameters *)parameters;
    return output == 0 && input < 2 && trillim_contiguous(
        p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_div(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimBinaryParameters *p = (const TrillimBinaryParameters *)parameters;
    TrillimBinaryTask task = {outputs[0], inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_binary_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_div_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Div(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_div_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimBinaryParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimBinaryParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimBinaryParameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(TrillimBinaryParameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(TrillimBinaryParameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimBinaryParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimBinaryParameters), TRILLIM_ALIGNOF(TrillimBinaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        trillim_binary_requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_div_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
