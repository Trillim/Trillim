#include "common.h"

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *a_strides;
    const size_t *b_strides;
    uint32_t dtype;
} TrillimBinaryParameters;

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

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *a_strides;
    const size_t *b_strides;
    uint32_t dtype;
    uint32_t predicate;
} TrillimCompareParameters;

typedef struct {
    void *output;
    const void *const *inputs;
    const void *parameters;
    const TrillimKernelContext *context;
} TrillimSelectionTask;

static inline bool trillim_compare(TrillimValue a, TrillimValue b, uint32_t dtype, uint32_t predicate) {
    bool equal, less;
    if (trillim_float(dtype)) {
        equal = a.f == b.f;
        less = a.f < b.f;
        switch (predicate) {
        case 0: return equal;
        case 1: return !equal;
        case 2: return less;
        case 3: return a.f <= b.f;
        case 4: return a.f > b.f;
        default: return a.f >= b.f;
        }
    }
    equal = a.u == b.u;
    less = trillim_unsigned(dtype) ? a.u < b.u : a.i < b.i;
    switch (predicate) {
    case 0: return equal;
    case 1: return !equal;
    case 2: return less;
    case 3: return less || equal;
    case 4: return !less && !equal;
    default: return !less;
    }
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimCompareParameters *p = (const TrillimCompareParameters *)parameters;
    TrillimBinaryParameters binary = {p->count, p->rank, p->shape, p->a_strides, p->b_strides, p->dtype};
    TrillimStatus status = trillim_binary_requirements(&binary, inputs, outputs, scratch);
    outputs[0] = p->count;
    return p->predicate < 6 ? status : TRILLIM_ERROR;
}

static TrillimStatus compare_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimSelectionTask *task = (const TrillimSelectionTask *)opaque;
    const TrillimCompareParameters *p = (const TrillimCompareParameters *)task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t i = begin; i < end; ++i) {
        TrillimValue a = trillim_load(task->inputs[0], trillim_offset(i, p->rank, p->shape, p->a_strides), p->dtype);
        TrillimValue b = trillim_load(task->inputs[1], trillim_offset(i, p->rank, p->shape, p->b_strides), p->dtype);
        ((uint8_t *)task->output)[i] = trillim_compare(a, b, p->dtype, p->predicate);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_compare(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const TrillimCompareParameters *p = (const TrillimCompareParameters *)parameters;
    TrillimSelectionTask task = {outputs[0], inputs, p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), compare_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_compare_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _compare as bind\n"
           "def pattern(a: Tensor, b: Tensor, *, predicate: str) -> tuple[Tensor, ...]:\n"
           "    return (Compare(a, b, predicate=predicate),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_compare_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimCompareParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimCompareParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimCompareParameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(TrillimCompareParameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(TrillimCompareParameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimCompareParameters, dtype), NULL},
        {"predicate", "uint32_t", offsetof(TrillimCompareParameters, predicate), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimCompareParameters), TRILLIM_ALIGNOF(TrillimCompareParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_compare_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
