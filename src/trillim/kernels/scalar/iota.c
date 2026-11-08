#include "common.h"
#include <math.h>
#include <string.h>

typedef struct {
    size_t count;
    int64_t start;
    int64_t step;
    uint32_t dtype;
} TrillimIotaParameters;

typedef struct {
    void *output;
    const void *parameters;
    const TrillimKernelContext *context;
} TrillimGenerationTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    (void)inputs;
    const TrillimIotaParameters *p = (const TrillimIotaParameters *)parameters;
    size_t bytes = trillim_itemsize(p->dtype);
    outputs[0] = 0;
    *scratch = 0;
    if ((p->dtype != TRILLIM_INT32 && p->dtype != TRILLIM_INT64) || p->step == 0 || p->count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->count * bytes;
    return TRILLIM_OK;
}

static TrillimStatus generate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimGenerationTask *task = (const TrillimGenerationTask *)opaque;
    const TrillimIotaParameters *p = (const TrillimIotaParameters *)task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t i = begin; i < end; ++i) {
        TrillimValue value;
        value.u = (uint64_t)p->start + (uint64_t)i * (uint64_t)p->step;
        trillim_store(task->output, i, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_iota(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)inputs;
    (void)scratch;
    const TrillimIotaParameters *p = (const TrillimIotaParameters *)parameters;
    TrillimGenerationTask task = {outputs[0], p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), generate, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_iota_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _iota as bind\n"
           "physical_inputs = ()\n"
           "def pattern(stop: Tensor, *, start = 0, step: int = 1, dtype = \"int64\") -> tuple[Tensor, ...]:\n"
           "    return (Iota(stop, start=start, step=step, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_iota_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimIotaParameters, count), NULL},
        {"start", "int64_t", offsetof(TrillimIotaParameters, start), NULL},
        {"step", "int64_t", offsetof(TrillimIotaParameters, step), NULL},
        {"dtype", "uint32_t", offsetof(TrillimIotaParameters, dtype), NULL},
    };
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimIotaParameters), TRILLIM_ALIGNOF(TrillimIotaParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 0, 1, requirements, NULL, NULL, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_iota_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
