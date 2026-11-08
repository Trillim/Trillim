#include "common.h"
#include <math.h>
#include <string.h>

typedef struct {
    size_t count;
    uint64_t seed;
    uint64_t offset;
    uint32_t distribution;
    uint32_t dtype;
} TrillimRandomParameters;

typedef struct {
    void *output;
    const void *parameters;
    const TrillimKernelContext *context;
} TrillimGenerationTask;

static inline uint64_t trillim_random_hash(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    (void)inputs;
    const TrillimRandomParameters *p = (const TrillimRandomParameters *)parameters;
    size_t bytes = trillim_itemsize(p->dtype);
    outputs[0] = 0;
    *scratch = 0;
    if (!trillim_float(p->dtype) || p->distribution > 1 || p->count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->count * bytes;
    return TRILLIM_OK;
}

static TrillimStatus generate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimGenerationTask *task = (const TrillimGenerationTask *)opaque;
    const TrillimRandomParameters *p = (const TrillimRandomParameters *)task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t i = begin; i < end; ++i) {
        uint64_t counter = p->offset + (uint64_t)i;
        uint64_t first = trillim_random_hash(p->seed ^ (counter * 2));
        TrillimValue value;
        if (p->distribution == 0) {
            value.f = (float)(first >> 40) * 0x1p-24f;
            float maximum = p->dtype == TRILLIM_FP16 ? 0x1.ffcp-1f :
                (p->dtype == TRILLIM_BF16 ? 0x1.fep-1f : 0x1.fffffep-1f);
            value.f = fminf(value.f, maximum);
        } else {
            uint64_t second = trillim_random_hash(p->seed ^ (counter * 2 + 1));
            double u = ((double)(first >> 11) + 0.5) * 0x1p-53;
            double v = ((double)(second >> 11) + 0.5) * 0x1p-53;
            value.f = (float)(sqrt(-2.0 * log(u)) * cos(6.2831853071795864769 * v));
        }
        trillim_store(task->output, i, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_random(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)inputs;
    (void)scratch;
    const TrillimRandomParameters *p = (const TrillimRandomParameters *)parameters;
    TrillimGenerationTask task = {outputs[0], p, context};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), generate, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_random_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _random as bind\n"
           "physical_inputs = ()\n"
           "def pattern(*, shape, seed, offset = 0, distribution = \"uniform\", dtype = \"fp32\") -> tuple[Tensor, ...]:\n"
           "    return (Random(shape, seed=seed, offset=offset, distribution=distribution, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_random_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimRandomParameters, count), NULL},
        {"seed", "uint64_t", offsetof(TrillimRandomParameters, seed), NULL},
        {"offset", "uint64_t", offsetof(TrillimRandomParameters, offset), NULL},
        {"distribution", "uint32_t", offsetof(TrillimRandomParameters, distribution), NULL},
        {"dtype", "uint32_t", offsetof(TrillimRandomParameters, dtype), NULL},
    };
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRandomParameters), TRILLIM_ALIGNOF(TrillimRandomParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 0, 1, requirements, NULL, NULL, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_random_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
