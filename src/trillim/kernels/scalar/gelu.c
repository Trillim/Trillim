#include "common.h"
#include <math.h>
typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_FP32 && p->input_dtype != TRILLIM_BF16) ||
        (p->output_dtype != TRILLIM_FP32 && p->output_dtype != TRILLIM_BF16))
        return TRILLIM_ERROR;
    if (p->count > SIZE_MAX / 4) return TRILLIM_ERROR;
    inputs[0] = p->count * trillim_itemsize(p->input_dtype);
    outputs[0] = p->count * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t begin = tile * 1024, end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t i = begin; i < end; ++i) {
        float x = trillim_load(t->input, i, p->input_dtype).f;
        TrillimValue value = {.f = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f))};
        trillim_store(t->output, i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_gelu(void *const *outputs, const void *const *inputs, void *scratch,
                                                 const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    return tiles == 1 ? apply(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_scalar_gelu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _gelu as bind\n"
           "def pattern(x: Tensor, *, approximate: str = 'none', dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Gelu(x, approximate=approximate) if dtype is None else Cast(Gelu(x, approximate=approximate), "
           "dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gelu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gelu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
