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
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype))
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
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _DTYPE_IDS, _rmsnorm\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if c.attrs.get('approximate', 'none') != 'none' or x.shape != output.shape:\n"
           "        raise ValueError('fused GELU requires the erf definition and matching shapes')\n"
           "    for value in (*c.inputs.values(), *c.outputs):\n"
           "        if value.dtype not in {'fp16', 'bf16', 'fp32'} or (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "    return {'count': _product(x.shape), 'input_dtype': _DTYPE_IDS[x.dtype], 'output_dtype': _DTYPE_IDS[output.dtype]}\n"
           "def pattern(x: Tensor, *, approximate: str = 'none', dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Gelu(x, approximate=approximate) if dtype is None else Cast(Gelu(x, approximate=approximate), dtype),)\n";
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
