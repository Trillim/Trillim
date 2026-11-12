#include "common.h"
typedef struct {
    size_t rows;
    size_t width;
    uint32_t input_dtype;
    uint32_t bias_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static float round_native(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype))
        return TRILLIM_ERROR;
    inputs[1] = 0;
    if (!trillim_float(p->bias_dtype) || !p->width || p->width > SIZE_MAX / 4 ||
        p->rows > SIZE_MAX / 4 / p->width)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus apply(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t offset = row * p->width;
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            TrillimValue result = {
                .f = trillim_load(task->input, offset + i, p->input_dtype).f +
                     trillim_load(task->bias, i, p->bias_dtype).f
            };
            uint32_t sum_dtype = p->input_dtype == p->bias_dtype ? p->input_dtype : TRILLIM_FP32;
            result.f = round_native(result.f, sum_dtype);
            trillim_store(task->output, offset + i, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_bias_add(void *const *outputs, const void *const *inputs, void *scratch,
                                                       const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->rows;
    return tiles == 1 ? apply(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_scalar_bias_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _DTYPE_IDS, _rmsnorm\n"
           "def bind(c):\n"
           "    x, bias, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    if not x.shape or bias.shape != x.shape[-1:] or output.shape != x.shape:\n"
           "        raise ValueError('bias must match the final channel axis')\n"
           "    for value in (*c.inputs.values(), *c.outputs):\n"
           "        if value.dtype not in {'fp16', 'bf16', 'fp32'} or (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'input_dtype': _DTYPE_IDS[x.dtype], 'bias_dtype': _DTYPE_IDS[bias.dtype], 'output_dtype': _DTYPE_IDS[output.dtype]}\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Add(a, b) if dtype is None else Cast(Add(a, b), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bias_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bias_add_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &caps;
}
