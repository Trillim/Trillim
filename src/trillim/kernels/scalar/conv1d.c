#include "common.h"
typedef struct {
    size_t channels, length, outputs, stride, output_length;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input, *weight, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    uint32_t types[] = {p->input_dtype, p->weight_dtype, p->bias_dtype, p->output_dtype};
    for (size_t i = 0; i < 4; ++i)
        if (!trillim_float(types[i])) return TRILLIM_ERROR;
    if ((p->stride != 1 && p->stride != 2) || !p->length || p->output_length != (p->length - 1) / p->stride + 1 ||
        p->channels > SIZE_MAX / 4 / p->length || p->channels > SIZE_MAX / 12 ||
        (p->channels && p->outputs > SIZE_MAX / 12 / p->channels) || p->outputs > SIZE_MAX / 4 / p->output_length)
        return TRILLIM_ERROR;
    inputs[0] = p->channels * p->length * trillim_itemsize(p->input_dtype);
    inputs[1] = p->outputs * p->channels * 3 * trillim_itemsize(p->weight_dtype);
    inputs[2] = p->outputs * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->outputs * p->output_length * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus convolve(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->output_length / 32 + (p->output_length % 32 != 0);
    size_t output = tile / tiles;
    size_t begin = tile % tiles * 32;
    size_t count = p->output_length - begin < 32 ? p->output_length - begin : 32;
    float sums[32];
    float bias = trillim_load(task->bias, output, p->bias_dtype).f;
    for (size_t i = 0; i < count; ++i) {
        sums[i] = bias;
    }
    for (size_t channel = 0; channel < p->channels; ++channel) {
        if ((channel & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t k = 0; k < 3; ++k) {
            float weight = trillim_load(task->weight, (output * p->channels + channel) * 3 + k,
                                        p->weight_dtype).f;
            for (size_t i = 0; i < count; ++i) {
                size_t position = (begin + i) * p->stride + k;
                if (position != 0 && position <= p->length) {
                    float value = trillim_load(task->input, channel * p->length + position - 1,
                                              p->input_dtype).f;
                    sums[i] += value * weight;
                }
            }
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t i = 0; i < count; ++i) {
        TrillimValue result = {.f = sums[i]};
        trillim_store(task->output, output * p->output_length + begin + i, p->output_dtype, result);
    }
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_conv1d(void *const *outputs, const void *const *inputs, void *scratch,
                                                     const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t tiles = p->output_length / 32 + (p->output_length % 32 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->outputs * tiles, convolve, &task);
}
TRILLIM_KERNEL const char *trillim_scalar_conv1d_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _DTYPE_IDS, _rmsnorm\n"
           "def bind(c):\n"
           "    x, weight, bias = (c.inputs[name] for name in ('x', 'weight', 'bias'))\n"
           "    stride = c.attrs.get('stride', 1)\n"
           "    if isinstance(stride, tuple) and len(stride) == 1:\n"
           "        stride = stride[0]\n"
           "    if (c.attrs.get('kernel') != (3,) or stride not in {1, 2}\n"
           "        or c.attrs.get('padding', 1) not in (1, (1,))\n"
           "        or c.attrs.get('dilation', 1) not in (1, (1,)) or c.attrs.get('groups', 1) != 1\n"
           "        or len(x.shape) != 2 or len(weight.shape) != 3\n"
           "        or weight.shape[1] != x.shape[0] or weight.shape[2].value != 3\n"
           "        or bias.shape != weight.shape[:1]):\n"
           "        raise ValueError('fused Conv1D requires padded width-three weights and stride 1 or 2')\n"
           "    for value in (*c.inputs.values(), *c.outputs):\n"
           "        if value.dtype not in {'fp16', 'bf16', 'fp32'} or (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "    return {'channels': x.shape[0], 'length': x.shape[1], 'outputs': weight.shape[0], 'stride': stride, 'output_length': c.outputs[0].shape[1], 'input_dtype': _DTYPE_IDS[x.dtype], 'weight_dtype': _DTYPE_IDS[weight.dtype], 'bias_dtype': _DTYPE_IDS[bias.dtype], 'output_dtype': _DTYPE_IDS[c.outputs[0].dtype]}\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, kernel: tuple[int, ...] = (3,), stride = 1, padding = 1, dilation = 1, groups: int = 1, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Conv(x, weight, bias=bias, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups) if dtype is None else Cast(Conv(x, weight, bias=bias, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_conv1d_parameters(void) {
    static const TrillimParameter fields[] = {
        {"channels", "size_t", offsetof(Parameters, channels), NULL},
        {"length", "size_t", offsetof(Parameters, length), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"stride", "size_t", offsetof(Parameters, stride), NULL},
        {"output_length", "size_t", offsetof(Parameters, output_length), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 9, fields, 3, 1, requirements, NULL, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_conv1d_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &caps;
}
