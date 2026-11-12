#include "common.h"
#include <math.h>

typedef struct {
    size_t inputs, width;
    uint32_t weight_dtype, bias_dtype;
} Parameters;

typedef struct {
    float *hidden_output, *cell_output;
    const float *input, *hidden, *cell;
    const void *input_weight, *recurrent_weight, *input_bias, *recurrent_bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    for (size_t i = 0; i < 7; ++i) {
        inputs[i] = 0;
    }
    outputs[0] = outputs[1] = *scratch = 0;
    if (!p->inputs || !p->width || !trillim_float(p->weight_dtype) ||
        !trillim_float(p->bias_dtype) || p->inputs > SIZE_MAX / sizeof(float) ||
        p->width > SIZE_MAX / 4 / sizeof(float) ||
        p->inputs > SIZE_MAX / trillim_itemsize(p->weight_dtype) / 4 / p->width ||
        p->width > SIZE_MAX / trillim_itemsize(p->weight_dtype) / 4 / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->inputs * sizeof(float);
    inputs[1] = inputs[2] = p->width * sizeof(float);
    inputs[3] = 4 * p->width * p->inputs * trillim_itemsize(p->weight_dtype);
    inputs[4] = 4 * p->width * p->width * trillim_itemsize(p->weight_dtype);
    inputs[5] = inputs[6] = 4 * p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = outputs[1] = p->width * sizeof(float);
    return TRILLIM_OK;
}

static float sigmoid(float x) {
    float e = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t begin = tile * 8;
    size_t end = p->width - begin < 8 ? p->width : begin + 8;
    for (size_t channel = begin; channel < end; ++channel) {
        float gates[4];
        for (size_t gate = 0; gate < 4; ++gate) {
            size_t row = gate * p->width + channel;
            float input_sum = 0.0f, recurrent_sum = 0.0f;
            for (size_t k = 0; k < p->inputs; ++k) {
                if ((k & 1023) == 0 &&
                    task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                input_sum += task->input[k] *
                    trillim_load(task->input_weight, row * p->inputs + k, p->weight_dtype).f;
            }
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 &&
                    task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                recurrent_sum += task->hidden[k] *
                    trillim_load(task->recurrent_weight, row * p->width + k, p->weight_dtype).f;
            }
            input_sum += trillim_load(task->input_bias, row, p->bias_dtype).f;
            recurrent_sum += trillim_load(task->recurrent_bias, row, p->bias_dtype).f;
            gates[gate] = input_sum + recurrent_sum;
        }
        float cell = sigmoid(gates[1]) * task->cell[channel] +
                     sigmoid(gates[0]) * tanhf(gates[2]);
        task->cell_output[channel] = cell;
        task->hidden_output[channel] = sigmoid(gates[3]) * tanhf(cell);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_lstm_cell(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {
        outputs[0], outputs[1], inputs[0], inputs[1], inputs[2],
        inputs[3], inputs[4], inputs[5], inputs[6], p, context
    };
    size_t tiles = p->width / 8 + (p->width % 8 != 0);
    return context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_lstm_cell_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype\n"
           "from trillim.compiler.passes.infer import _expression, _constant_expression\n"
           "def bind(context):\n"
           "    x, hidden, cell = (context.inputs[name] for name in ('x', 'hidden', 'cell'))\n"
           "    iw, rw, ib, rb = (context.inputs[name] for name in ('input_weight', 'recurrent_weight', 'input_bias', 'recurrent_bias'))\n"
           "    if len(x.shape) != 1 or len(hidden.shape) != 1 or cell.shape != hidden.shape:\n"
           "        raise ValueError('LSTMCell requires vector activation tensors')\n"
           "    if any(t.dtype != 'fp32' for t in (x, hidden, cell, *context.outputs)):\n"
           "        raise ValueError('LSTMCell requires FP32 activations')\n"
           "    if iw.dtype not in {'fp16', 'bf16', 'fp32'} or rw.dtype != iw.dtype or ib.dtype not in {'fp16', 'bf16', 'fp32'} or rb.dtype != ib.dtype:\n"
           "        raise ValueError('LSTMCell requires matching native weight and bias pairs')\n"
           "    gates = _expression('mul', hidden.shape[0], _constant_expression(4))\n"
           "    if iw.shape != (gates, x.shape[0]) or rw.shape != (gates, hidden.shape[0]) or ib.shape != (gates,) or rb.shape != (gates,):\n"
           "        raise ValueError('invalid LSTMCell weight shapes')\n"
           "    if len(context.outputs) != 2 or any(t.shape != hidden.shape for t in context.outputs):\n"
           "        raise ValueError('invalid LSTMCell output shapes')\n"
           "    if any(t.strides and t.strides != _contiguous_strides(t.shape) for t in (x, hidden, cell, iw, rw, ib, rb)):\n"
           "        raise ValueError('LSTMCell requires contiguous tensors')\n"
           "    return {'inputs': x.shape[0], 'width': hidden.shape[0], 'weight_dtype': _native_dtype(iw.dtype), 'bias_dtype': _native_dtype(ib.dtype)}\n"
           "def pattern(x: Tensor, hidden: Tensor, cell: Tensor, *, input_weight: Tensor, recurrent_weight: Tensor, input_bias: Tensor, recurrent_bias: Tensor) -> tuple[Tensor, ...]:\n"
           "    if input_bias is None or recurrent_bias is None:\n"
           "        raise ValueError('LSTMCell requires both bias vectors')\n"
           "    return LSTMCell(x, hidden, cell, input_weight=input_weight, recurrent_weight=recurrent_weight, input_bias=input_bias, recurrent_bias=recurrent_bias)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 1 && input == 2;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_lstm_cell_parameters(void) {
    static const TrillimParameter fields[] = {
        {"inputs", "size_t", offsetof(Parameters, inputs), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL}
    };
    static const size_t inputs[] = {4, 4, 4, 1, 1, 1, 1}, outputs[] = {4, 4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 7, 2,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_lstm_cell_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
