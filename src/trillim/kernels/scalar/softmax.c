#include "common.h"
#include <math.h>

typedef struct {
    size_t rows, width;
    uint32_t dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    float *scratch;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || !p->width ||
        p->rows > SIZE_MAX / sizeof(float) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    *scratch = p->rows * p->width * sizeof(float);
    return TRILLIM_OK;
}

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static TrillimStatus softmax(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float maximum = -INFINITY;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->dtype).f;
        maximum = maximum >= value ? maximum : value;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float difference = round_dtype(trillim_load(task->input, start + column, p->dtype).f - maximum, p->dtype);
        float numerator = round_dtype(expf(difference), p->dtype);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    sum = round_dtype(sum, p->dtype);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_softmax(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], scratch, p, context};
    if (p->rows == 1) {
        return softmax(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     softmax, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_softmax_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('Softmax requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('Softmax requires matching floating input and output')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Softmax requires contiguous input')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'dtype': _native_dtype(x.dtype)}\n"
           "def pattern(x: Tensor, *, axis: int = -1) -> tuple[Tensor, ...]:\n"
           "    return (Softmax(x, axis=axis),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_softmax_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_softmax_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
