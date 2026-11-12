#include "common.h"
#include <math.h>

typedef struct {
    size_t rows, width;
    uint32_t dtype;
    float eps;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || !p->width ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        p->rows > SIZE_MAX / trillim_itemsize(p->dtype) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
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

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float sum = 0.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(task->input, start + column, p->dtype).f;
        sum += round_dtype(x * x, p->dtype);
    }
    sum = round_dtype(sum, p->dtype);
    float added = round_dtype(sum + round_dtype(p->eps, p->dtype), p->dtype);
    float denominator = round_dtype(sqrtf(added), p->dtype);
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {
            .f = trillim_load(task->input, start + column, p->dtype).f / denominator
        };
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_l2norm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows == 1) {
        return normalize(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     normalize, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_l2norm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('L2Norm requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('L2Norm requires matching floating input and output')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('L2Norm requires contiguous input')\n"
           "    eps = context.attrs.get('eps', 1e-6)\n"
           "    if not isinstance(eps, (int, float)):\n"
           "        raise ValueError('invalid L2Norm epsilon')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'dtype': _native_dtype(x.dtype), 'eps': float(eps)}\n"
           "def pattern(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (L2Norm(x, axis=axis, eps=eps),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_l2norm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_l2norm_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
