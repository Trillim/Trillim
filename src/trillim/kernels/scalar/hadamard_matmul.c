#include "common.h"


typedef struct { size_t rows, row_stride; } Parameters;
typedef struct {
    const float *input;
    float *output;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->rows == 0) return TRILLIM_OK;
    size_t shape[] = {p->rows, 1024}, strides[] = {p->row_stride, 1}, span;
    if (!trillim_span(2, shape, strides, &span) || span > SIZE_MAX / sizeof(float) ||
        p->rows > SIZE_MAX / 1024 / sizeof(float)) return TRILLIM_ERROR;
    inputs[0] = span * sizeof(float);
    outputs[0] = p->rows * 1024 * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus transform(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    const float *x = task->input + row * task->p->row_stride;
    float *y = task->output + row * 1024;
    if (x != y) memcpy(y, x, 1024 * sizeof(float));
    for (size_t half = 1; half < 1024; half *= 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        for (size_t group = 0; group < 1024; group += half * 2) {
            size_t lane = 0;

            for (; lane < half; ++lane) {
                float a = y[group + lane], b = y[group + half + lane];
                y[group + lane] = a + b;
                y[group + half + lane] = a - b;
            }
        }
    }
    for (size_t lane = 0; lane < 1024; ++lane) y[lane] *= 0.03125f;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_hadamard_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (p->rows == 0) return TRILLIM_OK;
    Task task = {inputs[0], outputs[0], p, context};
    return p->rows == 1 ? transform(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, transform, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_hadamard_matmul_tdl(void) {
    return "from trillim.tdl import Tensor, Constant, Matmul, Mul\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "import numpy as np\n"
           "_H = np.ones((1, 1), dtype=np.float32)\n"
           "while _H.shape[0] < 1024:\n"
           "    _H = np.block([[_H, _H], [_H, -_H]])\n"
           "def pattern(a: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(Matmul(a, Constant(_H)), 1.0 / 32),)\n"
           "def bind(context):\n"
           "    x, y = context.inputs['a'], context.outputs[0]\n"
           "    one = _constant_expression(1)\n"
           "    width = _constant_expression(1024)\n"
           "    if x.dtype != 'fp32' or y.dtype != 'fp32' or len(x.shape) != 2 or x.shape[1] != width or y.shape != x.shape:\n"
           "        raise ValueError('requires FP32 normalized Sylvester-1024 multiplication')\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    if strides[-1] != one:\n"
           "        raise ValueError('requires contiguous transform blocks')\n"
           "    return dict(rows=x.shape[0], row_stride=strides[0])\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const Parameters *p = parameters;
    return output == 0 && input == 0 && (!p || p->rows <= 1 || p->row_stride == 1024);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_hadamard_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"row_stride", "size_t", offsetof(Parameters, row_stride), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_hadamard_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
