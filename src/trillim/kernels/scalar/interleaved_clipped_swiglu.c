#include "common.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t rows, width;
} Parameters;

typedef struct {
    float *output;
    const float *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->width % 4) {
        return TRILLIM_ERROR;
    }
    if (!p->width || !p->rows) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / (2 * sizeof(float)) / p->width) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
    inputs[0] = outputs[0] * 2;
    return TRILLIM_OK;
}

/* Match the staged FP32 arithmetic in the decomposed operations. */
static float activate(float gate, float up) {
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    return (gate * probability) * (up + 1.0f);
}

static TrillimStatus apply(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t width = task->p->width;
    const float *source = task->input + row * width * 2;
    float *output = task->output + row * width;
    for (size_t i = 0; i < width; ++i) {
        if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        output[i] = activate(source[2 * i], source[2 * i + 1]);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_interleaved_clipped_swiglu(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows > 4 && p->rows > 16384 / p->width) {
        return context->thread_pool->run(context->thread_pool->context, p->rows, apply, &task);
    }
    for (size_t row = 0; row < p->rows; ++row) {
        TrillimStatus status = apply(&task, row, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_interleaved_clipped_swiglu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if len(x.shape) != 3 or len(output.shape) != 3 or x.dtype != 'fp32' or output.dtype != 'fp32':\n"
           "        raise ValueError('interleaved clipped SwiGLU needs rank-three FP32 tensors')\n"
           "    width = output.shape[-1]\n"
           "    if width.value is None or width.value % 4 or x.shape[-1].value != width.value * 2 or x.shape[:-1] != output.shape[:-1]:\n"
           "        raise ValueError('interleaved clipped SwiGLU needs a gate and value per output channel')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('interleaved clipped SwiGLU needs contiguous projected rows')\n"
           "    return dict(rows=_product(output.shape[:-1]), width=width)\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    if not isinstance(x, Tensor) or x.dtype != 'fp32' or len(x.shape.dimensions) != 3:\n"
           "        raise ValueError('expected a rank-three FP32 tensor')\n"
           "    width = x.shape.dimensions[-1]\n"
           "    if width is None or width <= 0 or width % 4:\n"
           "        raise ValueError('expected a fixed channel width divisible by four')\n"
           "    gate_ids = Iota(width * 2, step=2)\n"
           "    up_ids = Iota(width * 2, start=1, step=2)\n"
           "    gate = Minimum(Take(x, gate_ids, axis=2, rank=3), 7.0)\n"
           "    up = Take(x, up_ids, axis=2, rank=3)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(up, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_interleaved_clipped_swiglu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_interleaved_clipped_swiglu_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
