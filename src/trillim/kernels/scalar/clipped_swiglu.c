#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
} Parameters;

typedef struct {
    float *output;
    const float *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rank && (!p->shape || !p->a_strides || !p->b_strides)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (!p->shape[axis]) {
            return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
        }
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    size_t a, b;
    if (count != p->count || count > SIZE_MAX / sizeof(float) ||
        !trillim_span(p->rank, p->shape, p->a_strides, &a) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b) ||
        a > SIZE_MAX / sizeof(float) || b > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a * sizeof(float);
    inputs[1] = b * sizeof(float);
    outputs[0] = count * sizeof(float);
    return TRILLIM_OK;
}

static float activate(float gate, float up) {
    /* Compare/Select clamps send NaN gates to +7 and NaN up values to -7. */
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float activated = gate * probability;
    float shifted = up + 1.0f;
    return activated * shifted;
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t index = tile * 2048;
    size_t end = p->count - index < 2048 ? p->count : index + 2048;
    while (index < end) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t a = trillim_offset(index, p->rank, p->shape, p->a_strides);
        size_t b = trillim_offset(index, p->rank, p->shape, p->b_strides);
        size_t a_step = p->rank ? p->a_strides[p->rank - 1] : 0;
        size_t b_step = p->rank ? p->b_strides[p->rank - 1] : 0;
        size_t width = p->rank ? p->shape[p->rank - 1] : 1;
        size_t remaining = width - index % width;
        size_t limit = end - index < remaining ? end : index + remaining;
        for (; index < limit; ++index, a += a_step, b += b_step) {
            if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            task->output[index] = activate(task->a[a], task->b[b]);
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (output != 0 || input >= 2) {
        return false;
    }
    if (!parameters) {
        return true;
    }
    const Parameters *p = parameters;
    return trillim_contiguous(p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_clipped_swiglu(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->count) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->count / 2048 + (p->count % 2048 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_clipped_swiglu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary\n"
           "def bind(context):\n"
           "    a, b, output = context.inputs['a'], context.inputs['b'], context.outputs[0]\n"
           "    if any(value.dtype != 'fp32' for value in (a, b, output)):\n"
           "        raise ValueError('clipped SwiGLU requires FP32 tensors')\n"
           "    parameters = _binary(context)\n"
           "    del parameters['dtype']\n"
           "    return parameters\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    gate = Minimum(a, 7.0)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(b, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_clipped_swiglu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_clipped_swiglu_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
