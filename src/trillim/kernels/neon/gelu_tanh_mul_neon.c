#include "common.h"
#include <arm_neon.h>

#if !defined(_WIN32)
#define COMPONENT_API __attribute__((visibility("default")))
#else
#define COMPONENT_API
#endif

extern COMPONENT_API TrillimStatus trillim_neon_gelu_tanh(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern COMPONENT_API const TrillimCapabilities *trillim_neon_gelu_tanh_capabilities(void);

/* One tile of the GELU kernel keeps its call on the current worker. */
enum { TILE = 1024 };

typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} GeluParameters;

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    float *output;
    const float *up;
    const float *gate;
    size_t count;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = inputs[1] = outputs[0] = p->count * sizeof(float);
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t begin = tile * TILE;
    size_t count = task->count - begin < TILE ? task->count - begin : TILE;
    float activation[TILE];
    const void *inputs[] = {task->gate + begin};
    void *outputs[] = {activation};
    GeluParameters gelu = {count, TRILLIM_FP32, TRILLIM_FP32};
    TrillimStatus status = trillim_neon_gelu_tanh(outputs, inputs, NULL, &gelu, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    const float *up = task->up + begin;
    float *output = task->output + begin;
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        vst1q_f32(output + i, vmulq_f32(vld1q_f32(up + i), vld1q_f32(activation + i)));
    }
    for (; i < count; ++i) {
        output[i] = up[i] * activation[i];
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gelu_tanh_mul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p->count, context};
    size_t tiles = p->count / TILE + (p->count % TILE != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_gelu_tanh_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    a, b, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    for value in (a, b, output):\n"
           "        strides = _contiguous_strides(value.shape)\n"
           "        if value.dtype != 'fp32' or value.shape != output.shape or (value.strides or strides) != strides:\n"
           "            raise ValueError('GELU gate requires matching contiguous FP32 tensors')\n"
           "    return {'count': _product(output.shape)}\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(a, Gelu(b, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gelu_tanh_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gelu_tanh_mul_capabilities(void) {
    return trillim_neon_gelu_tanh_capabilities();
}
