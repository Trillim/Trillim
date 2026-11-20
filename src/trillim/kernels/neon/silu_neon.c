#include "common.h"
#include <arm_neon.h>

#if !defined(_WIN32)
#define COMPONENT_API __attribute__((visibility("default")))
#else
#define COMPONENT_API
#endif

extern COMPONENT_API TrillimStatus trillim_neon_activation_sigmoid(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern COMPONENT_API const TrillimCapabilities *trillim_neon_activation_sigmoid_capabilities(void);

/* One tile of the Sigmoid kernel keeps its call on the current worker. */
enum { TILE = 1024 };

typedef struct {
    size_t count;
    uint32_t dtype;
} Parameters;

typedef struct {
    float *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->dtype != TRILLIM_BF16 && p->dtype != TRILLIM_FP32) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->dtype);
    outputs[0] = p->count * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t begin = tile * TILE;
    size_t count = task->p->count - begin < TILE ? task->p->count - begin : TILE;
    uint32_t dtype = task->p->dtype;
    const uint8_t *input = (const uint8_t *)task->input + begin * trillim_itemsize(dtype);
    float *output = task->output + begin;
    const void *inputs[] = {input};
    void *outputs[] = {output};
    Parameters sigmoid = {count, dtype};
    TrillimStatus status = trillim_neon_activation_sigmoid(outputs, inputs, NULL, &sigmoid, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        float32x4_t value;
        if (dtype == TRILLIM_FP32) {
            value = vld1q_f32((const float *)input + i);
        } else {
            uint16x4_t bits = vld1_u16((const uint16_t *)input + i);
            value = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
        }
        vst1q_f32(output + i, vmulq_f32(value, vld1q_f32(output + i)));
    }
    for (; i < count; ++i) {
        output[i] = trillim_load(input, i, dtype).f * output[i];
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_silu(
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
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / TILE + (p->count % TILE != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_silu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    strides = _contiguous_strides(x.shape)\n"
           "    if x.dtype not in {'bf16', 'fp32'} or output.dtype != 'fp32' or x.shape != output.shape or (x.strides or strides) != strides:\n"
           "        raise ValueError('SiLU requires contiguous BF16 or FP32 input and FP32 output')\n"
           "    return {'count': _product(x.shape), 'dtype': _native_dtype(x.dtype)}\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Silu(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_silu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_silu_capabilities(void) {
    return trillim_neon_activation_sigmoid_capabilities();
}
