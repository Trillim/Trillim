#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const float *up;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float gelu_tanh(float x, uint32_t dtype) {
    float square = round_dtype(x * x, dtype);
    float cube = round_dtype(square * x, dtype);
    float correction = round_dtype(cube * round_dtype(0.044715f, dtype), dtype);
    float sum = round_dtype(x + correction, dtype);
    float argument = round_dtype(sum * round_dtype(0.7978845608028654f, dtype), dtype);
    float scaled = round_dtype(argument * 2.0f, dtype);
    float e = round_dtype(expf(-fabsf(scaled)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    float probability = scaled >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
    float cdf = probability * 2.0f - 1.0f;
    float half = round_dtype(x * 0.5f, dtype);
    return half * (cdf + 1.0f);
}

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(TRILLIM_FP32) || !trillim_float(TRILLIM_FP32) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(TRILLIM_FP32);
    outputs[0] = inputs[1] = p->count * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;

    for (; i < end; ++i) {
        float input = trillim_load(task->input, i, TRILLIM_FP32).f;
        TrillimValue value = {.f = gelu_tanh(input, TRILLIM_FP32) * task->up[i]};
        trillim_store(task->output, i, TRILLIM_FP32, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_gelu_tanh_mul(
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
    Task task = {outputs[0], inputs[1], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_gelu_tanh_mul_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gelu_tanh_mul_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gelu_tanh_mul_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
