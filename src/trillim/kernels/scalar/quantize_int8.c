#include "common.h"
#include <math.h>

typedef struct {
    size_t tokens;
    size_t width;
    uint32_t input_dtype;
} QuantizeParameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input;
    const QuantizeParameters *parameters;
    const TrillimKernelContext *context;
} QuantizeTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const QuantizeParameters *p = parameters;
    inputs[0] = outputs[0] = outputs[1] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->input_dtype);
    if (!trillim_float(p->input_dtype) || (p->tokens && !p->width) ||
        p->tokens > SIZE_MAX / sizeof(float) ||
        (p->width && p->tokens > SIZE_MAX / bytes / p->width)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * bytes;
    outputs[0] = p->tokens * p->width;
    outputs[1] = p->tokens * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus quantize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const QuantizeTask *task = opaque;
    const QuantizeParameters *p = task->parameters;
    size_t start = row * p->width;
    float maximum = 1e-5f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        maximum = fmaxf(maximum, fabsf(value));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->input_dtype).f * scale;
        float magnitude = fminf(fabsf(value), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[start + column] = (int8_t)(value < 0 ? -code : code);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_quantize_int8(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const QuantizeParameters *p = parameters;
    QuantizeTask task = {outputs[0], outputs[1], inputs[0], p, context};
    return p->tokens == 1 ? quantize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, quantize, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_quantize_int8_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantize_int8 as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(x)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_quantize_int8_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(QuantizeParameters, tokens), NULL},
        {"width", "size_t", offsetof(QuantizeParameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(QuantizeParameters, input_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(QuantizeParameters), TRILLIM_ALIGNOF(QuantizeParameters),
        3, fields, 1, 2, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_quantize_int8_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
