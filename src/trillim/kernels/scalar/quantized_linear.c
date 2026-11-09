#include "common.h"
#include <math.h>

typedef struct {
    size_t tokens;
    size_t width;
    size_t outputs;
    uint32_t output_dtype;
} LinearParameters;

typedef struct {
    void *output;
    const int8_t *codes;
    const uint8_t *weight;
    const float *scales;
    const LinearParameters *parameters;
    const TrillimKernelContext *context;
} LinearTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const LinearParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!p->width || p->width > INT64_MAX / 128 || groups > SIZE_MAX / 32 ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        p->outputs > (SIZE_MAX - sizeof(float)) / (groups * 32) ||
        p->tokens > SIZE_MAX / bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * 32 + sizeof(float);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = p->outputs / 32 + (p->outputs % 32 != 0);
    size_t row = index / tiles;
    size_t begin = (index % tiles) * 32;
    size_t end = p->outputs - begin < 32 ? p->outputs : begin + 32;
    size_t packed_row = (p->width / 128 + (p->width % 128 != 0)) * 32;
    float gamma;
    memcpy(&gamma, task->weight + p->outputs * packed_row, sizeof(gamma));
    float scale = task->scales[row];
    if (!isfinite(scale) || scale <= 0) {
        return TRILLIM_ERROR;
    }
    for (size_t column = begin; column < end; ++column) {
        int64_t sum = 0;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t packed = task->weight[column * packed_row + (k / 128) * 32 + k % 32];
            int32_t code = (packed >> (6 - ((k % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            sum += (int64_t)task->codes[row * p->width + k] * (code - 1);
        }
        TrillimValue result;
        result.f = ((float)sum * gamma) / scale;
        trillim_store(task->output, row * p->outputs + column, p->output_dtype, result);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_quantized_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const LinearParameters *p = parameters;
    LinearTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t tiles = p->outputs / 32 + (p->outputs % 32 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->tokens * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_quantized_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantized_linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_quantized_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(LinearParameters, tokens), NULL},
        {"width", "size_t", offsetof(LinearParameters, width), NULL},
        {"outputs", "size_t", offsetof(LinearParameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(LinearParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(LinearParameters), TRILLIM_ALIGNOF(LinearParameters),
        4, fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_quantized_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
