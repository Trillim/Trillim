#include "common.h"


typedef struct {
    size_t tokens, width, outputs;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    void *output;
    const int8_t *input, *weight;
    const float *scales;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype) || !p->width || p->width > INT32_MAX / (128 * 127) ||
        p->width > SIZE_MAX - 2 || p->outputs > SIZE_MAX / (p->width + 2) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) return TRILLIM_ERROR;
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * (p->width + 2);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = tile / tiles, begin = (tile % tiles) * 16;
    size_t end = p->outputs - begin < 16 ? p->outputs : begin + 16;
    float multiplier = 1.0f / task->scales[row];
    for (size_t column = begin; column < end; ++column) {
        int64_t sum = 0;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            sum += (int32_t)task->input[row * p->width + k] * task->weight[column * p->width + k];
        }
        uint16_t bits;
        memcpy(&bits, (const uint8_t *)task->weight + p->outputs * p->width + column * 2, sizeof(bits));
        TrillimValue value = {.f = (float)sum * (trillim_bf16(bits) * multiplier)};
        trillim_store(task->output, row * p->outputs + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_row_int8_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->tokens * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_row_int8_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _row_int8_linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_row_int8_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_row_int8_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
