#include "common.h"
#include <math.h>
typedef struct {
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
    float eps;
} Parameters;
typedef struct {
    void *output;
    const void *input, *weight, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    uint32_t types[] = {p->input_dtype, p->weight_dtype, p->bias_dtype, p->output_dtype};
    for (size_t i = 0; i < 4; ++i)
        if (types[i] != TRILLIM_BF16 && types[i] != TRILLIM_FP32) return TRILLIM_ERROR;
    if (!p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width || !isfinite(p->eps) || p->eps < 0)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    inputs[2] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t offset = row * p->width;
    float sum = 0.0f;
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            sum += trillim_load(task->input, offset + i, p->input_dtype).f;
        }
    }
    float mean = sum / (float)p->width;
    float variance = 0.0f;
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            float centered = trillim_load(task->input, offset + i, p->input_dtype).f - mean;
            variance += centered * centered;
        }
    }
    float inverse = 1.0f / sqrtf(variance / (float)p->width + p->eps);
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            float centered = trillim_load(task->input, offset + i, p->input_dtype).f - mean;
            float weight = trillim_load(task->weight, i, p->weight_dtype).f;
            float bias = trillim_load(task->bias, i, p->bias_dtype).f;
            TrillimValue result = {.f = (centered * inverse) * weight + bias};
            trillim_store(task->output, offset + i, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_layernorm(void *const *outputs, const void *const *inputs, void *scratch,
                                                        const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0)
                        : context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}
TRILLIM_KERNEL const char *trillim_scalar_layernorm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _layernorm as bind\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, axes: tuple[int, ...] = (-1,), eps: float = 1e-5, "
           "dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (LayerNorm(x, weight=weight, bias=bias, axes=axes, eps=eps) if dtype is None else "
           "Cast(LayerNorm(x, weight=weight, bias=bias, axes=axes, eps=eps), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_layernorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 3, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_layernorm_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_LIBM};
    return &capabilities;
}
