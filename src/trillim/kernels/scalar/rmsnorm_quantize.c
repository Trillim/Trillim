#include "common.h"
#include <float.h>
#include <math.h>


typedef struct {
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, output_dtype;
    float eps, weight_offset;
} Parameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input, *weight;
    void *normalized;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = outputs[1] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP32) ||
        (p->weight_dtype != TRILLIM_BF16 && p->weight_dtype != TRILLIM_FP32) ||
        (p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32) ||
        !p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width) return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    outputs[0] = p->rows * p->width;
    outputs[1] = p->rows * sizeof(float);
    *scratch = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width, full = p->width / 8 * 8;
    float sums[8] = {0};
    for (size_t column = 0; column < full; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        sums[column % 8] += value * value;
    }
    float sum = 0;
    for (size_t i = 0; i < 8; ++i) sum += sums[i];
    for (size_t column = full; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        sum += value * value;
    }
    float inverse = 1.0f / sqrtf(sum / (float)p->width + p->eps), maximum = 1e-5f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        TrillimValue value = {.f = (trillim_load(task->input, start + column, p->input_dtype).f * inverse) *
            (trillim_load(task->weight, column, p->weight_dtype).f + p->weight_offset)};
        trillim_store(task->normalized, start + column, p->output_dtype, value);
        float rounded = trillim_load(task->normalized, start + column, p->output_dtype).f;
        if (!isfinite(rounded)) return TRILLIM_ERROR;
        maximum = fmaxf(maximum, fabsf(rounded));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float value = trillim_load(task->normalized, start + column, p->output_dtype).f * scale;
        task->codes[start + column] = (int8_t)nearbyintf(fminf(fmaxf(value, -127.0f), 127.0f));
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_rmsnorm_quantize(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], outputs[1], inputs[0], inputs[1], scratch, p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_rmsnorm_quantize_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm_quantize as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_rmsnorm_quantize_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 2, 2,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_rmsnorm_quantize_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
