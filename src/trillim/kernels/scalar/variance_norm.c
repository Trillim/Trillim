#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif


typedef struct {
    size_t rows;
    size_t width;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    float eps;
    int32_t correction;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const void *weight;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    inputs[1] = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        !trillim_float(p->weight_dtype) ||
        !p->width || p->width > SIZE_MAX / sizeof(float) ||
        p->rows > SIZE_MAX / sizeof(float) / p->width || !isfinite(p->eps) || p->eps < 0.0f) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    size_t offset = row * p->width;
    float total = 0.0f;
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            total += trillim_load(task->input, offset + i, p->input_dtype).f;
        }
    }
    if (p->input_dtype == TRILLIM_BF16) {
        total = trillim_bf16(trillim_to_bf16(total));
    } else if (p->input_dtype == TRILLIM_FP16) {
        total = trillim_fp16(trillim_to_fp16(total));
    }
    float mean = total / (float)p->width;
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
    float count = (float)((double)p->width - (double)p->correction);
    float denominator = sqrtf(variance / count + p->eps);
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            float value = trillim_load(task->input, offset + i, p->input_dtype).f;
            float weight = trillim_load(task->weight, i, p->weight_dtype).f;
            TrillimValue result = {.f = (value / denominator) * weight};
            trillim_store(task->output, offset + i, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_variance_norm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0)
                        : context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_variance_norm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm, _contiguous_strides\n"
           "def bind(c):\n"
           "    x = c.inputs['x']\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('normalization requires floating tensors')\n"
           "    w = c.inputs['weight']\n"
           "    if w.dtype not in {'fp16', 'bf16', 'fp32'} or (w.strides or _contiguous_strides(w.shape)) != _contiguous_strides(w.shape):\n"
           "        raise ValueError('variance normalization requires contiguous floating weight')\n"
           "    correction = c.attrs.get('correction', 1)\n"
           "    if not isinstance(correction, int):\n"
           "        raise TypeError('correction must be an integer')\n"
           "    p = _rmsnorm(c)\n"
           "    return {key: value for key, value in p.items() if key not in {'storage_rows', 'storage_columns', 'weight_offset'}} | {'correction': correction}\n"
           "def pattern(x: Tensor, weight: Tensor, *, correction: int = 1, eps: float = 1e-5, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    result = VarianceNorm(x, weight=weight, correction=correction, eps=eps)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_variance_norm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"correction", "int32_t", offsetof(Parameters, correction), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        2, 1, requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_variance_norm_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
