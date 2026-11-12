#include "common.h"
#include <math.h>
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif


#if defined(__APPLE__) && defined(__aarch64__)
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t rows;
    size_t width;
    uint32_t input_dtype;
    uint32_t output_dtype;
    float eps;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index, vreinterpret_u16_f16(vcvt_f16_f32(value)));
        return;
    }
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        !p->width || p->width > SIZE_MAX / sizeof(float) ||
        p->rows > SIZE_MAX / sizeof(float) / p->width || !isfinite(p->eps) || p->eps < 0.0f) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
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
    if (p->output_dtype == TRILLIM_FP32) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        float32x4_t center = vdupq_n_f32(mean);
        size_t i = 0;
        for (; p->width - i >= 4; i += 4) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t value = vsubq_f32(load4(task->input, offset + i, p->input_dtype), center);
            sum = vaddq_f32(sum, vmulq_f32(value, value));
        }
        variance = vaddvq_f32(sum);
        for (; i < p->width; ++i) {
            float value = trillim_load(task->input, offset + i, p->input_dtype).f - mean;
            variance += value * value;
        }
    } else {
        for (size_t begin = 0; begin < p->width; begin += 1024) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
            for (size_t i = begin; i < end; ++i) {
                float value = trillim_load(task->input, offset + i, p->input_dtype).f - mean;
                variance += value * value;
            }
        }
    }
    float count = (float)p->width;
    float denominator = sqrtf(variance / count + p->eps);
    float32x4_t divisor = vdupq_n_f32(denominator);
    size_t column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = load4(task->input, offset + column, p->input_dtype);
        value = vdivq_f32(vsubq_f32(value, vdupq_n_f32(mean)), divisor);
        store4(task->output, offset + column, p->output_dtype, value);
    }
    for (size_t begin = column; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            float value = trillim_load(task->input, offset + i, p->input_dtype).f;
            TrillimValue result = {.f = (value - mean) / denominator};
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

TRILLIM_KERNEL TrillimStatus trillim_neon_layernorm_plain(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0)
                        : context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_neon_layernorm_plain_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm, _contiguous_strides\n"
           "def bind(c):\n"
           "    x = c.inputs['x']\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('normalization requires floating tensors')\n"
           "    if c.attrs.get('axes', (-1,)) != (-1,):\n"
           "        raise ValueError('normalization requires the final channel axis')\n"
           "    p = _rmsnorm(c)\n"
           "    return {key: value for key, value in p.items() if key in {'rows', 'width', 'input_dtype', 'output_dtype', 'eps'}}\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...] = (-1,), eps: float = 1e-5, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    result = LayerNorm(x, axes=axes, eps=eps)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_layernorm_plain_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        1, 1, requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_layernorm_plain_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
