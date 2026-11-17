#include "common.h"
#include <math.h>
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

TRILLIM_AVX2 static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    __m256i bits = _mm256_cvtepu16_epi32(raw);
    if (dtype == TRILLIM_BF16) return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
    __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
    __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7c00));
    __m256 normal = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(magnitude, 13), _mm256_set1_epi32(0x38000000)));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude), _mm256_set1_ps(0x1p-24f));
    __m256 special = _mm256_castsi256_ps(_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x3ff)), 13), _mm256_set1_epi32(0x7f800000)));
    __m256 value = _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256())));
    value = _mm256_blendv_ps(value, special, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00))));
    return _mm256_xor_ps(value, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)), 16)));
}

TRILLIM_AVX2 static inline void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i absolute = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    __m256i nan = _mm256_cmpgt_epi32(absolute, _mm256_set1_epi32(0x7f800000));
    __m256i rounded;
    if (dtype == TRILLIM_BF16) {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)));
        rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    } else {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0xfff), _mm256_and_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(1)));
        rounded = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(absolute, bias), 13), _mm256_set1_epi32(112 << 10));
        rounded = _mm256_min_epi32(rounded, _mm256_set1_epi32(0x7c00));
        __m256 small = _mm256_mul_ps(_mm256_castsi256_ps(absolute), _mm256_set1_ps(0x1p24f));
        __m256i subnormal = _mm256_cvttps_epi32(_mm256_round_ps(small, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        rounded = _mm256_blendv_epi8(rounded, subnormal, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), absolute));
        __m256i payload = _mm256_or_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(0x200));
        payload = _mm256_and_si256(payload, _mm256_set1_epi32(0x3ff));
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(payload, _mm256_set1_epi32(0x7c00)), nan);
        rounded = _mm256_or_si256(rounded, _mm256_and_si256(upper, _mm256_set1_epi32(0x8000)));
    }
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index), _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

TRILLIM_AVX2 static inline float sum8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

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

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
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

TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
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
        __m256 sum = _mm256_set1_ps(0.0f);
        __m256 center = _mm256_set1_ps(mean);
        size_t i = 0;
        for (; p->width - i >= 8; i += 8) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 value = _mm256_sub_ps(load8(task->input, offset + i, p->input_dtype), center);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(value, value));
        }
        variance = sum8(sum);
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
    float count = (float)((double)p->width - (double)p->correction);
    float denominator = sqrtf(variance / count + p->eps);
    __m256 divisor = _mm256_set1_ps(denominator);
    size_t column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 value = load8(task->input, offset + column, p->input_dtype);
        value = _mm256_mul_ps(_mm256_div_ps(value, divisor), load8(task->weight, column, p->weight_dtype));
        store8(task->output, offset + column, p->output_dtype, value);
    }
    for (size_t begin = column; begin < p->width; begin += 1024) {
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

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_normalize_variance_avx2(
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

TRILLIM_KERNEL const char *trillim_x86_normalize_variance_avx2_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_normalize_variance_avx2_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_normalize_variance_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
