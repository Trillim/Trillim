#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>
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

TRILLIM_AVX2 static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t itemsize = trillim_itemsize(dtype);
    if (columns != 0 && rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (itemsize != 0) {
        if (count > SIZE_MAX / itemsize) {
            return false;
        }
        *bytes = count * itemsize;
        return true;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        if (rows > (SIZE_MAX - count) / 2) {
            return false;
        }
        *bytes = count + rows * 2;
        return true;
    }
    size_t group;
    size_t record;
    switch (dtype) {
    case TRILLIM_MAT_TERNARY: group = 128; record = 32; break;
    case TRILLIM_GRP_INT8: group = 32; record = 34; break;
    case TRILLIM_GRP_INT4: group = 32; record = 18; break;
    case TRILLIM_GRP_INT5: group = 32; record = 22; break;
    case TRILLIM_DUBGRP_INT6: group = 256; record = 210; break;
    case TRILLIM_GRP_BINARY: group = 128; record = 18; break;
    case TRILLIM_GRP_TERNARY: group = 128; record = 34; break;
    case TRILLIM_MXFP4: group = 32; record = 17; break;
    default: return false;
    }
    size_t groups = columns / group + (columns % group != 0);
    if (rows != 0 && groups > SIZE_MAX / record / rows) {
        return false;
    }
    *bytes = rows * groups * record;
    if (dtype == TRILLIM_MAT_TERNARY) {
        if (*bytes > SIZE_MAX - sizeof(float)) {
            return false;
        }
        *bytes += sizeof(float);
    }
    return true;
}

typedef struct {
    size_t rows;
    size_t width;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    float eps;
    float weight_offset;
} TrillimRMSNormParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimRMSNormParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRMSNormTask;

TRILLIM_AVX2 static inline TrillimStatus trillim_rms_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                     size_t *scratch) {
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if ((p->input_dtype != TRILLIM_FP16 && p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP32) ||
        (p->output_dtype != TRILLIM_FP16 && p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32)) {
        return TRILLIM_ERROR;
    }
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width || p->rows * p->width > SIZE_MAX / bytes || p->storage_columns == 0 ||
        p->storage_rows > SIZE_MAX / p->storage_columns || p->rows * p->width > p->storage_rows * p->storage_columns ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * bytes;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = (const TrillimRMSNormTask *)opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    size_t start = row * p->width;
    float sum = 0.0f;
    size_t column = 0;
    if (p->output_dtype != TRILLIM_FP32) {
        for (; p->width - column >= 8; column += 8) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            float lanes[8];
            _mm256_storeu_ps(lanes, load8(task->input, start + column, p->input_dtype));
            for (size_t lane = 0; lane < 8; ++lane) sum += lanes[lane] * lanes[lane];
        }
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column, p->input_dtype).f;
            sum += value * value;
        }
    } else {
        __m256 sum0 = _mm256_set1_ps(0.0f);
        __m256 sum1 = _mm256_set1_ps(0.0f);
        __m256 sum2 = _mm256_set1_ps(0.0f);
        __m256 sum3 = _mm256_set1_ps(0.0f);
        column = 0;
        for (; p->width - column >= 32; column += 32) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 a = load8(task->input, start + column, p->input_dtype);
            __m256 b = load8(task->input, start + column + 8, p->input_dtype);
            __m256 c = load8(task->input, start + column + 16, p->input_dtype);
            __m256 d = load8(task->input, start + column + 24, p->input_dtype);
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a, a));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(b, b));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(c, c));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(d, d));
        }
        sum = sum8(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column, p->input_dtype).f;
            sum += value * value;
        }
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    __m256 divisor = _mm256_set1_ps(denominator);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 value = _mm256_div_ps(load8(task->input, start + column, p->input_dtype), divisor);
        store8(task->output, start + column, p->output_dtype, value);
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = trillim_load(task->input, start + column, p->input_dtype).f / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRMSNormParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_rmsnorm_plain_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rmsnorm_plain_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm as bind\n"
           "def pattern(x: Tensor, *, eps: float = 1e-6, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, eps=eps, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_rmsnorm_plain_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(TrillimRMSNormParameters, rows), NULL},
        {"width", "size_t", offsetof(TrillimRMSNormParameters, width), NULL},
        {"storage_rows", "size_t", offsetof(TrillimRMSNormParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimRMSNormParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, output_dtype), NULL},
        {"eps", "float", offsetof(TrillimRMSNormParameters, eps), NULL},
        {"weight_offset", "float", offsetof(TrillimRMSNormParameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1, trillim_rms_requirements,
        can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rmsnorm_plain_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
