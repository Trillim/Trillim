#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))


TRILLIM_AVX2 static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

TRILLIM_AVX2 static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits,
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)))), 16);
    __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
        _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
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

static inline float trillim_read_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (trillim_itemsize(dtype) != 0) {
        TrillimValue value = trillim_load(data, index, dtype);
        if (trillim_float(dtype)) {
            return value.f;
        }
        return trillim_unsigned(dtype) ? (float)value.u : (float)value.i;
    }
    const uint8_t *values = (const uint8_t *)data;
    size_t row = index / columns;
    size_t column = index % columns;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        int8_t code;
        memcpy(&bits, values + rows * columns + row * 2, 2);
        memcpy(&code, values + index, 1);
        return (float)code * trillim_bf16(bits);
    }
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_BINARY ||
          dtype == TRILLIM_GRP_TERNARY) ? 128 : 32);
    size_t groups = columns / group + (columns % group != 0);
    size_t total = rows * groups;
    size_t block = row * groups + column / group;
    size_t offset = column % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        size_t code = (values[block * 32 + offset % 32] >> (6 - (offset / 32) * 2)) & 3;
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
        return code == 3 ? NAN : (float)((int32_t)code - 1) * scale;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        uint8_t code = (values[block * 16 + offset / 2] >> ((offset % 2) * 4)) & 15;
        uint8_t exponent = values[total * 16 + block];
        float value = magnitudes[code & 7];
        if ((code & 8) != 0) {
            value = -value;
        }
        return exponent == 255 ? NAN : ldexpf(value, (int)exponent - 127);
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 :
        (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    uint16_t bits;
    memcpy(&bits, values + scale_start + block * 2, 2);
    float scale = trillim_bf16(bits);
    if (dtype == TRILLIM_GRP_INT8) {
        int8_t code;
        memcpy(&code, values + block * 32 + offset, 1);
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        int32_t code = ((values[block * 16 + offset / 8] >> (7 - offset % 8)) & 1) ? 1 : -1;
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t code = (values[block * 16 + offset % 16] >> ((offset / 16) * 4)) & 15;
        if (dtype == TRILLIM_GRP_INT5) {
            code |= ((values[scale_start + total * 2 + block * 4 + offset / 8] >> (offset % 8)) & 1) << 4;
        }
        return (float)(code - (dtype == TRILLIM_GRP_INT4 ? 8 : 16)) * scale;
    }
    size_t half = offset / 128;
    size_t lane = (offset % 128) / 32;
    size_t low = block * 128 + half * 64 + (lane / 2) * 32 + offset % 32;
    size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
    int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
    code |= ((values[high] >> (lane * 2)) & 3) << 4;
    int8_t subscale;
    memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
    return (float)((code - 32) * subscale) * scale;
}

TRILLIM_AVX2 static inline __m256 trillim_avx2_load8(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    if (dtype == TRILLIM_BF16) {
        __m128i values = _mm_loadu_si128((const __m128i *)((const uint8_t *)data + index * 2));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(values), 16));
    }
    float values[8];
    for (size_t i = 0; i < 8; ++i) {
        values[i] = trillim_read_float(data, index + i, dtype, rows, columns);
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static inline float trillim_avx2_sum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
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
    const void *weight;
    const TrillimRMSNormParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRMSNormTask;

static inline TrillimStatus trillim_rms_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width || p->rows * p->width > SIZE_MAX / bytes ||
        p->storage_columns == 0 || p->storage_rows > SIZE_MAX / p->storage_columns ||
        p->rows * p->width > p->storage_rows * p->storage_columns ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (!trillim_payload_size(1, p->width, p->weight_dtype, &inputs[1])) {
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
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    size_t column = 0;
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
    float sum = trillim_avx2_sum(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        sum += value * value;
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    __m256 divisor = _mm256_set1_ps(denominator);
    __m256 offset = _mm256_set1_ps(p->weight_offset);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 value = _mm256_div_ps(load8(task->input, start + column, p->input_dtype), divisor);
        __m256 weight = trillim_avx2_load8(task->weight, column, p->weight_dtype, 1, p->width);
        value = _mm256_mul_ps(value, _mm256_add_ps(weight, offset));
        store8(task->output, start + column, p->output_dtype, value);
    }
    for (; column < p->width; ++column) {
        float weight = trillim_read_float(task->weight, column, p->weight_dtype, 1, p->width);
        TrillimValue value = {.f = (trillim_load(task->input, start + column, p->input_dtype).f / denominator) * (weight + p->weight_offset)};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRMSNormParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_rmsnorm_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], inputs[1], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rmsnorm_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rmsnorm as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_rmsnorm_avx2_parameters(void) {
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
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, trillim_rms_requirements,
        can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rmsnorm_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
