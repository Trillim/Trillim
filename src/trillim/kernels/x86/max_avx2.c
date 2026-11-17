#include "common.h"
#include <math.h>
#include <immintrin.h>

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

TRILLIM_AVX2 static inline __m256 select8(__m256i m, __m256 a, __m256 b) {
    return _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m));
}

TRILLIM_AVX2 static inline __m256i ge8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_GE_OQ));
}

TRILLIM_AVX2 static inline __m256i eq8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_EQ_OQ));
}

TRILLIM_AVX2 static inline __m256 max8(__m256 a, __m256 b) {
    return select8(ge8(a,b),a,b);
}

TRILLIM_AVX2 static inline float max_lane8(__m256 a) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_max_ps(s,_mm_movehl_ps(s,s));
    s = _mm_max_ss(s,_mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}

TRILLIM_AVX2 static inline uint32_t all8(__m256i a) {
    return _mm256_testc_si256(a,_mm256_set1_epi32(-1)) ? UINT32_MAX : 0;
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

TRILLIM_AVX2 static inline float trillim_read_float(
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

TRILLIM_AVX2 static inline TrillimStatus trillim_convert(TrillimValue input, uint32_t from, uint32_t to, TrillimValue *output) {
    if (trillim_float(to)) {
        output->f = trillim_float(from) ? input.f :
            (trillim_unsigned(from) ? (float)input.u : (float)input.i);
    } else if (to == TRILLIM_BOOL) {
        output->u = trillim_float(from) ? input.f != 0.0f : input.u != 0;
    } else if (trillim_float(from)) {
        if (!isfinite(input.f)) {
            return TRILLIM_ERROR;
        }
        size_t bits = trillim_itemsize(to) * 8;
        long double modulus = ldexpl(1.0L, (int)bits);
        long double value = fmodl(truncl((long double)input.f), modulus);
        if (value < 0.0L) {
            value += modulus;
        }
        output->u = (uint64_t)value;
    } else {
        output->u = input.u;
    }
    return TRILLIM_OK;
}

typedef struct {
    size_t outputs;
    size_t output_rank;
    size_t reduce_rank;
    size_t reduce_count;
    const size_t *output_shape;
    const size_t *output_strides;
    const size_t *reduce_shape;
    const size_t *reduce_strides;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimReductionParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimReductionParameters *parameters;
    const TrillimKernelContext *context;
    bool contiguous;
    size_t count;
    size_t block;
} TrillimReductionTask;

TRILLIM_AVX2 static bool shape_product(size_t rank, const size_t *shape, size_t *product) {
    if (rank && !shape) {
        return false;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!shape[axis]) {
            *product = 0;
            return true;
        }
    }
    size_t result = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (result > SIZE_MAX / shape[axis]) {
            return false;
        }
        result *= shape[axis];
    }
    *product = result;
    return true;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_reduction_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimReductionParameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t output_count, reduce_count;
    if (!bytes || p->input_dtype > TRILLIM_MXFP4 ||
        (p->output_rank && !p->output_strides) || (p->reduce_rank && !p->reduce_strides) ||
        !shape_product(p->output_rank, p->output_shape, &output_count) ||
        !shape_product(p->reduce_rank, p->reduce_shape, &reduce_count) ||
        output_count != p->outputs || reduce_count != p->reduce_count ||
        p->outputs > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->outputs * bytes;
    if (p->outputs && p->reduce_count) {
        size_t outer_span, inner_span;
        if ((p->storage_columns && p->storage_rows > SIZE_MAX / p->storage_columns) ||
            !trillim_span(p->output_rank, p->output_shape, p->output_strides, &outer_span) ||
            !trillim_span(p->reduce_rank, p->reduce_shape, p->reduce_strides, &inner_span)) {
            return TRILLIM_ERROR;
        }
        size_t capacity = p->storage_rows * p->storage_columns;
        if (!outer_span || !inner_span || outer_span - 1 > SIZE_MAX - inner_span ||
            outer_span - 1 + inner_span > capacity ||
            !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
            return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static inline bool trillim_greater_equal(TrillimValue a, TrillimValue b, uint32_t dtype) {
    if (trillim_float(dtype)) {
        return a.f >= b.f;
    }
    return trillim_unsigned(dtype) ? a.u >= b.u : a.i >= b.i;
}

TRILLIM_AVX2 static size_t effective_count(const TrillimReductionParameters *p) {
    if (!p->reduce_count) {
        return 0;
    }
    for (size_t axis = 0; axis < p->reduce_rank; ++axis) {
        if (p->reduce_shape[axis] > 1 && p->reduce_strides[axis] != 0) {
            return p->reduce_count;
        }
    }
    return 1;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_reduction_task(void *opaque, size_t output_index, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = (const TrillimReductionTask *)opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t base = trillim_offset(output_index, p->output_rank, p->output_shape, p->output_strides);
    TrillimValue accumulator;
    accumulator.u = 0;
    if (trillim_float(p->output_dtype)) {
        accumulator.f = 0.0f;
    }
    bool valid = false;
    uint32_t compare_dtype = trillim_itemsize(p->input_dtype) == 0 ? TRILLIM_FP32 : p->input_dtype;
    for (size_t start = 0; start < task->count;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = task->count - start < 1024 ? task->count : start + 1024;
        for (size_t index = start; index < end; ++index) {
            size_t source = base + trillim_offset(index, p->reduce_rank, p->reduce_shape, p->reduce_strides);
            TrillimValue value;
            if (trillim_itemsize(p->input_dtype) == 0) {
                value.f = trillim_read_float(task->input, source, p->input_dtype, p->storage_rows, p->storage_columns);
            } else {
                value = trillim_load(task->input, source, p->input_dtype);
            }
            if (!valid || !trillim_greater_equal(accumulator, value, compare_dtype)) {
                accumulator = value;
            }
            valid = true;
        }
        start = end;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (valid) {
        TrillimValue converted;
        TrillimStatus status = trillim_convert(accumulator, compare_dtype, p->output_dtype, &converted);
        if (status != TRILLIM_OK) {
            return status;
        }
        accumulator = converted;
    }
    trillim_store(task->output, output_index, p->output_dtype, accumulator);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static float native_float(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(value);
    }
#if defined(__aarch64__) && defined(__FLT16_MANT_DIG__)
    if ((value & 0x7c00) != 0x7c00) {
        __extension__ _Float16 half;
        memcpy(&half, &value, sizeof(half));
        return (float)half;
    }
#endif
    return trillim_fp16(value);
}

TRILLIM_AVX2 static TrillimStatus max_sequence(const TrillimReductionTask *task, size_t index, size_t base) {
    const TrillimReductionParameters *p = task->parameters;
    float maximum = task->count ? native_float(task->input, base, p->input_dtype) : 0.0f;
    for (size_t k = 1; k < task->count; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = task->contiguous ? k : (p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides));
        float value = native_float(task->input, base + offset, p->input_dtype);
        maximum = maximum >= value ? maximum : value;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue result = {.f = maximum};
    trillim_store(task->output, index, p->output_dtype, result);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus max_float(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t base = trillim_offset(index, p->output_rank, p->output_shape, p->output_strides);
    if (!task->contiguous || task->count < 8) {
        return max_sequence(task, index, base);
    }
    float maxima[8];
    unsigned has_nan = 0;
    for (size_t lane = 0; lane < 8; ++lane) {
        maxima[lane] = native_float(task->input, base + lane, p->input_dtype);
        has_nan |= isnan(maxima[lane]);
    }
    size_t k = 8;
    for (; task->count - k >= 8; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t lane = 0; lane < 8; ++lane) {
            float value = native_float(task->input, base + k + lane, p->input_dtype);
            has_nan |= isnan(value);
            maxima[lane] = maxima[lane] >= value ? maxima[lane] : value;
        }
    }
    float maximum = maxima[0];
    for (size_t lane = 1; lane < 8; ++lane) {
        maximum = maximum >= maxima[lane] ? maximum : maxima[lane];
    }
    for (; k < task->count; ++k) {
        float value = native_float(task->input, base + k, p->input_dtype);
        has_nan |= isnan(value);
        maximum = maximum >= value ? maximum : value;
    }
    if (has_nan) {
        return max_sequence(task, index, base);
    }
    if (maximum == 0.0f) {
        for (size_t i = 0; i < task->count; ++i) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float value = native_float(task->input, base + i, p->input_dtype);
            if (value == 0.0f) {
                maximum = value;
                break;
            }
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue result = {.f = maximum};
    trillim_store(task->output, index, p->output_dtype, result);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus max_columns(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t columns = p->output_shape[p->output_rank - 1];
    size_t tiles = columns / 64 + (columns % 64 != 0);
    size_t first = tile % tiles * 64;
    size_t count = columns - first < 64 ? columns - first : 64;
    size_t output = tile / tiles * columns + first;
    size_t base = trillim_offset(output, p->output_rank, p->output_shape, p->output_strides);
    float maxima[64];
    for (size_t lane = 0; lane < count; ++lane) {
        maxima[lane] = task->count ? native_float(task->input, base + lane, p->input_dtype) : 0.0f;
    }
    for (size_t k = 1; k < task->count; ++k) {
        if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
        for (size_t lane = 0; lane < count; ++lane) {
            float value = native_float(task->input, base + offset + lane, p->input_dtype);
            maxima[lane] = maxima[lane] >= value ? maxima[lane] : value;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t lane = 0; lane < count; ++lane) {
        TrillimValue result = {.f = maxima[lane]};
        trillim_store(task->output, output + lane, p->output_dtype, result);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus max_row_simd(void *opaque, size_t index, size_t worker) {
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    if (!task->contiguous || task->count < 32) {
        return max_float(opaque, index, worker);
    }
    size_t base = trillim_offset(index, p->output_rank, p->output_shape, p->output_strides);
    __m256 m0 = load8(task->input, base + 0, p->input_dtype);
    __m256i valid0 = eq8(m0, m0);
    __m256 m1 = load8(task->input, base + 8, p->input_dtype);
    __m256i valid1 = eq8(m1, m1);
    __m256 m2 = load8(task->input, base + 16, p->input_dtype);
    __m256i valid2 = eq8(m2, m2);
    __m256 m3 = load8(task->input, base + 24, p->input_dtype);
    __m256i valid3 = eq8(m3, m3);
    size_t k = 32;
    for (; task->count - k >= 32; k += 32) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 v0 = load8(task->input, base + k + 0, p->input_dtype);
        valid0 = _mm256_and_si256(valid0, eq8(v0, v0));
        m0 = max8(m0, v0);
        __m256 v1 = load8(task->input, base + k + 8, p->input_dtype);
        valid1 = _mm256_and_si256(valid1, eq8(v1, v1));
        m1 = max8(m1, v1);
        __m256 v2 = load8(task->input, base + k + 16, p->input_dtype);
        valid2 = _mm256_and_si256(valid2, eq8(v2, v2));
        m2 = max8(m2, v2);
        __m256 v3 = load8(task->input, base + k + 24, p->input_dtype);
        valid3 = _mm256_and_si256(valid3, eq8(v3, v3));
        m3 = max8(m3, v3);
    }
    __m256 maximum = max8(max8(m0, m1), max8(m2, m3));
    __m256i valid = _mm256_and_si256(_mm256_and_si256(valid0, valid1), _mm256_and_si256(valid2, valid3));
    for (; task->count - k >= 8; k += 8) {
        __m256 value = load8(task->input, base + k, p->input_dtype);
        valid = _mm256_and_si256(valid, eq8(value, value));
        maximum = max8(maximum, value);
    }
    if (all8(valid) != UINT32_MAX) {
        return max_sequence(task, index, base);
    }
    float result = max_lane8(maximum);
    for (; k < task->count; ++k) {
        float value = native_float(task->input, base + k, p->input_dtype);
        if (isnan(value)) {
            return max_sequence(task, index, base);
        }
        result = result >= value ? result : value;
    }
    if (result == 0.0f) {
        for (size_t i = 0; i < task->count; ++i) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float value = native_float(task->input, base + i, p->input_dtype);
            if (value == 0.0f) {
                result = value;
                break;
            }
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue value = {.f = result};
    trillim_store(task->output, index, p->output_dtype, value);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus max_columns_simd(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t columns = p->output_shape[p->output_rank - 1];
    size_t tiles = columns / 64 + (columns % 64 != 0);
    size_t first = tile % tiles * 64;
    size_t count = columns - first < 64 ? columns - first : 64;
    if (count < 8) {
        return max_columns(opaque, tile, worker);
    }
    size_t output = tile / tiles * columns + first;
    size_t base = trillim_offset(output, p->output_rank, p->output_shape, p->output_strides);
    size_t lane = 0;
    for (; count - lane >= 64; lane += 64) {
        __m256 m0 = task->count ? load8(task->input, base + lane + 0, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m1 = task->count ? load8(task->input, base + lane + 8, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m2 = task->count ? load8(task->input, base + lane + 16, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m3 = task->count ? load8(task->input, base + lane + 24, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m4 = task->count ? load8(task->input, base + lane + 32, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m5 = task->count ? load8(task->input, base + lane + 40, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m6 = task->count ? load8(task->input, base + lane + 48, p->input_dtype) : _mm256_set1_ps(0.0f);
        __m256 m7 = task->count ? load8(task->input, base + lane + 56, p->input_dtype) : _mm256_set1_ps(0.0f);
        for (size_t k = 1; k < task->count; ++k) {
            if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
                trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
            __m256 v0 = load8(task->input, base + offset + lane + 0, p->input_dtype);
            m0 = select8(ge8(m0, v0), m0, v0);
            __m256 v1 = load8(task->input, base + offset + lane + 8, p->input_dtype);
            m1 = select8(ge8(m1, v1), m1, v1);
            __m256 v2 = load8(task->input, base + offset + lane + 16, p->input_dtype);
            m2 = select8(ge8(m2, v2), m2, v2);
            __m256 v3 = load8(task->input, base + offset + lane + 24, p->input_dtype);
            m3 = select8(ge8(m3, v3), m3, v3);
            __m256 v4 = load8(task->input, base + offset + lane + 32, p->input_dtype);
            m4 = select8(ge8(m4, v4), m4, v4);
            __m256 v5 = load8(task->input, base + offset + lane + 40, p->input_dtype);
            m5 = select8(ge8(m5, v5), m5, v5);
            __m256 v6 = load8(task->input, base + offset + lane + 48, p->input_dtype);
            m6 = select8(ge8(m6, v6), m6, v6);
            __m256 v7 = load8(task->input, base + offset + lane + 56, p->input_dtype);
            m7 = select8(ge8(m7, v7), m7, v7);
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, output + lane + 0, p->output_dtype, m0);
        store8(task->output, output + lane + 8, p->output_dtype, m1);
        store8(task->output, output + lane + 16, p->output_dtype, m2);
        store8(task->output, output + lane + 24, p->output_dtype, m3);
        store8(task->output, output + lane + 32, p->output_dtype, m4);
        store8(task->output, output + lane + 40, p->output_dtype, m5);
        store8(task->output, output + lane + 48, p->output_dtype, m6);
        store8(task->output, output + lane + 56, p->output_dtype, m7);
    }
    for (; count - lane >= 8; lane += 8) {
        __m256 maximum = task->count ? load8(task->input, base + lane, p->input_dtype) : _mm256_set1_ps(0.0f);
        for (size_t k = 1; k < task->count; ++k) {
            if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
                trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
            __m256 value = load8(task->input, base + offset + lane, p->input_dtype);
            maximum = select8(ge8(maximum, value), maximum, value);
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, output + lane, p->output_dtype, maximum);
    }
    for (; lane < count; ++lane) {
        TrillimStatus status = max_sequence(task, output + lane, base + lane);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus max_rows(void *opaque, size_t tile, size_t worker) {
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t first = tile * task->block;
    size_t end = p->outputs - first < task->block ? p->outputs : first + task->block;
    for (size_t row = first; row < end; ++row) {
        TrillimStatus status = max_row_simd(opaque, row, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_max_simd_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimReductionParameters *p = (const TrillimReductionParameters *)parameters;
    TrillimReductionTask task = {outputs[0], inputs[0], p, context,
        trillim_contiguous(p->reduce_rank, p->reduce_shape, p->reduce_strides), effective_count(p), 1};
    if (!p->outputs) {
        return TRILLIM_OK;
    }
    bool small = !task.count || p->outputs <= 8192 / task.count;
    TrillimTask function = trillim_reduction_task;
    size_t tasks = p->outputs;
    if (trillim_float(p->input_dtype) && trillim_float(p->output_dtype)) {
        size_t block = task.count ? 16384 / task.count : 32;
        size_t workers = context->thread_pool->worker_count;
        size_t parallel = workers > 1 ? p->outputs / workers / 4 : 32;
        if (block > 32) {
            block = 32;
        }
        if (block > parallel) {
            block = parallel;
        }
        if (!block) {
            block = 1;
        }
        task.block = small ? p->outputs : block;
        tasks = p->outputs / task.block + (p->outputs % task.block != 0);
        function = max_rows;
        if (p->output_rank && p->output_strides[p->output_rank - 1] == 1) {
            size_t columns = p->output_shape[p->output_rank - 1];
            size_t tiles = columns / 64 + (columns % 64 != 0);
            tasks = p->outputs / columns * tiles;
            function = max_columns_simd;
        }
    }
    if (small) {
        for (size_t i = 0; i < tasks; ++i) {
            TrillimStatus status = function(&task, i, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return tasks == 1 ? function(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tasks, function, &task);
}

TRILLIM_KERNEL const char *trillim_x86_max_simd_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bind_reduction as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> tuple[Tensor, ...]:\n"
           "    return (Max(x, axes=axes, keepdims=keepdims),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_max_simd_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"outputs", "size_t", offsetof(TrillimReductionParameters, outputs), NULL},
        {"output_rank", "size_t", offsetof(TrillimReductionParameters, output_rank), NULL},
        {"reduce_rank", "size_t", offsetof(TrillimReductionParameters, reduce_rank), NULL},
        {"reduce_count", "size_t", offsetof(TrillimReductionParameters, reduce_count), NULL},
        {"output_shape", "size_t", offsetof(TrillimReductionParameters, output_shape), "output_rank"},
        {"output_strides", "size_t", offsetof(TrillimReductionParameters, output_strides), "output_rank"},
        {"reduce_shape", "size_t", offsetof(TrillimReductionParameters, reduce_shape), "reduce_rank"},
        {"reduce_strides", "size_t", offsetof(TrillimReductionParameters, reduce_strides), "reduce_rank"},
        {"storage_rows", "size_t", offsetof(TrillimReductionParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimReductionParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimReductionParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimReductionParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimReductionParameters), TRILLIM_ALIGNOF(TrillimReductionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_reduction_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_max_simd_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
