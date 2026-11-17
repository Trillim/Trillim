#include "common.h"
#include <immintrin.h>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t batches;
    size_t m;
    size_t n;
    size_t k;
    size_t rank;
    const size_t *batch_shape;
    const size_t *a_batch_strides;
    const size_t *b_batch_strides;
    size_t a_row_stride;
    size_t a_col_stride;
    size_t b_row_stride;
    size_t b_col_stride;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t out_dtype;
    size_t a_storage_rows;
    size_t a_storage_columns;
    size_t b_storage_rows;
    size_t b_storage_columns;
} TrillimMatmulParameters;

static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum,sum));
}
static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits),16));
    }
    if (dtype == TRILLIM_FP16) {
        __m128i packed = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        __m256i bits = _mm256_cvtepu16_epi32(packed);
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x8000)),16);
        __m256i exponent = _mm256_and_si256(bits,_mm256_set1_epi32(0x7c00));
        __m256i mantissa = _mm256_and_si256(bits,_mm256_set1_epi32(0x03ff));
        __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x7fff)),13),_mm256_set1_epi32(0x38000000));
        __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa),_mm256_set1_ps(0x1p-24f));
        __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000),_mm256_slli_epi32(mantissa,13));
        __m256 result = _mm256_blendv_ps(_mm256_castsi256_ps(normal),subnormal,_mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent,_mm256_setzero_si256())));
        result = _mm256_blendv_ps(result,_mm256_castsi256_ps(special),_mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent,_mm256_set1_epi32(0x7c00))));
        return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(result),sign));
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) values[lane] = trillim_load(data,index + lane,dtype).f;
    return _mm256_loadu_ps(values);
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

typedef struct {
    void *output;
    const void *a;
    const uint8_t *b;
    const TrillimMatmulParameters *p;
    size_t columns;
    const TrillimKernelContext *context;
} Task;

static size_t group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
}

static inline bool trillim_matrix_span(size_t base, size_t rows, size_t columns, size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - base) / row_stride) {
        return false;
    }
    base += (rows - 1) * row_stride;
    if (col_stride != 0 && columns - 1 > (SIZE_MAX - base) / col_stride) {
        return false;
    }
    return base + (columns - 1) * col_stride <= capacity;
}

static inline TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->out_dtype) ||
        p->b_dtype < TRILLIM_ROW_INT8 || p->b_dtype > TRILLIM_MXFP4 ||
        !p->k || p->k % 16 || p->a_col_stride != 1 || p->b_row_stride != 1 ||
        p->b_col_stride != p->k || !p->b_storage_columns || p->b_storage_columns % p->k) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->b_batch_strides[axis] % p->k) {
            return TRILLIM_ERROR;
        }
    }
    size_t bytes = trillim_itemsize(p->out_dtype);
    size_t a_span, b_span;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (bytes == 0 || !trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span)) {
        return TRILLIM_ERROR;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if (p->a_storage_columns == 0 || p->b_storage_columns == 0 ||
            p->a_storage_rows > SIZE_MAX / p->a_storage_columns ||
            p->b_storage_rows > SIZE_MAX / p->b_storage_columns ||
            !trillim_matrix_span(a_span, p->m, p->k, p->a_row_stride, p->a_col_stride, p->a_storage_rows * p->a_storage_columns) ||
            !trillim_matrix_span(b_span, p->k, p->n, p->b_row_stride, p->b_col_stride, p->b_storage_rows * p->b_storage_columns) ||
            !trillim_payload_size(p->a_storage_rows, p->a_storage_columns, p->a_dtype, &inputs[0]) ||
            !trillim_payload_size(p->b_storage_rows, p->b_storage_columns, p->b_dtype, &inputs[1])) {
            return TRILLIM_ERROR;
        }
    }
    outputs[0] = p->batches * p->m * p->n * bytes;
    return TRILLIM_OK;
}

static float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static float load_fp32(const void *data, size_t index) {
    float value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return value;
}

static float load_fp16(const void *data, size_t index) {
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return trillim_fp16(value);
}

static float load_bf16(const void *data, size_t index) {
    return read_bf16((const uint8_t *)data + index * sizeof(uint16_t));
}

static inline __m256 decode8(const uint8_t *values, size_t rows, size_t columns, uint32_t dtype, size_t row, size_t k) {
    __m256i ones = _mm256_set1_epi32(1), mask = _mm256_set1_epi32(15);
    __m256i codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = read_bf16(values + rows * columns + row * 2);
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + row * columns + k)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(codes),_mm256_set1_ps(scale));
    }
    size_t group = group_width(dtype), groups = columns / group + (columns % group != 0);
    size_t total = rows * groups, block = row * groups + k / group, offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        if (dtype == TRILLIM_MAT_TERNARY) memcpy(&scale,values + total * 32,4);
        else scale = read_bf16(values + total * 32 + block * 2);
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset % 32)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(6 - offset / 32 * 2))),_mm256_set1_epi32(3));
        __m256 decoded = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(codes,ones)),_mm256_set1_ps(scale));
        return _mm256_blendv_ps(decoded,_mm256_set1_ps(NAN),_mm256_castsi256_ps(_mm256_cmpeq_epi32(codes,_mm256_set1_epi32(3))));
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float palette[16] = {0,.5f,1,1.5f,2,3,4,6,-0.0f,-.5f,-1,-1.5f,-2,-3,-4,-6};
        uint32_t bytes;
        memcpy(&bytes,values + block * 16 + offset / 2,4);
        __m128i packed = _mm_cvtsi32_si128((int)bytes);
        __m128i low = _mm_and_si128(packed,_mm_set1_epi8(15));
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed,4),_mm_set1_epi8(15));
        codes = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low,high));
        uint8_t exponent = values[total * 16 + block];
        scale = exponent == 255 ? NAN : ldexpf(1.0f,(int)exponent - 127);
        return _mm256_mul_ps(_mm256_i32gather_ps(palette,codes,4),_mm256_set1_ps(scale));
    }
    size_t width = dtype == TRILLIM_GRP_INT8 ? 32 : dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16;
    size_t scale_start = total * width;
    scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset)));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        codes = _mm256_srlv_epi32(_mm256_set1_epi32(values[block * 16 + offset / 8]),_mm256_setr_epi32(7,6,5,4,3,2,1,0));
        codes = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(codes,ones),1),ones);
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 16 + offset % 16)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(offset / 16 * 4))),mask);
        if (dtype == TRILLIM_GRP_INT5) {
            __m256i high = _mm256_srlv_epi32(_mm256_set1_epi32(values[scale_start + total * 2 + block * 4 + offset / 8]),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
            codes = _mm256_or_si256(codes,_mm256_slli_epi32(_mm256_and_si256(high,ones),4));
        }
        codes = _mm256_sub_epi32(codes,_mm256_set1_epi32(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
    } else {
        size_t half = offset / 128, lane = offset % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + offset % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + low)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(lane % 2 * 4))),mask);
        __m256i upper = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + high)));
        upper = _mm256_and_si256(_mm256_srlv_epi32(upper,_mm256_set1_epi32((int)(lane * 2))),_mm256_set1_epi32(3));
        codes = _mm256_sub_epi32(_mm256_or_si256(codes,_mm256_slli_epi32(upper,4)),_mm256_set1_epi32(32));
        int8_t subscale;
        memcpy(&subscale,values + scale_start + total * 66 + block * 16 + offset / 16,1);
        codes = _mm256_mullo_epi32(codes,_mm256_set1_epi32(subscale));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes),_mm256_set1_ps(scale));
}
static void decode_scalar(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
    const TrillimMatmulParameters *p = task->p;
    const uint8_t *values = task->b;
    uint32_t dtype = p->b_dtype;
    if (dtype == TRILLIM_ROW_INT8) {
        float scale = read_bf16(values + p->b_storage_rows * p->b_storage_columns + column * 2);
        const int8_t *codes = (const int8_t *)values + column * p->b_storage_columns + k;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    size_t group = group_width(dtype);
    size_t groups = p->b_storage_columns / group + (p->b_storage_columns % group != 0);
    size_t total = p->b_storage_rows * groups;
    size_t block = column * groups + k / group;
    size_t offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            scale = read_bf16(values + total * 32 + block * 2);
        }
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 32 + (offset + i) % 32] >> (6 - ((offset + i) / 32) * 2)) & 3;
            decoded[i] = code == 3 ? NAN : (float)(code - 1) * scale;
        }
        return;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        uint8_t exponent = values[total * 16 + block];
        uint32_t bits = exponent != 0 ? (uint32_t)exponent << 23 : UINT32_C(0x00400000);
        float scale;
        memcpy(&scale, &bits, sizeof(scale));
        float palette[16];
        for (size_t i = 0; i < 8; ++i) {
            palette[i] = exponent == 255 ? NAN : magnitudes[i] * scale;
            palette[i + 8] = exponent == 255 ? NAN : -palette[i];
        }
        for (size_t i = 0; i < count; ++i) {
            uint8_t code = (values[block * 16 + (offset + i) / 2] >> (((offset + i) % 2) * 4)) & 15;
            decoded[i] = palette[code];
        }
        return;
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    float scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)values + block * 32 + offset;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        for (size_t i = 0; i < count; ++i) {
            bool positive = (values[block * 16 + (offset + i) / 8] >> (7 - (offset + i) % 8)) & 1;
            decoded[i] = positive ? scale : -scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t zero = dtype == TRILLIM_GRP_INT4 ? 8 : 16;
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 16 + (offset + i) % 16] >> (((offset + i) / 16) * 4)) & 15;
            if (dtype == TRILLIM_GRP_INT5) {
                code |= ((values[scale_start + total * 2 + block * 4 + (offset + i) / 8] >> ((offset + i) % 8)) & 1) << 4;
            }
            decoded[i] = (float)(code - zero) * scale;
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t position = offset + i;
        size_t half = position / 128;
        size_t lane = position % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + position % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + position % 32;
        int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
        code |= ((values[high] >> (lane * 2)) & 3) << 4;
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + position / 16, sizeof(subscale));
        decoded[i] = (float)((code - 32) * subscale) * scale;
    }
}
static void decode_block(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
    const TrillimMatmulParameters *p = task->p;
    if (count % 8 || k % 8) { decode_scalar(task,column,k,count,decoded); return; }
    for (size_t i = 0; i < count; i += 8) _mm256_storeu_ps(decoded + i,decode8(task->b,p->b_storage_rows,p->b_storage_columns,p->b_dtype,column,k+i));
}

#define TRILLIM_PACKED_MATMUL(NAME, LOAD) \
static TrillimStatus project_##NAME(void *opaque, size_t index, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const TrillimMatmulParameters *p = task->p; \
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0); \
    size_t columns = p->n / task->columns + (p->n % task->columns != 0); \
    size_t batch = index / columns / row_tiles; \
    size_t row = index / columns % row_tiles * 4; \
    size_t rows = p->m - row < 4 ? p->m - row : 4; \
    size_t first = index % columns * task->columns; \
    size_t end = p->n - first < task->columns ? p->n : first + task->columns; \
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride; \
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides); \
    size_t output = (batch * p->m + row) * p->n; \
    size_t group = group_width(p->b_dtype); \
    for (size_t column = first; column < end; ++column) { \
        size_t flat = b_base + column * p->k; \
        size_t storage_row = flat / p->b_storage_columns; \
        size_t storage_column = flat % p->b_storage_columns; \
        float sums[4] = {0}; \
        for (size_t k = 0; k < p->k;) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t count = group - (storage_column + k) % group; \
            if (count > p->k - k) { \
                count = p->k - k; \
            } \
            float decoded[256]; \
            decode_block(task, storage_row, storage_column + k, count, decoded); \
            size_t i = 0; \
            __m256 accumulators[4] = {_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()}; \
            for (; count - i >= 8; i += 8) { \
                __m256 weight = _mm256_loadu_ps(decoded + i); \
                for (size_t r = 0; r < rows; ++r) { \
                    __m256 x = load8(task->a,a_base + k + r * p->a_row_stride + i,p->a_dtype); \
                    accumulators[r] = _mm256_add_ps(accumulators[r],_mm256_mul_ps(x,weight)); \
                } \
            } \
            for (size_t r = 0; r < rows; ++r) sums[r] += sum8(accumulators[r]); \
            for (; i < count; ++i) { \
                float weight = load_fp32(decoded, i); \
                sums[0] = multiply_add(LOAD(task->a, a_base + k + i), weight, sums[0]); \
                if (rows > 1) { \
                    sums[1] = multiply_add(LOAD(task->a, a_base + p->a_row_stride + k + i), weight, sums[1]); \
                } \
                if (rows > 2) { \
                    sums[2] = multiply_add(LOAD(task->a, a_base + 2 * p->a_row_stride + k + i), weight, sums[2]); \
                } \
                if (rows > 3) { \
                    sums[3] = multiply_add(LOAD(task->a, a_base + 3 * p->a_row_stride + k + i), weight, sums[3]); \
                } \
            } \
            k += count; \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue result = {.f = sums[r]}; \
            trillim_store(task->output, output + r * p->n + column, p->out_dtype, result); \
        } \
    } \
    return TRILLIM_OK; \
}

TRILLIM_PACKED_MATMUL(fp32, load_fp32)
TRILLIM_PACKED_MATMUL(fp16, load_fp16)
TRILLIM_PACKED_MATMUL(bf16, load_bf16)

#undef TRILLIM_PACKED_MATMUL

TRILLIM_KERNEL TrillimStatus trillim_x86_block_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimMatmulParameters *p = parameters;
    if (!p->batches || !p->m || !p->n) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t columns = p->n / workers / 4;
    columns = columns / 16 * 16;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, columns, context};
    size_t rows = p->m / 4 + (p->m % 4 != 0);
    size_t tiles = p->n / columns + (p->n % columns != 0);
    TrillimTask project;
    switch (p->a_dtype) {
    case TRILLIM_FP32: project = project_fp32; break;
    case TRILLIM_BF16: project = project_bf16; break;
    case TRILLIM_FP16: project = project_fp16; break;
    default: return TRILLIM_ERROR;
    }
    size_t tasks = p->batches * rows * tiles;
    return tasks == 1 ? project(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tasks, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_block_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _matmul\n"
           "def bind(c):\n"
           "    if c.inputs['a'].dtype not in {'fp16', 'bf16', 'fp32'} or c.inputs['b'].dtype not in {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('packed matmul requires floating activations and compressed weights')\n"
           "    p = _matmul(c)\n"
           "    width = p['k'].value\n"
           "    columns = p['b_storage_columns'].value\n"
           "    if width is None or width <= 0 or width % 16 or columns is None or columns <= 0 or columns % width:\n"
           "        raise ValueError('packed matmul requires aligned fixed-width rows')\n"
           "    if p['a_col_stride'].value != 1 or p['b_row_stride'].value != 1 or p['b_col_stride'] != p['k']:\n"
           "        raise ValueError('packed matmul requires contiguous reduction axes and output-major weights')\n"
           "    def aligned(stride):\n"
           "        if stride.value is not None: return stride.value % width == 0\n"
           "        if stride.op == 'broadcast_stride': return aligned(stride.args[1])\n"
           "        if stride.op == 'mul': return any(aligned(part) for part in stride.args)\n"
           "        return False\n"
           "    if any(not aligned(stride) for stride in p['b_batch_strides']):\n"
           "        raise ValueError('packed matmul batch strides must preserve complete weight vectors')\n"
           "    return p\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_block_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(TrillimMatmulParameters, batches), NULL},
        {"m", "size_t", offsetof(TrillimMatmulParameters, m), NULL},
        {"n", "size_t", offsetof(TrillimMatmulParameters, n), NULL},
        {"k", "size_t", offsetof(TrillimMatmulParameters, k), NULL},
        {"rank", "size_t", offsetof(TrillimMatmulParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(TrillimMatmulParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(TrillimMatmulParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(TrillimMatmulParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(TrillimMatmulParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(TrillimMatmulParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(TrillimMatmulParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(TrillimMatmulParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(TrillimMatmulParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(TrillimMatmulParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(TrillimMatmulParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(TrillimMatmulParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(TrillimMatmulParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(TrillimMatmulParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(TrillimMatmulParameters, b_storage_columns), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimMatmulParameters), TRILLIM_ALIGNOF(TrillimMatmulParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_block_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
