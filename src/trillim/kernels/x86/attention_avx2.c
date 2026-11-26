#include "common.h"
#include <cpuid.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>

static inline float trillim_packed_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

/* An eight-lane read within one logical row. */
__attribute__((target("avx2"))) static inline __m256 trillim_packed_decode8(const uint8_t *values, size_t rows, size_t columns, uint32_t dtype, size_t row, size_t k) {
    if ((k & 7) != 0) {
        float lane_values[8];
        for (size_t i = 0; i < 8; ++i) {
            lane_values[i] = trillim_packed_read_float(values, row * columns + k + i, dtype, rows, columns);
        }
        return _mm256_loadu_ps(lane_values);
    }
    __m256i ones = _mm256_set1_epi32(1), mask = _mm256_set1_epi32(15);
    __m256i codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = trillim_packed_bf16(values + rows * columns + row * 2);
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + row * columns + k)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
    }
    size_t group = (dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY ? 128 : 32)), groups = columns / group + (columns % group != 0);
    size_t total = rows * groups, block = row * groups + k / group, offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            scale = trillim_packed_bf16(values + total * 32 + block * 2);
        }
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset % 32)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(6 - offset / 32 * 2))), _mm256_set1_epi32(3));
        __m256 decoded = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(codes, ones)), _mm256_set1_ps(scale));
        return _mm256_blendv_ps(decoded, _mm256_set1_ps(NAN), _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, _mm256_set1_epi32(3))));
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float palette[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
        uint32_t bytes;
        memcpy(&bytes, values + block * 16 + offset / 2, 4);
        __m128i packed = _mm_cvtsi32_si128((int)bytes);
        __m128i low = _mm_and_si128(packed, _mm_set1_epi8(15));
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
        codes = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
        uint8_t exponent = values[total * 16 + block];
        scale = trillim_mxfp4_scale(exponent);
        return _mm256_mul_ps(_mm256_i32gather_ps(palette, codes, 4), _mm256_set1_ps(scale));
    }
    size_t width = dtype == TRILLIM_GRP_INT8 ? 32 : dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16;
    size_t scale_start = total * width;
    scale = trillim_packed_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset)));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        codes = _mm256_srlv_epi32(_mm256_set1_epi32(values[block * 16 + offset / 8]), _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
        codes = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(codes, ones), 1), ones);
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 16 + offset % 16)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(offset / 16 * 4))), mask);
        if (dtype == TRILLIM_GRP_INT5) {
            __m256i high = _mm256_srlv_epi32(_mm256_set1_epi32(values[scale_start + total * 2 + block * 4 + offset / 8]), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            codes = _mm256_or_si256(codes, _mm256_slli_epi32(_mm256_and_si256(high, ones), 4));
        }
        codes = _mm256_sub_epi32(codes, _mm256_set1_epi32(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
    } else {
        size_t half = offset / 128, lane = offset % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + offset % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + low)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(lane % 2 * 4))), mask);
        __m256i upper = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + high)));
        upper = _mm256_and_si256(_mm256_srlv_epi32(upper, _mm256_set1_epi32((int)(lane * 2))), _mm256_set1_epi32(3));
        codes = _mm256_sub_epi32(_mm256_or_si256(codes, _mm256_slli_epi32(upper, 4)), _mm256_set1_epi32(32));
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + offset / 16, 1);
        codes = _mm256_mullo_epi32(codes, _mm256_set1_epi32(subscale));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
}

#define TRILLIM_AVX2 __attribute__((target("avx2")))

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
    if (dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4 &&
        columns && columns - index % columns >= 8) {
        return trillim_packed_decode8(data, rows, columns, dtype, index / columns, index % columns);
    }
    float values[8];
    for (size_t i = 0; i < 8; ++i) {
        values[i] = trillim_packed_read_float(data, index + i, dtype, rows, columns);
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
    size_t heads;
    size_t kv_heads;
    size_t queries;
    size_t length;
    size_t key_dim;
    size_t value_dim;
    size_t rank;
    const size_t *q_strides;
    const size_t *k_strides;
    const size_t *v_strides;
    size_t q_rows;
    size_t q_columns;
    size_t k_rows;
    size_t k_columns;
    size_t v_rows;
    size_t v_columns;
    size_t window;
    uint32_t output_dtype;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t causal;
    uint32_t windowed;
    uint32_t default_scale;
    float scale;
} TrillimAttentionParameters;

typedef struct {
    void *output;
    const void *query;
    const void *keys;
    const void *values;
    float *scratch;
    const TrillimAttentionParameters *parameters;
    const TrillimKernelContext *context;
} TrillimAttentionTask;

static bool trillim_attention_blocked(const TrillimAttentionParameters *p) {
    return p->heads && p->kv_heads && p->queries >= 4 && p->length >= 8 &&
        p->q_dtype == TRILLIM_FP32 && p->k_dtype == TRILLIM_FP32 &&
        p->v_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_FP32 &&
        !p->causal && !p->windowed && p->key_dim == 64 && p->value_dim == 64 &&
        p->q_strides[2] == 1 && p->k_strides[2] == 1 && p->v_strides[2] == 1;
}

static inline bool trillim_attention_span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t span;
    return (columns == 0 || rows <= SIZE_MAX / columns) &&
        trillim_span(3, shape, strides, &span) && span <= rows * columns;
}

static inline TrillimStatus trillim_attention_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimAttentionParameters *p = (const TrillimAttentionParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    inputs[2] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (p->heads == 0 || p->queries == 0) {
        return TRILLIM_OK;
    }
    if (p->rank != 3 || p->kv_heads == 0 || p->heads % p->kv_heads != 0 ||
        p->key_dim == 0 || p->value_dim == 0 || !trillim_float(p->output_dtype) || p->causal > 1 || p->windowed > 1 || p->default_scale > 1 ||
        p->queries > INT64_MAX || p->length > INT64_MAX ||
        p->heads > SIZE_MAX / p->queries || p->heads * p->queries > SIZE_MAX / trillim_itemsize(p->output_dtype) / p->value_dim) {
        return TRILLIM_ERROR;
    }
    size_t query_shape[] = {p->heads, p->queries, p->key_dim};
    size_t key_shape[] = {p->kv_heads, p->length, p->key_dim};
    size_t value_shape[] = {p->kv_heads, p->length, p->value_dim};
    if (!trillim_attention_span(query_shape, p->q_strides, p->q_rows, p->q_columns) ||
        !trillim_packed_payload_size(p->q_rows, p->q_columns, p->q_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->length != 0 && (!trillim_attention_span(key_shape, p->k_strides, p->k_rows, p->k_columns) ||
        !trillim_attention_span(value_shape, p->v_strides, p->v_rows, p->v_columns) ||
        !trillim_packed_payload_size(p->k_rows, p->k_columns, p->k_dtype, &inputs[1]) ||
        !trillim_packed_payload_size(p->v_rows, p->v_columns, p->v_dtype, &inputs[2]))) {
        return TRILLIM_ERROR;
    }
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    if (p->value_dim > SIZE_MAX - 64 || p->heads > SIZE_MAX / tiles ||
        p->heads * tiles > SIZE_MAX / sizeof(float) / (p->value_dim + 64)) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->queries * p->value_dim * trillim_itemsize(p->output_dtype);
    *scratch = p->heads * tiles * (p->value_dim + 64) * sizeof(float);
    if (trillim_attention_blocked(p)) {
        size_t width = (p->length + 7) & ~(size_t)7;
        size_t query_tiles = p->queries / 4 + (p->queries % 4 != 0);
        if (p->kv_heads > SIZE_MAX / 128 || p->heads > SIZE_MAX / 4 / query_tiles) {
            return TRILLIM_ERROR;
        }
        size_t rows = p->kv_heads * 128;
        size_t slots = p->heads * query_tiles < 64 ? p->heads * query_tiles : 64;
        size_t scores = slots * 4;
        if (scores > SIZE_MAX - rows || rows + scores > SIZE_MAX / sizeof(float) / width) {
            return TRILLIM_ERROR;
        }
        size_t blocked = (rows + scores) * width * sizeof(float);
        if (blocked > *scratch) {
            *scratch = blocked;
        }
    }
    return TRILLIM_OK;
}

/* A reduced-range polynomial also retains the subnormal tail of softmax. */
__attribute__((target("avx2,fma"))) static __m256 trillim_attention_exp(__m256 x) {
    __m256 original = x;
    x = _mm256_max_ps(x, _mm256_set1_ps(-104.0f));
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(0.00021219444005469058f), r);
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 720.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 120.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 24.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 6.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(0.5f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    __m256i integer = _mm256_cvttps_epi32(n);
    __m256i normal = _mm256_max_epi32(integer, _mm256_set1_epi32(-126));
    __m256i exponent = _mm256_slli_epi32(_mm256_add_epi32(normal, _mm256_set1_epi32(127)), 23);
    __m256i tail = _mm256_slli_epi32(_mm256_add_epi32(_mm256_sub_epi32(integer, normal), _mm256_set1_epi32(127)), 23);
    value = _mm256_mul_ps(_mm256_mul_ps(value, _mm256_castsi256_ps(exponent)), _mm256_castsi256_ps(tail));
    return _mm256_blendv_ps(value, original, _mm256_cmp_ps(original, original, _CMP_UNORD_Q));
}

TRILLIM_AVX2 static TrillimStatus trillim_attention_pack(void *opaque, size_t head, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t width = (p->length + 7) & ~(size_t)7;
    float *keys = task->scratch + head * 64 * width;
    float *values = task->scratch + p->kv_heads * 64 * width + head * 64 * width;
    const float *source = (const float *)task->keys + head * p->k_strides[0];
    for (size_t channel = 0; channel < 64; ++channel) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t key = 0; key < p->length; ++key) {
            keys[channel * width + key] = source[key * p->k_strides[1] + channel];
        }
        for (size_t key = p->length; key < width; ++key) {
            keys[channel * width + key] = 0;
        }
    }
    for (size_t key = 0; key < p->length; ++key) {
        if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        memcpy(values + key * 64, (const float *)task->values + head * p->v_strides[0] + key * p->v_strides[1], 64 * sizeof(float));
    }
    return TRILLIM_OK;
}

__attribute__((target("avx2,fma"))) static TrillimStatus trillim_attention_block(void *opaque, size_t tile, size_t slot) {
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t tiles = p->queries / 4 + (p->queries % 4 != 0);
    size_t head = tile / tiles, first = tile % tiles * 4;
    size_t rows = p->queries - first < 4 ? p->queries - first : 4;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t width = (p->length + 7) & ~(size_t)7;
    const float *keys = task->scratch + kv_head * 64 * width;
    const float *values = task->scratch + p->kv_heads * 64 * width + kv_head * 64 * width;
    float *scores = task->scratch + p->kv_heads * 128 * width + slot * 4 * width;
    const float *q[4];
    for (size_t row = 0; row < 4; ++row) {
        q[row] = (const float *)task->query + head * p->q_strides[0] + (first + (row < rows ? row : 0)) * p->q_strides[1];
    }
    __m256 scale = _mm256_set1_ps(p->default_scale ? 0.125f : p->scale);
    for (size_t key = 0; key < width; key += 8) {
        if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (size_t channel = 0; channel < 64; ++channel) {
            __m256 k = _mm256_loadu_ps(keys + channel * width + key);
            a = _mm256_fmadd_ps(k, _mm256_set1_ps(q[0][channel]), a);
            b = _mm256_fmadd_ps(k, _mm256_set1_ps(q[1][channel]), b);
            c = _mm256_fmadd_ps(k, _mm256_set1_ps(q[2][channel]), c);
            d = _mm256_fmadd_ps(k, _mm256_set1_ps(q[3][channel]), d);
        }
        _mm256_storeu_ps(scores + key, _mm256_mul_ps(a, scale));
        _mm256_storeu_ps(scores + width + key, _mm256_mul_ps(b, scale));
        _mm256_storeu_ps(scores + width * 2 + key, _mm256_mul_ps(c, scale));
        _mm256_storeu_ps(scores + width * 3 + key, _mm256_mul_ps(d, scale));
    }
    for (size_t row = 0; row < rows; ++row) {
        float *probability = scores + row * width;
        for (size_t key = p->length; key < width; ++key) {
            probability[key] = -INFINITY;
        }
        __m256 peak = _mm256_set1_ps(-INFINITY);
        for (size_t key = 0; key < width; key += 8) {
            peak = _mm256_max_ps(peak, _mm256_loadu_ps(probability + key));
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, peak);
        float maximum = -INFINITY;
        for (size_t lane = 0; lane < 8; ++lane) {
            maximum = fmaxf(maximum, lanes[lane]);
        }
        __m256 total = _mm256_setzero_ps();
        for (size_t key = 0; key < width; key += 8) {
            __m256 value = trillim_attention_exp(_mm256_sub_ps(_mm256_loadu_ps(probability + key), _mm256_set1_ps(maximum)));
            _mm256_storeu_ps(probability + key, value);
            total = _mm256_add_ps(total, value);
        }
        float inverse = 1.0f / trillim_avx2_sum(total);
        for (size_t key = 0; key < width; key += 8) {
            _mm256_storeu_ps(probability + key, _mm256_mul_ps(_mm256_loadu_ps(probability + key), _mm256_set1_ps(inverse)));
        }
    }
    for (size_t channel = 0; channel < 64; channel += 16) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0;
        __m256 c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        for (size_t key = 0; key < p->length; ++key) {
            if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 v0 = _mm256_loadu_ps(values + key * 64 + channel);
            __m256 v1 = _mm256_loadu_ps(values + key * 64 + channel + 8);
            __m256 p0 = _mm256_set1_ps(scores[key]);
            __m256 p1 = _mm256_set1_ps(scores[width + key]);
            __m256 p2 = _mm256_set1_ps(scores[width * 2 + key]);
            __m256 p3 = _mm256_set1_ps(scores[width * 3 + key]);
            a0 = _mm256_fmadd_ps(v0, p0, a0);
            a1 = _mm256_fmadd_ps(v1, p0, a1);
            b0 = _mm256_fmadd_ps(v0, p1, b0);
            b1 = _mm256_fmadd_ps(v1, p1, b1);
            c0 = _mm256_fmadd_ps(v0, p2, c0);
            c1 = _mm256_fmadd_ps(v1, p2, c1);
            d0 = _mm256_fmadd_ps(v0, p3, d0);
            d1 = _mm256_fmadd_ps(v1, p3, d1);
        }
        __m256 result[4][2] = {{a0, a1}, {b0, b1}, {c0, c1}, {d0, d1}};
        for (size_t row = 0; row < rows; ++row) {
            float *output = (float *)task->output + ((first + row) * p->heads + head) * 64 + channel;
            _mm256_storeu_ps(output, result[row][0]);
            _mm256_storeu_ps(output + 8, result[row][1]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_attention_blocks(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t count = p->heads * (p->queries / 4 + (p->queries % 4 != 0));
    /* Reuse score storage instead of materializing a quadratic attention matrix. */
    size_t slots = count < 64 ? count : 64;
    for (size_t tile = slot; tile < count; tile += slots) {
        TrillimStatus status = trillim_attention_block(opaque, tile, slot);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_attention_score(
    const TrillimAttentionTask *task, size_t query, size_t key, float *score
) {
    const TrillimAttentionParameters *p = task->parameters;
    float sum = 0.0f;
    size_t channel = 0;
    __m256 accumulator = _mm256_setzero_ps();
    __m256 sum1 = accumulator, sum2 = accumulator, sum3 = accumulator;
    for (; p->q_strides[2] == 1 && p->k_strides[2] == 1 && p->key_dim - channel >= 32; channel += 32) {
        if (p->key_dim > 1024 && (channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(
            trillim_avx2_load8(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns),
            trillim_avx2_load8(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
            trillim_avx2_load8(task->query, query + channel + 8, p->q_dtype, p->q_rows, p->q_columns),
            trillim_avx2_load8(task->keys, key + channel + 8, p->k_dtype, p->k_rows, p->k_columns)));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
            trillim_avx2_load8(task->query, query + channel + 16, p->q_dtype, p->q_rows, p->q_columns),
            trillim_avx2_load8(task->keys, key + channel + 16, p->k_dtype, p->k_rows, p->k_columns)));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
            trillim_avx2_load8(task->query, query + channel + 24, p->q_dtype, p->q_rows, p->q_columns),
            trillim_avx2_load8(task->keys, key + channel + 24, p->k_dtype, p->k_rows, p->k_columns)));
    }
    accumulator = _mm256_add_ps(_mm256_add_ps(accumulator, sum1), _mm256_add_ps(sum2, sum3));
    for (; p->q_strides[2] == 1 && p->k_strides[2] == 1 && p->key_dim - channel >= 8; channel += 8) {
        if (p->key_dim > 1024 && (channel & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 q = trillim_avx2_load8(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns);
        __m256 k = trillim_avx2_load8(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns);
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(q, k));
    }
    sum = trillim_avx2_sum(accumulator);
    for (; channel < p->key_dim; ++channel) {
        if (p->key_dim > 1024 && (channel & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_packed_read_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
        float k = trillim_packed_read_float(task->keys, key + channel * p->k_strides[2], p->k_dtype, p->k_rows, p->k_columns);
        sum += q * k;
    }
    *score = sum;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_attention_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = (const TrillimAttentionTask *)opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    size_t head = index / tiles;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t first = (index % tiles) * 16;
    size_t last = p->queries - first < 16 ? p->queries : first + 16;
    float *scores = task->scratch + index * (p->value_dim + 64);
    float *accumulator = scores + 64;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t qi = first; qi < last; ++qi) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            accumulator[channel] = 0.0f;
        }
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)qi;
        size_t end = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        size_t begin = 0;
        if (p->windowed && position >= 0 && (uint64_t)position + 1 > p->window) {
            begin = (size_t)position + 1 - p->window;
        }
        size_t query = head * p->q_strides[0] + qi * p->q_strides[1];
        float maximum = -INFINITY;
        float denominator = 0.0f;
        for (size_t block = begin; block < end; block += 64) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = end - block < 64 ? end - block : 64;
            float block_maximum = -INFINITY;
            for (size_t j = 0; j < count; ++j) {
                size_t key = kv_head * p->k_strides[0] + (block + j) * p->k_strides[1];
                TrillimStatus status = trillim_attention_score(task, query, key, &scores[j]);
                if (status != TRILLIM_OK) {
                    return status;
                }
                scores[j] *= scale;
                block_maximum = fmaxf(block_maximum, scores[j]);
            }
            float next_maximum = fmaxf(maximum, block_maximum);
            float previous = maximum == -INFINITY ? 0.0f : expf(maximum - next_maximum);
            float block_sum = 0.0f;
            for (size_t j = 0; j < count; ++j) {
                scores[j] = expf(scores[j] - next_maximum);
                block_sum += scores[j];
            }
            for (size_t channel = 0; channel < p->value_dim; ++channel) {
                if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                accumulator[channel] *= previous;
            }
            denominator = denominator * previous + block_sum;
            maximum = next_maximum;
            size_t first_channel = 0;
            for (; p->v_strides[2] == 1 && p->value_dim - first_channel >= 32; first_channel += 32) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                __m256 a = _mm256_loadu_ps(accumulator + first_channel);
                __m256 b = _mm256_loadu_ps(accumulator + first_channel + 8);
                __m256 c = _mm256_loadu_ps(accumulator + first_channel + 16);
                __m256 d = _mm256_loadu_ps(accumulator + first_channel + 24);
                for (size_t j = 0; j < count; ++j) {
                    size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1] + first_channel;
                    __m256 probability = _mm256_set1_ps(scores[j]);
                    a = _mm256_add_ps(a, _mm256_mul_ps(probability, trillim_avx2_load8(task->values, value, p->v_dtype, p->v_rows, p->v_columns)));
                    b = _mm256_add_ps(b, _mm256_mul_ps(probability, trillim_avx2_load8(task->values, value + 8, p->v_dtype, p->v_rows, p->v_columns)));
                    c = _mm256_add_ps(c, _mm256_mul_ps(probability, trillim_avx2_load8(task->values, value + 16, p->v_dtype, p->v_rows, p->v_columns)));
                    d = _mm256_add_ps(d, _mm256_mul_ps(probability, trillim_avx2_load8(task->values, value + 24, p->v_dtype, p->v_rows, p->v_columns)));
                }
                _mm256_storeu_ps(accumulator + first_channel, a);
                _mm256_storeu_ps(accumulator + first_channel + 8, b);
                _mm256_storeu_ps(accumulator + first_channel + 16, c);
                _mm256_storeu_ps(accumulator + first_channel + 24, d);
            }
            for (size_t j = 0; j < count; ++j) {
                size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1];
                size_t channel = first_channel;
                __m256 probability = _mm256_set1_ps(scores[j]);
                for (; p->v_strides[2] == 1 && p->value_dim - channel >= 8; channel += 8) {
                    if (p->value_dim > 1024 && (channel & 1023) == 0 &&
                        task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    __m256 v = trillim_avx2_load8(task->values, value + channel, p->v_dtype, p->v_rows, p->v_columns);
                    __m256 sum = _mm256_loadu_ps(accumulator + channel);
                    _mm256_storeu_ps(accumulator + channel, _mm256_add_ps(sum, _mm256_mul_ps(probability, v)));
                }
                for (; channel < p->value_dim; ++channel) {
                    if (p->value_dim > 1024 && (channel & 1023) == 0 &&
                        task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    float v = trillim_packed_read_float(task->values, value + channel * p->v_strides[2], p->v_dtype, p->v_rows, p->v_columns);
                    accumulator[channel] += scores[j] * v;
                }
            }
        }
        size_t output = (qi * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            TrillimValue value;
            value.f = accumulator[channel] / denominator;
            trillim_store(task->output, output + channel, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_attention_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimAttentionParameters *p = (const TrillimAttentionParameters *)parameters;
    TrillimAttentionTask task = {(float *)outputs[0], inputs[0], inputs[1], inputs[2], (float *)scratch, p, context};
    unsigned eax, ebx, ecx, edx;
    if (trillim_attention_blocked(p) && __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 12))) {
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, p->kv_heads, trillim_attention_pack, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        size_t tiles = p->queries / 4 + (p->queries % 4 != 0);
        size_t count = p->heads * tiles < 64 ? p->heads * tiles : 64;
        return context->thread_pool->run(context->thread_pool->context, count, trillim_attention_blocks, &task);
    }
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->heads * tiles, trillim_attention_task, &task);
}

TRILLIM_KERNEL const char *trillim_x86_attention_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention as bind\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_attention_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(TrillimAttentionParameters, heads), NULL},
        {"kv_heads", "size_t", offsetof(TrillimAttentionParameters, kv_heads), NULL},
        {"queries", "size_t", offsetof(TrillimAttentionParameters, queries), NULL},
        {"length", "size_t", offsetof(TrillimAttentionParameters, length), NULL},
        {"key_dim", "size_t", offsetof(TrillimAttentionParameters, key_dim), NULL},
        {"value_dim", "size_t", offsetof(TrillimAttentionParameters, value_dim), NULL},
        {"rank", "size_t", offsetof(TrillimAttentionParameters, rank), NULL},
        {"q_strides", "size_t", offsetof(TrillimAttentionParameters, q_strides), "rank"},
        {"k_strides", "size_t", offsetof(TrillimAttentionParameters, k_strides), "rank"},
        {"v_strides", "size_t", offsetof(TrillimAttentionParameters, v_strides), "rank"},
        {"q_rows", "size_t", offsetof(TrillimAttentionParameters, q_rows), NULL},
        {"q_columns", "size_t", offsetof(TrillimAttentionParameters, q_columns), NULL},
        {"k_rows", "size_t", offsetof(TrillimAttentionParameters, k_rows), NULL},
        {"k_columns", "size_t", offsetof(TrillimAttentionParameters, k_columns), NULL},
        {"v_rows", "size_t", offsetof(TrillimAttentionParameters, v_rows), NULL},
        {"v_columns", "size_t", offsetof(TrillimAttentionParameters, v_columns), NULL},
        {"window", "size_t", offsetof(TrillimAttentionParameters, window), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimAttentionParameters, output_dtype), NULL},
        {"q_dtype", "uint32_t", offsetof(TrillimAttentionParameters, q_dtype), NULL},
        {"k_dtype", "uint32_t", offsetof(TrillimAttentionParameters, k_dtype), NULL},
        {"v_dtype", "uint32_t", offsetof(TrillimAttentionParameters, v_dtype), NULL},
        {"causal", "uint32_t", offsetof(TrillimAttentionParameters, causal), NULL},
        {"windowed", "uint32_t", offsetof(TrillimAttentionParameters, windowed), NULL},
        {"default_scale", "uint32_t", offsetof(TrillimAttentionParameters, default_scale), NULL},
        {"scale", "float", offsetof(TrillimAttentionParameters, scale), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimAttentionParameters), TRILLIM_ALIGNOF(TrillimAttentionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 3, 1,
        trillim_attention_requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_attention_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
