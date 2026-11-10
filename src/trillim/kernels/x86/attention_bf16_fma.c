#include "common.h"
#include <immintrin.h>
#include <math.h>

#define SIMD __attribute__((target("avx2,fma")))

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

static inline bool trillim_attention_span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t span;
    return (columns == 0 || rows <= SIZE_MAX / columns) && trillim_span(3, shape, strides, &span) &&
           span <= rows * columns;
}

SIMD static inline __attribute__((always_inline)) __m256 load8(const void *data, size_t index) {
    __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

SIMD static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value), high = _mm256_srli_epi32(bits, 16);
    __m256i rounded =
        _mm256_srli_epi32(_mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                                                  _mm256_and_si256(high, _mm256_set1_epi32(1)))),
                          16);
    __m256i nan =
        _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(high, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
                     _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

static bool payload(size_t rows, size_t columns, size_t *bytes) {
    if (columns && rows > SIZE_MAX / 2 / columns) return false;
    *bytes = rows * columns * 2;
    return true;
}

static TrillimStatus trillim_attention_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                    size_t *scratch) {
    const TrillimAttentionParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!p->heads || !p->queries) return TRILLIM_OK;
    if (p->rank != 3 || !p->kv_heads || p->heads % p->kv_heads || (p->key_dim != 64 && p->key_dim != 128) ||
        p->value_dim != p->key_dim || p->q_dtype != TRILLIM_BF16 || p->k_dtype != TRILLIM_BF16 ||
        p->v_dtype != TRILLIM_BF16 || (p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32) ||
        p->q_strides[2] != 1 || p->k_strides[2] != 1 || p->v_strides[2] != 1 || p->causal > 1 || p->windowed > 1 ||
        p->default_scale > 1 || p->queries > INT64_MAX || p->length > INT64_MAX || p->heads > SIZE_MAX / p->queries ||
        p->heads * p->queries > SIZE_MAX / p->key_dim / trillim_itemsize(p->output_dtype))
        return TRILLIM_ERROR;
    size_t qshape[] = {p->heads, p->queries, p->key_dim}, kvshape[] = {p->kv_heads, p->length, p->key_dim};
    if (!trillim_attention_span(qshape, p->q_strides, p->q_rows, p->q_columns) ||
        !trillim_attention_span(kvshape, p->k_strides, p->k_rows, p->k_columns) ||
        !trillim_attention_span(kvshape, p->v_strides, p->v_rows, p->v_columns) ||
        !payload(p->q_rows, p->q_columns, &inputs[0]) || !payload(p->k_rows, p->k_columns, &inputs[1]) ||
        !payload(p->v_rows, p->v_columns, &inputs[2]))
        return TRILLIM_ERROR;
    size_t groups = p->queries / 16 + (p->queries % 16 != 0);
    if (groups > 4) groups = 4;
    if (p->heads > SIZE_MAX / groups || (p->length && p->heads * groups > SIZE_MAX / sizeof(float) / p->length))
        return TRILLIM_ERROR;
    *scratch = p->heads * groups * p->length * sizeof(float);
    if (!p->causal && !p->windowed && p->queries >= 4) {
        if (*scratch > SIZE_MAX / 4) return TRILLIM_ERROR;
        *scratch *= 4;
    }
    outputs[0] = p->heads * p->queries * p->key_dim * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

SIMD static __m256 exp_negative(__m256 x) {
    __m256 original = x;
    x = _mm256_max_ps(_mm256_set1_ps(-104.0f), _mm256_min_ps(x, _mm256_setzero_ps()));
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(0.00021219444005469058f), r);
    // Range reduction bounds |r| by ln(2)/2 for this Taylor polynomial.
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 720.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 120.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 24.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 6.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(0.5f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_cvttps_epi32(n);
    __m256i low = _mm256_min_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(126)), _mm256_setzero_si256());
    __m256i high = _mm256_sub_epi32(exponent, low);
    __m256 main_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(high, _mm256_set1_epi32(127)), 23));
    __m256 tail_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(low, _mm256_set1_epi32(127)), 23));
    value = _mm256_mul_ps(_mm256_mul_ps(value, main_scale), tail_scale);
    return _mm256_blendv_ps(value, original, _mm256_cmp_ps(original, original, _CMP_UNORD_Q));
}

SIMD static TrillimStatus normalize_scores(float *scores, size_t begin, size_t end, const TrillimKernelContext *context,
                                           float *inverse) {
    __m256 maximum = _mm256_set1_ps(-INFINITY);
    size_t key = begin;
    for (; end - key >= 8; key += 8) {
        if (((key - begin) & 255) == 0 && context->is_cancelled(context->cancellation_context))
            return TRILLIM_CANCELLED;
        maximum = _mm256_max_ps(maximum, _mm256_loadu_ps(scores + key));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, maximum);
    float peak = -INFINITY;
    for (size_t i = 0; i < 8; ++i)
        peak = fmaxf(peak, lanes[i]);
    for (; key < end; ++key)
        peak = fmaxf(peak, scores[key]);
    maximum = _mm256_set1_ps(peak);
    __m256 total = _mm256_setzero_ps();
    key = begin;
    for (; end - key >= 8; key += 8) {
        if (((key - begin) & 255) == 0 && context->is_cancelled(context->cancellation_context))
            return TRILLIM_CANCELLED;
        __m256 value = exp_negative(_mm256_sub_ps(_mm256_loadu_ps(scores + key), maximum));
        _mm256_storeu_ps(scores + key, value);
        total = _mm256_add_ps(total, value);
    }
    _mm256_storeu_ps(lanes, total);
    float denominator = 0;
    for (size_t i = 0; i < 8; ++i)
        denominator += lanes[i];
    for (; key < end; ++key) {
        scores[key] = expf(scores[key] - peak);
        denominator += scores[key];
    }
    *inverse = 1.0f / denominator;
    return TRILLIM_OK;
}

SIMD static inline __attribute__((always_inline)) float dot(const float *query, const void *keys, size_t key,
                                                            size_t width) {
    __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
    for (size_t i = 0; i < width; i += 32) {
        a = _mm256_fmadd_ps(_mm256_loadu_ps(query + i), load8(keys, key + i), a);
        b = _mm256_fmadd_ps(_mm256_loadu_ps(query + i + 8), load8(keys, key + i + 8), b);
        c = _mm256_fmadd_ps(_mm256_loadu_ps(query + i + 16), load8(keys, key + i + 16), c);
        d = _mm256_fmadd_ps(_mm256_loadu_ps(query + i + 24), load8(keys, key + i + 24), d);
    }
    __m256 total = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

SIMD static TrillimStatus attend(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = p->queries / 16 + (p->queries % 16 != 0);
    if (groups > 4) groups = 4;
    size_t head = index / groups, kvhead = head / (p->heads / p->kv_heads);
    size_t per_group = p->queries / groups + (p->queries % groups != 0);
    size_t first = index % groups * per_group;
    size_t last = p->queries - first < per_group ? p->queries : first + per_group;
    float *scores = task->scratch + index * p->length;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t qi = first; qi < last; ++qi) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float query[128];
        size_t qbase = head * p->q_strides[0] + qi * p->q_strides[1];
        for (size_t i = 0; i < p->key_dim; i += 8)
            _mm256_storeu_ps(query + i, load8(task->query, qbase + i));
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)qi;
        size_t end = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        size_t begin =
            p->windowed && position >= 0 && (size_t)position + 1 > p->window ? (size_t)position + 1 - p->window : 0;
        for (size_t key = begin; key < end; ++key) {
            if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                return TRILLIM_CANCELLED;
            scores[key] = dot(query, task->keys, kvhead * p->k_strides[0] + key * p->k_strides[1], p->key_dim) * scale;
        }
        float reciprocal;
        TrillimStatus status = normalize_scores(scores, begin, end, task->context, &reciprocal);
        if (status != TRILLIM_OK) return status;
        __m256 inverse = _mm256_set1_ps(reciprocal);
        size_t output = (qi * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; channel += 32) {
            __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
            for (size_t key = begin; key < end; ++key) {
                if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                    return TRILLIM_CANCELLED;
                size_t value = kvhead * p->v_strides[0] + key * p->v_strides[1] + channel;
                __m256 probability = _mm256_set1_ps(scores[key]);
                a = _mm256_fmadd_ps(probability, load8(task->values, value), a);
                b = _mm256_fmadd_ps(probability, load8(task->values, value + 8), b);
                c = _mm256_fmadd_ps(probability, load8(task->values, value + 16), c);
                d = _mm256_fmadd_ps(probability, load8(task->values, value + 24), d);
            }
            store8(task->output, output + channel, p->output_dtype, _mm256_mul_ps(a, inverse));
            store8(task->output, output + channel + 8, p->output_dtype, _mm256_mul_ps(b, inverse));
            store8(task->output, output + channel + 16, p->output_dtype, _mm256_mul_ps(c, inverse));
            store8(task->output, output + channel + 24, p->output_dtype, _mm256_mul_ps(d, inverse));
        }
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus attend4(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = p->queries / 16 + (p->queries % 16 != 0);
    if (groups > 4) groups = 4;
    size_t head = index / groups, kvhead = head / (p->heads / p->kv_heads);
    size_t per_group = p->queries / groups + (p->queries % groups != 0), first = index % groups * per_group;
    size_t last = p->queries - first < per_group ? p->queries : first + per_group;
    float *scores = task->scratch + index * p->length * 4;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t qi = first; qi < last; qi += 4) {
        size_t rows = last - qi < 4 ? last - qi : 4;
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        float query[4][128];
        for (size_t row = 0; row < 4; ++row) {
            size_t base = head * p->q_strides[0] + (qi + (row < rows ? row : 0)) * p->q_strides[1];
            for (size_t i = 0; i < p->key_dim; i += 8)
                _mm256_storeu_ps(query[row] + i, load8(task->query, base + i));
        }
        for (size_t key = 0; key < p->length; ++key) {
            if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                return TRILLIM_CANCELLED;
            __m256 a0 = _mm256_setzero_ps(), b0 = a0;
            __m256 a1 = _mm256_setzero_ps(), b1 = a1;
            __m256 a2 = _mm256_setzero_ps(), b2 = a2;
            __m256 a3 = _mm256_setzero_ps(), b3 = a3;
            size_t base = kvhead * p->k_strides[0] + key * p->k_strides[1];
            for (size_t i = 0; i < p->key_dim; i += 16) {
                __m256 k0 = load8(task->keys, base + i), k1 = load8(task->keys, base + i + 8);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(query[0] + i), k0, a0);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(query[0] + i + 8), k1, b0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(query[1] + i), k0, a1);
                b1 = _mm256_fmadd_ps(_mm256_loadu_ps(query[1] + i + 8), k1, b1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(query[2] + i), k0, a2);
                b2 = _mm256_fmadd_ps(_mm256_loadu_ps(query[2] + i + 8), k1, b2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(query[3] + i), k0, a3);
                b3 = _mm256_fmadd_ps(_mm256_loadu_ps(query[3] + i + 8), k1, b3);
            }
            {
                __m256 total = _mm256_add_ps(a0, b0);
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
                sum = _mm_hadd_ps(sum, sum);
                scores[0 * p->length + key] = _mm_cvtss_f32(_mm_hadd_ps(sum, sum)) * scale;
            }
            {
                __m256 total = _mm256_add_ps(a1, b1);
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
                sum = _mm_hadd_ps(sum, sum);
                scores[1 * p->length + key] = _mm_cvtss_f32(_mm_hadd_ps(sum, sum)) * scale;
            }
            {
                __m256 total = _mm256_add_ps(a2, b2);
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
                sum = _mm_hadd_ps(sum, sum);
                scores[2 * p->length + key] = _mm_cvtss_f32(_mm_hadd_ps(sum, sum)) * scale;
            }
            {
                __m256 total = _mm256_add_ps(a3, b3);
                __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
                sum = _mm_hadd_ps(sum, sum);
                scores[3 * p->length + key] = _mm_cvtss_f32(_mm_hadd_ps(sum, sum)) * scale;
            }
        }
        float inverses[4];
        for (size_t row = 0; row < rows; ++row) {
            TrillimStatus status =
                normalize_scores(scores + row * p->length, 0, p->length, task->context, &inverses[row]);
            if (status != TRILLIM_OK) return status;
        }
        for (size_t channel = 0; channel < p->value_dim; channel += 16) {
            __m256 a0 = _mm256_setzero_ps(), b0 = a0;
            __m256 a1 = _mm256_setzero_ps(), b1 = a1;
            __m256 a2 = _mm256_setzero_ps(), b2 = a2;
            __m256 a3 = _mm256_setzero_ps(), b3 = a3;
            for (size_t key = 0; key < p->length; ++key) {
                if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                    return TRILLIM_CANCELLED;
                size_t base = kvhead * p->v_strides[0] + key * p->v_strides[1] + channel;
                __m256 v0 = load8(task->values, base), v1 = load8(task->values, base + 8);
                {
                    __m256 probability = _mm256_set1_ps(scores[0 * p->length + key]);
                    a0 = _mm256_fmadd_ps(probability, v0, a0);
                    b0 = _mm256_fmadd_ps(probability, v1, b0);
                }
                {
                    __m256 probability = _mm256_set1_ps(scores[1 * p->length + key]);
                    a1 = _mm256_fmadd_ps(probability, v0, a1);
                    b1 = _mm256_fmadd_ps(probability, v1, b1);
                }
                {
                    __m256 probability = _mm256_set1_ps(scores[2 * p->length + key]);
                    a2 = _mm256_fmadd_ps(probability, v0, a2);
                    b2 = _mm256_fmadd_ps(probability, v1, b2);
                }
                {
                    __m256 probability = _mm256_set1_ps(scores[3 * p->length + key]);
                    a3 = _mm256_fmadd_ps(probability, v0, a3);
                    b3 = _mm256_fmadd_ps(probability, v1, b3);
                }
            }
            if (rows > 0) {
                size_t base = ((qi + 0) * p->heads + head) * p->value_dim + channel;
                __m256 inverse = _mm256_set1_ps(inverses[0]);
                store8(task->output, base, p->output_dtype, _mm256_mul_ps(a0, inverse));
                store8(task->output, base + 8, p->output_dtype, _mm256_mul_ps(b0, inverse));
            }
            if (rows > 1) {
                size_t base = ((qi + 1) * p->heads + head) * p->value_dim + channel;
                __m256 inverse = _mm256_set1_ps(inverses[1]);
                store8(task->output, base, p->output_dtype, _mm256_mul_ps(a1, inverse));
                store8(task->output, base + 8, p->output_dtype, _mm256_mul_ps(b1, inverse));
            }
            if (rows > 2) {
                size_t base = ((qi + 2) * p->heads + head) * p->value_dim + channel;
                __m256 inverse = _mm256_set1_ps(inverses[2]);
                store8(task->output, base, p->output_dtype, _mm256_mul_ps(a2, inverse));
                store8(task->output, base + 8, p->output_dtype, _mm256_mul_ps(b2, inverse));
            }
            if (rows > 3) {
                size_t base = ((qi + 3) * p->heads + head) * p->value_dim + channel;
                __m256 inverse = _mm256_set1_ps(inverses[3]);
                store8(task->output, base, p->output_dtype, _mm256_mul_ps(a3, inverse));
                store8(task->output, base + 8, p->output_dtype, _mm256_mul_ps(b3, inverse));
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_attention_bf16_fma(void *const *outputs, const void *const *inputs,
                                                            void *scratch, const void *parameters,
                                                            const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const TrillimAttentionParameters *p = parameters;
    if (!p->heads || !p->queries) return TRILLIM_OK;
    size_t groups = p->queries / 16 + (p->queries % 16 != 0);
    if (groups > 4) groups = 4;
    TrillimAttentionTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context};
    return context->thread_pool->run(context->thread_pool->context, p->heads * groups,
                                     !p->causal && !p->windowed && p->queries >= 4 ? attend4 : attend, &task);
}

TRILLIM_KERNEL const char *trillim_x86_attention_bf16_fma_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bf16_attention as bind\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: "
           "DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_attention_bf16_fma_parameters(void) {
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
    static const TrillimParameterSchema schema = {sizeof(TrillimAttentionParameters),
                                                  TRILLIM_ALIGNOF(TrillimAttentionParameters),
                                                  sizeof(fields) / sizeof(fields[0]),
                                                  fields,
                                                  3,
                                                  1,
                                                  trillim_attention_requirements,
                                                  NULL,
                                                  inputs,
                                                  outputs,
                                                  4};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_attention_bf16_fma_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &capabilities;
}
