#include "common.h"
#include <immintrin.h>
#include <math.h>

#define SIMD __attribute__((target("avx2,fma")))

enum { QUERY_TILE = 4, MAX_TASKS = 32 };

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

static size_t query_tiles(const TrillimAttentionParameters *p) {
    size_t queries = (p->heads / p->kv_heads) * p->queries;
    return queries / QUERY_TILE + (queries % QUERY_TILE != 0);
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
    size_t tasks = p->kv_heads * query_tiles(p);
    tasks = tasks < MAX_TASKS ? tasks : MAX_TASKS;
    if (p->length > SIZE_MAX / sizeof(float) / QUERY_TILE / tasks) {
        return TRILLIM_ERROR;
    }
    *scratch = tasks * QUERY_TILE * p->length * sizeof(float);
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
    maximum = _mm256_set1_ps(peak == -INFINITY ? 0.0f : peak);
    if (peak == -INFINITY) peak = 0.0f;
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
    *inverse = denominator > 0.0f ? 1.0f / denominator : 1.0f;
    return TRILLIM_OK;
}

typedef struct {
    float query[QUERY_TILE][128];
    float *scores[QUERY_TILE];
    size_t begin[QUERY_TILE], end[QUERY_TILE], output[QUERY_TILE];
    size_t rows, first_key, last_key;
} QueryTile;

__attribute__((target("avx2")))
static inline __m128 load4(const uint16_t *data) {
    __m128i bits = _mm_loadl_epi64((const __m128i *)data);
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(bits), 16));
}

__attribute__((target("avx2")))
static TrillimStatus compute_scores(const TrillimAttentionTask *task, QueryTile *tile,
                                   const uint16_t *keys, float scale) {
    const TrillimAttentionParameters *p = task->parameters;
    size_t stride = p->k_strides[1];
    size_t first = tile->first_key;
    for (size_t key = first; key < tile->last_key;) {
        if (((key - first) & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (tile->last_key - key < 4) {
            const uint16_t *k = keys + key * stride;
            for (size_t row = 0; row < tile->rows; ++row) {
                if (key >= tile->begin[row] && key < tile->end[row]) {
                    float sum = 0;
                    for (size_t channel = 0; channel < p->key_dim; ++channel) {
                        sum += tile->query[row][channel] * trillim_bf16(k[channel]);
                    }
                    tile->scores[row][key] = sum * scale;
                }
            }
            ++key;
            continue;
        }

        const uint16_t *base = keys + key * stride;
        __m128 sums[QUERY_TILE] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
        /* Each SIMD lane accumulates a key in channel order. */
        for (size_t channel = 0; channel < p->key_dim; channel += 4) {
            __m128 k0 = load4(base + channel);
            __m128 k1 = load4(base + stride + channel);
            __m128 k2 = load4(base + 2 * stride + channel);
            __m128 k3 = load4(base + 3 * stride + channel);
            _MM_TRANSPOSE4_PS(k0, k1, k2, k3);
            for (size_t row = 0; row < tile->rows; ++row) {
                const float *q = tile->query[row] + channel;
                __m128 sum = sums[row];
                sum = _mm_add_ps(sum, _mm_mul_ps(k0, _mm_set1_ps(q[0])));
                sum = _mm_add_ps(sum, _mm_mul_ps(k1, _mm_set1_ps(q[1])));
                sum = _mm_add_ps(sum, _mm_mul_ps(k2, _mm_set1_ps(q[2])));
                sums[row] = _mm_add_ps(sum, _mm_mul_ps(k3, _mm_set1_ps(q[3])));
            }
        }
        for (size_t row = 0; row < tile->rows; ++row) {
            _mm_storeu_ps(tile->scores[row] + key, _mm_mul_ps(sums[row], _mm_set1_ps(scale)));
        }
        key += 4;
    }
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

SIMD static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

SIMD static TrillimStatus fast_scores(const TrillimAttentionTask *task, const QueryTile *tile,
                                    const uint16_t *keys, float scale) {
    const TrillimAttentionParameters *p = task->parameters;
    size_t stride = p->k_strides[1];
    if (tile->rows == 1) {
        for (size_t key = tile->first_key; key < tile->last_key; ++key) {
            if (((key - tile->first_key) & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            tile->scores[0][key] = dot(tile->query[0], keys, key * stride, p->key_dim) * scale;
        }
        return TRILLIM_OK;
    }
    const float *queries[QUERY_TILE];
    for (size_t row = 0; row < QUERY_TILE; ++row) {
        queries[row] = tile->query[row < tile->rows ? row : 0];
    }
    for (size_t key = tile->first_key; key < tile->last_key; ++key) {
        if (((key - tile->first_key) & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 a0 = _mm256_setzero_ps(), b0 = a0;
        __m256 a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
        size_t base = key * stride;
        for (size_t channel = 0; channel < p->key_dim; channel += 16) {
            __m256 k0 = load8(keys, base + channel), k1 = load8(keys, base + channel + 8);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[0] + channel), k0, a0);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[0] + channel + 8), k1, b0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[1] + channel), k0, a1);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[1] + channel + 8), k1, b1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[2] + channel), k0, a2);
            b2 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[2] + channel + 8), k1, b2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[3] + channel), k0, a3);
            b3 = _mm256_fmadd_ps(_mm256_loadu_ps(queries[3] + channel + 8), k1, b3);
        }
        tile->scores[0][key] = sum8(_mm256_add_ps(a0, b0)) * scale;
        tile->scores[1][key] = sum8(_mm256_add_ps(a1, b1)) * scale;
        tile->scores[2][key] = sum8(_mm256_add_ps(a2, b2)) * scale;
        tile->scores[3][key] = sum8(_mm256_add_ps(a3, b3)) * scale;
    }
    return TRILLIM_OK;
}

SIMD static void bind_tile(const TrillimAttentionTask *task, size_t index, size_t tiles, float *scores,
                           QueryTile *tile, bool load_query) {
    const TrillimAttentionParameters *p = task->parameters;
    size_t group = p->heads / p->kv_heads;
    size_t kv_head = index / tiles;
    size_t first = index % tiles * QUERY_TILE;
    size_t rows = group * p->queries - first;
    tile->rows = rows < QUERY_TILE ? rows : QUERY_TILE;
    tile->first_key = p->length;
    tile->last_key = 0;
    if (scores) {
        for (size_t row = 0; row < QUERY_TILE; ++row) {
            tile->scores[row] = scores + row * p->length;
        }
    }
    for (size_t row = 0; row < tile->rows; ++row) {
        size_t head = kv_head * group + (first + row) / p->queries;
        size_t qi = (first + row) % p->queries;
        size_t distance = p->queries - qi;
        size_t position_end = distance > p->length ? 0 : p->length - distance + 1;
        tile->begin[row] = p->windowed && position_end > p->window ? position_end - p->window : 0;
        tile->end[row] = p->causal ? position_end : p->length;
        tile->output[row] = (qi * p->heads + head) * p->value_dim;
        if (!p->length) {
            continue;
        }
        if (tile->begin[row] < tile->first_key) tile->first_key = tile->begin[row];
        if (tile->end[row] > tile->last_key) tile->last_key = tile->end[row];
        if (!load_query) continue;
        size_t base = head * p->q_strides[0] + qi * p->q_strides[1];
        for (size_t channel = 0; channel < p->key_dim; channel += 8) {
            _mm256_storeu_ps(tile->query[row] + channel, load8(task->query, base + channel));
        }
    }
}

SIMD static TrillimStatus apply_values_one(const TrillimAttentionTask *task, const QueryTile *tile,
                                           size_t kv_head, float inverse, size_t first_channel, size_t last_channel) {
    const TrillimAttentionParameters *p = task->parameters;
    __m256 reciprocal = _mm256_set1_ps(inverse);
    for (size_t channel = first_channel; channel < last_channel; channel += 32) {
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (size_t key = tile->first_key; key < tile->last_key; ++key) {
            if (((key - tile->first_key) & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t base = kv_head * p->v_strides[0] + key * p->v_strides[1] + channel;
            __m256 probability = _mm256_set1_ps(tile->scores[0][key]);
            a = _mm256_fmadd_ps(probability, load8(task->values, base), a);
            b = _mm256_fmadd_ps(probability, load8(task->values, base + 8), b);
            c = _mm256_fmadd_ps(probability, load8(task->values, base + 16), c);
            d = _mm256_fmadd_ps(probability, load8(task->values, base + 24), d);
        }
        size_t output = tile->output[0] + channel;
        store8(task->output, output, p->output_dtype, _mm256_mul_ps(a, reciprocal));
        store8(task->output, output + 8, p->output_dtype, _mm256_mul_ps(b, reciprocal));
        store8(task->output, output + 16, p->output_dtype, _mm256_mul_ps(c, reciprocal));
        store8(task->output, output + 24, p->output_dtype, _mm256_mul_ps(d, reciprocal));
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus apply_values_same_mask(const TrillimAttentionTask *task, const QueryTile *tile,
                                                 size_t kv_head, const float *inverse,
                                                 size_t first_channel, size_t last_channel) {
    const TrillimAttentionParameters *p = task->parameters;
    const float *scores[QUERY_TILE];
    for (size_t row = 0; row < QUERY_TILE; ++row) {
        scores[row] = tile->scores[row < tile->rows ? row : 0];
    }
    for (size_t channel = first_channel; channel < last_channel; channel += 16) {
        __m256 a0 = _mm256_setzero_ps(), b0 = a0, a1 = a0, b1 = a0;
        __m256 a2 = a0, b2 = a0, a3 = a0, b3 = a0;
        for (size_t key = tile->first_key; key < tile->last_key; ++key) {
            if (((key - tile->first_key) & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t base = kv_head * p->v_strides[0] + key * p->v_strides[1] + channel;
            __m256 v0 = load8(task->values, base), v1 = load8(task->values, base + 8);
            __m256 w0 = _mm256_set1_ps(scores[0][key]), w1 = _mm256_set1_ps(scores[1][key]);
            __m256 w2 = _mm256_set1_ps(scores[2][key]), w3 = _mm256_set1_ps(scores[3][key]);
            a0 = _mm256_fmadd_ps(w0, v0, a0);
            b0 = _mm256_fmadd_ps(w0, v1, b0);
            a1 = _mm256_fmadd_ps(w1, v0, a1);
            b1 = _mm256_fmadd_ps(w1, v1, b1);
            a2 = _mm256_fmadd_ps(w2, v0, a2);
            b2 = _mm256_fmadd_ps(w2, v1, b2);
            a3 = _mm256_fmadd_ps(w3, v0, a3);
            b3 = _mm256_fmadd_ps(w3, v1, b3);
        }
        __m256 result[QUERY_TILE][2] = {{a0, b0}, {a1, b1}, {a2, b2}, {a3, b3}};
        for (size_t row = 0; row < tile->rows; ++row) {
            __m256 reciprocal = _mm256_set1_ps(inverse[row]);
            size_t output = tile->output[row] + channel;
            store8(task->output, output, p->output_dtype, _mm256_mul_ps(result[row][0], reciprocal));
            store8(task->output, output + 8, p->output_dtype, _mm256_mul_ps(result[row][1], reciprocal));
        }
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus apply_values(const TrillimAttentionTask *task, const QueryTile *tile, size_t kv_head,
                                       const float *inverse, size_t first_channel, size_t last_channel) {
    const TrillimAttentionParameters *p = task->parameters;
    if (tile->rows == 1) {
        return apply_values_one(task, tile, kv_head, inverse[0], first_channel, last_channel);
    }
    bool same_mask = true;
    for (size_t row = 0; row < tile->rows; ++row) {
        same_mask &= tile->begin[row] == tile->first_key && tile->end[row] == tile->last_key;
    }
    if (same_mask) {
        return apply_values_same_mask(task, tile, kv_head, inverse, first_channel, last_channel);
    }
    for (size_t channel = first_channel; channel < last_channel; channel += 16) {
        __m256 a[QUERY_TILE], b[QUERY_TILE];
        for (size_t row = 0; row < tile->rows; ++row) {
            a[row] = b[row] = _mm256_setzero_ps();
        }
        for (size_t key = tile->first_key; key < tile->last_key; ++key) {
            if (((key - tile->first_key) & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t base = kv_head * p->v_strides[0] + key * p->v_strides[1] + channel;
            __m256 v0 = load8(task->values, base), v1 = load8(task->values, base + 8);
            for (size_t row = 0; row < tile->rows; ++row) {
                if (key >= tile->begin[row] && key < tile->end[row]) {
                    __m256 weight = _mm256_set1_ps(tile->scores[row][key]);
                    a[row] = _mm256_fmadd_ps(weight, v0, a[row]);
                    b[row] = _mm256_fmadd_ps(weight, v1, b[row]);
                }
            }
        }
        for (size_t row = 0; row < tile->rows; ++row) {
            __m256 reciprocal = _mm256_set1_ps(inverse[row]);
            store8(task->output, tile->output[row] + channel, p->output_dtype, _mm256_mul_ps(a[row], reciprocal));
            store8(task->output, tile->output[row] + channel + 8, p->output_dtype, _mm256_mul_ps(b[row], reciprocal));
        }
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus attend_tile(const TrillimAttentionTask *task, size_t index, size_t tiles, float *scores) {
    const TrillimAttentionParameters *p = task->parameters;
    QueryTile tile;
    bind_tile(task, index, tiles, scores, &tile, true);
    if (!p->length) {
        size_t item = trillim_itemsize(p->output_dtype);
        for (size_t row = 0; row < tile.rows; ++row) {
            memset((uint8_t *)task->output + tile.output[row] * item, 0, p->value_dim * item);
        }
        return TRILLIM_OK;
    }
    size_t kv_head = index / tiles;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    const uint16_t *keys = (const uint16_t *)task->keys + kv_head * p->k_strides[0];
    TrillimStatus status;
    if (tile.rows == 1 || (!p->causal && !p->windowed && p->queries >= 4) ||
        (p->queries == 1 && tile.last_key - tile.first_key <= 128)) {
        status = fast_scores(task, &tile, keys, scale);
    } else {
        status = compute_scores(task, &tile, keys, scale);
    }
    if (status != TRILLIM_OK) return status;
    float inverse[QUERY_TILE];
    for (size_t row = 0; row < tile.rows; ++row) {
        status = normalize_scores(tile.scores[row], tile.begin[row], tile.end[row], task->context, inverse + row);
        if (status != TRILLIM_OK) return status;
    }
    return apply_values(task, &tile, kv_head, inverse, 0, p->value_dim);
}

typedef struct {
    const TrillimAttentionTask *task;
    size_t tiles, work, parts;
    float inverse[MAX_TASKS][QUERY_TILE];
} Decode;

enum { DECODE_KEY_TILE = 128, DECODE_CHANNEL_TILE = 32 };

SIMD static TrillimStatus decode_scores(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Decode *decode = opaque;
    const TrillimAttentionTask *task = decode->task;
    const TrillimAttentionParameters *p = task->parameters;
    size_t group = index % decode->work, part = index / decode->work;
    float *scores = task->scratch + group * QUERY_TILE * p->length;
    QueryTile tile;
    bind_tile(task, group, decode->tiles, scores, &tile, true);
    size_t blocks = (tile.last_key - tile.first_key) / DECODE_KEY_TILE;
    size_t tail = (tile.last_key - tile.first_key) % DECODE_KEY_TILE;
    blocks += tail != 0;
    size_t per_part = blocks / decode->parts, remainder = blocks % decode->parts;
    size_t first = part * per_part + (part < remainder ? part : remainder);
    size_t past = first + per_part + (part < remainder);
    tile.last_key = past == blocks ? tile.last_key : tile.first_key + past * DECODE_KEY_TILE;
    tile.first_key += first * DECODE_KEY_TILE;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    size_t kv_head = group / decode->tiles;
    const uint16_t *keys = (const uint16_t *)task->keys + kv_head * p->k_strides[0];
    return tile.rows == 1 ? fast_scores(task, &tile, keys, scale) : compute_scores(task, &tile, keys, scale);
}

SIMD static TrillimStatus decode_normalize(void *opaque, size_t group, size_t worker) {
    (void)worker;
    Decode *decode = opaque;
    const TrillimAttentionTask *task = decode->task;
    const TrillimAttentionParameters *p = task->parameters;
    QueryTile tile;
    bind_tile(task, group, decode->tiles, task->scratch + group * QUERY_TILE * p->length, &tile, false);
    for (size_t row = 0; row < tile.rows; ++row) {
        TrillimStatus status = normalize_scores(tile.scores[row], tile.begin[row], tile.end[row], task->context,
                                                &decode->inverse[group][row]);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus decode_values(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Decode *decode = opaque;
    const TrillimAttentionTask *task = decode->task;
    const TrillimAttentionParameters *p = task->parameters;
    size_t group = index % decode->work, first = index / decode->work * DECODE_CHANNEL_TILE;
    QueryTile tile;
    bind_tile(task, group, decode->tiles, task->scratch + group * QUERY_TILE * p->length, &tile, false);
    return apply_values(task, &tile, group / decode->tiles, decode->inverse[group], first, first + DECODE_CHANNEL_TILE);
}

static TrillimStatus attend(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t tiles = query_tiles(p), work = p->kv_heads * tiles;
    size_t tasks = work < MAX_TASKS ? work : MAX_TASKS;
    float *scores = p->length ? task->scratch + index * QUERY_TILE * p->length : NULL;
    for (size_t tile = index; tile < work; tile += tasks) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        TrillimStatus status = attend_tile(task, tile, tiles, scores);
        if (status != TRILLIM_OK) return status;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_attention_bf16_fma(void *const *outputs, const void *const *inputs,
                                                            void *scratch, const void *parameters,
                                                            const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const TrillimAttentionParameters *p = parameters;
    if (!p->heads || !p->queries) return TRILLIM_OK;
    size_t work = p->kv_heads * query_tiles(p);
    size_t tasks = work < MAX_TASKS ? work : MAX_TASKS;
    TrillimAttentionTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context};
    const TrillimThreadPool *pool = context->thread_pool;
    size_t visible = p->windowed && p->window < p->length ? p->window : p->length;
    if (p->queries == 1 && visible >= 2 * DECODE_KEY_TILE && work == tasks && work <= pool->worker_count / 2) {
        size_t parts = pool->worker_count / work;
        size_t blocks = visible / DECODE_KEY_TILE + (visible % DECODE_KEY_TILE != 0);
        if (parts > blocks) parts = blocks;
        Decode decode = {.task = &task, .tiles = query_tiles(p), .work = work, .parts = parts};
        TrillimStatus status = pool->run(pool->context, work * parts, decode_scores, &decode);
        if (status == TRILLIM_OK) {
            status = pool->run(pool->context, work, decode_normalize, &decode);
        }
        if (status == TRILLIM_OK) {
            status = pool->run(pool->context, work * (p->value_dim / DECODE_CHANNEL_TILE), decode_values, &decode);
        }
        return status;
    }
    return pool->run(pool->context, tasks, attend, &task);
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
