#include "common.h"
#include <immintrin.h>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif

#include <math.h>

static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum,sum));
}
static inline __m256 exp_negative(__m256 original) {
    __m256 x = _mm256_max_ps(_mm256_set1_ps(-104.0f), _mm256_min_ps(original, _mm256_set1_ps(0.0f)));
    __m256 exponent = _mm256_floor_ps(_mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f))));
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(exponent, _mm256_set1_ps(0.693359375f)));
    r = _mm256_add_ps(r, _mm256_mul_ps(exponent, _mm256_set1_ps(0.00021219444005469058f)));
    /* Range reduction keeps |r| <= ln(2)/2 for the degree-seven polynomial. */
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 720.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 120.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 24.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 6.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(value, r));
    __m256i n = _mm256_cvttps_epi32(exponent);
    __m256i low = _mm256_min_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(126)), _mm256_set1_epi32(0));
    __m256i high = _mm256_sub_epi32(n, low);
    __m256 main_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(high, _mm256_set1_epi32(127)), 23));
    __m256 tail_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(low, _mm256_set1_epi32(127)), 23));
    value = _mm256_mul_ps(_mm256_mul_ps(value, main_scale), tail_scale);
    return _mm256_blendv_ps(original, value, _mm256_cmp_ps(original, original, _CMP_EQ_OQ));
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


static bool span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t count;
    return (columns == 0 || rows <= SIZE_MAX / columns) &&
        trillim_span(3, shape, strides, &count) && count <= rows * columns;
}

static bool payload(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t width = trillim_itemsize(dtype);
    if (!trillim_float(dtype) || (columns && rows > SIZE_MAX / width / columns)) {
        return false;
    }
    *bytes = rows * columns * width;
    return true;
}

static size_t query_groups(const TrillimAttentionParameters *p) {
    size_t rows = (p->heads / p->kv_heads) * p->queries;
    return rows / 4 + (rows % 4 != 0);
}

static TrillimStatus trillim_attention_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const TrillimAttentionParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    if (p->rank != 3 || !p->kv_heads || p->heads % p->kv_heads || !p->key_dim || !p->value_dim ||
        (p->output_dtype != TRILLIM_FP32 &&
         (p->output_dtype != TRILLIM_BF16 || p->q_dtype != TRILLIM_BF16 ||
          p->k_dtype != TRILLIM_BF16 || p->v_dtype != TRILLIM_BF16 ||
          (p->length > 64 && p->queries != 1))) || p->q_strides[2] != 1 || p->k_strides[2] != 1 || p->v_strides[2] != 1 ||
        p->causal > 1 || p->windowed > 1 || p->default_scale > 1 || p->queries > INT64_MAX || p->length > INT64_MAX ||
        p->heads > SIZE_MAX / p->queries ||
        p->heads * p->queries > SIZE_MAX / trillim_itemsize(p->output_dtype) / p->value_dim) {
        return TRILLIM_ERROR;
    }
    size_t qshape[] = {p->heads, p->queries, p->key_dim};
    size_t kshape[] = {p->kv_heads, p->length, p->key_dim};
    size_t vshape[] = {p->kv_heads, p->length, p->value_dim};
    if (!span(qshape, p->q_strides, p->q_rows, p->q_columns) ||
        !payload(p->q_rows, p->q_columns, p->q_dtype, &inputs[0]) ||
        !trillim_float(p->k_dtype) || !trillim_float(p->v_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->length && (!span(kshape, p->k_strides, p->k_rows, p->k_columns) ||
        !span(vshape, p->v_strides, p->v_rows, p->v_columns) ||
        !payload(p->k_rows, p->k_columns, p->k_dtype, &inputs[1]) ||
        !payload(p->v_rows, p->v_columns, p->v_dtype, &inputs[2]))) {
        return TRILLIM_ERROR;
    }
    size_t groups = query_groups(p);
    if (p->key_dim > SIZE_MAX - 64 || p->value_dim > SIZE_MAX - 64 - p->key_dim) {
        return TRILLIM_ERROR;
    }
    size_t slots = p->kv_heads * groups < 128 ? p->kv_heads * groups : 128;
    if (slots > SIZE_MAX / sizeof(float) / 4 / (p->key_dim + p->value_dim + 64)) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->queries * p->value_dim * trillim_itemsize(p->output_dtype);
    *scratch = slots * 4 * (p->key_dim + p->value_dim + 64) * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus attend_tile(const TrillimAttentionTask *task, size_t index, size_t slot) {
    const TrillimAttentionParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = query_groups(p);
    size_t kv_head = index / groups;
    size_t rows_per_head = (p->heads / p->kv_heads) * p->queries;
    size_t first = index % groups * 4;
    size_t rows = rows_per_head - first < 4 ? rows_per_head - first : 4;
    size_t first_head = kv_head * (p->heads / p->kv_heads);
    float *queries = task->scratch + slot * 4 * (p->key_dim + p->value_dim + 64);
    float *accumulator = queries + 4 * p->key_dim;
    float *scores = accumulator + 4 * p->value_dim;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float denominator[4] = {0, 0, 0, 0};
    size_t begin[4], end[4];
    size_t first_key = p->length, last_key = 0;
    for (size_t row = 0; row < rows; ++row) {
        size_t head = first_head + (first + row) / p->queries;
        size_t position_index = (first + row) % p->queries;
        size_t query = head * p->q_strides[0] + position_index * p->q_strides[1];
        size_t channel = 0;
        for (; p->key_dim - channel >= 8; channel += 8) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            _mm256_storeu_ps(queries + row * p->key_dim + channel, load8(task->query, query + channel, p->q_dtype));
        }
        for (; channel < p->key_dim; ++channel) {
            queries[row * p->key_dim + channel] = trillim_load(task->query, query + channel, p->q_dtype).f;
        }
        for (channel = 0; channel < p->value_dim; ++channel) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            accumulator[row * p->value_dim + channel] = 0;
        }
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)position_index;
        begin[row] = p->windowed && position >= 0 && (uint64_t)position + 1 > p->window ?
            (size_t)position + 1 - p->window : 0;
        end[row] = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        if (begin[row] < first_key) {
            first_key = begin[row];
        }
        if (end[row] > last_key) {
            last_key = end[row];
        }
    }
    for (size_t block = first_key; block < last_key; block += 64) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = last_key - block < 64 ? last_key - block : 64;
        for (size_t key = 0; key < count; ++key) {
            size_t base = kv_head * p->k_strides[0] + (block + key) * p->k_strides[1];
            if (p->output_dtype == TRILLIM_BF16) {
                float sums[4] = {0};
                for (size_t channel = 0; channel < p->key_dim; ++channel) {
                    if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    float value = trillim_bf16(((const uint16_t *)task->keys)[base + channel]);
                    for (size_t row = 0; row < rows; ++row) {
                        sums[row] += queries[row * p->key_dim + channel] * value;
                    }
                }
                for (size_t row = 0; row < rows; ++row) {
                    scores[row * 64 + key] =
                        block + key >= begin[row] && block + key < end[row] ? sums[row] * scale : -INFINITY;
                }
                continue;
            }

            __m256 sums[4][2];
            for (size_t row = 0; row < rows; ++row) {
                sums[row][0] = sums[row][1] = _mm256_set1_ps(0);
            }
            size_t channel = 0;
            for (; p->key_dim - channel >= 16; channel += 16) {
                if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                __m256 k0 = load8(task->keys, base + channel, p->k_dtype);
                __m256 k1 = load8(task->keys, base + channel + 8, p->k_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    const float *q = queries + row * p->key_dim + channel;
                    sums[row][0] = _mm256_add_ps(sums[row][0], _mm256_mul_ps(_mm256_loadu_ps(q), k0));
                    sums[row][1] = _mm256_add_ps(sums[row][1], _mm256_mul_ps(_mm256_loadu_ps(q + 8), k1));
                }
            }
            for (; p->key_dim - channel >= 8; channel += 8) {
                __m256 k0 = load8(task->keys, base + channel, p->k_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    sums[row][0] = _mm256_add_ps(sums[row][0], _mm256_mul_ps(_mm256_loadu_ps(queries + row * p->key_dim + channel), k0));
                }
            }
            for (size_t row = 0; row < rows; ++row) {
                float sum = sum8(_mm256_add_ps(sums[row][0], sums[row][1]));
                for (size_t tail = channel; tail < p->key_dim; ++tail) {
                    sum += queries[row * p->key_dim + tail] * trillim_load(task->keys, base + tail, p->k_dtype).f;
                }
                scores[row * 64 + key] = block + key >= begin[row] && block + key < end[row] ? sum * scale : -INFINITY;
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float peak = -INFINITY;
            for (size_t key = 0; key < count; ++key) {
                peak = fmaxf(peak, scores[row * 64 + key]);
            }
            if (block >= end[row] || block + count <= begin[row]) {
                for (size_t key = 0; key < count; ++key) {
                    scores[row * 64 + key] = 0;
                }
                continue;
            }
            peak = fmaxf(peak, maximum[row]);
            float normalization_peak = peak == -INFINITY ? 0.0f : peak;
            float previous = maximum[row] == -INFINITY ? 0 : expf(maximum[row] - normalization_peak);
            maximum[row] = peak;
            __m256 total = _mm256_set1_ps(0);
            size_t key = 0;
            for (; p->output_dtype == TRILLIM_FP32 && count - key >= 8; key += 8) {
                float *target = scores + row * 64 + key;
                __m256 value = exp_negative(_mm256_sub_ps(_mm256_loadu_ps(target), _mm256_set1_ps(normalization_peak)));
                _mm256_storeu_ps(target, value);
                total = _mm256_add_ps(total, value);
            }
            float sum = sum8(total);
            for (; key < count; ++key) {
                float value = expf(scores[row * 64 + key] - normalization_peak);
                scores[row * 64 + key] = value;
                sum += value;
            }
            denominator[row] = denominator[row] * previous + sum;
            size_t channel = 0;
            for (; p->value_dim - channel >= 8; channel += 8) {
                if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float *output = accumulator + row * p->value_dim + channel;
                _mm256_storeu_ps(output, _mm256_mul_ps(_mm256_loadu_ps(output),_mm256_set1_ps(previous)));
            }
            for (; channel < p->value_dim; ++channel) {
                accumulator[row * p->value_dim + channel] *= previous;
            }
        }
        size_t channel = 0;
        for (; p->value_dim - channel >= 16; channel += 16) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 sums[4][2];
            for (size_t row = 0; row < rows; ++row) {
                const float *output = accumulator + row * p->value_dim + channel;
                sums[row][0] = _mm256_loadu_ps(output);
                sums[row][1] = _mm256_loadu_ps(output + 8);
            }
            for (size_t key = 0; key < count; ++key) {
                size_t base = kv_head * p->v_strides[0] + (block + key) * p->v_strides[1] + channel;
                __m256 v0 = load8(task->values, base, p->v_dtype);
                __m256 v1 = load8(task->values, base + 8, p->v_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    if (block + key >= begin[row] && block + key < end[row]) {
                        float probability = scores[row * 64 + key];
                        sums[row][0] = _mm256_add_ps(sums[row][0], _mm256_mul_ps(v0, _mm256_set1_ps(probability)));
                        sums[row][1] = _mm256_add_ps(sums[row][1], _mm256_mul_ps(v1, _mm256_set1_ps(probability)));
                    }
                }
            }
            for (size_t row = 0; row < rows; ++row) {
                float *output = accumulator + row * p->value_dim + channel;
                _mm256_storeu_ps(output, sums[row][0]);
                _mm256_storeu_ps(output + 8, sums[row][1]);
            }
        }
        for (; channel < p->value_dim; ++channel) {
            for (size_t key = 0; key < count; ++key) {
                size_t base = kv_head * p->v_strides[0] + (block + key) * p->v_strides[1] + channel;
                float value = trillim_load(task->values, base, p->v_dtype).f;
                for (size_t row = 0; row < rows; ++row) {
                    if (block + key >= begin[row] && block + key < end[row]) {
                        accumulator[row * p->value_dim + channel] += value * scores[row * 64 + key];
                    }
                }
            }
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        size_t head = first_head + (first + row) / p->queries;
        size_t output = (((first + row) % p->queries) * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            TrillimValue value = {.f = accumulator[row * p->value_dim + channel] / (denominator[row] > 0.0f ? denominator[row] : 1.0f)};
            trillim_store(task->output, output + channel, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus attend(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = query_groups(p);
    size_t tiles = p->kv_heads * groups;
    size_t slots = tiles < 128 ? tiles : 128;
    for (size_t index = slot; index < tiles;) {
        TrillimStatus status = attend_tile(task, index, slot);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (tiles - index <= slots) {
            break;
        }
        index += slots;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_attention_tiled(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimAttentionParameters *p = parameters;
    if (!p->heads || !p->queries) return TRILLIM_OK;
    TrillimAttentionTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context};
    size_t groups = query_groups(p);
    size_t slots = p->kv_heads * groups < 128 ? p->kv_heads * groups : 128;
    return context->thread_pool->run(context->thread_pool->context, slots, attend, &task);
}

TRILLIM_KERNEL const char *trillim_x86_attention_tiled_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention as bind\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_attention_tiled_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_attention_tiled_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
