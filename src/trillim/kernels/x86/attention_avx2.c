#define TRILLIM_ENABLE_PACKED_AVX2
#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

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
    if (dtype >= TRILLIM_ROW_INT8 && columns && columns - index % columns >= 8) {
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
