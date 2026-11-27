#include "common.h"
#include <math.h>
#include <string.h>
#ifdef __APPLE__
#include <dlfcn.h>
#include <stdatomic.h>
#endif

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t width = trillim_itemsize(dtype);
    if (!trillim_float(dtype) || (columns && rows > SIZE_MAX / width / columns)) {
        return false;
    }
    *bytes = rows * columns * width;
    return true;
}

static inline float trillim_read_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    (void)rows;
    (void)columns;
    return trillim_load(data, index, dtype).f;
}

#define HAS_MASK 1
#define HAS_SINKS 0
#define INPUT_COUNT 4

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
    uint32_t softcapped;
    float softcap;
    const size_t *mask_strides;
    size_t mask_rows;
    size_t mask_columns;
    uint32_t mask_dtype;
    uint32_t sink_dtype;
} TrillimAttentionParameters;

typedef struct {
    void *output;
    const void *query;
    const void *keys;
    const void *values;
    const void *mask;
    const void *sinks;
    float *scratch;
    const TrillimAttentionParameters *parameters;
    const TrillimKernelContext *context;
} TrillimAttentionTask;

#ifdef __APPLE__
#define BLAS_QUERY_TILE 32
#define BLAS_SLOTS 32
#define BLAS_ROW_WORK_LIMIT (1024 * 1024)

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);

static Sgemm sgemm;
static GetThreading get_threading;
static SetThreading set_threading;
static atomic_uint blas_state;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
        unsigned expected = 0;
        if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                    memory_order_acq_rel, memory_order_acquire)) {
            void *library = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                                   RTLD_LOCAL | RTLD_LAZY);
            if (library) {
                void *multiply = dlsym(library, "cblas_sgemm$NEWLAPACK");
                if (!multiply) {
                    multiply = dlsym(library, "cblas_sgemm");
                }
                void *get = dlsym(library, "BLASGetThreading");
                void *set = dlsym(library, "BLASSetThreading");
                if (multiply && get && set) {
                    memcpy(&sgemm, &multiply, sizeof(sgemm));
                    memcpy(&get_threading, &get, sizeof(get_threading));
                    memcpy(&set_threading, &set, sizeof(set_threading));
                } else {
                    dlclose(library);
                }
            }
            atomic_store_explicit(&blas_state, 2, memory_order_release);
        } else {
            while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            }
        }
    }
    return sgemm != NULL;
}

static bool blas_shape(const TrillimAttentionParameters *p) {
    /* Bound work between cancellation checks around each matrix product. */
    return p->output_dtype == TRILLIM_BF16 && p->q_dtype == TRILLIM_FP32 &&
        p->k_dtype == TRILLIM_FP32 && p->v_dtype == TRILLIM_BF16 &&
        p->queries >= BLAS_QUERY_TILE && p->length >= BLAS_QUERY_TILE &&
        p->key_dim && p->length <= BLAS_ROW_WORK_LIMIT / p->key_dim &&
        p->value_dim && p->length <= BLAS_ROW_WORK_LIMIT / p->value_dim &&
        !p->causal && !p->windowed && !p->softcapped;
}

static size_t blas_tiles(const TrillimAttentionParameters *p) {
    return p->queries / BLAS_QUERY_TILE + (p->queries % BLAS_QUERY_TILE != 0);
}

static size_t blas_slots(const TrillimAttentionParameters *p) {
    size_t jobs = p->heads * blas_tiles(p);
    return jobs < BLAS_SLOTS ? jobs : BLAS_SLOTS;
}
static bool direct_rows(const size_t *strides, size_t width) {
    return strides[2] == 1 && strides[1] >= width && strides[1] <= INT32_MAX;
}

static size_t packed_rows(size_t heads, size_t rows, size_t width, const size_t *strides) {
    return direct_rows(strides, width) ? 0 : heads * rows * width;
}

static bool append_workspace(size_t heads, size_t rows, size_t width, size_t *total) {
    if (rows != 0 && heads > (SIZE_MAX / sizeof(float) - *total) / rows / width) {
        return false;
    }
    *total += heads * rows * width;
    return true;
}

#endif

static inline bool trillim_attention_span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t span;
    return (columns == 0 || rows <= SIZE_MAX / columns) &&
        trillim_span(3, shape, strides, &span) && span <= rows * columns;
}

static inline TrillimStatus trillim_attention_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimAttentionParameters *p = (const TrillimAttentionParameters *)parameters;
    for (size_t index = 0; index < INPUT_COUNT; ++index) {
        inputs[index] = 0;
    }
    outputs[0] = 0;
    *scratch = 0;
    if (p->output_dtype != TRILLIM_FP32 &&
        (p->output_dtype != TRILLIM_BF16 || p->causal || p->windowed || p->softcapped ||
         p->q_dtype != TRILLIM_FP32 || p->k_dtype != TRILLIM_FP32 ||
         p->v_dtype != TRILLIM_BF16)) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->queries == 0) {
        return TRILLIM_OK;
    }
    if (p->rank != 3 || p->kv_heads == 0 || p->heads % p->kv_heads != 0 ||
        p->key_dim == 0 || p->value_dim == 0 || p->softcapped > 1 ||
        (p->softcapped && (!isfinite(p->softcap) || p->softcap <= 0)) ||
        !trillim_float(p->q_dtype) || !trillim_float(p->k_dtype) || !trillim_float(p->v_dtype) ||
        p->causal > 1 || p->windowed > 1 || p->default_scale > 1 ||
        p->queries > INT64_MAX || p->length > INT64_MAX ||
        p->heads > SIZE_MAX / p->queries || p->heads * p->queries > SIZE_MAX / trillim_itemsize(p->output_dtype) / p->value_dim) {
        return TRILLIM_ERROR;
    }
    size_t query_shape[] = {p->heads, p->queries, p->key_dim};
    size_t key_shape[] = {p->kv_heads, p->length, p->key_dim};
    size_t value_shape[] = {p->kv_heads, p->length, p->value_dim};
    if (!trillim_attention_span(query_shape, p->q_strides, p->q_rows, p->q_columns) ||
        !trillim_payload_size(p->q_rows, p->q_columns, p->q_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->length != 0 && (!trillim_attention_span(key_shape, p->k_strides, p->k_rows, p->k_columns) ||
        !trillim_attention_span(value_shape, p->v_strides, p->v_rows, p->v_columns) ||
        !trillim_payload_size(p->k_rows, p->k_columns, p->k_dtype, &inputs[1]) ||
        !trillim_payload_size(p->v_rows, p->v_columns, p->v_dtype, &inputs[2]))) {
        return TRILLIM_ERROR;
    }
    if (HAS_MASK && p->length != 0) {
        size_t mask_shape[] = {p->heads, p->queries, p->length};
        if (!trillim_attention_span(mask_shape, p->mask_strides, p->mask_rows, p->mask_columns) ||
            !trillim_payload_size(p->mask_rows, p->mask_columns, p->mask_dtype, &inputs[3])) {
            return TRILLIM_ERROR;
        }
    }
    if (HAS_SINKS && !trillim_payload_size(1, p->heads, p->sink_dtype, &inputs[3 + HAS_MASK])) {
        return TRILLIM_ERROR;
    }
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    if (p->value_dim > SIZE_MAX - 64 || p->heads > SIZE_MAX / tiles ||
        p->heads * tiles > SIZE_MAX / sizeof(float) / (p->value_dim + 64)) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->queries * p->value_dim * trillim_itemsize(p->output_dtype);
    *scratch = p->heads * tiles * (p->value_dim + 64) * sizeof(float);
#ifdef __APPLE__
    /* Reserve for smaller shapes even when the maximum exceeds the BLAS work limit. */
    if (p->output_dtype == TRILLIM_BF16 && p->q_dtype == TRILLIM_FP32 &&
        p->k_dtype == TRILLIM_FP32 && p->v_dtype == TRILLIM_BF16 &&
        p->queries >= BLAS_QUERY_TILE && p->length >= BLAS_QUERY_TILE &&
        !p->causal && !p->windowed && !p->softcapped) {
        size_t count = 0;
        if (!append_workspace(blas_slots(p), BLAS_QUERY_TILE, p->length + p->value_dim, &count) ||
            !append_workspace(p->kv_heads, p->length, p->value_dim, &count) ||
            !append_workspace(p->heads, p->queries, p->key_dim, &count) ||
            !append_workspace(p->kv_heads, p->length, p->key_dim, &count)) {
            return TRILLIM_ERROR;
        }
        size_t bytes = count * sizeof(float);
        if (bytes > *scratch) {
            *scratch = bytes;
        }
    }
#endif
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_score(
    const TrillimAttentionTask *task, size_t query, size_t key, float *score
) {
    const TrillimAttentionParameters *p = task->parameters;
    float sum = 0.0f;
    size_t channel = 0;
    for (; channel < p->key_dim; ++channel) {
        if (p->key_dim > 1024 && (channel & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_read_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
        float k = trillim_read_float(task->keys, key + channel * p->k_strides[2], p->k_dtype, p->k_rows, p->k_columns);
        sum += q * k;
    }
    *score = sum;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_task(void *opaque, size_t index, size_t worker) {
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
        for (size_t block = 0; block < p->length; block += 64) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->length - block < 64 ? p->length - block : 64;
            float block_maximum = -INFINITY;
            for (size_t j = 0; j < count; ++j) {
                size_t key = kv_head * p->k_strides[0] + (block + j) * p->k_strides[1];
                float score = -INFINITY;
                if (block + j >= begin && block + j < end) {
                    TrillimStatus status = trillim_attention_score(task, query, key, &score);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                    if (p->q_dtype == p->k_dtype) {
                        if (p->q_dtype == TRILLIM_FP16) {
                            score = trillim_fp16(trillim_to_fp16(score));
                        } else if (p->q_dtype == TRILLIM_BF16) {
                            score = trillim_bf16(trillim_to_bf16(score));
                        }
                    }
                    score *= scale;
                    if (p->softcapped) {
                        score = tanhf(score / p->softcap) * p->softcap;
                    }
                }
                if (HAS_MASK) {
                    size_t offset = head * p->mask_strides[0] + qi * p->mask_strides[1] + (block + j) * p->mask_strides[2];
                    score += trillim_load(task->mask, offset, p->mask_dtype).f;
                }
                scores[j] = score;
                block_maximum = fmaxf(block_maximum, score);
            }
            float next_maximum = fmaxf(maximum, block_maximum);
            float normalization_peak = next_maximum == -INFINITY ? 0.0f : next_maximum;
            float previous = maximum == -INFINITY ? 0.0f : expf(maximum - normalization_peak);
            float block_sum = 0.0f;
            for (size_t j = 0; j < count; ++j) {
                scores[j] = expf(scores[j] - (next_maximum == -INFINITY ? 0.0f : next_maximum));
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
            for (size_t j = 0; j < count; ++j) {
                size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1];
                size_t channel = 0;
                for (; channel < p->value_dim; ++channel) {
                    if (p->value_dim > 1024 && (channel & 1023) == 0 &&
                        task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    float v = trillim_read_float(task->values, value + channel * p->v_strides[2], p->v_dtype, p->v_rows, p->v_columns);
                    accumulator[channel] += scores[j] * v;
                }
            }
        }
        denominator = denominator > 0.0f ? denominator : 1.0f;
        if (HAS_SINKS) {
            float sink = trillim_load(task->sinks, head, p->sink_dtype).f;
            float peak = fmaxf(maximum, sink);
            if (peak == -INFINITY) {
                peak = 0;
            }
            float previous = maximum == -INFINITY ? 0 : expf(maximum - peak);
            denominator = denominator * previous + expf(sink - peak);
            for (size_t channel = 0; channel < p->value_dim; ++channel) {
                if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                accumulator[channel] *= previous;
            }
        }
        size_t output = (qi * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            TrillimValue value;
            value.f = accumulator[channel] / (denominator > 0.0f ? denominator : 1.0f);
            trillim_store(task->output, output + channel, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

#ifdef __APPLE__
typedef struct {
    const void *source;
    const size_t *strides;
    float *destination;
    size_t rows;
    size_t width;
    uint32_t dtype;
    const TrillimKernelContext *context;
} PackTask;

static size_t row_tiles(size_t rows) {
    return rows / BLAS_QUERY_TILE + (rows % BLAS_QUERY_TILE != 0);
}

static TrillimStatus pack_head_rows(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const PackTask *task = opaque;
    size_t tiles = row_tiles(task->rows);
    size_t head = index / tiles;
    size_t first = (index % tiles) * BLAS_QUERY_TILE;
    size_t count = task->rows - first;
    if (count > BLAS_QUERY_TILE) {
        count = BLAS_QUERY_TILE;
    }
    for (size_t row = first; row < first + count; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = head * task->strides[0] + row * task->strides[1];
        float *dest = task->destination + (head * task->rows + row) * task->width;
        for (size_t channel = 0; channel < task->width; ++channel) {
            dest[channel] = trillim_load(task->source, offset + channel * task->strides[2], task->dtype).f;
        }
    }
    return TRILLIM_OK;
}

static void normalize_scores(float *scores, const void *mask, size_t mask_offset,
                             const TrillimAttentionParameters *p, float scale) {
    float maximum = -INFINITY;
    for (size_t key = 0; key < p->length; ++key) {
        float bias = trillim_load(mask, mask_offset + key * p->mask_strides[2], p->mask_dtype).f;
        scores[key] = scores[key] * scale + bias;
        maximum = fmaxf(maximum, scores[key]);
    }
    if (maximum == -INFINITY) {
        maximum = 0;
    }
    float total = 0;
    for (size_t key = 0; key < p->length; ++key) {
        scores[key] = expf(scores[key] - maximum);
        total += scores[key];
    }
    if (total > 0) {
        float inverse = 1.0f / total;
        for (size_t key = 0; key < p->length; ++key) {
            scores[key] *= inverse;
        }
    }
}

static TrillimStatus blas_tile(const TrillimAttentionTask *task, size_t index,
                              float *scores, float *result) {
    const TrillimAttentionParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = blas_tiles(p);
    size_t head = index / tiles;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t first = (index % tiles) * BLAS_QUERY_TILE;
    size_t count = p->queries - first;
    if (count > BLAS_QUERY_TILE) {
        count = BLAS_QUERY_TILE;
    }
    bool direct_q = direct_rows(p->q_strides, p->key_dim);
    bool direct_k = direct_rows(p->k_strides, p->key_dim);
    size_t values_count = p->kv_heads * p->length * p->value_dim;
    size_t queries_count = packed_rows(p->heads, p->queries, p->key_dim, p->q_strides);
    const float *q = direct_q
        ? (const float *)task->query + head * p->q_strides[0] + first * p->q_strides[1]
        : task->scratch + values_count + (head * p->queries + first) * p->key_dim;
    const float *k = direct_k
        ? (const float *)task->keys + kv_head * p->k_strides[0]
        : task->scratch + values_count + queries_count + (kv_head * p->length) * p->key_dim;
    const float *v = task->scratch + kv_head * p->length * p->value_dim;
    size_t query_stride = direct_q ? p->q_strides[1] : p->key_dim;
    size_t key_stride = direct_k ? p->k_strides[1] : p->key_dim;
    sgemm(101, 111, 112, (int)count, (int)p->length, (int)p->key_dim, 1.0f,
          q, (int)query_stride, k, (int)key_stride, 0.0f, scores, (int)p->length);
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t row = 0; row < count; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = head * p->mask_strides[0] + (first + row) * p->mask_strides[1];
        normalize_scores(scores + row * p->length, task->mask, offset, p, scale);
    }
    sgemm(101, 111, 111, (int)count, (int)p->value_dim, (int)p->length, 1.0f,
          scores, (int)p->length, v, (int)p->value_dim, 0.0f, result, (int)p->value_dim);
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t row = 0; row < count; ++row) {
        size_t offset = ((first + row) * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            trillim_store(task->output, offset + channel, p->output_dtype,
                (TrillimValue){.f = result[row * p->value_dim + channel]});
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus blas_task(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t slots = blas_slots(p);
    size_t jobs = p->heads * blas_tiles(p);
    float *scores = task->scratch + p->kv_heads * p->length * p->value_dim +
        packed_rows(p->heads, p->queries, p->key_dim, p->q_strides) +
        packed_rows(p->kv_heads, p->length, p->key_dim, p->k_strides) +
        slot * BLAS_QUERY_TILE * (p->length + p->value_dim);
    float *result = scores + BLAS_QUERY_TILE * p->length;
    unsigned previous = get_threading();
    bool restore = previous != 1 && set_threading(1) == 0;
    TrillimStatus status = TRILLIM_OK;
    for (size_t index = slot; index < jobs; index += slots) {
        status = blas_tile(task, index, scores, result);
        if (status != TRILLIM_OK) {
            break;
        }
    }
    if (restore) {
        (void)set_threading(previous);
    }
    return status;
}
#endif

TRILLIM_KERNEL TrillimStatus trillim_scalar_attention_mask(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimAttentionParameters *p = (const TrillimAttentionParameters *)parameters;
    TrillimAttentionTask task = {
        outputs[0], inputs[0], inputs[1], inputs[2], HAS_MASK ? inputs[3] : NULL,
        HAS_SINKS ? inputs[3 + HAS_MASK] : NULL, scratch, p, context
    };
#ifdef __APPLE__
    if (p->heads && blas_shape(p) && initialize_blas()) {
        const TrillimThreadPool *pool = context->thread_pool;
        float *packed = scratch;
        PackTask values = {inputs[2], p->v_strides, packed, p->length, p->value_dim, p->v_dtype, context};
        TrillimStatus status = pool->run(pool->context, p->kv_heads * row_tiles(p->length), pack_head_rows, &values);
        if (status != TRILLIM_OK) {
            return status;
        }
        packed += p->kv_heads * p->length * p->value_dim;
        if (!direct_rows(p->q_strides, p->key_dim)) {
            PackTask queries = {inputs[0], p->q_strides, packed, p->queries, p->key_dim, p->q_dtype, context};
            status = pool->run(pool->context, p->heads * row_tiles(p->queries), pack_head_rows, &queries);
            if (status != TRILLIM_OK) {
                return status;
            }
            packed += p->heads * p->queries * p->key_dim;
        }
        if (!direct_rows(p->k_strides, p->key_dim)) {
            PackTask keys = {inputs[1], p->k_strides, packed, p->length, p->key_dim, p->k_dtype, context};
            status = pool->run(pool->context, p->kv_heads * row_tiles(p->length), pack_head_rows, &keys);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return context->thread_pool->run(context->thread_pool->context, blas_slots(p), blas_task, &task);
    }
#endif
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->heads * tiles, trillim_attention_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_attention_mask_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention, _broadcast_strides, _contiguous_strides, _storage_shape, _product, _DTYPE_IDS\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "import math\n"
           "physical_inputs = ('q', 'k', 'v', 'mask')\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, *, causal: bool = True, scale = None, window = None, softcap = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, softcap=softcap, dtype=dtype, mask=mask),)\n"
           "def bind(context):\n"
           "    result = _attention(context)\n"
           "    softcap = context.attrs.get('softcap')\n"
           "    if context.outputs[0].dtype != 'fp32' and (\n"
           "        context.outputs[0].dtype != 'bf16' or result['causal'] or result['windowed'] or softcap is not None or\n"
           "        tuple(context.inputs[name].dtype for name in ('q', 'k', 'v')) != ('fp32', 'fp32', 'bf16')\n"
           "    ):\n"
           "        raise ValueError('unsupported masked attention precision or window')\n"
           "    if softcap is not None and (not isinstance(softcap, (int, float)) or not math.isfinite(softcap) or softcap <= 0):\n"
           "        raise ValueError('attention softcap must be finite and positive')\n"
           "    result.update(softcapped=int(softcap is not None), softcap=float(softcap) if softcap is not None else 1.0,\n"
           "                  mask_strides=(_constant_expression(0),) * 3, mask_rows=0, mask_columns=0, mask_dtype=_DTYPE_IDS['fp32'], sink_dtype=_DTYPE_IDS['fp32'])\n"
           "    for name in ('q', 'k', 'v'):\n"
           "        if context.inputs[name].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "            raise ValueError('extended attention requires native floating tensors')\n"
           "    mask = context.inputs['mask']\n"
           "    q, k = context.inputs['q'], context.inputs['k']\n"
           "    if mask.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('attention mask must use native floating storage')\n"
           "    result['mask_strides'] = _broadcast_strides(mask.shape, (q.shape[1], q.shape[0], k.shape[0]), mask.strides)\n"
           "    result['mask_rows'], result['mask_columns'] = _storage_shape(mask)\n"
           "    result['mask_dtype'] = _DTYPE_IDS[mask.dtype]\n"
           "    return result\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_attention_mask_parameters(void) {
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
        {"softcapped", "uint32_t", offsetof(TrillimAttentionParameters, softcapped), NULL},
        {"softcap", "float", offsetof(TrillimAttentionParameters, softcap), NULL},
        {"mask_strides", "size_t", offsetof(TrillimAttentionParameters, mask_strides), "rank"},
        {"mask_rows", "size_t", offsetof(TrillimAttentionParameters, mask_rows), NULL},
        {"mask_columns", "size_t", offsetof(TrillimAttentionParameters, mask_columns), NULL},
        {"mask_dtype", "uint32_t", offsetof(TrillimAttentionParameters, mask_dtype), NULL},
        {"sink_dtype", "uint32_t", offsetof(TrillimAttentionParameters, sink_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimAttentionParameters), TRILLIM_ALIGNOF(TrillimAttentionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, INPUT_COUNT, 1,
        trillim_attention_requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_attention_mask_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
