#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

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

static inline float32x4_t exp_negative(float32x4_t original) {
    float32x4_t x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(original, vdupq_n_f32(0.0f)));
    float32x4_t exponent = vrndmq_f32(vfmaq_n_f32(vdupq_n_f32(0.5f), x, 1.4426950408889634f));
    float32x4_t r = vfmsq_n_f32(x, exponent, 0.693359375f);
    r = vfmaq_n_f32(r, exponent, 0.00021219444005469058f);
    /* Range reduction keeps |r| <= ln(2)/2 for the degree-seven polynomial. */
    float32x4_t value = vdupq_n_f32(1.0f / 5040.0f);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(0.5f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    int32x4_t n = vcvtq_s32_f32(exponent);
    int32x4_t low = vminq_s32(vaddq_s32(n, vdupq_n_s32(126)), vdupq_n_s32(0));
    int32x4_t high = vsubq_s32(n, low);
    float32x4_t main_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(high, vdupq_n_s32(127)), 23));
    float32x4_t tail_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(low, vdupq_n_s32(127)), 23));
    value = vmulq_f32(vmulq_f32(value, main_scale), tail_scale);
    return vbslq_f32(vceqq_f32(original, original), value, original);
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
    if (p->output_dtype != TRILLIM_FP32) {
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
    return TRILLIM_OK;
}

static inline float32x4_t trillim_neon_load4(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)data + index)));
    }
    if (dtype == TRILLIM_BF16) {
        return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16((const uint16_t *)data + index), 16));
    }
    float lanes[4];
    for (size_t i = 0; i < 4; ++i) {
        lanes[i] = trillim_read_float(data, index + i, dtype, rows, columns);
    }
    return vld1q_f32(lanes);
}

static inline float scalar_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (dtype == TRILLIM_FP32) {
        return ((const float *)data)[index];
    }
    if (dtype == TRILLIM_FP16) {
        return (float)((const float16_t *)data)[index];
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(((const uint16_t *)data)[index]);
    }
    return trillim_read_float(data, index, dtype, rows, columns);
}

static inline TrillimStatus trillim_attention_score(
    const TrillimAttentionTask *task, size_t query, size_t key, float *score
) {
    const TrillimAttentionParameters *p = task->parameters;
    /* Preserve the intermediate QK Matmul cast's rounding boundaries. */
    if (p->q_dtype == p->k_dtype && p->q_dtype != TRILLIM_FP32) {
        float sum = 0;
        for (size_t channel = 0; channel < p->key_dim; ++channel) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float q = scalar_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
            float k = scalar_float(task->keys, key + channel * p->k_strides[2], p->k_dtype, p->k_rows, p->k_columns);
            sum += q * k;
        }
        *score = sum;
        return TRILLIM_OK;
    }
    size_t channel = 0;
    float32x4_t a = vdupq_n_f32(0.0f), b = a, c = a, d = a;
    if (p->q_strides[2] == 1 && p->k_strides[2] == 1) {
        for (; p->key_dim - channel >= 16; channel += 16) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            a = vfmaq_f32(a,
                trillim_neon_load4(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns),
                trillim_neon_load4(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns));
            b = vfmaq_f32(b,
                trillim_neon_load4(task->query, query + channel + 4, p->q_dtype, p->q_rows, p->q_columns),
                trillim_neon_load4(task->keys, key + channel + 4, p->k_dtype, p->k_rows, p->k_columns));
            c = vfmaq_f32(c,
                trillim_neon_load4(task->query, query + channel + 8, p->q_dtype, p->q_rows, p->q_columns),
                trillim_neon_load4(task->keys, key + channel + 8, p->k_dtype, p->k_rows, p->k_columns));
            d = vfmaq_f32(d,
                trillim_neon_load4(task->query, query + channel + 12, p->q_dtype, p->q_rows, p->q_columns),
                trillim_neon_load4(task->keys, key + channel + 12, p->k_dtype, p->k_rows, p->k_columns));
        }
        for (; p->key_dim - channel >= 4; channel += 4) {
            a = vfmaq_f32(a,
                trillim_neon_load4(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns),
                trillim_neon_load4(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns));
        }
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; channel < p->key_dim; ++channel) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_read_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
        float k = trillim_read_float(task->keys, key + channel * p->k_strides[2], p->k_dtype, p->k_rows, p->k_columns);
        sum += q * k;
    }
    *score = sum;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_scores4(
    const TrillimAttentionTask *task, size_t query, size_t key, float *scores
) {
    const TrillimAttentionParameters *p = task->parameters;
    float32x4_t sums = vdupq_n_f32(0);
    for (size_t channel = 0; channel < p->key_dim; ++channel) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = key + channel * p->k_strides[2];
        float32x4_t keys;
        if (p->k_dtype == TRILLIM_FP16 || p->k_dtype == TRILLIM_BF16) {
            const uint16_t *input = task->keys;
            uint16x4_t bits = {
                input[offset], input[offset + p->k_strides[1]],
                input[offset + 2 * p->k_strides[1]], input[offset + 3 * p->k_strides[1]]
            };
            keys = p->k_dtype == TRILLIM_FP16 ? vcvt_f32_f16(vreinterpret_f16_u16(bits)) :
                vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
        } else {
            float lanes[] = {
                scalar_float(task->keys, offset, p->k_dtype, p->k_rows, p->k_columns),
                scalar_float(task->keys, offset + p->k_strides[1], p->k_dtype, p->k_rows, p->k_columns),
                scalar_float(task->keys, offset + 2 * p->k_strides[1], p->k_dtype, p->k_rows, p->k_columns),
                scalar_float(task->keys, offset + 3 * p->k_strides[1], p->k_dtype, p->k_rows, p->k_columns)
            };
            keys = vld1q_f32(lanes);
        }
        float q = scalar_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
        sums = vfmaq_n_f32(sums, keys, q);
    }
    vst1q_f32(scores, sums);
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
                bool grouped = (p->q_dtype == p->k_dtype && p->q_dtype != TRILLIM_FP32) &&
                    count - (j / 4) * 4 >= 4;
                if (grouped && j % 4 == 0) {
                    TrillimStatus status = trillim_attention_scores4(task, query, key, scores + j);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                float score = -INFINITY;
                if (block + j >= begin && block + j < end) {
                    if (grouped) {
                        score = scores[j];
                    } else {
                        TrillimStatus status = trillim_attention_score(task, query, key, &score);
                        if (status != TRILLIM_OK) {
                            return status;
                        }
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
            float32x4_t block_total = vdupq_n_f32(0);
            size_t j = 0;
            for (; p->output_dtype == TRILLIM_FP32 && count - j >= 4; j += 4) {
                float32x4_t values = exp_negative(vsubq_f32(vld1q_f32(scores + j), vdupq_n_f32(normalization_peak)));
                vst1q_f32(scores + j, values);
                block_total = vaddq_f32(block_total, values);
            }
            float block_sum = vaddvq_f32(block_total);
            for (; j < count; ++j) {
                scores[j] = expf(scores[j] - normalization_peak);
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
            size_t channel = 0;
            if (p->v_strides[2] == 1) {
                for (; p->value_dim - channel >= 16; channel += 16) {
                    if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    float32x4_t a = vld1q_f32(accumulator + channel);
                    float32x4_t b = vld1q_f32(accumulator + channel + 4);
                    float32x4_t c = vld1q_f32(accumulator + channel + 8);
                    float32x4_t d = vld1q_f32(accumulator + channel + 12);
                    for (size_t j = 0; j < count; ++j) {
                        size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1] + channel;
                        a = vfmaq_n_f32(a, trillim_neon_load4(task->values, value, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
                        b = vfmaq_n_f32(b, trillim_neon_load4(task->values, value + 4, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
                        c = vfmaq_n_f32(c, trillim_neon_load4(task->values, value + 8, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
                        d = vfmaq_n_f32(d, trillim_neon_load4(task->values, value + 12, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
                    }
                    vst1q_f32(accumulator + channel, a);
                    vst1q_f32(accumulator + channel + 4, b);
                    vst1q_f32(accumulator + channel + 8, c);
                    vst1q_f32(accumulator + channel + 12, d);
                }
                for (; p->value_dim - channel >= 4; channel += 4) {
                    float32x4_t a = vld1q_f32(accumulator + channel);
                    for (size_t j = 0; j < count; ++j) {
                        size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1] + channel;
                        a = vfmaq_n_f32(a, trillim_neon_load4(task->values, value, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
                    }
                    vst1q_f32(accumulator + channel, a);
                }
            }
            for (; channel < p->value_dim; ++channel) {
                if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                for (size_t j = 0; j < count; ++j) {
                    size_t value = kv_head * p->v_strides[0] + (block + j) * p->v_strides[1] + channel * p->v_strides[2];
                    accumulator[channel] += scores[j] * trillim_read_float(task->values, value, p->v_dtype, p->v_rows, p->v_columns);
                }
            }
        }
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

TRILLIM_KERNEL TrillimStatus trillim_neon_attention_mask(
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
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->heads * tiles, trillim_attention_task, &task);
}

TRILLIM_KERNEL const char *trillim_neon_attention_mask_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention, _broadcast_strides, _contiguous_strides, _storage_shape, _product, _DTYPE_IDS\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "import math\n"
           "physical_inputs = ('q', 'k', 'v', 'mask')\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, *, causal: bool = True, scale = None, window = None, softcap = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, softcap=softcap, dtype=dtype, mask=mask),)\n"
           "def bind(context):\n"
           "    if context.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('extended attention requires FP32 output to preserve final-cast semantics')\n"
           "    result = _attention(context)\n"
           "    softcap = context.attrs.get('softcap')\n"
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_attention_mask_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_attention_mask_capabilities(void) {
    /* NEON is mandatory in the AArch64 baseline used by Apple builds. */
#if defined(__APPLE__)
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &capabilities;
}
