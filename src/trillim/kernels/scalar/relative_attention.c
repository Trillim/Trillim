#include "common.h"
#include <math.h>

#define USE_NEON 0
#if USE_NEON
#include <arm_neon.h>
#endif

#define QUERY_BLOCK 4
#define KEY_BLOCK 32

#if USE_NEON
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

#endif

typedef struct {
    size_t length;
    size_t relative_length;
    size_t heads;
    size_t key_dim;
    size_t value_dim;
    size_t rank;
    const size_t *q_strides;
    const size_t *k_strides;
    const size_t *v_strides;
    const size_t *r_strides;
    size_t content_head_stride;
    size_t content_channel_stride;
    size_t position_head_stride;
    size_t position_channel_stride;
} RelativeParameters;

typedef struct {
    float *output;
    const float *query;
    const float *keys;
    const float *values;
    const float *relative;
    const float *content_bias;
    const float *position_bias;
    float *scratch;
    const RelativeParameters *parameters;
    const TrillimKernelContext *context;
} RelativeTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const RelativeParameters *p = parameters;
    for (size_t input = 0; input < 6; ++input) {
        inputs[input] = 0;
    }
    outputs[0] = 0;
    *scratch = 0;
    if (!p->length || !p->heads) {
        return TRILLIM_OK;
    }
    if (p->rank != 3 || !p->key_dim || !p->value_dim || p->length > SIZE_MAX / 2 ||
        p->relative_length < p->length * 2 - 1 || p->heads > SIZE_MAX / p->length ||
        p->heads * p->length > SIZE_MAX / sizeof(float) / p->value_dim) {
        return TRILLIM_ERROR;
    }
    size_t key_shape[] = {p->length, p->heads, p->key_dim};
    size_t value_shape[] = {p->length, p->heads, p->value_dim};
    size_t relative_shape[] = {p->relative_length, p->heads, p->key_dim};
    size_t bias_shape[] = {p->heads, p->key_dim};
    size_t content_strides[] = {p->content_head_stride, p->content_channel_stride};
    size_t position_strides[] = {p->position_head_stride, p->position_channel_stride};
    const size_t *shapes[] = {key_shape, key_shape, value_shape, relative_shape, bias_shape, bias_shape};
    const size_t *strides[] = {p->q_strides, p->k_strides, p->v_strides, p->r_strides, content_strides, position_strides};
    for (size_t input = 0; input < 6; ++input) {
        size_t span;
        if (!trillim_span(input < 4 ? 3 : 2, shapes[input], strides[input], &span) || span > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        inputs[input] = span * sizeof(float);
    }
    if (p->value_dim > SIZE_MAX - KEY_BLOCK || p->key_dim > (SIZE_MAX - KEY_BLOCK - p->value_dim) / 2) {
        return TRILLIM_ERROR;
    }
    size_t groups = p->length / QUERY_BLOCK + (p->length % QUERY_BLOCK != 0);
    size_t jobs = p->heads * groups;
    size_t slots = jobs < 128 ? jobs : 128;
    size_t row_size = 2 * p->key_dim + p->value_dim + KEY_BLOCK;
    if (row_size > SIZE_MAX / sizeof(float) / QUERY_BLOCK / slots) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->length * p->heads * p->value_dim * sizeof(float);
    *scratch = slots * QUERY_BLOCK * row_size * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus dot_pair(
    const RelativeTask *task, const float *content_query, const float *position_query,
    size_t key, size_t relative, float *score
) {
    const RelativeParameters *p = task->parameters;
    float content = 0, position = 0;
    size_t channel = 0;
#if USE_NEON
    if (p->k_strides[2] == 1 && p->r_strides[2] == 1) {
        float32x4_t content_sum = vdupq_n_f32(0), position_sum = vdupq_n_f32(0);
        for (; p->key_dim - channel >= 4; channel += 4) {
            if (p->key_dim > 256 && (channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            content_sum = vfmaq_f32(content_sum, vld1q_f32(content_query + channel), vld1q_f32(task->keys + key + channel));
            position_sum = vfmaq_f32(position_sum, vld1q_f32(position_query + channel), vld1q_f32(task->relative + relative + channel));
        }
        content = vaddvq_f32(content_sum);
        position = vaddvq_f32(position_sum);
    }
#endif
    for (; channel < p->key_dim; ++channel) {
        if (p->key_dim > 256 && (channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        content += content_query[channel] * task->keys[key + channel * p->k_strides[2]];
        position += position_query[channel] * task->relative[relative + channel * p->r_strides[2]];
    }
    *score = content + position;
    return TRILLIM_OK;
}

static TrillimStatus run_tile(const RelativeTask *task, size_t index, size_t slot) {
    const RelativeParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->length / QUERY_BLOCK + (p->length % QUERY_BLOCK != 0);
    size_t head = index / groups;
    size_t first = index % groups * QUERY_BLOCK;
    size_t rows = p->length - first < QUERY_BLOCK ? p->length - first : QUERY_BLOCK;
    size_t row_size = 2 * p->key_dim + p->value_dim + KEY_BLOCK;
    float *workspace = task->scratch + slot * QUERY_BLOCK * row_size;
    float maximum[QUERY_BLOCK], denominator[QUERY_BLOCK];
    float divisor = sqrtf((float)p->key_dim);
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum[row] = -INFINITY;
        denominator[row] = 0;
        float *content_query = workspace + row * row_size;
        float *position_query = content_query + p->key_dim;
        float *accumulator = position_query + p->key_dim;
        size_t query = (first + row) * p->q_strides[0] + head * p->q_strides[1];
        /* Bias additions round before the two independent dot products. */
        for (size_t channel = 0; channel < p->key_dim; ++channel) {
            if ((channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float value = task->query[query + channel * p->q_strides[2]];
            content_query[channel] = value + task->content_bias[head * p->content_head_stride + channel * p->content_channel_stride];
            position_query[channel] = value + task->position_bias[head * p->position_head_stride + channel * p->position_channel_stride];
        }
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if (p->value_dim > 256 && (channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            accumulator[channel] = 0;
        }
    }
    for (size_t block = 0; block < p->length; block += KEY_BLOCK) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->length - block < KEY_BLOCK ? p->length - block : KEY_BLOCK;
        for (size_t row = 0; row < rows; ++row) {
            float *content_query = workspace + row * row_size;
            float *position_query = content_query + p->key_dim;
            float *accumulator = position_query + p->key_dim;
            float *scores = accumulator + p->value_dim;
            float peak = -INFINITY;
            for (size_t key = 0; key < count; ++key) {
                size_t relative_row = p->length - 1 - (first + row) + block + key;
                TrillimStatus status = dot_pair(task, content_query, position_query,
                    (block + key) * p->k_strides[0] + head * p->k_strides[1],
                    relative_row * p->r_strides[0] + head * p->r_strides[1], scores + key);
                if (status != TRILLIM_OK) {
                    return status;
                }
                scores[key] /= divisor;
                peak = fmaxf(peak, scores[key]);
            }
            peak = fmaxf(peak, maximum[row]);
            float normalization_peak = peak == -INFINITY ? 0 : peak;
            float previous = maximum[row] == -INFINITY ? 0 : expf(maximum[row] - normalization_peak);
            maximum[row] = peak;
            float sum = 0;
            size_t key = 0;
#if USE_NEON
            float32x4_t vector_sum = vdupq_n_f32(0);
            for (; count - key >= 4; key += 4) {
                float32x4_t weights = exp_negative(vsubq_f32(vld1q_f32(scores + key), vdupq_n_f32(normalization_peak)));
                vst1q_f32(scores + key, weights);
                vector_sum = vaddq_f32(vector_sum, weights);
            }
            sum = vaddvq_f32(vector_sum);
#endif
            for (; key < count; ++key) {
                scores[key] = expf(scores[key] - normalization_peak);
                sum += scores[key];
            }
            denominator[row] = denominator[row] * previous + sum;
            for (size_t channel = 0; channel < p->value_dim; ++channel) {
                if (p->value_dim > 256 && (channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                accumulator[channel] *= previous;
            }
        }
        for (size_t key = 0; key < count; ++key) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t value = (block + key) * p->v_strides[0] + head * p->v_strides[1];
            for (size_t row = 0; row < rows; ++row) {
                float *accumulator = workspace + row * row_size + 2 * p->key_dim;
                float weight = accumulator[p->value_dim + key];
                size_t channel = 0;
#if USE_NEON
                if (p->v_strides[2] == 1) {
                    for (; p->value_dim - channel >= 4; channel += 4) {
                        if (p->value_dim > 256 && (channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                            return TRILLIM_CANCELLED;
                        }
                        vst1q_f32(accumulator + channel, vfmaq_n_f32(vld1q_f32(accumulator + channel), vld1q_f32(task->values + value + channel), weight));
                    }
                }
#endif
                for (; channel < p->value_dim; ++channel) {
                    if ((channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    accumulator[channel] += weight * task->values[value + channel * p->v_strides[2]];
                }
            }
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const float *accumulator = workspace + row * row_size + 2 * p->key_dim;
        float reciprocal = denominator[row] > 0 ? 1.0f / denominator[row] : NAN;
        float *output = task->output + ((first + row) * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            if (p->value_dim > 256 && (channel & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            output[channel] = accumulator[channel] * reciprocal;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus run_task(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const RelativeTask *task = opaque;
    const RelativeParameters *p = task->parameters;
    size_t groups = p->length / QUERY_BLOCK + (p->length % QUERY_BLOCK != 0);
    size_t jobs = p->heads * groups;
    size_t slots = jobs < 128 ? jobs : 128;
    for (size_t index = slot; index < jobs;) {
        TrillimStatus status = run_tile(task, index, slot);
        if (status != TRILLIM_OK || jobs - index <= slots) {
            return status;
        }
        index += slots;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_relative_attention(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const RelativeParameters *p = parameters;
    if (!p->length || !p->heads) {
        return TRILLIM_OK;
    }
    RelativeTask task = {outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], scratch, p, context};
    size_t groups = p->length / QUERY_BLOCK + (p->length % QUERY_BLOCK != 0);
    size_t jobs = p->heads * groups;
    size_t slots = jobs < 128 ? jobs : 128;
    return context->thread_pool->run(context->thread_pool->context, slots, run_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_relative_attention_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "physical_inputs = ('q', 'k', 'v', 'relative_keys', 'content_bias', 'position_bias')\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, relative_keys: Tensor, *, content_bias: Tensor, position_bias: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (RelativeAttention(q, k, v, relative_keys, content_bias=content_bias, position_bias=position_bias),)\n"
           "def bind(context):\n"
           "    if any(context.inputs[name].dtype != 'fp32' for name in physical_inputs) or context.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('relative attention requires FP32 tensors')\n"
           "    q, k, v, r, u, b = (context.inputs[name] for name in physical_inputs)\n"
           "    if len(q.shape) != 3 or k.shape != q.shape or len(v.shape) != 3 or v.shape[:2] != q.shape[:2] or len(r.shape) != 3 or r.shape[1:] != q.shape[1:] or u.shape != q.shape[1:] or b.shape != q.shape[1:]:\n"
           "        raise ValueError('relative attention shapes disagree')\n"
           "    us, bs = (tensor.strides or _contiguous_strides(tensor.shape) for tensor in (u, b))\n"
           "    result = dict(length=q.shape[0], relative_length=r.shape[0], heads=q.shape[1], key_dim=q.shape[2], value_dim=v.shape[2], rank=3, content_head_stride=us[0], content_channel_stride=us[1], position_head_stride=bs[0], position_channel_stride=bs[1])\n"
           "    for name, tensor in (('q', q), ('k', k), ('v', v), ('r', r)):\n"
           "        result[name + '_strides'] = tensor.strides or _contiguous_strides(tensor.shape)\n"
           "    return result\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_relative_attention_parameters(void) {
    static const TrillimParameter fields[] = {
        {"length", "size_t", offsetof(RelativeParameters, length), NULL},
        {"relative_length", "size_t", offsetof(RelativeParameters, relative_length), NULL},
        {"heads", "size_t", offsetof(RelativeParameters, heads), NULL},
        {"key_dim", "size_t", offsetof(RelativeParameters, key_dim), NULL},
        {"value_dim", "size_t", offsetof(RelativeParameters, value_dim), NULL},
        {"rank", "size_t", offsetof(RelativeParameters, rank), NULL},
        {"q_strides", "size_t", offsetof(RelativeParameters, q_strides), "rank"},
        {"k_strides", "size_t", offsetof(RelativeParameters, k_strides), "rank"},
        {"v_strides", "size_t", offsetof(RelativeParameters, v_strides), "rank"},
        {"r_strides", "size_t", offsetof(RelativeParameters, r_strides), "rank"},
        {"content_head_stride", "size_t", offsetof(RelativeParameters, content_head_stride), NULL},
        {"content_channel_stride", "size_t", offsetof(RelativeParameters, content_channel_stride), NULL},
        {"position_head_stride", "size_t", offsetof(RelativeParameters, position_head_stride), NULL},
        {"position_channel_stride", "size_t", offsetof(RelativeParameters, position_channel_stride), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(RelativeParameters), TRILLIM_ALIGNOF(RelativeParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 6, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_relative_attention_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
