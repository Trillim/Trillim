#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Older Apple assemblers discard .arch when C is emitted as assembly text. */
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_asm_v\\reg,\\reg\n"
    ".endr\n"
);
#define BF16
static inline float32x4_t dot_bf16(float32x4_t sum, uint16x8_t a, uint16x8_t b) {
    __asm__ volatile(".inst 0x6e40fc00 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
/* Function targets disappear before the builder's separate assembly step. */
__asm__(".arch armv8.6-a+bf16");
#if defined(__clang__)
#define BF16 __attribute__((target("bf16")))
#else
#define BF16 __attribute__((target("+bf16")))
#endif
BF16 static inline float32x4_t dot_bf16(float32x4_t sum, uint16x8_t a, uint16x8_t b) {
    return vbfdotq_f32(sum, vreinterpretq_bf16_u16(a), vreinterpretq_bf16_u16(b));
}
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#include <string.h>

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

static inline float trillim_read_float(
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
} TrillimAttentionParameters;

typedef struct {
    void *output;
    const void *query;
    const void *keys;
    const void *values;
    float *scratch;
    const TrillimAttentionParameters *parameters;
    const TrillimKernelContext *context;
    bool use_bf16;
    atomic_bool *ordinary;
} TrillimAttentionTask;

static inline bool trillim_attention_span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t span;
    return (columns == 0 || rows <= SIZE_MAX / columns) &&
        trillim_span(3, shape, strides, &span) && span <= rows * columns;
}

static size_t attention_partitions(const TrillimAttentionParameters *p) {
    size_t jobs = p->heads * p->queries;
    if (!jobs) {
        return 1;
    }
    size_t target = 128 / jobs + (128 % jobs != 0);
    size_t available = p->length / 256;
    if (!available) {
        available = 1;
    }
    if (target > available) {
        target = available;
    }
    return target < 16 ? target : 16;
}

static bool optimized_layout(const TrillimAttentionParameters *p) {
    return p->output_dtype == TRILLIM_FP32 && trillim_float(p->q_dtype) && trillim_float(p->k_dtype) &&
        trillim_float(p->v_dtype) && p->q_strides[2] == 1 && p->k_strides[2] == 1 && p->v_strides[2] == 1 &&
        p->key_dim <= 4096 && p->value_dim <= 4096;
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
        !trillim_payload_size(p->q_rows, p->q_columns, p->q_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->length != 0 && (!trillim_attention_span(key_shape, p->k_strides, p->k_rows, p->k_columns) ||
        !trillim_attention_span(value_shape, p->v_strides, p->v_rows, p->v_columns) ||
        !trillim_payload_size(p->k_rows, p->k_columns, p->k_dtype, &inputs[1]) ||
        !trillim_payload_size(p->v_rows, p->v_columns, p->v_dtype, &inputs[2]))) {
        return TRILLIM_ERROR;
    }
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    if (p->value_dim > SIZE_MAX - 64 || p->heads > SIZE_MAX / tiles ||
        p->heads * tiles > SIZE_MAX / sizeof(float) / (p->value_dim + 64)) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->queries * p->value_dim * trillim_itemsize(p->output_dtype);
    *scratch = p->heads * tiles * (p->value_dim + 64) * sizeof(float);
    if (p->output_dtype == TRILLIM_FP32 && trillim_float(p->q_dtype) && trillim_float(p->k_dtype) && trillim_float(p->v_dtype)) {
        size_t key_dim = p->key_dim < 4096 ? p->key_dim : 4096;
        size_t value_dim = p->value_dim < 4096 ? p->value_dim : 4096;
        size_t tile_groups = p->queries / 4 + (p->queries % 4 != 0);
        size_t tile_slots = p->heads * tile_groups < 128 ? p->heads * tile_groups : 128;
        size_t tile_bytes = tile_slots * 4 * (key_dim + value_dim + 64) * sizeof(float);
        if (tile_bytes > *scratch) {
            *scratch = tile_bytes;
        }
        if (p->length >= 512) {
            /* Reserve a monotonic bound for every smaller decode shape. */
            size_t rows = p->queries < 4 ? p->queries : 4;
            size_t jobs = p->heads * rows;
            size_t slots = jobs < 16 ? jobs * 16 : 256;
            if (jobs > slots) {
                slots = jobs;
            }
            if (slots > SIZE_MAX / sizeof(float) / (value_dim + 66)) {
                return TRILLIM_ERROR;
            }
            size_t bytes = slots * (value_dim + 66) * sizeof(float);
            if (bytes > *scratch) {
                *scratch = bytes;
            }
        }
    }
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

static bool available_bf16(void) {
    static atomic_int cached = ATOMIC_VAR_INIT(0);
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_BF16", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP2_BF16)
    available = (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

static TrillimStatus check_bf16(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!atomic_load_explicit(task->ordinary, memory_order_relaxed)) {
        return TRILLIM_OK;
    }
    size_t queries = p->queries * p->heads;
    size_t total = queries + p->length * p->kv_heads;
    size_t first = index * 128;
    size_t last = total - first < 128 ? total : first + 128;
    uint16x8_t minimum = vdupq_n_u16(UINT16_MAX), maximum = vdupq_n_u16(0);
    for (size_t row = first; row < last; ++row) {
        size_t base;
        const uint16_t *data;
        if (row < queries) {
            base = row / p->heads * p->q_strides[1] + row % p->heads * p->q_strides[0];
            data = task->query;
        } else {
            size_t key = row - queries;
            base = key / p->kv_heads * p->k_strides[1] + key % p->kv_heads * p->k_strides[0];
            data = task->keys;
        }
        size_t channel = 0;
        for (; p->key_dim - channel >= 8; channel += 8) {
            if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint16x8_t bits = vandq_u16(vld1q_u16(data + base + channel), vdupq_n_u16(0x7fff));
            maximum = vmaxq_u16(maximum, bits);
            bits = vbslq_u16(vceqq_u16(bits, vdupq_n_u16(0)), vdupq_n_u16(UINT16_MAX), bits);
            minimum = vminq_u16(minimum, bits);
        }
        for (; channel < p->key_dim; ++channel) {
            uint16_t bits = data[base + channel] & 0x7fff;
            if (bits && (bits < 0x2f80 || bits >= 0x4f80)) {
                atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
    }
    /* Zeros and [2^-32, 2^32) keep BFDOT products and sums normal and finite. */
    if (vminvq_u16(minimum) < 0x2f80 || vmaxvq_u16(maximum) >= 0x4f80) {
        atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
    }
    return TRILLIM_OK;
}

BF16 static TrillimStatus score_bf16(
    const TrillimAttentionTask *task, size_t query, size_t key, float *score
) {
    const TrillimAttentionParameters *p = task->parameters;
    const uint16_t *q = (const uint16_t *)task->query + query;
    const uint16_t *k = (const uint16_t *)task->keys + key;
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    size_t channel = 0;
    for (; p->key_dim - channel >= 32; channel += 32) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        a = dot_bf16(a, vld1q_u16(q + channel), vld1q_u16(k + channel));
        b = dot_bf16(b, vld1q_u16(q + channel + 8), vld1q_u16(k + channel + 8));
        c = dot_bf16(c, vld1q_u16(q + channel + 16), vld1q_u16(k + channel + 16));
        d = dot_bf16(d, vld1q_u16(q + channel + 24), vld1q_u16(k + channel + 24));
    }
    for (; p->key_dim - channel >= 8; channel += 8) {
        a = dot_bf16(a, vld1q_u16(q + channel), vld1q_u16(k + channel));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; channel < p->key_dim; ++channel) {
        sum += trillim_bf16(q[channel]) * trillim_bf16(k[channel]);
    }
    *score = sum;
    return TRILLIM_OK;
}

BF16 static TrillimStatus score_bf16_rows(
    const TrillimAttentionTask *task, size_t head, size_t first, size_t rows, size_t key, float *scores
) {
    const TrillimAttentionParameters *p = task->parameters;
    const uint16_t *query = task->query;
    const uint16_t *keys = (const uint16_t *)task->keys + key;
    const uint16_t *q0 = query + head * p->q_strides[0] + first * p->q_strides[1];
    const uint16_t *q1 = q0 + (rows > 1 ? 1 * p->q_strides[1] : 0);
    const uint16_t *q2 = q0 + (rows > 2 ? 2 * p->q_strides[1] : 0);
    const uint16_t *q3 = q0 + (rows > 3 ? 3 * p->q_strides[1] : 0);
    float32x4_t a0 = vdupq_n_f32(0), b0 = a0;
    float32x4_t a1 = vdupq_n_f32(0), b1 = a1;
    float32x4_t a2 = vdupq_n_f32(0), b2 = a2;
    float32x4_t a3 = vdupq_n_f32(0), b3 = a3;
    size_t channel = 0;
    for (; p->key_dim - channel >= 16; channel += 16) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t k0 = vld1q_u16(keys + channel), k1 = vld1q_u16(keys + channel + 8);
        a0 = dot_bf16(a0, vld1q_u16(q0 + channel), k0);
        b0 = dot_bf16(b0, vld1q_u16(q0 + channel + 8), k1);
        a1 = dot_bf16(a1, vld1q_u16(q1 + channel), k0);
        b1 = dot_bf16(b1, vld1q_u16(q1 + channel + 8), k1);
        a2 = dot_bf16(a2, vld1q_u16(q2 + channel), k0);
        b2 = dot_bf16(b2, vld1q_u16(q2 + channel + 8), k1);
        a3 = dot_bf16(a3, vld1q_u16(q3 + channel), k0);
        b3 = dot_bf16(b3, vld1q_u16(q3 + channel + 8), k1);
    }
    for (; p->key_dim - channel >= 8; channel += 8) {
        uint16x8_t k0 = vld1q_u16(keys + channel);
        a0 = dot_bf16(a0, vld1q_u16(q0 + channel), k0);
        a1 = dot_bf16(a1, vld1q_u16(q1 + channel), k0);
        a2 = dot_bf16(a2, vld1q_u16(q2 + channel), k0);
        a3 = dot_bf16(a3, vld1q_u16(q3 + channel), k0);
    }
    float sums[] = {
        vaddvq_f32(vaddq_f32(a0, b0)), vaddvq_f32(vaddq_f32(a1, b1)), vaddvq_f32(vaddq_f32(a2, b2)), vaddvq_f32(vaddq_f32(a3, b3))
    };
    for (; channel < p->key_dim; ++channel) {
        float k = trillim_bf16(keys[channel]);
        size_t base = head * p->q_strides[0] + first * p->q_strides[1] + channel;
        for (size_t row = 0; row < rows; ++row) {
            sums[row] += trillim_bf16(query[base + row * p->q_strides[1]]) * k;
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        scores[row] = sums[row];
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_score(
    const TrillimAttentionTask *task, size_t query, size_t key, float *score
) {
    const TrillimAttentionParameters *p = task->parameters;
    if (task->use_bf16) {
        return score_bf16(task, query, key, score);
    }
    /* Low precision outputs must preserve the reference's rounding boundaries. */
    if (p->output_dtype != TRILLIM_FP32) {
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
        for (size_t block = begin; block < end; block += 64) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = end - block < 64 ? end - block : 64;
            float block_maximum = -INFINITY;
            for (size_t j = 0; j < count; ++j) {
                size_t key = kv_head * p->k_strides[0] + (block + j) * p->k_strides[1];
                if (p->output_dtype != TRILLIM_FP32 && count - (j / 4) * 4 >= 4) {
                    if (j % 4 == 0) {
                        TrillimStatus status = trillim_attention_scores4(task, query, key, scores + j);
                        if (status != TRILLIM_OK) {
                            return status;
                        }
                    }
                } else {
                    TrillimStatus status = trillim_attention_score(task, query, key, &scores[j]);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                scores[j] *= scale;
                block_maximum = fmaxf(block_maximum, scores[j]);
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

static inline float32x4_t tiled_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static TrillimStatus trillim_attention_tile(const TrillimAttentionTask *task, size_t index, size_t slot) {
    const TrillimAttentionParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->queries / 4 + (p->queries % 4 != 0);
    size_t head = index / groups;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t first = index % groups * 4;
    size_t rows = p->queries - first < 4 ? p->queries - first : 4;
    float *queries = task->scratch + slot * 4 * (p->key_dim + p->value_dim + 64);
    float *accumulator = queries + 4 * p->key_dim;
    float *scores = accumulator + 4 * p->value_dim;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float denominator[4] = {0, 0, 0, 0};
    size_t begin[4], end[4];
    size_t first_key = p->length, last_key = 0;
    for (size_t row = 0; row < rows; ++row) {
        size_t query = head * p->q_strides[0] + (first + row) * p->q_strides[1];
        size_t channel = 0;
        for (; p->key_dim - channel >= 4; channel += 4) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            vst1q_f32(queries + row * p->key_dim + channel, tiled_load4(task->query, query + channel, p->q_dtype));
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
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)(first + row);
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
            if (task->use_bf16) {
                float sums[4];
                size_t base = kv_head * p->k_strides[0] + (block + key) * p->k_strides[1];
                TrillimStatus status = score_bf16_rows(task, head, first, rows, base, sums);
                if (status != TRILLIM_OK) {
                    return status;
                }
                for (size_t row = 0; row < rows; ++row) {
                    scores[row * 64 + key] = block + key >= begin[row] && block + key < end[row] ? sums[row] * scale : -INFINITY;
                }
                continue;
            }
            float32x4_t sums[4][2];
            for (size_t row = 0; row < rows; ++row) {
                sums[row][0] = sums[row][1] = vdupq_n_f32(0);
            }
            size_t base = kv_head * p->k_strides[0] + (block + key) * p->k_strides[1];
            size_t channel = 0;
            for (; p->key_dim - channel >= 8; channel += 8) {
                if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float32x4_t k0 = tiled_load4(task->keys, base + channel, p->k_dtype);
                float32x4_t k1 = tiled_load4(task->keys, base + channel + 4, p->k_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    const float *q = queries + row * p->key_dim + channel;
                    sums[row][0] = vfmaq_f32(sums[row][0], vld1q_f32(q), k0);
                    sums[row][1] = vfmaq_f32(sums[row][1], vld1q_f32(q + 4), k1);
                }
            }
            for (; p->key_dim - channel >= 4; channel += 4) {
                float32x4_t k0 = tiled_load4(task->keys, base + channel, p->k_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    sums[row][0] = vfmaq_f32(sums[row][0], vld1q_f32(queries + row * p->key_dim + channel), k0);
                }
            }
            for (size_t row = 0; row < rows; ++row) {
                float sum = vaddvq_f32(vaddq_f32(sums[row][0], sums[row][1]));
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
            float32x4_t total = vdupq_n_f32(0);
            size_t key = 0;
            for (; count - key >= 4; key += 4) {
                float *target = scores + row * 64 + key;
                float32x4_t value = exp_negative(vsubq_f32(vld1q_f32(target), vdupq_n_f32(normalization_peak)));
                vst1q_f32(target, value);
                total = vaddq_f32(total, value);
            }
            float sum = vaddvq_f32(total);
            for (; key < count; ++key) {
                float value = expf(scores[row * 64 + key] - normalization_peak);
                scores[row * 64 + key] = value;
                sum += value;
            }
            denominator[row] = denominator[row] * previous + sum;
            size_t channel = 0;
            for (; p->value_dim - channel >= 4; channel += 4) {
                if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float *output = accumulator + row * p->value_dim + channel;
                vst1q_f32(output, vmulq_n_f32(vld1q_f32(output), previous));
            }
            for (; channel < p->value_dim; ++channel) {
                accumulator[row * p->value_dim + channel] *= previous;
            }
        }
        size_t channel = 0;
        for (; p->value_dim - channel >= 8; channel += 8) {
            if ((channel & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t sums[4][2];
            for (size_t row = 0; row < rows; ++row) {
                const float *output = accumulator + row * p->value_dim + channel;
                sums[row][0] = vld1q_f32(output);
                sums[row][1] = vld1q_f32(output + 4);
            }
            for (size_t key = 0; key < count; ++key) {
                size_t base = kv_head * p->v_strides[0] + (block + key) * p->v_strides[1] + channel;
                float32x4_t v0 = tiled_load4(task->values, base, p->v_dtype);
                float32x4_t v1 = tiled_load4(task->values, base + 4, p->v_dtype);
                for (size_t row = 0; row < rows; ++row) {
                    if (block + key >= begin[row] && block + key < end[row]) {
                        float probability = scores[row * 64 + key];
                        sums[row][0] = vfmaq_n_f32(sums[row][0], v0, probability);
                        sums[row][1] = vfmaq_n_f32(sums[row][1], v1, probability);
                    }
                }
            }
            for (size_t row = 0; row < rows; ++row) {
                float *output = accumulator + row * p->value_dim + channel;
                vst1q_f32(output, sums[row][0]);
                vst1q_f32(output + 4, sums[row][1]);
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
        size_t output = ((first + row) * p->heads + head) * p->value_dim;
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

static TrillimStatus trillim_attention_tiled_task(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = p->queries / 4 + (p->queries % 4 != 0);
    size_t tiles = p->heads * groups;
    size_t slots = tiles < 128 ? tiles : 128;
    for (size_t index = slot; index < tiles;) {
        TrillimStatus status = trillim_attention_tile(task, index, slot);
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

static inline TrillimStatus trillim_attention_decode_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = (const TrillimAttentionTask *)opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t parts = attention_partitions(p);
    size_t part = index % parts;
    size_t head = index / parts / p->queries;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t first = index / parts % p->queries;
    size_t last = first + 1;
    size_t blocks = p->length / 64 + (p->length % 64 != 0);
    size_t first_block = blocks / parts * part + (part < blocks % parts ? part : blocks % parts);
    size_t last_block = first_block + blocks / parts + (part < blocks % parts);
    size_t partition_begin = first_block * 64;
    size_t partition_end = last_block * 64 < p->length ? last_block * 64 : p->length;
    float *scores = task->scratch + index * (p->value_dim + 66);
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
        if (begin < partition_begin) {
            begin = partition_begin;
        }
        if (end > partition_end) {
            end = partition_end;
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
        accumulator[p->value_dim] = maximum;
        accumulator[p->value_dim + 1] = denominator;
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_attention_merge(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t parts = attention_partitions(p);
    size_t head = index / p->queries;
    size_t query = index % p->queries;
    const float *partial = task->scratch + index * parts * (p->value_dim + 66) + 64;
    float maximum = -INFINITY;
    for (size_t part = 0; part < parts; ++part) {
        maximum = fmaxf(maximum, partial[part * (p->value_dim + 66) + p->value_dim]);
    }
    float weights[16];
    float denominator = 0;
    for (size_t part = 0; part < parts; ++part) {
        const float *source = partial + part * (p->value_dim + 66);
        weights[part] = source[p->value_dim] == -INFINITY ? 0 : expf(source[p->value_dim] - maximum);
        denominator += source[p->value_dim + 1] * weights[part];
    }
    float reciprocal = 1.0f / (denominator > 0 ? denominator : 1.0f);
    float *output = (float *)task->output + (query * p->heads + head) * p->value_dim;
    size_t channel = 0;
    for (; p->value_dim - channel >= 4; channel += 4) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t sum = vdupq_n_f32(0);
        for (size_t part = 0; part < parts; ++part) {
            sum = vfmaq_n_f32(sum, vld1q_f32(partial + part * (p->value_dim + 66) + channel), weights[part]);
        }
        vst1q_f32(output + channel, vmulq_n_f32(sum, reciprocal));
    }
    for (; channel < p->value_dim; ++channel) {
        float sum = 0;
        for (size_t part = 0; part < parts; ++part) {
            sum += partial[part * (p->value_dim + 66) + channel] * weights[part];
        }
        output[channel] = sum * reciprocal;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_attention(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimAttentionParameters *p = (const TrillimAttentionParameters *)parameters;
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    atomic_bool ordinary = ATOMIC_VAR_INIT(true);
    TrillimAttentionTask task = {
        outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context, false, &ordinary
    };
    bool optimized = optimized_layout(p);
    size_t visible = p->windowed && p->window < p->length ? p->window : p->length;
    bool checked_rows = !p->length || p->kv_heads <= (SIZE_MAX - p->heads * p->queries) / p->length;
    if (optimized && visible != 0 && p->q_dtype == TRILLIM_BF16 && p->k_dtype == TRILLIM_BF16 && checked_rows &&
        (p->queries >= 16 || (visible >= 256 && (!p->windowed || p->window >= p->length / 4))) && available_bf16()) {
        size_t rows = p->queries * p->heads + p->length * p->kv_heads;
        size_t groups = rows / 128 + (rows % 128 != 0);
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, groups, check_bf16, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        task.use_bf16 = atomic_load_explicit(&ordinary, memory_order_relaxed);
    }
    if (optimized && p->queries <= 4 && visible >= 1024 && context->thread_pool->worker_count > p->heads) {
        size_t jobs = p->heads * p->queries;
        TrillimStatus status = context->thread_pool->run(
            context->thread_pool->context, jobs * attention_partitions(p), trillim_attention_decode_task, &task
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        return context->thread_pool->run(context->thread_pool->context, jobs, trillim_attention_merge, &task);
    }
    if (optimized && visible != 0 && (p->queries >= 16 || !task.use_bf16)) {
        size_t groups = p->queries / 4 + (p->queries % 4 != 0);
        size_t slots = p->heads * groups < 128 ? p->heads * groups : 128;
        return context->thread_pool->run(context->thread_pool->context, slots, trillim_attention_tiled_task, &task);
    }
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->heads * tiles, trillim_attention_task, &task);
}

TRILLIM_KERNEL const char *trillim_neon_attention_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention as bind\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_attention_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_attention_capabilities(void) {
    /* NEON is mandatory in the AArch64 baseline used by Apple builds. */
#if defined(__APPLE__)
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &capabilities;
}
