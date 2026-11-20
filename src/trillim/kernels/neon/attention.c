#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>

typedef struct {
    const uint8_t *data;
    uint32_t dtype;
    unsigned group_shift;
    size_t first_element, first_block, blocks;
    float row_scale;
} TrillimPackedNeonRow;

static inline TrillimPackedNeonRow trillim_packed_neon_row(
    const void *weight, uint32_t dtype, size_t column, size_t width, size_t outputs
) {
    TrillimPackedNeonRow row = {.data = weight, .dtype = dtype, .first_element = column * width};
    if (width == 0) {
        return row;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        memcpy(&bits, row.data + outputs * width + column * sizeof(bits), sizeof(bits));
        row.row_scale = trillim_bf16(bits);
    } else if (dtype >= TRILLIM_MAT_TERNARY && dtype <= TRILLIM_MXFP4) {
        row.group_shift = 5;
        if (dtype == TRILLIM_DUBGRP_INT6) {
            row.group_shift = 8;
        } else if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
            row.group_shift = 7;
        }
        size_t group = (size_t)1 << row.group_shift;
        size_t groups = (width >> row.group_shift) + (width % group != 0);
        row.first_block = column * groups;
        row.blocks = outputs * groups;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&row.row_scale, row.data + row.blocks * 32, sizeof(row.row_scale));
        }
    }
    return row;
}

static inline float32x4_t trillim_neon_load4(const void *data, size_t index, uint32_t dtype) {
    const uint8_t *bytes = data;
    if (dtype == TRILLIM_FP32) {
        float32x4_t values;
        memcpy(&values, bytes + index * sizeof(float), sizeof(values));
        return values;
    }
    uint16x4_t bits;
    memcpy(&bits, bytes + index * sizeof(uint16_t), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(bits));
}

static inline float32x4_t trillim_packed_neon_convert4(int16x4_t codes, float scale) {
    return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(codes)), scale);
}

/* k is sixteen-aligned and the read stays within the logical row. */
static inline __attribute__((always_inline)) void trillim_packed_neon_decode16(
    const TrillimPackedNeonRow *row, size_t k, float32x4_t result[4]
) {
    const uint8_t *values = row->data;
    uint32_t dtype = row->dtype;
    size_t total = row->blocks;
    size_t block = row->first_block + (k >> row->group_shift);
    size_t offset = k & (((size_t)1 << row->group_shift) - 1);
    int8x16_t codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = row->row_scale;
        codes = vld1q_s8((const int8_t *)values + row->first_element + k);
    } else if (dtype == TRILLIM_MXFP4) {
        uint8x8_t packed = vld1_u8(values + block * 16 + offset / 2);
        uint8x8x2_t split = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
        uint8x16_t code = vcombine_u8(split.val[0], split.val[1]);
        /* Exact E2M1 values assembled from their two BF16 bytes. */
        static const uint8_t low_bits[16] = {0, 0, 128, 192, 0, 64, 128, 192, 0, 0, 128, 192, 0, 64, 128, 192};
        static const uint8_t high_bits[16] = {0, 63, 63, 63, 64, 64, 64, 64, 128, 191, 191, 191, 192, 192, 192, 192};
        uint8x16_t low = vqtbl1q_u8(vld1q_u8(low_bits), code);
        uint8x16_t high = vqtbl1q_u8(vld1q_u8(high_bits), code);
        uint16x8_t a = vreinterpretq_u16_u8(vzip1q_u8(low, high));
        uint16x8_t b = vreinterpretq_u16_u8(vzip2q_u8(low, high));
        result[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16));
        result[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a), 16));
        result[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
        result[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16));
        scale = trillim_mxfp4_scale(values[total * 16 + block]);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vmulq_n_f32(result[i], scale);
        }
        return;
    } else if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        uint8x16_t packed = vld1q_u8(values + block * 32 + offset % 32);
        uint8x16_t shifted = vshlq_u8(packed, vdupq_n_s8(-(int8_t)(6 - (offset / 32) * 2)));
        codes = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(shifted, vdupq_n_u8(3))), vdupq_n_s8(1));
        if (dtype == TRILLIM_MAT_TERNARY) {
            scale = row->row_scale;
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
    } else {
        size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
        size_t scale_start = total * value_width;
        uint16_t bits;
        memcpy(&bits, values + scale_start + block * 2, 2);
        scale = trillim_bf16(bits);
        if (dtype == TRILLIM_GRP_INT8) {
            codes = vld1q_s8((const int8_t *)values + block * 32 + offset);
        } else if (dtype == TRILLIM_GRP_BINARY) {
            const uint8_t *packed = values + block * 16 + offset / 8;
            uint8x16_t bytes = vcombine_u8(vdup_n_u8(packed[0]), vdup_n_u8(packed[1]));
            static const int8_t shifts[16] = {-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0};
            uint8x16_t code = vandq_u8(vshlq_u8(bytes, vld1q_s8(shifts)), vdupq_n_u8(1));
            codes = vsubq_s8(vreinterpretq_s8_u8(vshlq_n_u8(code, 1)), vdupq_n_s8(1));
        } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
            uint8x16_t packed = vld1q_u8(values + block * 16);
            uint8x16_t code = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(int8_t)((offset / 16) * 4))), vdupq_n_u8(15));
            if (dtype == TRILLIM_GRP_INT5) {
                const uint8_t *high = values + scale_start + total * 2 + block * 4 + offset / 8;
                uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
                static const int8_t shifts[16] = {0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7};
                uint8x16_t upper = vshlq_n_u8(vandq_u8(vshlq_u8(bytes, vld1q_s8(shifts)), vdupq_n_u8(1)), 4);
                code = vorrq_u8(code, upper);
            }
            codes = vsubq_s8(vreinterpretq_s8_u8(code), vdupq_n_s8(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
        } else {
            size_t half = offset / 128;
            size_t lane = (offset % 128) / 32;
            uint8x16_t packed = vld1q_u8(values + block * 128 + half * 64 + (lane / 2) * 32 + offset % 32);
            uint8x16_t high = vld1q_u8(values + scale_start + total * 2 + block * 64 + half * 32 + offset % 32);
            uint8x16_t low = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(int8_t)((lane % 2) * 4))), vdupq_n_u8(15));
            uint8x16_t upper = vshlq_n_u8(vandq_u8(vshlq_u8(high, vdupq_n_s8(-(int8_t)(lane * 2))), vdupq_n_u8(3)), 4);
            codes = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, upper)), vdupq_n_s8(32));
            int8_t subscale;
            memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
            int16x8_t a = vmulq_n_s16(vmovl_s8(vget_low_s8(codes)), subscale);
            int16x8_t b = vmulq_n_s16(vmovl_s8(vget_high_s8(codes)), subscale);
            result[0] = trillim_packed_neon_convert4(vget_low_s16(a), scale);
            result[1] = trillim_packed_neon_convert4(vget_high_s16(a), scale);
            result[2] = trillim_packed_neon_convert4(vget_low_s16(b), scale);
            result[3] = trillim_packed_neon_convert4(vget_high_s16(b), scale);
            return;
        }
    }
    int16x8_t low = vmovl_s8(vget_low_s8(codes)), high = vmovl_s8(vget_high_s8(codes));
    result[0] = trillim_packed_neon_convert4(vget_low_s16(low), scale);
    result[1] = trillim_packed_neon_convert4(vget_high_s16(low), scale);
    result[2] = trillim_packed_neon_convert4(vget_low_s16(high), scale);
    result[3] = trillim_packed_neon_convert4(vget_high_s16(high), scale);
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        int32x4_t code[4] = {vmovl_s16(vget_low_s16(low)), vmovl_s16(vget_high_s16(low)),
            vmovl_s16(vget_low_s16(high)), vmovl_s16(vget_high_s16(high))};
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vbslq_f32(vceqq_s32(code[i], vdupq_n_s32(2)), vdupq_n_f32(NAN), result[i]);
        }
    }
}

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
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#define BLAS_MAX_QUERIES 128

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
    size_t blas_queries;
    size_t blas_keys;
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

static size_t legacy_query_tile(const TrillimAttentionParameters *p, size_t workers) {
    size_t tiles = p->queries / 16 + (p->queries % 16 != 0);
    if (p->output_dtype == TRILLIM_BF16 && p->q_dtype == TRILLIM_BF16 &&
        p->k_dtype == TRILLIM_BF16 && p->v_dtype == TRILLIM_BF16 &&
        p->heads <= 4 && p->queries <= 64 && p->heads * tiles < workers) {
        return 4;
    }
    return 16;
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
    if (p->output_dtype == TRILLIM_BF16 && p->q_dtype == TRILLIM_BF16 &&
        p->k_dtype == TRILLIM_BF16 && p->v_dtype == TRILLIM_BF16) {
        size_t queries = p->queries < 64 ? p->queries : 64;
        size_t short_tiles = queries / 4 + (queries % 4 != 0);
        if (tiles < short_tiles) {
            tiles = short_tiles;
        }
    }
    if (p->value_dim > SIZE_MAX - 64 || p->heads > SIZE_MAX / tiles ||
        p->heads * tiles > SIZE_MAX / sizeof(float) / (p->value_dim + 64)) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->queries * p->value_dim * trillim_itemsize(p->output_dtype);
    *scratch = p->heads * tiles * (p->value_dim + 64) * sizeof(float);
    size_t fallback_bytes = *scratch;
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
        if (p->q_dtype == TRILLIM_FP32 && p->k_dtype == TRILLIM_FP32 && p->v_dtype == TRILLIM_FP32 &&
            p->queries >= 32 && p->length >= 64) {
            /* Library availability and smaller execution shapes cannot reduce this envelope. */
            static const size_t blas_tiles[][2] = {{64, 128}, {64, 256}, {128, 256}};
            size_t blas_value_dim = p->value_dim < 256 ? p->value_dim : 256;
            for (size_t tile = 0; tile < sizeof(blas_tiles) / sizeof(blas_tiles[0]); ++tile) {
                size_t queries = blas_tiles[tile][0], keys = blas_tiles[tile][1];
                size_t groups = p->queries / queries + (p->queries % queries != 0);
                size_t slots = p->heads * groups < 128 ? p->heads * groups : 128;
                size_t bytes = slots * queries * (blas_value_dim + keys) * sizeof(float);
                if (bytes > SIZE_MAX - fallback_bytes) {
                    return TRILLIM_ERROR;
                }
                bytes += fallback_bytes;
                if (bytes > *scratch) {
                    *scratch = bytes;
                }
            }
        }
    }
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

static inline float32x4_t attention_load4(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (trillim_float(dtype)) {
        return trillim_neon_load4(data, index, dtype);
    }
    float lanes[4];
    for (size_t i = 0; i < 4; ++i) {
        lanes[i] = trillim_packed_read_float(data, index + i, dtype, rows, columns);
    }
    return vld1q_f32(lanes);
}

static TrillimStatus trillim_attention_pack(void *opaque, size_t head, size_t worker) {
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

static TrillimStatus trillim_attention_block(void *opaque, size_t tile, size_t slot) {
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
    float32x4_t scale = vdupq_n_f32(p->default_scale ? 0.125f : p->scale);
    for (size_t key = 0; key < width; key += 4) {
        if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
        for (size_t channel = 0; channel < 64; ++channel) {
            float32x4_t k = vld1q_f32(keys + channel * width + key);
            a = vfmaq_n_f32(a, k, q[0][channel]);
            b = vfmaq_n_f32(b, k, q[1][channel]);
            c = vfmaq_n_f32(c, k, q[2][channel]);
            d = vfmaq_n_f32(d, k, q[3][channel]);
        }
        vst1q_f32(scores + key, vmulq_f32(a, scale));
        vst1q_f32(scores + width + key, vmulq_f32(b, scale));
        vst1q_f32(scores + width * 2 + key, vmulq_f32(c, scale));
        vst1q_f32(scores + width * 3 + key, vmulq_f32(d, scale));
    }
    for (size_t row = 0; row < rows; ++row) {
        float *probability = scores + row * width;
        for (size_t key = p->length; key < width; ++key) {
            probability[key] = -INFINITY;
        }
        float32x4_t peak = vdupq_n_f32(-INFINITY);
        for (size_t key = 0; key < width; key += 4) {
            peak = vmaxq_f32(peak, vld1q_f32(probability + key));
        }
        float maximum = vmaxvq_f32(peak);
        float32x4_t total = vdupq_n_f32(0);
        for (size_t key = 0; key < width; key += 4) {
            float32x4_t value = exp_negative(vsubq_f32(vld1q_f32(probability + key), vdupq_n_f32(maximum)));
            vst1q_f32(probability + key, value);
            total = vaddq_f32(total, value);
        }
        float inverse = 1.0f / vaddvq_f32(total);
        for (size_t key = 0; key < width; key += 4) {
            vst1q_f32(probability + key, vmulq_f32(vld1q_f32(probability + key), vdupq_n_f32(inverse)));
        }
    }
    for (size_t channel = 0; channel < 64; channel += 8) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0;
        float32x4_t c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        for (size_t key = 0; key < p->length; ++key) {
            if ((key & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t v0 = vld1q_f32(values + key * 64 + channel);
            float32x4_t v1 = vld1q_f32(values + key * 64 + channel + 4);
            float32x4_t p0 = vdupq_n_f32(scores[key]);
            float32x4_t p1 = vdupq_n_f32(scores[width + key]);
            float32x4_t p2 = vdupq_n_f32(scores[width * 2 + key]);
            float32x4_t p3 = vdupq_n_f32(scores[width * 3 + key]);
            a0 = vfmaq_f32(a0, v0, p0);
            a1 = vfmaq_f32(a1, v1, p0);
            b0 = vfmaq_f32(b0, v0, p1);
            b1 = vfmaq_f32(b1, v1, p1);
            c0 = vfmaq_f32(c0, v0, p2);
            c1 = vfmaq_f32(c1, v1, p2);
            d0 = vfmaq_f32(d0, v0, p3);
            d1 = vfmaq_f32(d1, v1, p3);
        }
        float32x4_t result[4][2] = {{a0, a1}, {b0, b1}, {c0, c1}, {d0, d1}};
        for (size_t row = 0; row < rows; ++row) {
            float *output = (float *)task->output + ((first + row) * p->heads + head) * 64 + channel;
            vst1q_f32(output, result[row][0]);
            vst1q_f32(output + 4, result[row][1]);
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

static inline void attention_load16(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns, float32x4_t values[4]
) {
    if (dtype >= TRILLIM_ROW_INT8 && columns != 0) {
        size_t column = index % columns;
        if (column % 16 == 0 && columns - column >= 16) {
            TrillimPackedNeonRow row = trillim_packed_neon_row(data, dtype, index / columns, columns, rows);
            trillim_packed_neon_decode16(&row, column, values);
            return;
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        values[i] = attention_load4(data, index + i * 4, dtype, rows, columns);
    }
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
    return trillim_packed_read_float(data, index, dtype, rows, columns);
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
            float32x4_t q[4], k[4];
            attention_load16(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns, q);
            attention_load16(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns, k);
            a = vfmaq_f32(a, q[0], k[0]);
            b = vfmaq_f32(b, q[1], k[1]);
            c = vfmaq_f32(c, q[2], k[2]);
            d = vfmaq_f32(d, q[3], k[3]);
        }
        for (; p->key_dim - channel >= 4; channel += 4) {
            a = vfmaq_f32(a,
                attention_load4(task->query, query + channel, p->q_dtype, p->q_rows, p->q_columns),
                attention_load4(task->keys, key + channel, p->k_dtype, p->k_rows, p->k_columns));
        }
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; channel < p->key_dim; ++channel) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_packed_read_float(task->query, query + channel * p->q_strides[2], p->q_dtype, p->q_rows, p->q_columns);
        float k = trillim_packed_read_float(task->keys, key + channel * p->k_strides[2], p->k_dtype, p->k_rows, p->k_columns);
        sum += q * k;
    }
    *score = sum;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_scores4_bf16(
    const TrillimAttentionTask *task, size_t query, size_t key, float *scores
) {
    const TrillimAttentionParameters *p = task->parameters;
    const uint16_t *q = (const uint16_t *)task->query + query;
    const uint16_t *k0 = (const uint16_t *)task->keys + key;
    size_t stride = p->k_strides[1];
    const uint16_t *k1 = k0 + stride;
    const uint16_t *k2 = k1 + stride;
    const uint16_t *k3 = k2 + stride;
    float32x4_t sums = vdupq_n_f32(0.0f);
    size_t channel = 0;
    for (; p->key_dim - channel >= 4; channel += 4) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x4_t a = vld1_u16(k0 + channel), b = vld1_u16(k1 + channel);
        uint16x4_t c = vld1_u16(k2 + channel), d = vld1_u16(k3 + channel);
        /* Transpose four contiguous loads while keeping each key's accumulation in channel order. */
        uint32x2_t ab_even = vreinterpret_u32_u16(vtrn1_u16(a, b));
        uint32x2_t ab_odd = vreinterpret_u32_u16(vtrn2_u16(a, b));
        uint32x2_t cd_even = vreinterpret_u32_u16(vtrn1_u16(c, d));
        uint32x2_t cd_odd = vreinterpret_u32_u16(vtrn2_u16(c, d));
        uint16x4_t keys0 = vreinterpret_u16_u32(vtrn1_u32(ab_even, cd_even));
        uint16x4_t keys1 = vreinterpret_u16_u32(vtrn1_u32(ab_odd, cd_odd));
        uint16x4_t keys2 = vreinterpret_u16_u32(vtrn2_u32(ab_even, cd_even));
        uint16x4_t keys3 = vreinterpret_u16_u32(vtrn2_u32(ab_odd, cd_odd));
        float32x4_t queries = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(q + channel), 16));
        sums = vfmaq_laneq_f32(sums, vreinterpretq_f32_u32(vshll_n_u16(keys0, 16)), queries, 0);
        sums = vfmaq_laneq_f32(sums, vreinterpretq_f32_u32(vshll_n_u16(keys1, 16)), queries, 1);
        sums = vfmaq_laneq_f32(sums, vreinterpretq_f32_u32(vshll_n_u16(keys2, 16)), queries, 2);
        sums = vfmaq_laneq_f32(sums, vreinterpretq_f32_u32(vshll_n_u16(keys3, 16)), queries, 3);
    }
    for (; channel < p->key_dim; ++channel) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x4_t keys = {k0[channel], k1[channel], k2[channel], k3[channel]};
        sums = vfmaq_n_f32(sums, vreinterpretq_f32_u32(vshll_n_u16(keys, 16)), trillim_bf16(q[channel]));
    }
    vst1q_f32(scores, sums);
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_attention_scores4(
    const TrillimAttentionTask *task, size_t query, size_t key, float *scores
) {
    const TrillimAttentionParameters *p = task->parameters;
    if (p->q_dtype == TRILLIM_BF16 && p->k_dtype == TRILLIM_BF16 &&
        p->q_strides[2] == 1 && p->k_strides[2] == 1) {
        return trillim_attention_scores4_bf16(task, query, key, scores);
    }
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
    size_t query_tile = legacy_query_tile(p, task->context->thread_pool->worker_count);
    size_t tiles = p->queries / query_tile + (p->queries % query_tile != 0);
    size_t head = index / tiles;
    size_t kv_head = head / (p->heads / p->kv_heads);
    size_t first = (index % tiles) * query_tile;
    size_t last = p->queries - first < query_tile ? p->queries : first + query_tile;
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
                        float32x4_t v[4];
                        attention_load16(task->values, value, p->v_dtype, p->v_rows, p->v_columns, v);
                        a = vfmaq_n_f32(a, v[0], scores[j]);
                        b = vfmaq_n_f32(b, v[1], scores[j]);
                        c = vfmaq_n_f32(c, v[2], scores[j]);
                        d = vfmaq_n_f32(d, v[3], scores[j]);
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
                        a = vfmaq_n_f32(a, attention_load4(task->values, value, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
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
                    accumulator[channel] += scores[j] * trillim_packed_read_float(task->values, value, p->v_dtype, p->v_rows, p->v_columns);
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
                        float32x4_t v[4];
                        attention_load16(task->values, value, p->v_dtype, p->v_rows, p->v_columns, v);
                        a = vfmaq_n_f32(a, v[0], scores[j]);
                        b = vfmaq_n_f32(b, v[1], scores[j]);
                        c = vfmaq_n_f32(c, v[2], scores[j]);
                        d = vfmaq_n_f32(d, v[3], scores[j]);
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
                        a = vfmaq_n_f32(a, attention_load4(task->values, value, p->v_dtype, p->v_rows, p->v_columns), scores[j]);
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
                    accumulator[channel] += scores[j] * trillim_packed_read_float(task->values, value, p->v_dtype, p->v_rows, p->v_columns);
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

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
#ifdef __APPLE__
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
static GetThreading get_threading;
static SetThreading set_threading;
#endif
#if !defined(_WIN32)
static atomic_uint blas_state;
#endif

static bool initialize_blas(const TrillimKernelContext *context) {
#if defined(_WIN32)
    (void)context;
    return false;
#else
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1, memory_order_acq_rel, memory_order_acquire)) {
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
        static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
#endif
        for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
            void *library = dlopen(names[index], RTLD_LOCAL | RTLD_LAZY);
            if (!library) {
                continue;
            }
#ifdef __APPLE__
            void *address = dlsym(library, "cblas_sgemm$NEWLAPACK");
            if (!address) {
                address = dlsym(library, "cblas_sgemm");
            }
#else
            void *address = dlsym(library, "cblas_sgemm");
#endif
            if (address) {
                _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                void *getter = dlsym(library, "BLASGetThreading");
                void *setter = dlsym(library, "BLASSetThreading");
                if (getter && setter) {
                    memcpy(&get_threading, &getter, sizeof(get_threading));
                    memcpy(&set_threading, &setter, sizeof(set_threading));
                } else {
                    sgemm = NULL;
                }
#endif
                if (sgemm) {
                    break;
                }
            }
            dlclose(library);
        }
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            if (context->is_cancelled(context->cancellation_context)) {
                return false;
            }
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
#endif
}

static size_t blas_first_key(const TrillimAttentionParameters *p) {
    if (p->windowed && p->length >= p->queries && p->length - p->queries + 1 > p->window) {
        return p->length - p->queries + 1 - p->window;
    }
    return 0;
}

static TrillimStatus ordinary_inputs(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!atomic_load_explicit(task->ordinary, memory_order_relaxed)) {
        return TRILLIM_OK;
    }
    size_t queries = p->heads * p->queries;
    size_t first_key = blas_first_key(p);
    size_t keys = p->kv_heads * (p->length - first_key);
    size_t total = queries + 2 * keys;
    size_t first = index * 128;
    size_t last = total - first < 128 ? total : first + 128;
    uint32x4_t minimum = vdupq_n_u32(UINT32_MAX), maximum = vdupq_n_u32(0);
    for (size_t row = first; row < last; ++row) {
        if ((row & 7) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const float *data;
        size_t base, channels;
        if (row < queries) {
            base = row / p->heads * p->q_strides[1] + row % p->heads * p->q_strides[0];
            data = task->query;
            channels = p->key_dim;
        } else {
            bool values = row - queries >= keys;
            size_t key = row - queries - (values ? keys : 0);
            const size_t *strides = values ? p->v_strides : p->k_strides;
            base = (key / p->kv_heads + first_key) * strides[1] + key % p->kv_heads * strides[0];
            data = values ? task->values : task->keys;
            channels = values ? p->value_dim : p->key_dim;
        }
        size_t channel = 0;
        for (; channels - channel >= 4; channel += 4) {
            uint32x4_t bits = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(data + base + channel)), vdupq_n_u32(0x7fffffff));
            maximum = vmaxq_u32(maximum, bits);
            bits = vbslq_u32(vceqq_u32(bits, vdupq_n_u32(0)), vdupq_n_u32(UINT32_MAX), bits);
            minimum = vminq_u32(minimum, bits);
        }
        for (; channel < channels; ++channel) {
            uint32_t bits;
            memcpy(&bits, data + base + channel, sizeof(bits));
            bits &= UINT32_C(0x7fffffff);
            if (bits && (bits < UINT32_C(0x2f800000) || bits >= UINT32_C(0x4f800000))) {
                atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
    }
    if (vminvq_u32(minimum) < UINT32_C(0x2f800000) || vmaxvq_u32(maximum) >= UINT32_C(0x4f800000)) {
        atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
    }
    return TRILLIM_OK;
}

static TrillimStatus blas_tile(const TrillimAttentionTask *task, size_t index, size_t slot) {
    const TrillimAttentionParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->queries / task->blas_queries + (p->queries % task->blas_queries != 0);
    size_t head = index / groups, kv_head = head / (p->heads / p->kv_heads);
    size_t first = index % groups * task->blas_queries;
    size_t rows = p->queries - first < task->blas_queries ? p->queries - first : task->blas_queries;
    size_t fallback_groups = p->queries / 16 + (p->queries % 16 != 0);
    size_t fallback_size = p->heads * fallback_groups * (p->value_dim + 64);
    float *scores = task->scratch + fallback_size + slot * task->blas_queries * (p->value_dim + task->blas_keys);
    float *accumulator = scores + task->blas_queries * task->blas_keys;
    float maximum[BLAS_MAX_QUERIES], denominator[BLAS_MAX_QUERIES];
    size_t begin[BLAS_MAX_QUERIES], end[BLAS_MAX_QUERIES];
    size_t first_key = p->length, last_key = 0;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum[row] = -INFINITY;
        denominator[row] = 0;
        memset(accumulator + row * p->value_dim, 0, p->value_dim * sizeof(float));
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)(first + row);
        end[row] = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        begin[row] = p->windowed && position >= 0 && (uint64_t)position + 1 > p->window ?
            (size_t)position + 1 - p->window : 0;
        if (begin[row] < first_key) {
            first_key = begin[row];
        }
        if (end[row] > last_key) {
            last_key = end[row];
        }
    }
    const float *query = (const float *)task->query + head * p->q_strides[0] + first * p->q_strides[1];
    for (size_t block = first_key; block < last_key; block += task->blas_keys) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = last_key - block < task->blas_keys ? last_key - block : task->blas_keys;
        const float *keys = (const float *)task->keys + kv_head * p->k_strides[0] + block * p->k_strides[1];
        sgemm(101, 111, 112, (int)rows, (int)count, (int)p->key_dim, scale,
              query, (int)p->q_strides[1], keys, (int)p->k_strides[1], 0, scores, task->blas_keys);
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t row = 0; row < rows; ++row) {
            float peak = -INFINITY;
            float *probabilities = scores + row * task->blas_keys;
            for (size_t key = 0; key < count; ++key) {
                if (block + key < begin[row] || block + key >= end[row]) {
                    probabilities[key] = -INFINITY;
                }
                peak = fmaxf(peak, probabilities[key]);
            }
            peak = fmaxf(peak, maximum[row]);
            float normalization_peak = peak == -INFINITY ? 0 : peak;
            float previous = maximum[row] == -INFINITY ? 0 : expf(maximum[row] - normalization_peak);
            maximum[row] = peak;
            float32x4_t sum = vdupq_n_f32(0);
            size_t key = 0;
            for (; count - key >= 4; key += 4) {
                float32x4_t value = exp_negative(vsubq_f32(vld1q_f32(probabilities + key), vdupq_n_f32(normalization_peak)));
                vst1q_f32(probabilities + key, value);
                sum = vaddq_f32(sum, value);
            }
            float block_sum = vaddvq_f32(sum);
            for (; key < count; ++key) {
                probabilities[key] = expf(probabilities[key] - normalization_peak);
                block_sum += probabilities[key];
            }
            denominator[row] = denominator[row] * previous + block_sum;
            size_t channel = 0;
            float *output = accumulator + row * p->value_dim;
            for (; p->value_dim - channel >= 4; channel += 4) {
                vst1q_f32(output + channel, vmulq_n_f32(vld1q_f32(output + channel), previous));
            }
            for (; channel < p->value_dim; ++channel) {
                output[channel] *= previous;
            }
        }
        const float *values = (const float *)task->values + kv_head * p->v_strides[0] + block * p->v_strides[1];
        sgemm(101, 111, 111, (int)rows, (int)p->value_dim, (int)count, 1,
              scores, task->blas_keys, values, (int)p->v_strides[1], 1, accumulator, (int)p->value_dim);
    }
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float reciprocal = 1.0f / (denominator[row] > 0 ? denominator[row] : 1.0f);
        float *output = (float *)task->output + ((first + row) * p->heads + head) * p->value_dim;
        const float *input = accumulator + row * p->value_dim;
        size_t channel = 0;
        for (; p->value_dim - channel >= 4; channel += 4) {
            vst1q_f32(output + channel, vmulq_n_f32(vld1q_f32(input + channel), reciprocal));
        }
        for (; channel < p->value_dim; ++channel) {
            output[channel] = input[channel] * reciprocal;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus fallback_blas_tile(void *opaque, size_t index, size_t worker) {
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = p->queries / task->blas_queries + (p->queries % task->blas_queries != 0);
    size_t fallback_groups = p->queries / 16 + (p->queries % 16 != 0);
    size_t head = index / groups, first = index % groups * task->blas_queries;
    size_t last = p->queries - first < task->blas_queries ? p->queries : first + task->blas_queries;
    for (size_t query = first; query < last; query += 16) {
        TrillimStatus status = trillim_attention_task(opaque, head * fallback_groups + query / 16, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus blas_task(void *opaque, size_t slot, size_t worker) {
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t groups = p->queries / task->blas_queries + (p->queries % task->blas_queries != 0);
    size_t tiles = p->heads * groups;
    size_t slots = tiles < 128 ? tiles : 128;
    bool enabled = true;
#ifdef __APPLE__
    unsigned previous = get_threading();
    if (previous != 1) {
        enabled = set_threading(1) == 0;
    }
#endif
    TrillimStatus status = TRILLIM_OK;
    for (size_t index = slot; index < tiles;) {
        status = enabled ? blas_tile(task, index, slot) : fallback_blas_tile(opaque, index, worker);
        if (status != TRILLIM_OK || tiles - index <= slots) {
            break;
        }
        index += slots;
    }
#ifdef __APPLE__
    if (enabled && previous != 1) {
        (void)set_threading(previous);
    }
#endif
    return status;
}

static bool attention_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus trillim_attention_gqa4(const TrillimAttentionTask *task) {
    const TrillimAttentionParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    const uint16_t *query[4];
    float *scores[4], *accumulator[4];
    float maximum[4], denominator[4];
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t head = 0; head < 4; ++head) {
        query[head] = (const uint16_t *)task->query + head * p->q_strides[0];
        scores[head] = task->scratch + head * (p->value_dim + 64);
        accumulator[head] = scores[head] + 64;
        memset(accumulator[head], 0, p->value_dim * sizeof(float));
        maximum[head] = -INFINITY;
        denominator[head] = 0;
    }
    /* SIMD lanes are heads, preserving each dot product's channel order. */
    float transposed[256 * 4];
    for (size_t channel = 0; channel < 256; channel += 4) {
        float32x4x4_t q = {{
            vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(query[0] + channel), 16)),
            vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(query[1] + channel), 16)),
            vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(query[2] + channel), 16)),
            vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(query[3] + channel), 16))
        }};
        vst4q_f32(transposed + channel * 4, q);
    }
    size_t begin = p->windowed && p->length > p->window ? p->length - p->window : 0;
    for (size_t block = begin; block < p->length; block += 64) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->length - block < 64 ? p->length - block : 64;
        size_t key = 0;
        for (; count - key >= 4; key += 4) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const uint16_t *k0 = (const uint16_t *)task->keys + (block + key) * p->k_strides[1];
            const uint16_t *k1 = k0 + p->k_strides[1];
            const uint16_t *k2 = k1 + p->k_strides[1];
            const uint16_t *k3 = k2 + p->k_strides[1];
            float32x4_t sum0 = vdupq_n_f32(0), sum1 = sum0, sum2 = sum0, sum3 = sum0;
            for (size_t channel = 0; channel < 256; channel += 4) {
                float32x4_t q0 = vld1q_f32(transposed + channel * 4);
                float32x4_t q1 = vld1q_f32(transposed + channel * 4 + 4);
                float32x4_t q2 = vld1q_f32(transposed + channel * 4 + 8);
                float32x4_t q3 = vld1q_f32(transposed + channel * 4 + 12);
                float32x4_t a = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(k0 + channel), 16));
                float32x4_t b = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(k1 + channel), 16));
                float32x4_t c = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(k2 + channel), 16));
                float32x4_t d = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(k3 + channel), 16));
#define UPDATE_KEY(S, K) \
                S = vfmaq_laneq_f32(S, q0, K, 0); \
                S = vfmaq_laneq_f32(S, q1, K, 1); \
                S = vfmaq_laneq_f32(S, q2, K, 2); \
                S = vfmaq_laneq_f32(S, q3, K, 3)
                UPDATE_KEY(sum0, a);
                UPDATE_KEY(sum1, b);
                UPDATE_KEY(sum2, c);
                UPDATE_KEY(sum3, d);
#undef UPDATE_KEY
            }
            float32x4_t ab0 = vtrn1q_f32(sum0, sum1), ab1 = vtrn2q_f32(sum0, sum1);
            float32x4_t cd0 = vtrn1q_f32(sum2, sum3), cd1 = vtrn2q_f32(sum2, sum3);
            vst1q_f32(scores[0] + key, vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)));
            vst1q_f32(scores[1] + key, vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)));
            vst1q_f32(scores[2] + key, vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)));
            vst1q_f32(scores[3] + key, vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)));
        }
        for (; key < count; ++key) {
            for (size_t head = 0; head < 4; ++head) {
                TrillimStatus status = trillim_attention_score(task, head * p->q_strides[0],
                    (block + key) * p->k_strides[1], scores[head] + key);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
        }
        for (size_t head = 0; head < 4; ++head) {
            float peak = -INFINITY;
            for (size_t key = 0; key < count; ++key) {
                if (isnan(scores[head][key])) {
                    return context->thread_pool->run(
                        context->thread_pool->context, 4, trillim_attention_task, (void *)task
                    );
                }
                scores[head][key] *= scale;
                peak = fmaxf(peak, scores[head][key]);
            }
            float next = fmaxf(maximum[head], peak);
            float normalizer = next == -INFINITY ? 0.0f : next;
            float previous = maximum[head] == -INFINITY ? 0.0f : expf(maximum[head] - normalizer);
            float sum = 0;
            for (size_t key = 0; key < count; ++key) {
                scores[head][key] = expf(scores[head][key] - normalizer);
                sum += scores[head][key];
            }
            for (size_t channel = 0; channel < 256; ++channel) {
                accumulator[head][channel] *= previous;
            }
            denominator[head] = denominator[head] * previous + sum;
            maximum[head] = next;
        }
        for (size_t channel = 0; channel < 256; channel += 8) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a0 = vld1q_f32(accumulator[0] + channel), b0 = vld1q_f32(accumulator[0] + channel + 4);
            float32x4_t a1 = vld1q_f32(accumulator[1] + channel), b1 = vld1q_f32(accumulator[1] + channel + 4);
            float32x4_t a2 = vld1q_f32(accumulator[2] + channel), b2 = vld1q_f32(accumulator[2] + channel + 4);
            float32x4_t a3 = vld1q_f32(accumulator[3] + channel), b3 = vld1q_f32(accumulator[3] + channel + 4);
            for (size_t key = 0; key < count; ++key) {
                const uint16_t *value = (const uint16_t *)task->values + (block + key) * p->v_strides[1] + channel;
                uint16x8_t bits = vld1q_u16(value);
                float32x4_t v0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16));
                float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16));
                a0 = vfmaq_n_f32(a0, v0, scores[0][key]);
                b0 = vfmaq_n_f32(b0, v1, scores[0][key]);
                a1 = vfmaq_n_f32(a1, v0, scores[1][key]);
                b1 = vfmaq_n_f32(b1, v1, scores[1][key]);
                a2 = vfmaq_n_f32(a2, v0, scores[2][key]);
                b2 = vfmaq_n_f32(b2, v1, scores[2][key]);
                a3 = vfmaq_n_f32(a3, v0, scores[3][key]);
                b3 = vfmaq_n_f32(b3, v1, scores[3][key]);
            }
            vst1q_f32(accumulator[0] + channel, a0);
            vst1q_f32(accumulator[0] + channel + 4, b0);
            vst1q_f32(accumulator[1] + channel, a1);
            vst1q_f32(accumulator[1] + channel + 4, b1);
            vst1q_f32(accumulator[2] + channel, a2);
            vst1q_f32(accumulator[2] + channel + 4, b2);
            vst1q_f32(accumulator[3] + channel, a3);
            vst1q_f32(accumulator[3] + channel + 4, b3);
        }
    }
    for (size_t head = 0; head < 4; ++head) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t channel = 0; channel < 256; ++channel) {
            TrillimValue value = {.f = accumulator[head][channel] /
                (denominator[head] > 0.0f ? denominator[head] : 1.0f)};
            trillim_store(task->output, head * 256 + channel, TRILLIM_BF16, value);
        }
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
        outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context, false, &ordinary, 64, 128
    };
    if (trillim_attention_blocked(p)) {
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, p->kv_heads, trillim_attention_pack, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        size_t tiles = p->queries / 4 + (p->queries % 4 != 0);
        size_t count = p->heads * tiles < 64 ? p->heads * tiles : 64;
        return context->thread_pool->run(context->thread_pool->context, count, trillim_attention_blocks, &task);
    }
    if (p->queries == 1 && p->heads == 4 && p->kv_heads == 1 &&
        p->key_dim == 256 && p->value_dim == 256 && p->length >= 8 && p->length <= 512 &&
        (!p->windowed || p->window >= 8) &&
        p->output_dtype == TRILLIM_BF16 && p->q_dtype == TRILLIM_BF16 &&
        p->k_dtype == TRILLIM_BF16 && p->v_dtype == TRILLIM_BF16 &&
        p->q_strides[2] == 1 && p->k_strides[2] == 1 && p->v_strides[2] == 1 &&
        (((uintptr_t)inputs[0] | (uintptr_t)inputs[1] | (uintptr_t)inputs[2]) & 1) == 0 &&
        attention_default_fpcr()) {
        return trillim_attention_gqa4(&task);
    }
    bool optimized = optimized_layout(p);
    size_t visible = p->windowed && p->window < p->length ? p->window : p->length;
    task.blas_queries = p->queries >= 512 && visible >= 512 ? 128 : 64;
    task.blas_keys = visible >= 256 ? 256 : 128;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    bool blas_rows = !p->length || p->kv_heads <= (SIZE_MAX - p->heads * p->queries) / 2 / p->length;
    if (optimized && p->q_dtype == TRILLIM_FP32 && p->k_dtype == TRILLIM_FP32 && p->v_dtype == TRILLIM_FP32 &&
        p->queries >= 32 && visible >= 64 && p->key_dim >= 64 && p->key_dim <= 256 &&
        p->value_dim >= 64 && p->value_dim <= 256 && p->q_strides[1] >= p->key_dim &&
        p->k_strides[1] >= p->key_dim && p->v_strides[1] >= p->value_dim && p->q_strides[1] <= INT32_MAX &&
        p->k_strides[1] <= INT32_MAX && p->v_strides[1] <= INT32_MAX && blas_rows &&
        isfinite(scale) && fabsf(scale) <= 1 && initialize_blas(context)) {
        size_t rows = p->heads * p->queries + 2 * p->kv_heads * (p->length - blas_first_key(p));
        size_t groups = rows / 128 + (rows % 128 != 0);
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, groups, ordinary_inputs, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (atomic_load_explicit(&ordinary, memory_order_relaxed)) {
            size_t groups = p->queries / task.blas_queries + (p->queries % task.blas_queries != 0);
            size_t slots = p->heads * groups < 128 ? p->heads * groups : 128;
            return context->thread_pool->run(context->thread_pool->context, slots, blas_task, &task);
        }
    }
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
    size_t query_tile = legacy_query_tile(p, context->thread_pool->worker_count);
    size_t tiles = p->queries / query_tile + (p->queries % query_tile != 0);
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
