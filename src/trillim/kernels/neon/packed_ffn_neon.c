
#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#if defined(__APPLE__)
#include <stdlib.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if !defined(_WIN32)
#define COMPONENT_API __attribute__((visibility("default")))
#else
#define COMPONENT_API
#endif

#if defined(__APPLE__) || defined(__linux__)
extern TrillimStatus trillim_neon_try_packed_ffn_int8(
        void *, const void *, const void *, const void *, const void *, float (*)(float),
    const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8")
#endif
;

extern TrillimStatus trillim_neon_try_packed_linear_bf16(
    void *, const void *, const void *, const void *, const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_linear_bf16")
#endif
;

extern float32x4_t trillim_neon_silu4_prefill(float32x4_t)
#if defined(__linux__)
    __asm__(".trillim_neon_silu4_prefill")
#endif
;
extern TrillimStatus trillim_neon_try_packed_ffn_int8_prefill(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    float32x4_t (*)(float32x4_t), const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8_prefill")
#endif
;
#endif

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

#if defined(__APPLE__)
extern TrillimStatus trillim_neon_try_packed_linear_prefill(
    void *, const void *, const void *, const void *, const TrillimKernelContext *, bool *
);
#endif

__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_ffn_fhm_v\\reg,\\reg\n"
    ".endr\n"
);

static inline float16x8_t signed_half(int16x8_t codes) {
    /* SCVTF Vd.8H, Vn.8H requires full FP16 arithmetic. */
    float16x8_t half;
    __asm__(".inst 0x4e79d800 | trillim_ffn_fhm_%0 | (trillim_ffn_fhm_%1 << 5)" : "=w"(half) : "w"(codes));
    return half;
}

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Older Apple assemblers discard function feature directives. */
#define FHM
static inline float32x4_t fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x4e20ec00 | trillim_ffn_fhm_%0 | (trillim_ffn_fhm_%1 << 5) | (trillim_ffn_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline float32x4_t fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x6e20cc00 | trillim_ffn_fhm_%0 | (trillim_ffn_fhm_%1 << 5) | (trillim_ffn_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
__asm__(".arch armv8.6-a+fp16fml");
#if defined(__clang__)
#define FHM __attribute__((target("fp16fml")))
#else
#define FHM __attribute__((target("+fp16fml")))
#endif
FHM static inline float32x4_t fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_low_f16(sum, a, b);
}
FHM static inline float32x4_t fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_high_f16(sum, a, b);
}
#endif

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

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { TILE_SIZE = 16 };

typedef struct {
    const uint8_t *codes;
    const uint8_t *scales;
    const uint8_t *high;
    const uint8_t *subscales;
} PackedGroupRow;

static inline bool packed_group_dtype(uint32_t dtype) {
    return dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5 ||
        dtype == TRILLIM_DUBGRP_INT6 || dtype == TRILLIM_GRP_INT8;
}

static inline PackedGroupRow packed_group_row(
    const uint8_t *data, size_t width, size_t outputs, size_t column, uint32_t dtype
) {
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    size_t groups = width / group + (width % group != 0);
    size_t block = column * groups, total = outputs * groups;
    size_t code_bytes = dtype == TRILLIM_DUBGRP_INT6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    PackedGroupRow row = {
        .codes = data + block * code_bytes,
        .scales = data + total * code_bytes + block * 2,
        .high = NULL,
        .subscales = NULL
    };
    if (dtype == TRILLIM_GRP_INT5) {
        row.high = data + total * (16 + 2) + block * 4;
    } else if (dtype == TRILLIM_DUBGRP_INT6) {
        row.high = data + total * (128 + 2) + block * 64;
        row.subscales = data + total * (128 + 2 + 64) + block * 16;
    }
    return row;
}

static inline float32x4_t packed_group_scaled(int16x4_t codes, float scale, bool factored) {
    float32x4_t values = vcvtq_f32_s32(vmovl_s16(codes));
    return factored ? values : vmulq_n_f32(values, scale);
}

static inline void packed_group_widen(
    int8x16_t codes, bool scaled, int8_t subscale, float scale, bool factored, float32x4_t result[4]
) {
    int16x8_t low = vmovl_s8(vget_low_s8(codes));
    int16x8_t high = vmovl_high_s8(codes);
    if (scaled) {
        low = vmulq_n_s16(low, subscale);
        high = vmulq_n_s16(high, subscale);
    }
    result[0] = packed_group_scaled(vget_low_s16(low), scale, factored);
    result[1] = packed_group_scaled(vget_high_s16(low), scale, factored);
    result[2] = packed_group_scaled(vget_low_s16(high), scale, factored);
    result[3] = packed_group_scaled(vget_high_s16(high), scale, factored);
}

static inline float16x8x2_t packed_group_q4_half(uint8x16_t codes) {
    /* The sixteen centered codes have zero low bytes in FP16. */
    static const uint8_t fp16_high[16] = {
        200, 199, 198, 197, 196, 194, 192, 188, 0, 60, 64, 66, 68, 69, 70, 71
    };
    uint8x16_t bits = vqtbl1q_u8(vld1q_u8(fp16_high), codes);
    return (float16x8x2_t){{
        vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(bits), 8)),
        vreinterpretq_f16_u16(vshll_high_n_u8(bits, 8))
    }};
}

static inline void packed_group_q4(uint8x16_t codes, float scale, bool factored, float32x4_t result[4]) {
    float16x8x2_t half = packed_group_q4_half(codes);
    result[0] = vcvt_f32_f16(vget_low_f16(half.val[0]));
    result[1] = vcvt_high_f32_f16(half.val[0]);
    result[2] = vcvt_f32_f16(vget_low_f16(half.val[1]));
    result[3] = vcvt_high_f32_f16(half.val[1]);
    if (!factored) {
        for (size_t lane = 0; lane < 4; ++lane) {
            result[lane] = vmulq_n_f32(result[lane], scale);
        }
    }
}

static inline uint8x16_t packed_group_q5_high(const uint8_t *high) {
    static const uint8_t masks[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
    return vandq_u8(vtstq_u8(bytes, vld1q_u8(masks)), vdupq_n_u8(16));
}

static inline int8x16x2_t centered_q5(uint8x16_t low, uint8x16_t high, const uint8_t *bits) {
    int8x16_t center = vdupq_n_s8(16);
    return (int8x16x2_t){{
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, packed_group_q5_high(bits))), center),
        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(high, packed_group_q5_high(bits + 2))), center)
    }};
}

static inline float packed_group_scale(const PackedGroupRow *row, size_t k, uint32_t dtype) {
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    uint16_t bits;
    memcpy(&bits, row->scales + (k / group) * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline __attribute__((always_inline)) void packed_group_decode32(
    const PackedGroupRow *row, size_t k, uint32_t dtype, bool factored, float32x4_t result[8]
) {
    float scale = factored ? 0.0f : packed_group_scale(row, k, dtype);
    if (dtype == TRILLIM_DUBGRP_INT6) {
        size_t block = k / 256, segment = (k % 256) / 32;
        const uint8_t *codes = row->codes + block * 128 + segment / 4 * 64 + segment % 4 / 2 * 32;
        const uint8_t *high = row->high + block * 64 + segment / 4 * 32;
        const int8_t *subscales = (const int8_t *)row->subscales + k / 16;
        int8x16_t zero = vdupq_n_s8(32);
        int8x16_t shift = vdupq_n_s8(-(int8_t)(segment % 4 * 2));
        uint8x16_t packed0 = vld1q_u8(codes), packed1 = vld1q_u8(codes + 16);
        uint8x16_t lo0 = segment & 1 ? vshrq_n_u8(packed0, 4) : vandq_u8(packed0, vdupq_n_u8(15));
        uint8x16_t lo1 = segment & 1 ? vshrq_n_u8(packed1, 4) : vandq_u8(packed1, vdupq_n_u8(15));
        uint8x16_t upper0 = vshlq_u8(vld1q_u8(high), shift);
        uint8x16_t upper1 = vshlq_u8(vld1q_u8(high + 16), shift);
        upper0 = vshlq_n_u8(vandq_u8(upper0, vdupq_n_u8(3)), 4);
        upper1 = vshlq_n_u8(vandq_u8(upper1, vdupq_n_u8(3)), 4);
        int8x16_t a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, upper0)), zero);
        int8x16_t b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, upper1)), zero);
        packed_group_widen(a, true, subscales[0], scale, factored, result);
        packed_group_widen(b, true, subscales[1], scale, factored, result + 4);
        return;
    }
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)row->codes + k;
        packed_group_widen(vld1q_s8(codes), false, 1, scale, factored, result);
        packed_group_widen(vld1q_s8(codes + 16), false, 1, scale, factored, result + 4);
        return;
    }
    uint8x16_t packed = vld1q_u8(row->codes + k / 2);
    uint8x16_t a = vandq_u8(packed, vdupq_n_u8(15)), b = vshrq_n_u8(packed, 4);
    if (dtype == TRILLIM_GRP_INT4) {
        packed_group_q4(a, scale, factored, result);
        packed_group_q4(b, scale, factored, result + 4);
        return;
    }
    int8x16x2_t centered = centered_q5(a, b, row->high + k / 8);
    packed_group_widen(centered.val[0], false, 1, scale, factored, result);
    packed_group_widen(centered.val[1], false, 1, scale, factored, result + 4);
}

enum { GROUP_ROWS = 2 };

static inline __attribute__((always_inline)) TrillimStatus packed_group_dot(
    const void *input, uint32_t input_dtype, size_t row, size_t rows, size_t width, uint32_t dtype,
    const PackedGroupRow *first, const PackedGroupRow *second, const TrillimKernelContext *context,
    float result[GROUP_ROWS], float other[GROUP_ROWS]
) {
    const bool pair = second != NULL;
    float32x4_t a[GROUP_ROWS][4] = {{0}}, b[GROUP_ROWS][4] = {{0}};
    for (size_t k = 0; k < width; k += 32) {
        if ((k & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t aw[8], bw[8];
        packed_group_decode32(first, k, dtype, false, aw);
        if (pair) {
            packed_group_decode32(second, k, dtype, false, bw);
        }
        for (size_t j = 0; j < 8; ++j) {
            float32x4_t x = trillim_neon_load4(input, row * width + k + j * 4, input_dtype);
            a[0][j % 4] = vaddq_f32(a[0][j % 4], vmulq_f32(x, aw[j]));
            if (pair) {
                b[0][j % 4] = vaddq_f32(b[0][j % 4], vmulq_f32(x, bw[j]));
            }
            if (rows > 1) {
                x = trillim_neon_load4(input, (row + 1) * width + k + j * 4, input_dtype);
                a[1][j % 4] = vaddq_f32(a[1][j % 4], vmulq_f32(x, aw[j]));
                if (pair) {
                    b[1][j % 4] = vaddq_f32(b[1][j % 4], vmulq_f32(x, bw[j]));
                }
            }
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        result[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][2]), vaddq_f32(a[r][1], a[r][3])));
        if (pair) {
            other[r] = vaddvq_f32(vaddq_f32(vaddq_f32(b[r][0], b[r][2]), vaddq_f32(b[r][1], b[r][3])));
        }
    }
    return TRILLIM_OK;
}


typedef struct {
    size_t tokens, width, outputs;
    size_t hidden_width;
    uint32_t input_dtype, output_dtype;
    uint32_t gate_dtype, up_dtype, down_dtype;
} Parameters;

typedef struct ExactInput ExactInput;

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
    const uint16_t *half_input;
    const ExactInput *exact_input;
} Task;

static bool packed(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    *scratch = outputs[0] = 0;
    memset(inputs, 0, 4 * sizeof(*inputs));
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) ||
        !packed(p->gate_dtype) ||
        !packed(p->up_dtype) ||
        !packed(p->down_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (!trillim_packed_payload_size(p->tokens, p->outputs, p->output_dtype, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->width, p->input_dtype, inputs)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->up_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->gate_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    if (p->hidden_width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->hidden_width, p->down_dtype, inputs + 3)) {
        return TRILLIM_ERROR;
    }
    if (!trillim_packed_payload_size(p->tokens, p->hidden_width, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus dot(
    const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, const TrillimKernelContext *context, float *result
) {
    if (width == 0) {
        *result = 0.0f;
        return TRILLIM_OK;
    }
    size_t x_start = row * width;
    TrillimPackedNeonRow w = trillim_packed_neon_row(weight, weight_dtype, column, width, outputs);
    size_t k = 0;
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; width - k >= 16; k += 16) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t decoded[4];
        trillim_packed_neon_decode16(&w, k, decoded);
        for (size_t i = 0; i < 4; ++i) {
            float32x4_t x = trillim_neon_load4(input, x_start + k + i * 4, input_dtype);
            sums[i] = vaddq_f32(sums[i], vmulq_f32(x, decoded[i]));
        }
    }
    float32x4_t a = vaddq_f32(sums[0], sums[2]);
    float32x4_t b = vaddq_f32(sums[1], sums[3]);
    float sum = vaddvq_f32(vaddq_f32(a, b));
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        float decoded = trillim_packed_read_float(weight, w.first_element + k, weight_dtype, outputs, width);
        sum += x * decoded;
    }
    *result = sum;
    return TRILLIM_OK;
}

static TrillimStatus dot_pair(
    const void *input, uint32_t input_dtype, size_t row,
    const void *first_weight, uint32_t first_dtype, const void *second_weight, uint32_t second_dtype,
    size_t column, size_t width, size_t outputs, const TrillimKernelContext *context,
    float *first_result, float *second_result
) {
    if (width == 0) {
        *first_result = *second_result = 0.0f;
        return TRILLIM_OK;
    }
    size_t x_start = row * width, w_start = column * width;
    TrillimPackedNeonRow first = trillim_packed_neon_row(first_weight, first_dtype, column, width, outputs);
    TrillimPackedNeonRow second = trillim_packed_neon_row(second_weight, second_dtype, column, width, outputs);
    size_t k = 0;
    float32x4_t a[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t b[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; width - k >= 16; k += 16) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t aw[4], bw[4];
        trillim_packed_neon_decode16(&first, k, aw);
        trillim_packed_neon_decode16(&second, k, bw);
        for (size_t i = 0; i < 4; ++i) {
            float32x4_t x = trillim_neon_load4(input, x_start + k + i * 4, input_dtype);
            a[i] = vaddq_f32(a[i], vmulq_f32(x, aw[i]));
            b[i] = vaddq_f32(b[i], vmulq_f32(x, bw[i]));
        }
    }
    float first_sum = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3])));
    float second_sum = vaddvq_f32(vaddq_f32(vaddq_f32(b[0], b[2]), vaddq_f32(b[1], b[3])));
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        first_sum += x * trillim_packed_read_float(first_weight, w_start + k, first_dtype, outputs, width);
        second_sum += x * trillim_packed_read_float(second_weight, w_start + k, second_dtype, outputs, width);
    }
    *first_result = first_sum;
    *second_result = second_sum;
    return TRILLIM_OK;
}

static float silu(float x) {
    float e = expf(-fabsf(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * sigmoid;
}

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->hidden_width;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->hidden_width, column = index % p->hidden_width;
        float value, gate;
        TrillimStatus status = dot_pair(
            task->inputs[0], p->input_dtype, row,
            task->inputs[1], p->up_dtype, task->inputs[2], p->gate_dtype,
            column, p->width, p->hidden_width, task->context, &value, &gate
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        ((float *)task->hidden)[index] = value * silu(gate);
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->outputs;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->outputs, column = index % p->outputs;
        float value;
        TrillimStatus status = dot(
            task->hidden, TRILLIM_FP32, row, task->inputs[3], p->down_dtype,
            column, p->hidden_width, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_hidden_groups(
    void *opaque, size_t tile, uint32_t dtype
) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->hidden_width / TILE_SIZE + (p->hidden_width % TILE_SIZE != 0);
    size_t row = tile / tiles * GROUP_ROWS, first = tile % tiles * TILE_SIZE;
    size_t rows = p->tokens - row < GROUP_ROWS ? p->tokens - row : GROUP_ROWS;
    size_t end = p->hidden_width - first < TILE_SIZE ? p->hidden_width : first + TILE_SIZE;
    for (size_t column = first; column < end; ++column) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        PackedGroupRow a = packed_group_row(task->inputs[1], p->width, p->hidden_width, column, dtype);
        PackedGroupRow b = packed_group_row(task->inputs[2], p->width, p->hidden_width, column, dtype);
        float up[GROUP_ROWS], gate[GROUP_ROWS];
        TrillimStatus status = packed_group_dot(
            task->inputs[0], p->input_dtype, row, rows, p->width, dtype, &a, &b, task->context, up, gate
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        for (size_t r = 0; r < rows; ++r) {
            ((float *)task->hidden)[(row + r) * p->hidden_width + column] = up[r] * silu(gate[r]);
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

/* These input and scale bounds leave headroom for a normal FP32 grouped dot. */
enum { MIN_GROUP_EXPONENT = 87, MAX_GROUP_EXPONENT = 167, MAX_FACTORED_WIDTH = 1 << 16 };
enum { MAX_FHM_WIDTH = 8192 };
/* Bound both the Q6 widening dots and their FP32 SiLU product. */
enum { MAX_Q6_FHM_SCALE_EXPONENT = 147 };

static bool available_fhm(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value != 0) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_FHM", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP_ASIMDFHM)
    available = (getauxval(AT_HWCAP) & HWCAP_ASIMDFHM) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

static bool available_half_conversion(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_FP16", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP_ASIMDHP)
    available = (getauxval(AT_HWCAP) & HWCAP_ASIMDHP) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

FHM static inline float16x8x2_t signed_codes_half(int8x16_t codes) {
    return (float16x8x2_t){{
        signed_half(vmovl_s8(vget_low_s8(codes))),
        signed_half(vmovl_high_s8(codes)),
    }};
}

static inline bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus prepare_fhm_input(const Task *task, uint16_t *half, bool *exact) {
    size_t width = task->p->width;
    if (task->p->input_dtype == TRILLIM_FP16) {
        memcpy(half, task->inputs[0], width * sizeof(*half));
        *exact = true;
        return TRILLIM_OK;
    }
    uint32x4_t changed = vdupq_n_u32(0);
    for (size_t k = 0; k < width; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t lo = trillim_neon_load4(task->inputs[0], k, TRILLIM_BF16);
        float32x4_t hi = trillim_neon_load4(task->inputs[0], k + 4, TRILLIM_BF16);
        float16x8_t packed = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
        uint32x4_t widened_lo = vreinterpretq_u32_f32(vcvt_f32_f16(vget_low_f16(packed)));
        uint32x4_t widened_hi = vreinterpretq_u32_f32(vcvt_high_f32_f16(packed));
        changed = vorrq_u32(changed, veorq_u32(widened_lo, vreinterpretq_u32_f32(lo)));
        changed = vorrq_u32(changed, veorq_u32(widened_hi, vreinterpretq_u32_f32(hi)));
        vst1q_u16(half + k, vreinterpretq_u16_f16(packed));
    }
    *exact = vmaxvq_u32(changed) == 0;
    return TRILLIM_OK;
}

static TrillimStatus bounded_float16(
    const uint8_t *values, size_t count, uint32_t dtype, unsigned max_exponent,
    const TrillimKernelContext *context, bool *bounded
) {
    const unsigned low = dtype == TRILLIM_FP16 ? 0 : MIN_GROUP_EXPONENT << 7;
    const unsigned span = dtype == TRILLIM_FP16 ? 0x7bff :
        ((max_exponent - MIN_GROUP_EXPONENT + 1) << 7) - 1;
    uint16x8_t rejected = vdupq_n_u16(0);
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t bits = vreinterpretq_u16_u8(vld1q_u8(values + i * 2));
        uint16x8_t magnitude = vandq_u16(bits, vdupq_n_u16(0x7fff));
        uint16x8_t outside = vcgtq_u16(vsubq_u16(magnitude, vdupq_n_u16(low)), vdupq_n_u16(span));
        rejected = vorrq_u16(rejected, vbicq_u16(outside, vceqzq_u16(magnitude)));
    }
    *bounded = vmaxvq_u16(rejected) == 0;
    for (; i < count && *bounded; ++i) {
        uint16_t bits;
        memcpy(&bits, values + i * 2, sizeof(bits));
        unsigned magnitude = bits & 0x7fff;
        *bounded = magnitude == 0 || magnitude - low <= span;
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus paired_factored_dot(
    const Task *task, uint32_t dtype, const PackedGroupRow weights[4], float result[4]
) {
    size_t width = task->p->width;
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < width; k += group) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t length = width - k < group ? width - k : group;
        float32x4_t partial[4] = {0};
        for (size_t offset = 0; offset < length; offset += 32) {
            float32x4_t input[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                input[lane] = trillim_neon_load4(task->inputs[0], k + offset + 4 * lane, task->p->input_dtype);
            }
            for (size_t c = 0; c < 4; ++c) {
                float32x4_t codes[8];
                packed_group_decode32(&weights[c], k + offset, dtype, true, codes);
                float32x4_t low = partial[c];
                float32x4_t high = vdupq_n_f32(0.0f);
                for (size_t lane = 0; lane < 4; ++lane) {
                    low = vfmaq_f32(low, input[lane], codes[lane]);
                    high = vfmaq_f32(high, input[lane + 4], codes[lane + 4]);
                }
                partial[c] = vaddq_f32(low, high);
            }
        }
        for (size_t c = 0; c < 4; ++c) {
            sums[c] = vfmaq_n_f32(sums[c], partial[c], packed_group_scale(&weights[c], k, dtype));
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

FHM static inline __attribute__((always_inline)) TrillimStatus paired_fhm_dot(
    const Task *task, uint32_t dtype, const PackedGroupRow weights[4], float result[4]
) {
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < task->p->width; k += 32) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8x4_t bits = vld1q_u16_x4(task->half_input + k);
        float16x8_t x0 = vreinterpretq_f16_u16(bits.val[0]);
        float16x8_t x1 = vreinterpretq_f16_u16(bits.val[1]);
        float16x8_t x2 = vreinterpretq_f16_u16(bits.val[2]);
        float16x8_t x3 = vreinterpretq_f16_u16(bits.val[3]);
        for (size_t c = 0; c < 4; ++c) {
            uint8x16_t codes = vld1q_u8(weights[c].codes + k / 2);
            uint8x16_t a = vandq_u8(codes, vdupq_n_u8(15)), b = vshrq_n_u8(codes, 4);
            float16x8x2_t low, high;
            if (dtype == TRILLIM_GRP_INT4) {
                low = packed_group_q4_half(a);
                high = packed_group_q4_half(b);
            } else {
                int8x16x2_t centered = centered_q5(a, b, weights[c].high + k / 8);
                low = signed_codes_half(centered.val[0]);
                high = signed_codes_half(centered.val[1]);
            }
            float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
            lo = fhm_low(lo, x0, low.val[0]);
            hi = fhm_low(hi, x2, high.val[0]);
            lo = fhm_high(lo, x0, low.val[0]);
            hi = fhm_high(hi, x2, high.val[0]);
            lo = fhm_low(lo, x1, low.val[1]);
            hi = fhm_low(hi, x3, high.val[1]);
            lo = fhm_high(lo, x1, low.val[1]);
            hi = fhm_high(hi, x3, high.val[1]);
            float scale = packed_group_scale(&weights[c], k, dtype);
            sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), scale);
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

FHM static TrillimStatus paired_fhm_dot_int4(const Task *task, const PackedGroupRow weights[4], float result[4]) {
    return paired_fhm_dot(task, TRILLIM_GRP_INT4, weights, result);
}

FHM static TrillimStatus paired_fhm_dot_int5(const Task *task, const PackedGroupRow weights[4], float result[4]) {
    return paired_fhm_dot(task, TRILLIM_GRP_INT5, weights, result);
}

FHM static TrillimStatus paired_fhm_dot_int6(
    const Task *task, const PackedGroupRow weights[4], float result[4]
) {
    size_t width = task->p->width;
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < width; k += 256) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t block = k / 256;
        size_t length = width - k < 256 ? width - k : 256;
        float32x4_t partial[4] = {0};
        for (size_t offset = 0; offset < length; offset += 32) {
            size_t segment = offset / 32;
            uint16x8x4_t bits = vld1q_u16_x4(task->half_input + k + offset);
            float16x8_t x0 = vreinterpretq_f16_u16(bits.val[0]);
            float16x8_t x1 = vreinterpretq_f16_u16(bits.val[1]);
            float16x8_t x2 = vreinterpretq_f16_u16(bits.val[2]);
            float16x8_t x3 = vreinterpretq_f16_u16(bits.val[3]);
            int8x16_t shift = vdupq_n_s8(-(int8_t)(segment % 4 * 2));
            for (size_t c = 0; c < 4; ++c) {
                const uint8_t *codes = weights[c].codes + block * 128 + segment / 4 * 64 + segment % 4 / 2 * 32;
                const uint8_t *high = weights[c].high + block * 64 + segment / 4 * 32;
                const int8_t *subscales = (const int8_t *)weights[c].subscales + block * 16 + segment * 2;
                uint8x16x2_t packed = vld1q_u8_x2(codes);
                uint8x16x2_t upper = vld1q_u8_x2(high);
                uint8x16_t lo0 = segment & 1 ? vshrq_n_u8(packed.val[0], 4) : vandq_u8(packed.val[0], vdupq_n_u8(15));
                uint8x16_t lo1 = segment & 1 ? vshrq_n_u8(packed.val[1], 4) : vandq_u8(packed.val[1], vdupq_n_u8(15));
                uint8x16_t hi0 = vshlq_n_u8(vandq_u8(vshlq_u8(upper.val[0], shift), vdupq_n_u8(3)), 4);
                uint8x16_t hi1 = vshlq_n_u8(vandq_u8(vshlq_u8(upper.val[1], shift), vdupq_n_u8(3)), 4);
                int8x16_t center = vdupq_n_s8(32);
                float16x8x2_t low = signed_codes_half(vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo0, hi0)), center));
                float16x8x2_t high_half = signed_codes_half(vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), center));
                float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
                lo = fhm_low(lo, x0, low.val[0]);
                hi = fhm_low(hi, x2, high_half.val[0]);
                lo = fhm_high(lo, x0, low.val[0]);
                hi = fhm_high(hi, x2, high_half.val[0]);
                lo = fhm_low(lo, x1, low.val[1]);
                hi = fhm_low(hi, x3, high_half.val[1]);
                lo = fhm_high(lo, x1, low.val[1]);
                hi = fhm_high(hi, x3, high_half.val[1]);
                /* Apply each subscale in FP32 after the dot of its 16 integer codes. */
                float32x4_t sum = vfmaq_n_f32(partial[c], lo, (float)subscales[0]);
                partial[c] = vfmaq_n_f32(sum, hi, (float)subscales[1]);
            }
        }
        for (size_t c = 0; c < 4; ++c) {
            sums[c] = vfmaq_n_f32(sums[c], partial[c], packed_group_scale(&weights[c], k, TRILLIM_DUBGRP_INT6));
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set exact_i8mm_v\\reg,\\reg\n"
    ".endr\n"
);
static inline int32x4_t dot_i8mm(int32x4_t sum, int8x16_t x, int8x16_t w) {
    __asm__(".inst 0x4e80a400 | exact_i8mm_%0 | (exact_i8mm_%1 << 5) | (exact_i8mm_%2 << 16)"
            : "+w"(sum) : "w"(x), "w"(w));
    return sum;
}

enum { EXACT_GROUP = 32, EXACT_MAX_GROUPS = 8192 / EXACT_GROUP };

typedef struct {
    int8x16_t pair[4];
} ExactDigitPair;

struct ExactInput {
    ExactDigitPair low[EXACT_MAX_GROUPS];
    ExactDigitPair high[EXACT_MAX_GROUPS];
    float factor[EXACT_MAX_GROUPS];
    uint8_t digits[EXACT_MAX_GROUPS];
};

static bool available_i8mm(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_I8MM", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP2_I8MM)
    available = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

static unsigned encode_exact_group(const uint16_t bits[32], ExactDigitPair *low,
                                  ExactDigitPair *high, float *factor) {
    unsigned minimum = MAX_GROUP_EXPONENT, maximum = 0;
    for (size_t i = 0; i < 32; ++i) {
        if (!bits[i]) {
            continue;
        }
        unsigned exponent = (bits[i] >> 7) & 255;
        if (exponent < MIN_GROUP_EXPONENT || exponent > MAX_GROUP_EXPONENT) {
            return 0;
        }
        minimum = exponent < minimum ? exponent : minimum;
        maximum = exponent > maximum ? exponent : maximum;
    }
    if (!maximum) {
        minimum = maximum = 127;
    }
    unsigned span = maximum - minimum;
    /* A 32-element Q5 dot with this exponent span fits in int32_t. */
    if (span > 14) {
        return 0;
    }
    uint32_t power = (minimum - 7) << 23;
    memcpy(factor, &power, sizeof(power));
    /* Align the BF16 mantissas to the lowest exponent and split the exact integers
       into signed base-128 digits for the two or four SMMLA input rows. */
    unsigned digits = span <= 6 ? 2 : 4;
    for (size_t j = 0; j < 4; ++j) {
        int8_t lower[16], upper[16];
        for (size_t i = 0; i < 8; ++i) {
            uint16_t raw = bits[j * 8 + i];
            unsigned exponent = (raw >> 7) & 255;
            uint32_t value = raw ? ((raw & 127) | 128) << (exponent - minimum) : 0;
            int sign = (raw >> 15) ? -1 : 1;
            lower[i] = (int)(value & 127) * sign;
            lower[i + 8] = (int)((value >> 7) & 127) * sign;
            if (digits == 4) {
                upper[i] = (int)((value >> 14) & 127) * sign;
                upper[i + 8] = (int)(value >> 21) * sign;
            }
        }
        low->pair[j] = vld1q_s8(lower);
        if (digits == 4) {
            high->pair[j] = vld1q_s8(upper);
        }
    }
    return digits;
}

static inline int8x16x2_t unpack_q5(const PackedGroupRow *row, size_t k) {
    uint8x16_t packed = vld1q_u8(row->codes + k / 2);
    uint8x16_t low = vandq_u8(packed, vdupq_n_u8(15));
    uint8x16_t high = vshrq_n_u8(packed, 4);
    return centered_q5(low, high, row->high + k / 8);
}

static inline void q5_weight_pair(const PackedGroupRow *a, const PackedGroupRow *b, size_t k,
                                  int8x16_t weights[4]) {
    int8x16x2_t first = unpack_q5(a, k), second = unpack_q5(b, k);
    for (size_t j = 0; j < 2; ++j) {
        int64x2_t x = vreinterpretq_s64_s8(first.val[j]), y = vreinterpretq_s64_s8(second.val[j]);
        weights[2 * j] = vreinterpretq_s8_s64(vzip1q_s64(x, y));
        weights[2 * j + 1] = vreinterpretq_s8_s64(vzip2q_s64(x, y));
    }
}

static inline int32x4_t dot_exact_pair(const ExactDigitPair *x, const int8x16_t w[4]) {
    int32x4_t a = dot_i8mm(vdupq_n_s32(0), x->pair[0], w[0]);
    int32x4_t b = dot_i8mm(vdupq_n_s32(0), x->pair[1], w[1]);
    a = dot_i8mm(a, x->pair[2], w[2]);
    b = dot_i8mm(b, x->pair[3], w[3]);
    return vaddq_s32(a, b);
}

static inline int32x4_t exact_digits_to_sum(int32x4_t a, int32x4_t b) {
    int32x4_t low = vcombine_s32(vget_low_s32(a), vget_low_s32(b));
    int32x4_t high = vcombine_s32(vget_high_s32(a), vget_high_s32(b));
    return vaddq_s32(low, vshlq_n_s32(high, 7));
}

static inline int32x4_t q5_exact_group(const PackedGroupRow w[4], size_t k, const ExactDigitPair *low,
                                      const ExactDigitPair *high, bool wide) {
    int8x16_t w01[4], w23[4];
    q5_weight_pair(&w[0], &w[1], k, w01);
    q5_weight_pair(&w[2], &w[3], k, w23);
    int32x4_t a = dot_exact_pair(low, w01), b = dot_exact_pair(low, w23);
    int32x4_t result = exact_digits_to_sum(a, b);
    if (wide) {
        a = dot_exact_pair(high, w01);
        b = dot_exact_pair(high, w23);
        int32x4_t upper = exact_digits_to_sum(a, b);
        result = vaddq_s32(result, vshlq_n_s32(upper, 14));
    }
    return result;
}

static inline float32x4_t q5_group_fallback(const Task *task, const PackedGroupRow weights[4], size_t k) {
    float32x4_t x[8], sums[4];
    for (size_t j = 0; j < 8; ++j) {
        x[j] = trillim_neon_load4(task->inputs[0], k + j * 4, TRILLIM_BF16);
    }
    for (size_t c = 0; c < 4; ++c) {
        float32x4_t codes[8];
        packed_group_decode32(&weights[c], k, TRILLIM_GRP_INT5, true, codes);
        float32x4_t low = vdupq_n_f32(0), high = vdupq_n_f32(0);
        for (size_t j = 0; j < 4; ++j) {
            low = vfmaq_f32(low, x[j], codes[j]);
            high = vfmaq_f32(high, x[j + 4], codes[j + 4]);
        }
        sums[c] = vaddq_f32(low, high);
    }
    return vpaddq_f32(vpaddq_f32(sums[0], sums[1]), vpaddq_f32(sums[2], sums[3]));
}

static TrillimStatus paired_i8mm_dot(const Task *task, const PackedGroupRow w[4], float result[4]) {
    const ExactInput *input = task->exact_input;
    size_t groups = task->p->width / 32;
    float32x4_t sum = vdupq_n_f32(0);
    for (size_t j = 0; j < groups; ++j) {
        if ((j & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits[4];
        for (size_t c = 0; c < 4; ++c) {
            memcpy(&bits[c], w[c].scales + j * sizeof(bits[c]), sizeof(bits[c]));
        }
        float32x4_t scale = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(bits), 16));
        size_t k = j * 32;
        unsigned digits = input->digits[j];
        float32x4_t partial;
        if (digits) {
            const ExactDigitPair *low = input->low + j, *high = input->high + j;
            partial = vcvtq_f32_s32(q5_exact_group(w, k, low, high, digits == 4));
            scale = vmulq_n_f32(scale, input->factor[j]);
        } else {
            partial = q5_group_fallback(task, w, k);
        }
        sum = vfmaq_f32(sum, partial, scale);
    }
    vst1q_f32(result, sum);
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_hidden_factored(
    void *opaque, size_t tile, uint32_t dtype
) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t first = tile * TILE_SIZE;
    size_t count = p->hidden_width - first < TILE_SIZE ? p->hidden_width - first : TILE_SIZE;
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    size_t groups = p->width / group + (p->width % group != 0);
    bool fhm = dtype != TRILLIM_GRP_INT8 && task->half_input != NULL && default_fpcr();
    bool bounded;
    for (size_t i = 1; i <= 2; ++i) {
        PackedGroupRow row = packed_group_row(task->inputs[i], p->width, p->hidden_width, first, dtype);
        unsigned max_exponent = fhm && dtype == TRILLIM_DUBGRP_INT6 ? MAX_Q6_FHM_SCALE_EXPONENT : MAX_GROUP_EXPONENT;
        TrillimStatus status = bounded_float16(
            row.scales, count * groups, TRILLIM_BF16, max_exponent, task->context, &bounded
        );
        if (status == TRILLIM_OK && !bounded && max_exponent == MAX_Q6_FHM_SCALE_EXPONENT) {
            fhm = false;
            status = bounded_float16(
                row.scales, count * groups, TRILLIM_BF16, MAX_GROUP_EXPONENT, task->context, &bounded
            );
        }
        if (status != TRILLIM_OK) {
            return status;
        }
        if (!bounded) {
            return project_hidden_groups(opaque, tile, dtype);
        }
    }
    float *hidden = task->hidden;
    size_t end = first + count;
    bool i8mm = dtype == TRILLIM_GRP_INT5 && task->exact_input != NULL && default_fpcr();
    for (size_t column = first; column < end; column += 2) {
        size_t next = column + (column + 1 < end);
        PackedGroupRow packed[4] = {
            packed_group_row(task->inputs[1], p->width, p->hidden_width, column, dtype),
            packed_group_row(task->inputs[2], p->width, p->hidden_width, column, dtype),
            packed_group_row(task->inputs[1], p->width, p->hidden_width, next, dtype),
            packed_group_row(task->inputs[2], p->width, p->hidden_width, next, dtype),
        };
        float dot[4];
        TrillimStatus status;
        if (i8mm) {
            status = paired_i8mm_dot(task, packed, dot);
        } else if (fhm) {
            if (dtype == TRILLIM_GRP_INT4) {
                status = paired_fhm_dot_int4(task, packed, dot);
            } else if (dtype == TRILLIM_GRP_INT5) {
                status = paired_fhm_dot_int5(task, packed, dot);
            } else {
                status = paired_fhm_dot_int6(task, packed, dot);
            }
        } else {
            status = paired_factored_dot(task, dtype, packed, dot);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
        hidden[column] = dot[0] * silu(dot[1]);
        if (next != column) {
            hidden[next] = dot[2] * silu(dot[3]);
        }
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_down_groups(
    void *opaque, size_t tile, uint32_t dtype
) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / TILE_SIZE + (p->outputs % TILE_SIZE != 0);
    size_t row = tile / tiles * GROUP_ROWS, first = tile % tiles * TILE_SIZE;
    size_t rows = p->tokens - row < GROUP_ROWS ? p->tokens - row : GROUP_ROWS;
    size_t end = p->outputs - first < TILE_SIZE ? p->outputs : first + TILE_SIZE;
    for (size_t column = first; column < end; ++column) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        PackedGroupRow weight = packed_group_row(task->inputs[3], p->hidden_width, p->outputs, column, dtype);
        float result[GROUP_ROWS];
        TrillimStatus status = packed_group_dot(
            task->hidden, TRILLIM_FP32, row, rows, p->hidden_width, dtype, &weight, NULL, task->context, result, NULL
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        for (size_t r = 0; r < rows; ++r) {
            size_t index = (row + r) * p->outputs + column;
            trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = result[r]});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#define DEFINE_GROUP_PROJECT(NAME, DTYPE) \
static TrillimStatus NAME##_##DTYPE(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    return NAME(opaque, tile, DTYPE); \
}

DEFINE_GROUP_PROJECT(project_hidden_groups, TRILLIM_GRP_INT4)
DEFINE_GROUP_PROJECT(project_hidden_groups, TRILLIM_GRP_INT5)
DEFINE_GROUP_PROJECT(project_hidden_groups, TRILLIM_DUBGRP_INT6)
DEFINE_GROUP_PROJECT(project_hidden_groups, TRILLIM_GRP_INT8)
DEFINE_GROUP_PROJECT(project_hidden_factored, TRILLIM_GRP_INT4)
DEFINE_GROUP_PROJECT(project_hidden_factored, TRILLIM_GRP_INT5)
DEFINE_GROUP_PROJECT(project_hidden_factored, TRILLIM_DUBGRP_INT6)
DEFINE_GROUP_PROJECT(project_hidden_factored, TRILLIM_GRP_INT8)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT4)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT5)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_DUBGRP_INT6)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT8)

#undef DEFINE_GROUP_PROJECT

static TrillimTask project_hidden_groups_operation(uint32_t dtype, bool factored) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return factored ? project_hidden_factored_TRILLIM_GRP_INT4 : project_hidden_groups_TRILLIM_GRP_INT4;
    case TRILLIM_GRP_INT5: return factored ? project_hidden_factored_TRILLIM_GRP_INT5 : project_hidden_groups_TRILLIM_GRP_INT5;
    case TRILLIM_DUBGRP_INT6: return factored ? project_hidden_factored_TRILLIM_DUBGRP_INT6 : project_hidden_groups_TRILLIM_DUBGRP_INT6;
    case TRILLIM_GRP_INT8: return factored ? project_hidden_factored_TRILLIM_GRP_INT8 : project_hidden_groups_TRILLIM_GRP_INT8;
    default: return NULL;
    }
}

static TrillimTask project_down_groups_operation(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return project_down_groups_TRILLIM_GRP_INT4;
    case TRILLIM_GRP_INT5: return project_down_groups_TRILLIM_GRP_INT5;
    case TRILLIM_DUBGRP_INT6: return project_down_groups_TRILLIM_DUBGRP_INT6;
    case TRILLIM_GRP_INT8: return project_down_groups_TRILLIM_GRP_INT8;
    default: return NULL;
    }
}

static size_t group_tiles(size_t tokens, size_t outputs) {
    size_t rows = tokens / GROUP_ROWS + (tokens % GROUP_ROWS != 0);
    size_t columns = outputs / TILE_SIZE + (outputs % TILE_SIZE != 0);
    return rows * columns;
}

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

static __attribute__((noinline)) TrillimStatus try_run_hidden_i8mm(
    Task *task, size_t tiles, TrillimTask operation, bool *handled
) {
    *handled = false;
    ExactInput input;
    size_t groups = task->p->width / 32, eligible = 0;
    for (size_t j = 0; j < groups; ++j) {
        if ((j & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits[32];
        memcpy(bits, (const uint8_t *)task->inputs[0] + j * sizeof(bits), sizeof(bits));
        input.digits[j] = encode_exact_group(bits, input.low + j, input.high + j, input.factor + j);
        eligible += input.digits[j] != 0;
    }
    if (eligible * 4 < groups * 3) {
        return TRILLIM_OK;
    }
    *handled = true;
    task->exact_input = &input;
    const TrillimThreadPool *pool = task->context->thread_pool;
    TrillimStatus status = tiles == 1 ? operation(task, 0, 0) : pool->run(pool->context, tiles, operation, task);
    task->exact_input = NULL;
    return status;
}

static TrillimStatus run_hidden(
    Task *task, size_t tiles, TrillimTask operation, bool try_fhm
) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    const Parameters *p = task->p;
    if (operation == project_hidden_factored_TRILLIM_GRP_INT5 && p->input_dtype == TRILLIM_BF16 &&
        p->width <= EXACT_MAX_GROUPS * EXACT_GROUP && p->hidden_width > 1024 &&
        available_i8mm() && default_fpcr()) {
        bool handled;
        TrillimStatus status = try_run_hidden_i8mm(task, tiles, operation, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
    if (try_fhm) {
        /* BF16 enters this path only after a bit-exact FP16 round trip. */
        uint16_t half_input[MAX_FHM_WIDTH];
        bool exact;
        TrillimStatus status = prepare_fhm_input(task, half_input, &exact);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (exact) {
            task->half_input = half_input;
            status = tiles == 1 ? operation(task, 0, 0) : pool->run(pool->context, tiles, operation, task);
            task->half_input = NULL;
            return status;
        }
    }
    return tiles == 1 ? operation(task, 0, 0) : pool->run(pool->context, tiles, operation, task);
}

#if defined(__APPLE__)
enum { PREFILL_ELEMENTS = 1 << 21, MIN_PREFILL_WEIGHTS = 1 << 21, GATE_TILE = 1024 };

typedef struct {
    float *hidden;
    const float *gate;
    size_t count;
    const TrillimKernelContext *context;
} PrefillGateTask;

static TrillimStatus apply_prefill_gate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const PrefillGateTask *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t start = tile * GATE_TILE;
    size_t count = task->count - start < GATE_TILE ? task->count - start : GATE_TILE;
    for (size_t i = start; i < start + count; ++i) {
        task->hidden[i] *= silu(task->gate[i]);
    }
    return TRILLIM_OK;
}

static TrillimStatus try_prefill(
    const Task *task, size_t scratch_size, bool *handled
) {
    const Parameters *p = task->p;
    *handled = false;
    if (p->tokens < 32 || p->width < 256 || p->width % 32 != 0 ||
        p->hidden_width < 1024 || p->hidden_width % 32 != 0 || p->outputs < 1024 || p->outputs > INT32_MAX ||
        p->tokens > PREFILL_ELEMENTS / p->width || p->tokens > PREFILL_ELEMENTS / p->hidden_width ||
        p->width * p->hidden_width < MIN_PREFILL_WEIGHTS || p->hidden_width * p->outputs < MIN_PREFILL_WEIGHTS ||
        p->output_dtype != TRILLIM_FP32 || (uintptr_t)task->output % _Alignof(float) != 0 ||
        p->up_dtype != p->gate_dtype || p->up_dtype != p->down_dtype || !packed_group_dtype(p->up_dtype) ||
        !default_fpcr()) {
        return TRILLIM_OK;
    }
    /* The up projection uses the existing FP32 hidden scratch. */
    float *gate = malloc(scratch_size);
    if (gate == NULL) {
        return TRILLIM_OK;
    }
    const TrillimKernelContext *context = task->context;
    LinearParameters expand = {p->tokens, p->width, p->hidden_width, p->input_dtype, p->up_dtype, TRILLIM_FP32};
    bool projected;
    TrillimStatus status = trillim_neon_try_packed_linear_prefill(
        task->hidden, task->inputs[0], task->inputs[1], &expand, context, &projected
    );
    if (status == TRILLIM_OK && projected) {
        status = trillim_neon_try_packed_linear_prefill(
            gate, task->inputs[0], task->inputs[2], &expand, context, &projected
        );
    }
    if (status == TRILLIM_OK && projected) {
        PrefillGateTask product = {task->hidden, gate, p->tokens * p->hidden_width, context};
        size_t tiles = product.count / GATE_TILE + (product.count % GATE_TILE != 0);
        const TrillimThreadPool *pool = context->thread_pool;
        status = pool->run(pool->context, tiles, apply_prefill_gate, &product);
    }
    free(gate);
    if (status != TRILLIM_OK || !projected) {
        return status;
    }
    LinearParameters down = {p->tokens, p->hidden_width, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return trillim_neon_try_packed_linear_prefill(
        task->output, task->hidden, task->inputs[3], &down, context, handled
    );
}
#endif

#if defined(__APPLE__) || defined(__linux__)
#if defined(__APPLE__)
__asm__(".private_extern _trillim_neon_try_packed_ffn_bf16");
#else
__asm__(".hidden .trillim_neon_try_packed_ffn_bf16\n.globl .trillim_neon_try_packed_ffn_bf16");
static TrillimStatus trillim_neon_try_packed_ffn_bf16(
    void *, const void *const *, void *, const void *, const TrillimKernelContext *, bool *
) __asm__(".trillim_neon_try_packed_ffn_bf16");
#endif
static __attribute__((used)) TrillimStatus trillim_neon_try_packed_ffn_bf16(
    void *output, const void *const *inputs, void *hidden, const void *parameters,
    const TrillimKernelContext *context, bool *handled
) {
    *handled = false;
    const Parameters *p = parameters;
    if (p->tokens == 1 && (p->hidden_width < 1024 ||
        p->hidden_width > MAX_FACTORED_WIDTH || p->hidden_width % 32 != 0)) {
        return TRILLIM_OK;
    }
    const LinearParameters expand = {p->tokens, p->width, p->hidden_width, p->input_dtype, p->up_dtype, TRILLIM_FP32};
    bool expanded;
    const TrillimStatus status = p->tokens == 1
        ? trillim_neon_try_packed_ffn_int8(
            hidden, inputs[0], inputs[1], inputs[2], &expand, silu, context, &expanded)
        : trillim_neon_try_packed_ffn_int8_prefill(
            hidden, inputs[0], inputs[1], inputs[2], &expand, silu, trillim_neon_silu4_prefill, context, &expanded);
    if (status != TRILLIM_OK || !expanded) {
        return status;
    }
    const LinearParameters down = {p->tokens, p->hidden_width, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return trillim_neon_try_packed_linear_bf16(output, hidden, inputs[3], &down, context, handled);
}
#endif

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[4], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs, scratch, p, context, NULL, NULL};
    bool hidden_handled = false;
#if defined(__APPLE__) || defined(__linux__)
    const bool int8_prefill = p->tokens >= 4 && p->output_dtype == TRILLIM_FP32 &&
        p->up_dtype == p->gate_dtype && p->up_dtype == p->down_dtype &&
        (p->up_dtype == TRILLIM_DUBGRP_INT6 || p->up_dtype == TRILLIM_GRP_INT4 || p->up_dtype == TRILLIM_GRP_INT8);
    if (scratch_size != 0 && p->up_dtype == p->gate_dtype && (p->tokens == 1 || int8_prefill)) {
        LinearParameters expand = {p->tokens, p->width, p->hidden_width, p->input_dtype, p->up_dtype, TRILLIM_FP32};
        status = p->tokens == 1
            ? trillim_neon_try_packed_ffn_int8(
                task.hidden, inputs[0], inputs[1], inputs[2], &expand, silu, context, &hidden_handled)
            : trillim_neon_try_packed_ffn_int8_prefill(
                task.hidden, inputs[0], inputs[1], inputs[2], &expand, silu,
                trillim_neon_silu4_prefill, context, &hidden_handled);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (int8_prefill && hidden_handled) {
            LinearParameters down = {p->tokens, p->hidden_width, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
            const void *down_inputs[] = {scratch, inputs[3]};
            return trillim_neon_packed_linear(outputs, down_inputs, NULL, &down, context);
        }
    }
#endif
#if defined(__APPLE__)
    bool prefill;
    status = try_prefill(&task, scratch_size, &prefill);
    if (status != TRILLIM_OK || prefill) {
        return status;
    }
#endif
    if (scratch_size != 0 && !hidden_handled) {
        size_t count = p->tokens * p->hidden_width;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        TrillimTask operation = project_hidden;
        bool try_fhm = false;
        if (p->width != 0 && p->width % 32 == 0 && p->gate_dtype == p->up_dtype &&
            packed_group_dtype(p->up_dtype)) {
            bool factored = false;
            if (p->tokens == 1 && (p->input_dtype == TRILLIM_FP16 || p->input_dtype == TRILLIM_BF16) &&
                p->width >= 1024 && p->width <= MAX_FACTORED_WIDTH) {
                status = bounded_float16(inputs[0], p->width, p->input_dtype, MAX_GROUP_EXPONENT, context, &factored);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            operation = project_hidden_groups_operation(p->up_dtype, factored);
            tiles = group_tiles(p->tokens, p->hidden_width);
            try_fhm = factored && p->up_dtype != TRILLIM_GRP_INT8 &&
                p->width <= MAX_FHM_WIDTH &&
                p->hidden_width > 1024 && available_fhm() && default_fpcr() &&
                (p->up_dtype == TRILLIM_GRP_INT4 || available_half_conversion());
        }
        status = run_hidden(&task, tiles, operation, try_fhm);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (p->tokens == 1 && p->output_dtype == TRILLIM_FP32 && p->hidden_width >= 1024 &&
        p->hidden_width <= MAX_FACTORED_WIDTH && p->hidden_width % 32 == 0 && packed_group_dtype(p->down_dtype)) {
        LinearParameters down = {1, p->hidden_width, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
        const void *down_inputs[] = {scratch, inputs[3]};
        return trillim_neon_packed_linear(outputs, down_inputs, NULL, &down, context);
    }
    size_t count = p->tokens * p->outputs;
    size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
    TrillimTask operation = project;
    if (p->hidden_width != 0 && p->hidden_width % 32 == 0 && packed_group_dtype(p->down_dtype)) {
        operation = project_down_groups_operation(p->down_dtype);
        tiles = group_tiles(p->tokens, p->outputs);
    }
    return tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = up, gate, down\n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if any(matrix.dtype not in packed for matrix in matrices):\n"
           "        raise ValueError('packed FFN requires compressed weights')\n"
           "    if gate.shape != up.shape or x.shape[-1] != up.shape[1] or down.shape[1] != up.shape[0]:\n"
           "        raise ValueError('incompatible packed FFN dimensions')\n"
           "    if output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=down.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        hidden_width=up.shape[0],\n"
           "        gate_dtype=_DTYPE_IDS[gate.dtype],\n"
           "        up_dtype=_DTYPE_IDS[up.dtype],\n"
           "        down_dtype=_DTYPE_IDS[down.dtype],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is not Silu:\n"
           "        raise ValueError('packed FFN requires an unbiased SiLU gate')\n"
           "    result = FFN(x, up=up, gate=gate, down=down)\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_ffn_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
