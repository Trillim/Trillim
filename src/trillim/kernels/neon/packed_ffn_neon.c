
#include "common.h"
#include <arm_neon.h>
#include <math.h>

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

static inline float32x4_t packed_group_scaled(int16x4_t codes, float scale) {
    return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(codes)), scale);
}

static inline void packed_group_widen(
    int8x16_t codes, bool scaled, int8_t subscale, float scale, float32x4_t result[4]
) {
    int16x8_t low = vmovl_s8(vget_low_s8(codes));
    int16x8_t high = vmovl_high_s8(codes);
    if (scaled) {
        low = vmulq_n_s16(low, subscale);
        high = vmulq_n_s16(high, subscale);
    }
    result[0] = packed_group_scaled(vget_low_s16(low), scale);
    result[1] = packed_group_scaled(vget_high_s16(low), scale);
    result[2] = packed_group_scaled(vget_low_s16(high), scale);
    result[3] = packed_group_scaled(vget_high_s16(high), scale);
}

static inline void packed_group_q4(uint8x16_t codes, float scale, float32x4_t result[4]) {
    /* The sixteen centered codes have zero low bytes in FP16. */
    static const uint8_t fp16_high[16] = {
        200, 199, 198, 197, 196, 194, 192, 188, 0, 60, 64, 66, 68, 69, 70, 71
    };
    uint8x16_t bits = vqtbl1q_u8(vld1q_u8(fp16_high), codes);
    float16x8_t lo = vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(bits), 8));
    float16x8_t hi = vreinterpretq_f16_u16(vshll_high_n_u8(bits, 8));
    result[0] = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(lo)), scale);
    result[1] = vmulq_n_f32(vcvt_high_f32_f16(lo), scale);
    result[2] = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(hi)), scale);
    result[3] = vmulq_n_f32(vcvt_high_f32_f16(hi), scale);
}

static inline uint8x16_t packed_group_q5_high(const uint8_t *high) {
    static const uint8_t masks[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
    return vandq_u8(vtstq_u8(bytes, vld1q_u8(masks)), vdupq_n_u8(16));
}

static inline __attribute__((always_inline)) void packed_group_decode32(
    const PackedGroupRow *row, size_t k, uint32_t dtype, float32x4_t result[8]
) {
    uint16_t scale_bits;
    if (dtype == TRILLIM_DUBGRP_INT6) {
        size_t block = k / 256, segment = (k % 256) / 32;
        const uint8_t *codes = row->codes + block * 128 + segment / 4 * 64 + segment % 4 / 2 * 32;
        const uint8_t *high = row->high + block * 64 + segment / 4 * 32;
        const int8_t *subscales = (const int8_t *)row->subscales + k / 16;
        memcpy(&scale_bits, row->scales + block * 2, sizeof(scale_bits));
        float scale = trillim_bf16(scale_bits);
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
        packed_group_widen(a, true, subscales[0], scale, result);
        packed_group_widen(b, true, subscales[1], scale, result + 4);
        return;
    }
    memcpy(&scale_bits, row->scales + k / 16, sizeof(scale_bits));
    float scale = trillim_bf16(scale_bits);
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)row->codes + k;
        packed_group_widen(vld1q_s8(codes), false, 1, scale, result);
        packed_group_widen(vld1q_s8(codes + 16), false, 1, scale, result + 4);
        return;
    }
    uint8x16_t packed = vld1q_u8(row->codes + k / 2);
    uint8x16_t a = vandq_u8(packed, vdupq_n_u8(15)), b = vshrq_n_u8(packed, 4);
    if (dtype == TRILLIM_GRP_INT4) {
        packed_group_q4(a, scale, result);
        packed_group_q4(b, scale, result + 4);
        return;
    }
    const uint8_t *high = row->high + k / 8;
    int8x16_t low = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(a, packed_group_q5_high(high))), vdupq_n_s8(16));
    int8x16_t upper = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(b, packed_group_q5_high(high + 2))), vdupq_n_s8(16));
    packed_group_widen(low, false, 1, scale, result);
    packed_group_widen(upper, false, 1, scale, result + 4);
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
        packed_group_decode32(first, k, dtype, aw);
        if (pair) {
            packed_group_decode32(second, k, dtype, bw);
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

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
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
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT4)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT5)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_DUBGRP_INT6)
DEFINE_GROUP_PROJECT(project_down_groups, TRILLIM_GRP_INT8)

#undef DEFINE_GROUP_PROJECT

static TrillimTask project_hidden_groups_operation(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return project_hidden_groups_TRILLIM_GRP_INT4;
    case TRILLIM_GRP_INT5: return project_hidden_groups_TRILLIM_GRP_INT5;
    case TRILLIM_DUBGRP_INT6: return project_hidden_groups_TRILLIM_DUBGRP_INT6;
    case TRILLIM_GRP_INT8: return project_hidden_groups_TRILLIM_GRP_INT8;
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
    Task task = {outputs[0], inputs, scratch, p, context};
    if (scratch_size != 0) {
        size_t count = p->tokens * p->hidden_width;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        TrillimTask operation = project_hidden;
        if (p->width != 0 && p->width % 32 == 0 && p->gate_dtype == p->up_dtype &&
            packed_group_dtype(p->up_dtype)) {
            operation = project_hidden_groups_operation(p->up_dtype);
            tiles = group_tiles(p->tokens, p->hidden_width);
        }
        status = tiles == 1 ? operation(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
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
