#include "common.h"
#include <arm_neon.h>
#include <math.h>

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *weight;
    const Parameters *p;
    size_t columns;
    const TrillimKernelContext *context;
} Task;

static size_t group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        p->weight_dtype < TRILLIM_ROW_INT8 || p->weight_dtype > TRILLIM_MXFP4) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t input_bytes = trillim_itemsize(p->input_dtype), output_bytes = trillim_itemsize(p->output_dtype);
    if ((p->width && p->tokens > SIZE_MAX / input_bytes / p->width) ||
        p->tokens > SIZE_MAX / output_bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * input_bytes;
    outputs[0] = p->tokens * p->outputs * output_bytes;
    if (!p->width) {
        return TRILLIM_OK;
    }
    if (p->weight_dtype == TRILLIM_ROW_INT8) {
        if (p->width > SIZE_MAX - 2 || p->outputs > SIZE_MAX / (p->width + 2)) {
            return TRILLIM_ERROR;
        }
        inputs[1] = p->outputs * (p->width + 2);
        return TRILLIM_OK;
    }
    static const size_t records[] = {32, 34, 18, 22, 210, 18, 34, 17};
    size_t group = group_width(p->weight_dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    size_t record = records[p->weight_dtype - TRILLIM_MAT_TERNARY];
    size_t extra = p->weight_dtype == TRILLIM_MAT_TERNARY ? sizeof(float) : 0;
    if (groups > SIZE_MAX / record || p->outputs > (SIZE_MAX - extra) / (groups * record)) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->outputs * groups * record + extra;
    return TRILLIM_OK;
}

static float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_BF16) {
        return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(bits));
}

static float32x4_t convert4(int16x4_t codes, float scale) {
    return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(codes)), scale);
}

static inline __attribute__((always_inline)) void decode16(
    const Task *task, size_t column, size_t k, uint32_t dtype, float32x4_t result[4]
) {
    const Parameters *p = task->p;
    const uint8_t *values = task->weight;
    size_t group = group_width(dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    size_t total = p->outputs * groups;
    size_t block = column * groups + k / group;
    size_t offset = k % group;
    int8x16_t codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        memcpy(&bits, values + p->outputs * p->width + column * 2, 2);
        scale = trillim_bf16(bits);
        codes = vld1q_s8((const int8_t *)values + column * p->width + k);
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
        uint8_t exponent = values[total * 16 + block];
        if (exponent == 255) {
            for (size_t i = 0; i < 4; ++i) {
                uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(result[i]), vdupq_n_u32(UINT32_C(0x80000000)));
                result[i] = vreinterpretq_f32_u32(vorrq_u32(sign, vdupq_n_u32(UINT32_C(0x7fc00000))));
            }
            return;
        }
        uint32_t scale_bits = exponent ? (uint32_t)exponent << 23 : UINT32_C(0x00400000);
        memcpy(&scale, &scale_bits, sizeof(scale));
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vmulq_n_f32(result[i], scale);
        }
        return;
    } else if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        uint8x16_t packed = vld1q_u8(values + block * 32 + offset % 32);
        uint8x16_t shifted = vshlq_u8(packed, vdupq_n_s8(-(int8_t)(6 - (offset / 32) * 2)));
        codes = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(shifted, vdupq_n_u8(3))), vdupq_n_s8(1));
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
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
            uint16_t packed;
            memcpy(&packed, values + block * 16 + offset / 8, 2);
            uint8x16_t bytes = vcombine_u8(vdup_n_u8((uint8_t)packed), vdup_n_u8((uint8_t)(packed >> 8)));
            static const int8_t shifts[16] = {-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0};
            uint8x16_t code = vandq_u8(vshlq_u8(bytes, vld1q_s8(shifts)), vdupq_n_u8(1));
            codes = vsubq_s8(vreinterpretq_s8_u8(vshlq_n_u8(code, 1)), vdupq_n_s8(1));
        } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
            uint8x16_t packed = vld1q_u8(values + block * 16);
            uint8x16_t code = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(int8_t)((offset / 16) * 4))), vdupq_n_u8(15));
            if (dtype == TRILLIM_GRP_INT5) {
                uint16_t high;
                memcpy(&high, values + scale_start + total * 2 + block * 4 + offset / 8, 2);
                uint8x16_t bytes = vcombine_u8(vdup_n_u8((uint8_t)high), vdup_n_u8((uint8_t)(high >> 8)));
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
            result[0] = convert4(vget_low_s16(a), scale);
            result[1] = convert4(vget_high_s16(a), scale);
            result[2] = convert4(vget_low_s16(b), scale);
            result[3] = convert4(vget_high_s16(b), scale);
            return;
        }
    }
    int16x8_t low = vmovl_s8(vget_low_s8(codes)), high = vmovl_s8(vget_high_s8(codes));
    result[0] = convert4(vget_low_s16(low), scale);
    result[1] = convert4(vget_high_s16(low), scale);
    result[2] = convert4(vget_low_s16(high), scale);
    result[3] = convert4(vget_high_s16(high), scale);
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        int32x4_t code[4] = {vmovl_s16(vget_low_s16(low)), vmovl_s16(vget_high_s16(low)),
            vmovl_s16(vget_low_s16(high)), vmovl_s16(vget_high_s16(high))};
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vbslq_f32(vceqq_s32(code[i], vdupq_n_s32(2)), vdupq_n_f32(NAN), result[i]);
        }
    }
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

static inline __attribute__((always_inline)) void store_result(
    void *output, size_t index, uint32_t dtype, TrillimValue value
) {
    if (dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)output + index * 4, &value.f, sizeof(value.f));
    } else {
        uint16_t bits = dtype == TRILLIM_BF16 ? trillim_to_bf16(value.f) : trillim_to_fp16(value.f);
        memcpy((uint8_t *)output + index * 2, &bits, sizeof(bits));
    }
}

#define DEFINE_PROJECT(NAME, DTYPE, MTILE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const Parameters *p = task->p; \
    size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0); \
    size_t row = tile / tiles * MTILE, first = tile % tiles * task->columns; \
    size_t rows = p->tokens - row < MTILE ? p->tokens - row : MTILE; \
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns; \
    for (size_t column = first; column < end; column += 2) { \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        size_t second = column + 1 < end ? column + 1 : column; \
        float32x4_t s00 = vdupq_n_f32(0), s01 = s00, s10 = s00, s11 = s00; \
        float32x4_t s20 = s00, s21 = s00, s30 = s00, s31 = s00; \
        float32x4_t s40 = s00, s41 = s00, s50 = s00, s51 = s00; \
        float32x4_t s60 = s00, s61 = s00, s70 = s00, s71 = s00; \
        size_t k = 0; \
        for (; k + 16 <= p->width; k += 16) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t w0[4], w1[4]; \
            decode16(task, column, k, DTYPE, w0); \
            decode16(task, second, k, DTYPE, w1); \
            for (size_t j = 0; j < 4; ++j) { \
                float32x4_t x = load4(task->input, row * p->width + k + j * 4, p->input_dtype); \
                s00 = vfmaq_f32(s00, x, w0[j]); \
                s01 = vfmaq_f32(s01, x, w1[j]); \
                if (rows > 1) { \
                    x = load4(task->input, (row + 1) * p->width + k + j * 4, p->input_dtype); \
                    s10 = vfmaq_f32(s10, x, w0[j]); \
                    s11 = vfmaq_f32(s11, x, w1[j]); \
                } \
                if (rows > 2) { \
                    x = load4(task->input, (row + 2) * p->width + k + j * 4, p->input_dtype); \
                    s20 = vfmaq_f32(s20, x, w0[j]); \
                    s21 = vfmaq_f32(s21, x, w1[j]); \
                } \
                if (rows > 3) { \
                    x = load4(task->input, (row + 3) * p->width + k + j * 4, p->input_dtype); \
                    s30 = vfmaq_f32(s30, x, w0[j]); \
                    s31 = vfmaq_f32(s31, x, w1[j]); \
                } \
                if (rows > 4) { \
                    x = load4(task->input, (row + 4) * p->width + k + j * 4, p->input_dtype); \
                    s40 = vfmaq_f32(s40, x, w0[j]); \
                    s41 = vfmaq_f32(s41, x, w1[j]); \
                } \
                if (rows > 5) { \
                    x = load4(task->input, (row + 5) * p->width + k + j * 4, p->input_dtype); \
                    s50 = vfmaq_f32(s50, x, w0[j]); \
                    s51 = vfmaq_f32(s51, x, w1[j]); \
                } \
                if (rows > 6) { \
                    x = load4(task->input, (row + 6) * p->width + k + j * 4, p->input_dtype); \
                    s60 = vfmaq_f32(s60, x, w0[j]); \
                    s61 = vfmaq_f32(s61, x, w1[j]); \
                } \
                if (rows > 7) { \
                    x = load4(task->input, (row + 7) * p->width + k + j * 4, p->input_dtype); \
                    s70 = vfmaq_f32(s70, x, w0[j]); \
                    s71 = vfmaq_f32(s71, x, w1[j]); \
                } \
            } \
        } \
        float sums[8][2] = {{vaddvq_f32(s00), vaddvq_f32(s01)}, {vaddvq_f32(s10), vaddvq_f32(s11)}, \
            {vaddvq_f32(s20), vaddvq_f32(s21)}, {vaddvq_f32(s30), vaddvq_f32(s31)}, \
            {vaddvq_f32(s40), vaddvq_f32(s41)}, {vaddvq_f32(s50), vaddvq_f32(s51)}, \
            {vaddvq_f32(s60), vaddvq_f32(s61)}, {vaddvq_f32(s70), vaddvq_f32(s71)}}; \
        for (; k < p->width; ++k) { \
            float w0 = trillim_read_float(task->weight, column * p->width + k, p->weight_dtype, p->outputs, p->width); \
            float w1 = trillim_read_float(task->weight, second * p->width + k, p->weight_dtype, p->outputs, p->width); \
            for (size_t r = 0; r < rows; ++r) { \
                float x = trillim_load(task->input, (row + r) * p->width + k, p->input_dtype).f; \
                sums[r][0] += x * w0; \
                sums[r][1] += x * w1; \
            } \
        } \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue value = {.f = sums[r][0]}; \
            store_result(task->output, (row + r) * p->outputs + column, p->output_dtype, value); \
            if (second != column) { \
                value.f = sums[r][1]; \
                store_result(task->output, (row + r) * p->outputs + second, p->output_dtype, value); \
            } \
        } \
    } \
    return TRILLIM_OK; \
}

DEFINE_PROJECT(project_row_int8_4, TRILLIM_ROW_INT8, 4)
DEFINE_PROJECT(project_mat_ternary_4, TRILLIM_MAT_TERNARY, 4)
DEFINE_PROJECT(project_grp_int8_4, TRILLIM_GRP_INT8, 4)
DEFINE_PROJECT(project_grp_int4_4, TRILLIM_GRP_INT4, 4)
DEFINE_PROJECT(project_grp_int5_4, TRILLIM_GRP_INT5, 4)
DEFINE_PROJECT(project_dubgrp_int6_4, TRILLIM_DUBGRP_INT6, 4)
DEFINE_PROJECT(project_grp_binary_4, TRILLIM_GRP_BINARY, 4)
DEFINE_PROJECT(project_grp_ternary_4, TRILLIM_GRP_TERNARY, 4)
DEFINE_PROJECT(project_mxfp4_4, TRILLIM_MXFP4, 4)
DEFINE_PROJECT(project_mat_ternary_8, TRILLIM_MAT_TERNARY, 8)
DEFINE_PROJECT(project_grp_int5_8, TRILLIM_GRP_INT5, 8)
DEFINE_PROJECT(project_dubgrp_int6_8, TRILLIM_DUBGRP_INT6, 8)
DEFINE_PROJECT(project_grp_binary_8, TRILLIM_GRP_BINARY, 8)
DEFINE_PROJECT(project_grp_ternary_8, TRILLIM_GRP_TERNARY, 8)
DEFINE_PROJECT(project_mxfp4_8, TRILLIM_MXFP4, 8)

#undef DEFINE_PROJECT

static TrillimTask projection(uint32_t dtype, size_t rows) {
    if (rows == 8) {
        switch (dtype) {
        case TRILLIM_MAT_TERNARY: return project_mat_ternary_8;
        case TRILLIM_GRP_INT5: return project_grp_int5_8;
        case TRILLIM_DUBGRP_INT6: return project_dubgrp_int6_8;
        case TRILLIM_GRP_BINARY: return project_grp_binary_8;
        case TRILLIM_GRP_TERNARY: return project_grp_ternary_8;
        case TRILLIM_MXFP4: return project_mxfp4_8;
        default: return NULL;
        }
    }
    switch (dtype) {
    case TRILLIM_ROW_INT8: return project_row_int8_4;
    case TRILLIM_MAT_TERNARY: return project_mat_ternary_4;
    case TRILLIM_GRP_INT8: return project_grp_int8_4;
    case TRILLIM_GRP_INT4: return project_grp_int4_4;
    case TRILLIM_GRP_INT5: return project_grp_int5_4;
    case TRILLIM_DUBGRP_INT6: return project_dubgrp_int6_4;
    case TRILLIM_GRP_BINARY: return project_grp_binary_4;
    case TRILLIM_GRP_TERNARY: return project_grp_ternary_4;
    case TRILLIM_MXFP4: return project_mxfp4_4;
    default: return NULL;
    }
}

static TrillimStatus project_ordered(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0);
    size_t row = tile / tiles * 4, first = tile % tiles * task->columns;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        float sums[4][2] = {{0}};
        size_t k = 0;
        for (; k + 16 <= p->width; k += 16) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a[4], b[4];
            float w0[16], w1[16];
            decode16(task, column, k, p->weight_dtype, a);
            decode16(task, second, k, p->weight_dtype, b);
            for (size_t j = 0; j < 4; ++j) {
                vst1q_f32(w0 + j * 4, a[j]);
                vst1q_f32(w1 + j * 4, b[j]);
            }
            for (size_t r = 0; r < rows; ++r) {
                float input[16];
                for (size_t j = 0; j < 4; ++j) {
                    vst1q_f32(input + j * 4,
                        load4(task->input, (row + r) * p->width + k + j * 4, p->input_dtype));
                }
                for (size_t j = 0; j < 16; ++j) {
                    sums[r][0] = fmaf(input[j], w0[j], sums[r][0]);
                    sums[r][1] = fmaf(input[j], w1[j], sums[r][1]);
                }
            }
        }
        for (; k < p->width; ++k) {
            float w0 = trillim_read_float(task->weight, column * p->width + k, p->weight_dtype, p->outputs, p->width);
            float w1 = trillim_read_float(task->weight, second * p->width + k, p->weight_dtype, p->outputs, p->width);
            for (size_t r = 0; r < rows; ++r) {
                float x = trillim_load(task->input, (row + r) * p->width + k, p->input_dtype).f;
                sums[r][0] = fmaf(x, w0, sums[r][0]);
                sums[r][1] = fmaf(x, w1, sums[r][1]);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue value = {.f = sums[r][0]};
            store_result(task->output, (row + r) * p->outputs + column, p->output_dtype, value);
            if (second != column) {
                value.f = sums[r][1];
                store_result(task->output, (row + r) * p->outputs + second, p->output_dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t task_factor = p->tokens >= 8 ? 1 : 4;
    size_t columns = p->outputs / workers / task_factor;
    columns = columns / 2 * 2;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, columns, context};
    bool large_tile = p->tokens >= 16 && p->output_dtype == TRILLIM_FP32 &&
        (p->weight_dtype == TRILLIM_MAT_TERNARY || p->weight_dtype >= TRILLIM_GRP_INT5);
    size_t row_tile = large_tile ? 8 : 4;
    TrillimTask operation = p->output_dtype == TRILLIM_FP32 ? projection(p->weight_dtype, row_tile) : project_ordered;
    size_t rows = p->tokens / row_tile + (p->tokens % row_tile != 0);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    if (p->width <= 65536 / (p->tokens * p->outputs)) {
        task.columns = p->outputs;
        for (size_t tile = 0; tile < rows; ++tile) {
            TrillimStatus status = operation(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, rows * tiles, operation, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "def bind(context):\n"
           "    source, weight = context.inputs['x'], context.inputs['weight']\n"
           "    if source.dtype not in {'fp16', 'bf16', 'fp32'} or weight.dtype not in {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}:\n"
           "        raise ValueError('packed linear requires floating activations and compressed weights')\n"
           "    for tensor in (source, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('packed linear requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('packed linear requires a complete packed weight matrix')\n"
           "    values = _linear(context)\n"
           "    return dict(tokens=values['m'], width=values['k'], outputs=values['n'], input_dtype=values['a_dtype'], weight_dtype=values['b_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
