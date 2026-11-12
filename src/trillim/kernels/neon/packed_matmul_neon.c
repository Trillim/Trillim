#include "common.h"
#include <math.h>
#include <string.h>
#include <arm_neon.h>

typedef struct {
    size_t batches;
    size_t m;
    size_t n;
    size_t k;
    size_t rank;
    const size_t *batch_shape;
    const size_t *a_batch_strides;
    const size_t *b_batch_strides;
    size_t a_row_stride;
    size_t a_col_stride;
    size_t b_row_stride;
    size_t b_col_stride;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t out_dtype;
    size_t a_storage_rows;
    size_t a_storage_columns;
    size_t b_storage_rows;
    size_t b_storage_columns;
} TrillimMatmulParameters;

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

typedef TrillimMatmulParameters Parameters;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *weight;
    const Parameters *p;
    size_t columns;
    const TrillimKernelContext *context;
} Task;

static inline bool trillim_matrix_span(size_t base, size_t rows, size_t columns, size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - base) / row_stride) {
        return false;
    }
    base += (rows - 1) * row_stride;
    if (col_stride != 0 && columns - 1 > (SIZE_MAX - base) / col_stride) {
        return false;
    }
    return base + (columns - 1) * col_stride <= capacity;
}

static inline TrillimStatus trillim_matmul_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    size_t bytes = trillim_itemsize(p->out_dtype);
    size_t a_span, b_span;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->out_dtype) ||
        p->b_dtype < TRILLIM_ROW_INT8 || p->b_dtype > TRILLIM_MXFP4 ||
        !p->k || p->k % 16 || p->a_col_stride != 1 || p->b_row_stride != 1 ||
        p->b_col_stride != p->k || p->b_storage_columns % p->k ||
        bytes == 0 || !trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span)) {
        return TRILLIM_ERROR;
    }
    for (size_t i = 0; i < p->rank; ++i) {
        if (p->b_batch_strides[i] % p->k) {
            return TRILLIM_ERROR;
        }
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if (p->a_storage_columns == 0 || p->b_storage_columns == 0 ||
            p->a_storage_rows > SIZE_MAX / p->a_storage_columns ||
            p->b_storage_rows > SIZE_MAX / p->b_storage_columns ||
            !trillim_matrix_span(a_span, p->m, p->k, p->a_row_stride, p->a_col_stride, p->a_storage_rows * p->a_storage_columns) ||
            !trillim_matrix_span(b_span, p->k, p->n, p->b_row_stride, p->b_col_stride, p->b_storage_rows * p->b_storage_columns) ||
            !trillim_payload_size(p->a_storage_rows, p->a_storage_columns, p->a_dtype, &inputs[0]) ||
            !trillim_payload_size(p->b_storage_rows, p->b_storage_columns, p->b_dtype, &inputs[1])) {
            return TRILLIM_ERROR;
        }
    }
    outputs[0] = p->batches * p->m * p->n * bytes;
    return TRILLIM_OK;
}

static size_t group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
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
    size_t groups = p->b_storage_columns / group + (p->b_storage_columns % group != 0);
    size_t total = p->b_storage_rows * groups;
    size_t block = column * groups + k / group;
    size_t offset = k % group;
    int8x16_t codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        memcpy(&bits, values + p->b_storage_rows * p->b_storage_columns + column * 2, 2);
        scale = trillim_bf16(bits);
        codes = vld1q_s8((const int8_t *)values + column * p->b_storage_columns + k);
    } else if (dtype == TRILLIM_MXFP4) {
        uint8x8_t packed = vld1_u8(values + block * 16 + offset / 2);
        uint8x8x2_t split = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
        uint8x16_t code = vcombine_u8(split.val[0], split.val[1]);
        static const uint8_t magnitudes[16] = {0, 1, 2, 3, 4, 6, 8, 12};
        uint8x16_t magnitude = vqtbl1q_u8(vld1q_u8(magnitudes), vandq_u8(code, vdupq_n_u8(7)));
        uint8_t exponent = values[total * 16 + block];
        uint32_t scale_bits = exponent == 255 ? UINT32_C(0x7fc00000) :
            (exponent > 1 ? (uint32_t)(exponent - 1) << 23 : UINT32_C(0x00200000) << exponent);
        memcpy(&scale, &scale_bits, sizeof(scale));
        int16x8_t low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(magnitude)));
        int16x8_t high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(magnitude)));
        uint16x8_t sign0 = vshll_n_u8(vget_low_u8(vandq_u8(code, vdupq_n_u8(8))), 12);
        uint16x8_t sign1 = vshll_n_u8(vget_high_u8(vandq_u8(code, vdupq_n_u8(8))), 12);
        result[0] = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(convert4(vget_low_s16(low), scale)), vshll_n_u16(vget_low_u16(sign0), 16)));
        result[1] = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(convert4(vget_high_s16(low), scale)), vshll_n_u16(vget_high_u16(sign0), 16)));
        result[2] = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(convert4(vget_low_s16(high), scale)), vshll_n_u16(vget_low_u16(sign1), 16)));
        result[3] = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(convert4(vget_high_s16(high), scale)), vshll_n_u16(vget_high_u16(sign1), 16)));
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

#define DEFINE_PROJECT(NAME, DTYPE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const Parameters *p = task->p; \
    size_t tiles = p->n / task->columns + (p->n % task->columns != 0); \
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0); \
    size_t batch = tile / (tiles * row_tiles); \
    size_t local = tile % (tiles * row_tiles); \
    size_t row = local / tiles * 4, first = local % tiles * task->columns; \
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides); \
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides); \
    size_t rows = p->m - row < 4 ? p->m - row : 4; \
    size_t end = p->n - first < task->columns ? p->n : first + task->columns; \
    for (size_t column = first; column < end; column += 2) { \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        size_t second = column + 1 < end ? column + 1 : column; \
        size_t b0 = b_base + column * p->b_col_stride; \
        size_t b1 = b_base + second * p->b_col_stride; \
        size_t r0 = b0 / p->b_storage_columns, c0 = b0 % p->b_storage_columns; \
        size_t r1 = b1 / p->b_storage_columns, c1 = b1 % p->b_storage_columns; \
        float32x4_t s00 = vdupq_n_f32(0), s01 = s00, s10 = s00, s11 = s00; \
        float32x4_t s20 = s00, s21 = s00, s30 = s00, s31 = s00; \
        size_t k = 0; \
        for (; k + 16 <= p->k; k += 16) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t w0[4], w1[4]; \
            decode16(task, r0, c0 + k, DTYPE, w0); \
            decode16(task, r1, c1 + k, DTYPE, w1); \
            for (size_t j = 0; j < 4; ++j) { \
                float32x4_t x = load4(task->input, a_base + row * p->a_row_stride + k + j * 4, p->a_dtype); \
                s00 = vfmaq_f32(s00, x, w0[j]); \
                s01 = vfmaq_f32(s01, x, w1[j]); \
                if (rows > 1) { \
                    x = load4(task->input, a_base + (row + 1) * p->a_row_stride + k + j * 4, p->a_dtype); \
                    s10 = vfmaq_f32(s10, x, w0[j]); \
                    s11 = vfmaq_f32(s11, x, w1[j]); \
                } \
                if (rows > 2) { \
                    x = load4(task->input, a_base + (row + 2) * p->a_row_stride + k + j * 4, p->a_dtype); \
                    s20 = vfmaq_f32(s20, x, w0[j]); \
                    s21 = vfmaq_f32(s21, x, w1[j]); \
                } \
                if (rows > 3) { \
                    x = load4(task->input, a_base + (row + 3) * p->a_row_stride + k + j * 4, p->a_dtype); \
                    s30 = vfmaq_f32(s30, x, w0[j]); \
                    s31 = vfmaq_f32(s31, x, w1[j]); \
                } \
            } \
        } \
        float sums[4][2] = {{vaddvq_f32(s00), vaddvq_f32(s01)}, {vaddvq_f32(s10), vaddvq_f32(s11)}, \
            {vaddvq_f32(s20), vaddvq_f32(s21)}, {vaddvq_f32(s30), vaddvq_f32(s31)}}; \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue value = {.f = sums[r][0]}; \
            store_result(task->output, (batch * p->m + row + r) * p->n + column, p->out_dtype, value); \
            if (second != column) { \
                value.f = sums[r][1]; \
                store_result(task->output, (batch * p->m + row + r) * p->n + second, p->out_dtype, value); \
            } \
        } \
    } \
    return TRILLIM_OK; \
}

DEFINE_PROJECT(project_row_int8, TRILLIM_ROW_INT8)
DEFINE_PROJECT(project_mat_ternary, TRILLIM_MAT_TERNARY)
DEFINE_PROJECT(project_grp_int8, TRILLIM_GRP_INT8)
DEFINE_PROJECT(project_grp_int4, TRILLIM_GRP_INT4)
DEFINE_PROJECT(project_grp_int5, TRILLIM_GRP_INT5)
DEFINE_PROJECT(project_dubgrp_int6, TRILLIM_DUBGRP_INT6)
DEFINE_PROJECT(project_grp_binary, TRILLIM_GRP_BINARY)
DEFINE_PROJECT(project_grp_ternary, TRILLIM_GRP_TERNARY)
DEFINE_PROJECT(project_mxfp4, TRILLIM_MXFP4)

#undef DEFINE_PROJECT

static TrillimTask projection(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_ROW_INT8: return project_row_int8;
    case TRILLIM_MAT_TERNARY: return project_mat_ternary;
    case TRILLIM_GRP_INT8: return project_grp_int8;
    case TRILLIM_GRP_INT4: return project_grp_int4;
    case TRILLIM_GRP_INT5: return project_grp_int5;
    case TRILLIM_DUBGRP_INT6: return project_dubgrp_int6;
    case TRILLIM_GRP_BINARY: return project_grp_binary;
    case TRILLIM_GRP_TERNARY: return project_grp_ternary;
    case TRILLIM_MXFP4: return project_mxfp4;
    default: return NULL;
    }
}

static TrillimStatus project_ordered(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->n / task->columns + (p->n % task->columns != 0);
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0);
    size_t batch = tile / (tiles * row_tiles);
    size_t local = tile % (tiles * row_tiles);
    size_t row = local / tiles * 4, first = local % tiles * task->columns;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides);
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    size_t rows = p->m - row < 4 ? p->m - row : 4;
    size_t end = p->n - first < task->columns ? p->n : first + task->columns;
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        size_t b0 = b_base + column * p->b_col_stride;
        size_t b1 = b_base + second * p->b_col_stride;
        size_t r0 = b0 / p->b_storage_columns, c0 = b0 % p->b_storage_columns;
        size_t r1 = b1 / p->b_storage_columns, c1 = b1 % p->b_storage_columns;
        float sums[4][2] = {{0}};
        size_t k = 0;
        for (; k + 16 <= p->k; k += 16) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a[4], b[4];
            float w0[16], w1[16];
            decode16(task, r0, c0 + k, p->b_dtype, a);
            decode16(task, r1, c1 + k, p->b_dtype, b);
            for (size_t j = 0; j < 4; ++j) {
                vst1q_f32(w0 + j * 4, a[j]);
                vst1q_f32(w1 + j * 4, b[j]);
            }
            for (size_t r = 0; r < rows; ++r) {
                float input[16];
                for (size_t j = 0; j < 4; ++j) {
                    vst1q_f32(input + j * 4,
                        load4(task->input, a_base + (row + r) * p->a_row_stride + k + j * 4, p->a_dtype));
                }
                for (size_t j = 0; j < 16; ++j) {
                    sums[r][0] = fmaf(input[j], w0[j], sums[r][0]);
                    sums[r][1] = fmaf(input[j], w1[j], sums[r][1]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue value = {.f = sums[r][0]};
            store_result(task->output, (batch * p->m + row + r) * p->n + column, p->out_dtype, value);
            if (second != column) {
                value.f = sums[r][1];
                store_result(task->output, (batch * p->m + row + r) * p->n + second, p->out_dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->batches || !p->m || !p->n) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t columns = p->n / workers / 2;
    columns = columns / 2 * 2;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, columns, context};
    TrillimTask operation = p->out_dtype == TRILLIM_FP32 ? projection(p->b_dtype) : project_ordered;
    size_t rows = p->m / 4 + (p->m % 4 != 0);
    size_t tiles = p->n / columns + (p->n % columns != 0);
    if (p->k <= 65536 / (p->batches * p->m * p->n)) {
        task.columns = p->n;
        for (size_t tile = 0; tile < p->batches * rows; ++tile) {
            TrillimStatus status = operation(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, p->batches * rows * tiles, operation, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _matmul\n"
           "def bind(c):\n"
           "    if c.inputs['a'].dtype not in {'fp16', 'bf16', 'fp32'} or c.inputs['b'].dtype not in {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('packed matmul requires floating activations and compressed weights')\n"
           "    p = _matmul(c)\n"
           "    width = p['k'].value\n"
           "    columns = p['b_storage_columns'].value\n"
           "    if width is None or width <= 0 or width % 16 or columns is None or columns <= 0 or columns % width:\n"
           "        raise ValueError('packed matmul requires aligned fixed-width rows')\n"
           "    if p['a_col_stride'].value != 1 or p['b_row_stride'].value != 1 or p['b_col_stride'] != p['k']:\n"
           "        raise ValueError('packed matmul requires contiguous reduction axes and output-major weights')\n"
           "    def aligned(stride):\n"
           "        if stride.value is not None: return stride.value % width == 0\n"
           "        if stride.op == 'broadcast_stride': return aligned(stride.args[1])\n"
           "        if stride.op == 'mul': return any(aligned(part) for part in stride.args)\n"
           "        return False\n"
           "    if any(not aligned(stride) for stride in p['b_batch_strides']):\n"
           "        raise ValueError('packed matmul batch strides must preserve complete weight vectors')\n"
           "    return p\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(TrillimMatmulParameters, batches), NULL},
        {"m", "size_t", offsetof(TrillimMatmulParameters, m), NULL},
        {"n", "size_t", offsetof(TrillimMatmulParameters, n), NULL},
        {"k", "size_t", offsetof(TrillimMatmulParameters, k), NULL},
        {"rank", "size_t", offsetof(TrillimMatmulParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(TrillimMatmulParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(TrillimMatmulParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(TrillimMatmulParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(TrillimMatmulParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(TrillimMatmulParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(TrillimMatmulParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(TrillimMatmulParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(TrillimMatmulParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(TrillimMatmulParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(TrillimMatmulParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(TrillimMatmulParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(TrillimMatmulParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(TrillimMatmulParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(TrillimMatmulParameters, b_storage_columns), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimMatmulParameters), TRILLIM_ALIGNOF(TrillimMatmulParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, trillim_matmul_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
