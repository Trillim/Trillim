#include "common.h"
#include <arm_neon.h>
#include <float.h>

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

static inline float trillim_packed_round_even(float value) {
    float integer;
    float fraction = fabsf(modff(value, &integer));
    if (fraction > 0.5f || (fraction == 0.5f && fmodf(integer, 2.0f) != 0.0f)) {
        integer += copysignf(1.0f, value);
    }
    return integer;
}

static inline int32_t trillim_packed_quantize(float value, float scale, int32_t bound) {
    float normalized = scale == 0.0f ? value : value / scale;
    normalized = fminf((float)(bound - 1), fmaxf((float)-bound, normalized));
    return (int32_t)trillim_packed_round_even(normalized);
}

static float trillim_packed_absolute_sum128(const float *input) {
    float lanes[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        lanes[lane] = fabsf(input[lane]);
    }
    for (size_t i = 8; i < 128; i += 8) {
        for (size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] += fabsf(input[i + lane]);
        }
    }
    return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
           ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
}

static float trillim_packed_max_abs16(const float *input) {
    float maximum = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        maximum = fmaxf(maximum, fabsf(input[i]));
    }
    return maximum;
}

static float trillim_packed_encode_int6(const float *input, int8_t *subscales, uint8_t *codes) {
    float ideal[16];
    size_t anchor = 0;
    for (size_t sub = 0; sub < 16; ++sub) {
        ideal[sub] = trillim_packed_max_abs16(input + sub * 16) / 31.0f;
        if (ideal[sub] > ideal[anchor]) {
            anchor = sub;
        }
    }
    float master = trillim_bf16(trillim_to_bf16(ideal[anchor] / 127.0f));
    for (size_t sub = 0; sub < 16; ++sub) {
        const float *chunk = input + sub * 16;
        float ratio = master == 0.0f ? ideal[sub] : ideal[sub] / master;
        int32_t subscale = (int32_t)fminf(127.0f, trillim_packed_round_even(ratio));
        bool changed = false;
        for (size_t i = 0; i < 16; ++i) {
            int32_t code = trillim_packed_quantize(chunk[i], master * (float)subscale, 32);
            codes[sub * 16 + i] = (uint8_t)(code + 32);
            changed |= fabsf((float)(code * subscale) * master - chunk[i]) > 1e-5f;
        }
        subscales[sub] = (int8_t)subscale;
        if (!changed || sub == anchor) {
            continue;
        }
        int32_t products[16];
        bool on_lattice = true;
        for (size_t i = 0; i < 16; ++i) {
            float ratio_value = master == 0.0f ? chunk[i] : chunk[i] / master;
            products[i] = (int32_t)trillim_packed_round_even(ratio_value);
            on_lattice &= fabsf((float)products[i] * master - chunk[i]) <= 1e-5f;
        }
        if (!on_lattice) {
            continue;
        }
        /* Keep values already representable by a different subscale. */
        int32_t selected = 0;
        int32_t best_distance = 128;
        for (int32_t candidate = 1; candidate <= 127; ++candidate) {
            int32_t distance = candidate - subscale;
            distance = distance < 0 ? -distance : distance;
            if (distance >= best_distance) {
                continue;
            }
            bool valid = true;
            for (size_t i = 0; i < 16; ++i) {
                if (products[i] < -32 * candidate || products[i] > 31 * candidate ||
                    products[i] % candidate != 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                selected = candidate;
                best_distance = distance;
            }
        }
        if (selected != 0) {
            subscales[sub] = (int8_t)selected;
            for (size_t i = 0; i < 16; ++i) {
                codes[sub * 16 + i] = (uint8_t)(products[i] / selected + 32);
            }
        }
    }
    return master;
}

typedef struct {
    const float *source;
    size_t rows;
    size_t columns;
    uint32_t dtype;
    uint8_t *output;
    const TrillimKernelContext *context;
    size_t group_width;
    size_t groups;
    size_t value_width;
    float gamma;
} TrillimPackedEncodeTask;

static TrillimStatus trillim_packed_encode_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimPackedEncodeTask *task = opaque;
    const TrillimKernelContext *context = task->context;
    size_t rows = task->rows, columns = task->columns;
    const float *source = task->source;
    uint8_t *values = task->output;
    float maximum = 1e-5f;
    for (size_t column = 0; column < columns; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum = fmaxf(maximum, fabsf(source[row * columns + column]));
    }
    uint16_t scale = trillim_to_bf16(maximum / 127.0f);
    memcpy(values + rows * columns + row * 2, &scale, 2);
    float multiplier = 127.0f / maximum;
    for (size_t column = 0; column < columns; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_packed_round_even(source[row * columns + column] * multiplier);
        int8_t code = (int8_t)fminf(127.0f, fmaxf(-127.0f, q));
        memcpy(values + row * columns + column, &code, 1);
    }
    return TRILLIM_OK;
}

static void trillim_packed_encode_block(const TrillimPackedEncodeTask *task, size_t block) {
    size_t columns = task->columns, group = task->group_width, value_width = task->value_width;
    size_t total = task->rows * task->groups;
    size_t row = block / task->groups;
    size_t start = block % task->groups * group;
    size_t valid = columns - start < group ? columns - start : group;
    uint32_t dtype = task->dtype;
    const float *source = task->source;
    uint8_t *values = task->output;
    memset(values + block * value_width, 0, value_width);
    if (dtype == TRILLIM_GRP_INT5) {
        memset(values + total * (16 + 2) + block * 4, 0, 4);
    } else if (dtype == TRILLIM_DUBGRP_INT6) {
        memset(values + total * (128 + 2) + block * 64, 0, 64);
    }
    float input[256];
    float minimum = 0.0f, positive = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        input[i] = i < valid ? source[row * columns + start + i] : 0.0f;
        minimum = fminf(minimum, input[i]);
        positive = fmaxf(positive, input[i]);
    }
    positive = fabsf(positive);
    float negative = fabsf(minimum);
    float maximum = fmaxf(positive, negative);
    float scale = maximum;
    int32_t bound = dtype == TRILLIM_GRP_INT4 ? 8 :
        (dtype == TRILLIM_GRP_INT5 ? 16 : 128);
    uint8_t int6_codes[256];
    int exponent = 0;
    if (dtype == TRILLIM_GRP_INT8) {
        scale = maximum / 127.0f;
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        float ideal = fmaxf(positive / (float)(bound - 1), negative / (float)bound);
        scale = trillim_bf16(trillim_to_bf16(ideal));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        scale = trillim_packed_absolute_sum128(input) / (float)valid;
    } else if (dtype == TRILLIM_DUBGRP_INT6) {
        int8_t subscales[16];
        scale = trillim_packed_encode_int6(input, subscales, int6_codes);
        memcpy(values + total * (128 + 2 + 64) + block * 16, subscales, sizeof(subscales));
    } else if (dtype == TRILLIM_MXFP4) {
        if (maximum != 0.0f) {
            frexpf(maximum, &exponent);
            exponent -= 3;
            exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
        }
        values[total * 16 + block] = (uint8_t)(exponent + 127);
    }
    if (dtype != TRILLIM_MAT_TERNARY && dtype != TRILLIM_MXFP4) {
        uint16_t bits = trillim_to_bf16(scale);
        memcpy(values + total * value_width + block * 2, &bits, 2);
    }
    for (size_t i = 0; i < group; ++i) {
        if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
            float magnitude = dtype == TRILLIM_MAT_TERNARY ? task->gamma : maximum;
            float normalized = magnitude == 0.0f ? 0.0f : input[i] / magnitude;
            int32_t code;
            if (dtype == TRILLIM_MAT_TERNARY) {
                code = (int32_t)trillim_packed_round_even(fminf(1.0f, fmaxf(-1.0f, normalized))) + 1;
            } else {
                code = normalized >= 0.5f ? 2 : (normalized <= -0.5f ? 0 : 1);
            }
            values[block * 32 + i % 32] |= (uint8_t)(code << (6 - (i / 32) * 2));
        } else if (dtype == TRILLIM_GRP_BINARY) {
            values[block * 16 + i / 8] |= (uint8_t)((input[i] >= 0.0f) << (7 - i % 8));
        } else if (dtype == TRILLIM_MXFP4) {
            static const float midpoints[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
            float magnitude = fabsf(ldexpf(input[i], -exponent));
            uint8_t code = 0;
            while (code < 7 && magnitude > midpoints[code]) {
                ++code;
            }
            if (code < 7 && magnitude == midpoints[code] && (code & 1)) {
                ++code;
            }
            code |= signbit(input[i]) ? 8 : 0;
            values[block * 16 + i / 2] |= (uint8_t)(code << ((i % 2) * 4));
        } else if (dtype == TRILLIM_DUBGRP_INT6) {
            uint8_t code = int6_codes[i];
            size_t half = i / 128;
            size_t lane = (i % 128) / 32;
            values[block * 128 + half * 64 + (lane / 2) * 32 + i % 32] |= (uint8_t)((code & 15) << ((lane % 2) * 4));
            values[total * (128 + 2) + block * 64 + half * 32 + i % 32] |= (uint8_t)((code >> 4) << (lane * 2));
        } else {
            int32_t quantized = trillim_packed_quantize(input[i], scale, bound);
            if (dtype == TRILLIM_GRP_INT8) {
                int8_t code = (int8_t)(quantized < -127 ? -127 : quantized);
                memcpy(values + block * 32 + i, &code, 1);
            } else {
                uint8_t code = (uint8_t)(quantized + bound);
                values[block * 16 + i % 16] |= (uint8_t)((code & 15) << ((i / 16) * 4));
                if (dtype == TRILLIM_GRP_INT5) {
                    values[total * (16 + 2) + block * 4 + i / 8] |= (uint8_t)((code >> 4) << (i % 8));
                }
            }
        }
    }
}

static TrillimStatus trillim_packed_encode_blocks(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimPackedEncodeTask *task = opaque;
    size_t count = task->rows * task->groups, first = tile * 4;
    size_t end = count - first < 4 ? count : first + 4;
    for (size_t block = first; block < end; ++block) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_packed_encode_block(task, block);
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_packed_encode_quant(
    const float *source, size_t rows, size_t columns, uint32_t dtype,
    void *output, const TrillimKernelContext *context
) {
    TrillimPackedEncodeTask task = {
        .source = source, .rows = rows, .columns = columns,
        .dtype = dtype, .output = output, .context = context
    };
    TrillimTask operation = trillim_packed_encode_row;
    size_t tiles = rows;
    if (dtype != TRILLIM_ROW_INT8) {
        task.group_width = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
            ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY ||
              dtype == TRILLIM_GRP_BINARY) ? 128 : 32);
        task.groups = columns / task.group_width + (columns % task.group_width != 0);
        size_t count = rows * task.groups;
        task.value_width = dtype == TRILLIM_GRP_INT8 || dtype == TRILLIM_MAT_TERNARY ||
            dtype == TRILLIM_GRP_TERNARY ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
        if (dtype == TRILLIM_MAT_TERNARY) {
            size_t elements = rows * columns;
            for (size_t start = 0; start < elements;) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = elements - start < 1024 ? elements : start + 1024;
                for (; start < end; ++start) {
                    task.gamma = fmaxf(task.gamma, fabsf(source[start]));
                }
            }
            memcpy(task.output + count * 32, &task.gamma, sizeof(task.gamma));
        }
        tiles = count / 4 + (count % 4 != 0);
        operation = trillim_packed_encode_blocks;
    }
    return tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
}

typedef struct {
    size_t rows;
    size_t width;
    size_t tokens;
    size_t output_rows;
    size_t output_columns;
    uint32_t dtype;
    uint32_t index_dtype;
} Parameters;

typedef struct {
    void *destination;
    const uint8_t *weight;
    const void *indices;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static bool index_dtype(uint32_t dtype) {
    return dtype == TRILLIM_INT32 || dtype == TRILLIM_UINT32 ||
           dtype == TRILLIM_INT64 || dtype == TRILLIM_UINT64;
}

static bool copyable_mxfp4(const Parameters *p) {
    return p->dtype == TRILLIM_MXFP4 && p->width % 32 == 0 && p->output_columns % 32 == 0;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0 ||
        p->dtype < TRILLIM_ROW_INT8 || p->dtype > TRILLIM_MXFP4 ||
        !index_dtype(p->index_dtype) ||
        !trillim_packed_payload_size(p->rows, p->width, p->dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t id_size = trillim_itemsize(p->index_dtype);
    if (p->tokens > SIZE_MAX / id_size || p->tokens > SIZE_MAX / p->width ||
        !p->output_columns || p->output_rows > SIZE_MAX / p->output_columns ||
        p->output_rows * p->output_columns != p->tokens * p->width ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->tokens * id_size;
    if (!copyable_mxfp4(p)) {
        if (p->tokens * p->width > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->tokens * p->width * sizeof(float);
    }
    return TRILLIM_OK;
}

static size_t tiles_per_row(size_t width) {
    return width / 1024 + (width % 1024 != 0);
}

static uint64_t row_index(const Task *task, size_t token) {
    uint32_t dtype = task->parameters->index_dtype;
    TrillimValue coordinate = trillim_load(task->indices, token, dtype);
    return trillim_unsigned(dtype) ? coordinate.u : (uint64_t)coordinate.i;
}

enum { MXFP4_GROUPS_PER_TILE = 32768 };

static size_t mxfp4_tiles_per_row(size_t groups) {
    return groups / MXFP4_GROUPS_PER_TILE + (groups % MXFP4_GROUPS_PER_TILE != 0);
}

static TrillimStatus copy_mxfp4(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t groups = p->width / 32;
    size_t tiles = mxfp4_tiles_per_row(groups);
    size_t token = tile / tiles;
    uint64_t row = row_index(task, token);
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t first = tile % tiles * MXFP4_GROUPS_PER_TILE;
    size_t count = groups - first < MXFP4_GROUPS_PER_TILE ? groups - first : MXFP4_GROUPS_PER_TILE;
    size_t source = (size_t)row * groups + first;
    size_t destination = token * groups + first;
    const uint8_t *codes = task->weight + source * 16;
    const uint8_t *scales = task->weight + p->rows * groups * 16 + source;

    uint8x16_t vector_maximum = vdupq_n_u8(0);
    size_t i = 0;
    for (; count - i >= 16; i += 16) {
        vector_maximum = vmaxq_u8(vector_maximum, vld1q_u8(scales + i));
    }
    uint8_t maximum = vmaxvq_u8(vector_maximum);
    for (; i < count; ++i) {
        if (scales[i] > maximum) {
            maximum = scales[i];
        }
    }
    if (maximum >= 253) {
        for (i = 0; i < count; ++i) {
            if ((i & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t exponent = scales[i];
            if (exponent == 255) {
                return TRILLIM_ERROR;
            }
            if (exponent < 253) {
                continue;
            }
            /* The largest magnitudes overflow FP32 at these two exponents. */
            uint8x16_t packed = vld1q_u8(codes + i * 16);
            uint8x16_t mask = vdupq_n_u8(7);
            uint8x16_t low = vandq_u8(packed, mask);
            uint8x16_t high = vandq_u8(vshrq_n_u8(packed, 4), mask);
            uint8_t limit = exponent == 253 ? 6 : 4;
            if (vmaxvq_u8(vmaxq_u8(low, high)) >= limit) {
                return TRILLIM_ERROR;
            }
        }
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    uint8_t *output = task->destination;
    memcpy(output + destination * 16, codes, count * 16);
    memcpy(output + p->tokens * groups * 16 + destination, scales, count);
    return TRILLIM_OK;
}

static TrillimStatus gather(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = tiles_per_row(p->width);
    size_t token = tile / tiles;
    size_t column = (tile % tiles) * 1024;
    size_t end = p->width - column < 1024 ? p->width : column + 1024;
    uint64_t row = row_index(task, token);
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t source = (size_t)row * p->width;
    float *decoded = task->destination;
    size_t destination = token * p->width;
    if (end - column >= 16) {
        TrillimPackedNeonRow packed_row = trillim_packed_neon_row(
            task->weight, p->dtype, (size_t)row, p->width, p->rows
        );
        float32x4_t bound = vdupq_n_f32(FLT_MAX);
        for (; end - column >= 16; column += 16) {
            float32x4_t values[4];
            trillim_packed_neon_decode16(&packed_row, column, values);
            uint32x4_t finite = vcleq_f32(vabsq_f32(values[0]), bound);
            for (size_t lane = 1; lane < 4; ++lane) {
                finite = vandq_u32(finite, vcleq_f32(vabsq_f32(values[lane]), bound));
            }
            if (vminvq_u32(finite) == 0) {
                return TRILLIM_ERROR;
            }
            for (size_t lane = 0; lane < 4; ++lane) {
                vst1q_f32(decoded + destination + column + lane * 4, values[lane]);
            }
        }
    }
    for (; column < end; ++column) {
        float value = trillim_packed_read_float(task->weight, source + column, p->dtype, p->rows, p->width);
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        decoded[destination + column] = value;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_gather_neon(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    bool copy = copyable_mxfp4(p);
    Task task = {copy ? outputs[0] : scratch, inputs[0], inputs[1], p, context};
    size_t tiles = p->tokens * (copy ? mxfp4_tiles_per_row(p->width / 32) : tiles_per_row(p->width));
    TrillimTask operation = copy ? copy_mxfp4 : gather;
    TrillimStatus status = tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
    if (status != TRILLIM_OK || copy) {
        return status;
    }
    return trillim_packed_encode_quant(scratch, p->output_rows, p->output_columns,
                                      p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_gather_neon_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'indices')\n"
           "def bind(context):\n"
           "    weight, indices = context.inputs['x'], context.inputs['indices']\n"
           "    output = context.outputs[0]\n"
           "    if _DTYPE_IDS[weight.dtype] < _DTYPE_IDS['row_int8'] or output.dtype != weight.dtype:\n"
           "        raise ValueError('packed gather requires the same packed input and output dtype')\n"
           "    if not weight.shape:\n"
           "        raise ValueError('packed gather requires a source row axis')\n"
           "    rows, width = _storage_shape(weight)\n"
           "    if (rows, width) != (weight.shape[0], _product(weight.shape[1:])):\n"
           "        raise ValueError('packed gather requires a complete canonical source')\n"
           "    if not indices.shape or indices.shape[-1].value != 1 or indices.dtype not in ('int32', 'uint32', 'int64', 'uint64'):\n"
           "        raise ValueError('packed gather requires single integer coordinates')\n"
           "    for tensor in (weight, indices):\n"
           "        if tensor.strides and tensor.strides != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('packed gather requires contiguous inputs')\n"
           "    output_rows, output_columns = _storage_shape(output)\n"
           "    return dict(rows=rows, width=width, tokens=_product(indices.shape[:-1]), output_rows=output_rows, output_columns=output_columns, dtype=_DTYPE_IDS[weight.dtype], index_dtype=_DTYPE_IDS[indices.dtype])\n"
           "def pattern(x: Tensor, indices: Tensor, *, fill=0) -> tuple[Tensor, ...]:\n"
           "    return (Gather(x, indices=indices, fill=fill),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_gather_neon_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"output_rows", "size_t", offsetof(Parameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(Parameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"index_dtype", "uint32_t", offsetof(Parameters, index_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_gather_neon_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &capabilities;
}
