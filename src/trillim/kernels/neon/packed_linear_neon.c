#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Older Apple assemblers discard feature directives when compiling assembly text. */
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_fhm_v\\reg,\\reg\n"
    ".endr\n"
);
#define FHM
static inline float32x4_t fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x4e20ec00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline float32x4_t fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x6e20cc00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
/* Function targets disappear before the builder's separate assembly step. */
__asm__(".arch armv8.6-a+bf16+fp16fml");
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

/* Apple Clang 17 crashes while optimizing the BFDOT loop. */
#if !defined(__apple_build_version__) || __clang_major__ > 17
#define TRILLIM_PACKED_BFDOT 1
#if defined(__clang__)
#define BF16 __attribute__((target("bf16")))
#else
#define BF16 __attribute__((target("+bf16")))
#endif
BF16 static inline float32x4_t dot_bf16(float32x4_t sum, uint16x8_t a, uint16x8_t b) {
    return vbfdotq_f32(sum, vreinterpretq_bf16_u16(a), vreinterpretq_bf16_u16(b));
}
#endif

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

typedef struct {
    const uint8_t *codes;
    const uint8_t *scales;
    const uint8_t *high;
    const uint8_t *subscales;
} PackedGroupRow;

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
    float32x4_t value = vcvtq_f32_s32(vmovl_s16(codes));
    return factored ? value : vmulq_n_f32(value, scale);
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
    float16x8x2_t values = {{
        vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(bits), 8)),
        vreinterpretq_f16_u16(vshll_high_n_u8(bits, 8)),
    }};
    return values;
}

static inline void packed_group_q4(uint8x16_t codes, float scale, bool factored, float32x4_t result[4]) {
    float16x8x2_t values = packed_group_q4_half(codes);
    result[0] = vcvt_f32_f16(vget_low_f16(values.val[0]));
    result[1] = vcvt_high_f32_f16(values.val[0]);
    result[2] = vcvt_f32_f16(vget_low_f16(values.val[1]));
    result[3] = vcvt_high_f32_f16(values.val[1]);
    if (!factored) {
        for (size_t j = 0; j < 4; ++j) {
            result[j] = vmulq_n_f32(result[j], scale);
        }
    }
}

static inline uint8x16_t packed_group_q5_high(const uint8_t *high) {
    static const uint8_t masks[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
    return vandq_u8(vtstq_u8(bytes, vld1q_u8(masks)), vdupq_n_u8(16));
}

static inline __attribute__((always_inline)) uint8x16x2_t packed_group_codes(
    const PackedGroupRow *row, size_t k, uint32_t dtype
) {
    if (dtype == TRILLIM_GRP_INT8) {
        return vld1q_u8_x2(row->codes + k);
    }
    const uint8x16_t mask = vdupq_n_u8(15);
    if (dtype == TRILLIM_DUBGRP_INT6) {
        size_t block = k / 256, segment = (k % 256) / 32;
        const uint8_t *codes = row->codes + block * 128 + segment / 4 * 64 + segment % 4 / 2 * 32;
        const uint8_t *high = row->high + block * 64 + segment / 4 * 32;
        uint8x16x2_t packed = vld1q_u8_x2(codes);
        uint8x16x2_t upper = vld1q_u8_x2(high);
        int8x16_t shift = vdupq_n_s8((int8_t)(4 - (int)(segment % 4) * 2));
        for (size_t i = 0; i < 2; ++i) {
            uint8x16_t lo = segment & 1 ? vshrq_n_u8(packed.val[i], 4) : vandq_u8(packed.val[i], mask);
            uint8x16_t hi = vandq_u8(vshlq_u8(upper.val[i], shift), vdupq_n_u8(48));
            packed.val[i] = vorrq_u8(lo, hi);
        }
        return packed;
    }
    uint8x16_t packed = vld1q_u8(row->codes + k / 2);
    uint8x16x2_t result = {{vandq_u8(packed, mask), vshrq_n_u8(packed, 4)}};
    if (dtype == TRILLIM_GRP_INT5) {
        const uint8_t *high = row->high + k / 8;
        result.val[0] = vorrq_u8(result.val[0], packed_group_q5_high(high));
        result.val[1] = vorrq_u8(result.val[1], packed_group_q5_high(high + 2));
    }
    return result;
}

static inline float packed_group_scale(const PackedGroupRow *row, size_t k, uint32_t dtype) {
    size_t index = dtype == TRILLIM_DUBGRP_INT6 ? k / 256 : k / 32;
    uint16_t bits;
    memcpy(&bits, row->scales + index * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline __attribute__((always_inline)) void packed_group_decode32(
    const PackedGroupRow *row, size_t k, uint32_t dtype, bool factored, float32x4_t result[8]
) {
    uint8x16x2_t codes = packed_group_codes(row, k, dtype);
    float scale = factored ? 0 : packed_group_scale(row, k, dtype);
    if (dtype == TRILLIM_GRP_INT4) {
        packed_group_q4(codes.val[0], scale, factored, result);
        packed_group_q4(codes.val[1], scale, factored, result + 4);
        return;
    }
    bool has_subscale = dtype == TRILLIM_DUBGRP_INT6;
    uint8_t zero = has_subscale ? 32 : (dtype == TRILLIM_GRP_INT5 ? 16 : 0);
    const int8_t *subscales = has_subscale ? (const int8_t *)row->subscales + k / 16 : NULL;
    for (size_t i = 0; i < 2; ++i) {
        int8x16_t centered = vreinterpretq_s8_u8(vsubq_u8(codes.val[i], vdupq_n_u8(zero)));
        packed_group_widen(centered, has_subscale, has_subscale ? subscales[i] : 1, scale, factored, result + i * 4);
    }
}

#define DEFINE_PROJECT(NAME, DTYPE, MTILE, KSTEP) \
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
        PackedGroupRow group0, group1; \
        if (KSTEP == 32) { \
            group0 = packed_group_row(task->weight, p->width, p->outputs, column, DTYPE); \
            group1 = packed_group_row(task->weight, p->width, p->outputs, second, DTYPE); \
        } \
        size_t k = 0; \
        for (; p->width - k >= KSTEP; k += KSTEP) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t w0[KSTEP / 4], w1[KSTEP / 4]; \
            if (KSTEP == 32) { \
                packed_group_decode32(&group0, k, DTYPE, false, w0); \
                packed_group_decode32(&group1, k, DTYPE, false, w1); \
            } else { \
                decode16(task, column, k, DTYPE, w0); \
                decode16(task, second, k, DTYPE, w1); \
            } \
            for (size_t j = 0; j < KSTEP / 4; ++j) { \
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

DEFINE_PROJECT(project_row_int8_4, TRILLIM_ROW_INT8, 4, 16)
DEFINE_PROJECT(project_mat_ternary_4, TRILLIM_MAT_TERNARY, 4, 16)
DEFINE_PROJECT(project_grp_int8_4, TRILLIM_GRP_INT8, 4, 16)
DEFINE_PROJECT(project_grp_int4_4, TRILLIM_GRP_INT4, 4, 16)
DEFINE_PROJECT(project_grp_int5_4, TRILLIM_GRP_INT5, 4, 16)
DEFINE_PROJECT(project_dubgrp_int6_4, TRILLIM_DUBGRP_INT6, 4, 16)
DEFINE_PROJECT(project_grp_binary_4, TRILLIM_GRP_BINARY, 4, 16)
DEFINE_PROJECT(project_grp_ternary_4, TRILLIM_GRP_TERNARY, 4, 16)
DEFINE_PROJECT(project_mxfp4_4, TRILLIM_MXFP4, 4, 16)
DEFINE_PROJECT(project_mat_ternary_8, TRILLIM_MAT_TERNARY, 8, 16)
DEFINE_PROJECT(project_grp_int5_8, TRILLIM_GRP_INT5, 8, 16)
DEFINE_PROJECT(project_dubgrp_int6_8, TRILLIM_DUBGRP_INT6, 8, 16)
DEFINE_PROJECT(project_grp_binary_8, TRILLIM_GRP_BINARY, 8, 16)
DEFINE_PROJECT(project_grp_ternary_8, TRILLIM_GRP_TERNARY, 8, 16)
DEFINE_PROJECT(project_mxfp4_8, TRILLIM_MXFP4, 8, 16)

DEFINE_PROJECT(project_grp_int4_group, TRILLIM_GRP_INT4, 4, 32)
DEFINE_PROJECT(project_grp_int5_group, TRILLIM_GRP_INT5, 8, 32)
DEFINE_PROJECT(project_dubgrp_int6_group, TRILLIM_DUBGRP_INT6, 8, 32)

#undef DEFINE_PROJECT

#define DEFINE_GROUP_DECODE(NAME, DTYPE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const Parameters *p = task->p; \
    size_t first = tile * task->columns; \
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns; \
    for (size_t column = first; column < end; column += 4) { \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        size_t count = end - column < 4 ? end - column : 4; \
        PackedGroupRow group[4]; \
        float32x4_t sums[4]; \
        for (size_t c = 0; c < 4; ++c) { \
            group[c] = packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), DTYPE); \
            sums[c] = vdupq_n_f32(0); \
        } \
        for (size_t k = 0; k < p->width; k += 32) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t x[8]; \
            for (size_t j = 0; j < 8; ++j) { \
                x[j] = load4(task->input, k + j * 4, p->input_dtype); \
            } \
            for (size_t c = 0; c < 4; ++c) { \
                float32x4_t weights[8], sum = sums[c]; \
                packed_group_decode32(&group[c], k, DTYPE, false, weights); \
                for (size_t j = 0; j < 8; ++j) { \
                    sum = vfmaq_f32(sum, x[j], weights[j]); \
                } \
                sums[c] = sum; \
            } \
        } \
        for (size_t c = 0; c < count; ++c) { \
            store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])}); \
        } \
    } \
    return TRILLIM_OK; \
}

DEFINE_GROUP_DECODE(project_grp_int4_decode, TRILLIM_GRP_INT4)
DEFINE_GROUP_DECODE(project_grp_int5_decode, TRILLIM_GRP_INT5)
DEFINE_GROUP_DECODE(project_dubgrp_int6_decode, TRILLIM_DUBGRP_INT6)
DEFINE_GROUP_DECODE(project_grp_int8_decode, TRILLIM_GRP_INT8)

#undef DEFINE_GROUP_DECODE

/* Nonzero inputs and scales in [2^-40, 2^41) keep factored dots in the normal FP32 range. */
enum { MIN_GROUP_EXPONENT = 87, MAX_GROUP_EXPONENT = 167, MAX_FACTORED_WIDTH = 1 << 16 };

static TrillimStatus bounded_group_scales(
    const uint8_t *scales, size_t count, const TrillimKernelContext *context, bool *valid
) {
    const unsigned minimum = MIN_GROUP_EXPONENT << 7;
    const unsigned span = ((MAX_GROUP_EXPONENT - MIN_GROUP_EXPONENT + 1) << 7) - 1;
    uint16x8_t rejected = vdupq_n_u16(0);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t raw = vreinterpretq_u16_u8(vld1q_u8(scales + i * 2));
        uint16x8_t magnitude = vandq_u16(raw, vdupq_n_u16(0x7fff));
        uint16x8_t outside = vcgtq_u16(vsubq_u16(magnitude, vdupq_n_u16(minimum)), vdupq_n_u16(span));
        rejected = vorrq_u16(rejected, vbicq_u16(outside, vceqzq_u16(magnitude)));
    }
    for (; i < count; ++i) {
        uint16_t raw;
        memcpy(&raw, scales + i * 2, sizeof(raw));
        unsigned magnitude = raw & 0x7fff;
        if (magnitude != 0 && magnitude - minimum > span) {
            *valid = false;
            return TRILLIM_OK;
        }
    }
    *valid = vmaxvq_u16(rejected) == 0;
    return TRILLIM_OK;
}

#if defined(TRILLIM_PACKED_BFDOT)
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

/* Every centered Q5 code is exactly representable in BF16. */
static const uint8_t code_high_32[32] = {
    193, 193, 193, 193, 193, 193, 193, 193, 193, 192, 192, 192, 192, 192, 192, 191,
    0, 63, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65,
};
static const uint8_t code_low_32[32] = {
    128, 112, 96, 80, 64, 48, 32, 16, 0, 224, 192, 160, 128, 64, 0, 128,
    0, 128, 0, 64, 128, 160, 192, 224, 0, 16, 32, 48, 64, 80, 96, 112,
};
static inline uint16x8x2_t bf16_from_q5(uint8x16_t codes) {
    uint8x16_t lo = vqtbl2q_u8(vld1q_u8_x2(code_low_32), codes);
    uint8x16_t hi = vqtbl2q_u8(vld1q_u8_x2(code_high_32), codes);
    uint16x8x2_t weights = {{
        vreinterpretq_u16_u8(vzip1q_u8(lo, hi)),
        vreinterpretq_u16_u8(vzip2q_u8(lo, hi))
    }};
    return weights;
}

#endif

static TrillimStatus check_factored_input(const Task *task, bool *bounded) {
    const Parameters *p = task->p;
    const uint32x4_t minimum = vdupq_n_u32((uint32_t)MIN_GROUP_EXPONENT << 23);
    const uint32x4_t span = vdupq_n_u32(((uint32_t)(MAX_GROUP_EXPONENT - MIN_GROUP_EXPONENT + 1) << 23) - 1);
    uint32x4_t rejected = vdupq_n_u32(0);
    for (size_t k = 0; k < p->width; k += 4) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(load4(task->input, k, p->input_dtype)),
            vdupq_n_u32(UINT32_C(0x7fffffff)));
        uint32x4_t outside = vcgtq_u32(vsubq_u32(magnitude, minimum), span);
        rejected = vorrq_u32(rejected, vbicq_u32(outside, vceqzq_u32(magnitude)));
    }
    *bounded = vmaxvq_u32(rejected) == 0;
    return TRILLIM_OK;
}

#if defined(TRILLIM_PACKED_BFDOT)
BF16 static TrillimStatus project_bf16_int5(void *opaque, size_t tile, size_t worker) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t groups = p->width / 32;
    PackedGroupRow first_row = packed_group_row(task->weight, p->width, p->outputs, first, TRILLIM_GRP_INT5);
    bool valid;
    TrillimStatus status = bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!valid) {
        return project_grp_int5_decode(opaque, tile, worker);
    }
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), TRILLIM_GRP_INT5);
        }
        for (size_t k = 0; k < p->width; k += 32) {
            if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint16x8_t x[4];
            for (size_t j = 0; j < 4; ++j) {
                x[j] = vld1q_u16((const uint16_t *)task->input + k + j * 8);
            }
            float32x4_t partial[4] = {0};
            for (size_t c = 0; c < 4; ++c) {
                uint8x16x2_t codes = packed_group_codes(&row[c], k, TRILLIM_GRP_INT5);
                uint16x8x2_t lo = bf16_from_q5(codes.val[0]);
                uint16x8x2_t hi = bf16_from_q5(codes.val[1]);
                float32x4_t sum = dot_bf16(vdupq_n_f32(0), x[0], lo.val[0]);
                sum = dot_bf16(sum, x[1], lo.val[1]);
                sum = dot_bf16(sum, x[2], hi.val[0]);
                sum = dot_bf16(sum, x[3], hi.val[1]);
                partial[c] = sum;
            }
            for (size_t c = 0; c < 4; ++c) {
                float scale = packed_group_scale(&row[c], k, TRILLIM_GRP_INT5);
                sums[c] = vfmaq_n_f32(sums[c], partial[c], scale);
            }
        }
        for (size_t c = 0; c < count; ++c) {
            store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

#endif

/* Keep activation and code products in FP32 and apply each scale after its block dot. */
static inline __attribute__((always_inline)) TrillimStatus project_factored_group(
    void *opaque, size_t tile, uint32_t dtype, TrillimTask fallback
) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    size_t groups = p->width / group + (p->width % group != 0);
    PackedGroupRow first_row = packed_group_row(task->weight, p->width, p->outputs, first, dtype);
    bool valid;
    TrillimStatus status = bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!valid) {
        return fallback(opaque, tile, 0);
    }
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), dtype);
        }
        for (size_t k = 0; k < p->width; k += group) {
            if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t partial[4] = {0};
            size_t length = p->width - k < group ? p->width - k : group;
            for (size_t offset = 0; offset < length; offset += 32) {
                float32x4_t x[8];
                for (size_t j = 0; j < 8; ++j) {
                    x[j] = load4(task->input, k + offset + j * 4, p->input_dtype);
                }
                for (size_t c = 0; c < 4; ++c) {
                    float32x4_t weights[8];
                    packed_group_decode32(&row[c], k + offset, dtype, true, weights);
                    float32x4_t lo = partial[c], hi = vdupq_n_f32(0);
                    for (size_t j = 0; j < 4; ++j) {
                        lo = vfmaq_f32(lo, x[j], weights[j]);
                        hi = vfmaq_f32(hi, x[j + 4], weights[j + 4]);
                    }
                    partial[c] = vaddq_f32(lo, hi);
                }
            }
            for (size_t c = 0; c < 4; ++c) {
                sums[c] = vfmaq_n_f32(sums[c], partial[c], packed_group_scale(&row[c], k, dtype));
            }
        }
        for (size_t c = 0; c < count; ++c) {
            store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

#define DEFINE_FACTORED_PROJECT(NAME, DTYPE, FALLBACK) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    return project_factored_group(opaque, tile, DTYPE, FALLBACK); \
}

DEFINE_FACTORED_PROJECT(project_factored_int4, TRILLIM_GRP_INT4, project_grp_int4_decode)
DEFINE_FACTORED_PROJECT(project_factored_int5, TRILLIM_GRP_INT5, project_grp_int5_decode)
DEFINE_FACTORED_PROJECT(project_factored_int6, TRILLIM_DUBGRP_INT6, project_dubgrp_int6_decode)
DEFINE_FACTORED_PROJECT(project_factored_int8, TRILLIM_GRP_INT8, project_grp_int8_decode)

#undef DEFINE_FACTORED_PROJECT

static TrillimTask factored_projection(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return project_factored_int4;
    case TRILLIM_GRP_INT5: return project_factored_int5;
    case TRILLIM_DUBGRP_INT6: return project_factored_int6;
    case TRILLIM_GRP_INT8: return project_factored_int8;
    default: return NULL;
    }
}

enum { MAX_FHM_WIDTH = 8192 };

typedef struct {
    Task projection;
    const uint8_t *input;
} FhmTask;

static bool available_fhm(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
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

static bool fhm_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus prepare_fhm_input(const Task *task, uint16_t *input, bool *valid) {
    const Parameters *p = task->p;
    uint32x4_t changed = vdupq_n_u32(0);
    for (size_t k = 0; k < p->width; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t low = load4(task->input, k, p->input_dtype);
        float32x4_t high = load4(task->input, k + 4, p->input_dtype);
        float16x8_t half = vcvt_high_f16_f32(vcvt_f16_f32(low), high);
        uint32x4_t low_bits = vreinterpretq_u32_f32(vcvt_f32_f16(vget_low_f16(half)));
        uint32x4_t high_bits = vreinterpretq_u32_f32(vcvt_high_f32_f16(half));
        changed = vorrq_u32(changed, veorq_u32(vreinterpretq_u32_f32(low), low_bits));
        changed = vorrq_u32(changed, veorq_u32(vreinterpretq_u32_f32(high), high_bits));
        vst1q_u16(input + k, vreinterpretq_u16_f16(half));
    }
    *valid = vmaxvq_u32(changed) == 0;
    return TRILLIM_OK;
}

FHM static TrillimStatus project_fhm_int4(void *opaque, size_t tile, size_t worker) {
    FhmTask *fhm = opaque;
    const Task *task = &fhm->projection;
    const Parameters *p = task->p;
    if (!fhm_default_fpcr()) {
        return project_factored_int4(&fhm->projection, tile, worker);
    }
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t groups = p->width / 32;
    PackedGroupRow first_row = packed_group_row(task->weight, p->width, p->outputs, first, TRILLIM_GRP_INT4);
    bool valid;
    TrillimStatus status = bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!valid) {
        return project_grp_int4_decode(&fhm->projection, tile, worker);
    }
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), TRILLIM_GRP_INT4);
        }
        for (size_t k = 0; k < p->width; k += 32) {
            if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8x16x4_t input = vld1q_u8_x4(fhm->input + k * sizeof(uint16_t));
            float16x8_t x0 = vreinterpretq_f16_u8(input.val[0]), x1 = vreinterpretq_f16_u8(input.val[1]);
            float16x8_t x2 = vreinterpretq_f16_u8(input.val[2]), x3 = vreinterpretq_f16_u8(input.val[3]);
            for (size_t c = 0; c < 4; ++c) {
                uint8x16x2_t codes = packed_group_codes(&row[c], k, TRILLIM_GRP_INT4);
                float16x8x2_t low = packed_group_q4_half(codes.val[0]);
                float16x8x2_t high = packed_group_q4_half(codes.val[1]);
                float32x4_t lo = vdupq_n_f32(0), hi = lo;
                lo = fhm_low(lo, x0, low.val[0]);
                hi = fhm_low(hi, x2, high.val[0]);
                lo = fhm_high(lo, x0, low.val[0]);
                hi = fhm_high(hi, x2, high.val[0]);
                lo = fhm_low(lo, x1, low.val[1]);
                hi = fhm_low(hi, x3, high.val[1]);
                lo = fhm_high(lo, x1, low.val[1]);
                hi = fhm_high(hi, x3, high.val[1]);
                float scale = packed_group_scale(&row[c], k, TRILLIM_GRP_INT4);
                sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), scale);
            }
        }
        for (size_t c = 0; c < count; ++c) {
            store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

static __attribute__((noinline)) TrillimStatus run_fhm_int4(const Task *task, size_t tiles, bool *handled) {
    *handled = false;
    if (!available_fhm() || !fhm_default_fpcr()) {
        return TRILLIM_OK;
    }
    /* Widening FMA uses the half-width activations only after a bit-exact round trip. */
    uint16_t input[MAX_FHM_WIDTH];
    FhmTask fhm = {*task, task->input};
    if (task->p->input_dtype != TRILLIM_FP16) {
        TrillimStatus status = prepare_fhm_input(task, input, handled);
        if (status != TRILLIM_OK || !*handled) {
            return status;
        }
        fhm.input = (const uint8_t *)input;
    }
    *handled = true;
    const TrillimThreadPool *pool = task->context->thread_pool;
    return pool->run(pool->context, tiles, project_fhm_int4, &fhm);
}

static TrillimTask group_projection(uint32_t dtype, size_t rows) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return rows == 1 ? project_grp_int4_decode : project_grp_int4_group;
    case TRILLIM_GRP_INT5:
        if (rows == 1) {
            return project_grp_int5_decode;
        }
        return rows == 8 ? project_grp_int5_group : NULL;
    case TRILLIM_DUBGRP_INT6:
        if (rows == 1) {
            return project_dubgrp_int6_decode;
        }
        return rows == 8 ? project_dubgrp_int6_group : NULL;
    case TRILLIM_GRP_INT8: return rows == 1 ? project_grp_int8_decode : NULL;
    default: return NULL;
    }
}

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
    if (p->output_dtype == TRILLIM_FP32 && p->width != 0 && p->width % 32 == 0) {
        size_t group_rows = p->tokens == 1 ? 1 : row_tile;
        TrillimTask groups = group_projection(p->weight_dtype, group_rows);
        if (groups != NULL) {
            operation = groups;
            row_tile = group_rows;
            if (row_tile == 1) {
                task.columns = columns / 4 * 4;
            }
        }
    }
    if (row_tile == 1 && p->width >= 1024 && p->width <= MAX_FACTORED_WIDTH && p->outputs >= 16) {
        TrillimTask factored = factored_projection(p->weight_dtype);
        if (factored != NULL) {
            bool bounded;
            TrillimStatus status = check_factored_input(&task, &bounded);
            if (status != TRILLIM_OK) {
                return status;
            }
            if (bounded) {
                operation = factored;
#if defined(TRILLIM_PACKED_BFDOT)
                if (p->weight_dtype == TRILLIM_GRP_INT5 && p->input_dtype == TRILLIM_BF16 && available_bf16()) {
                    operation = project_bf16_int5;
                }
#endif
            }
        }
    }
    size_t rows = p->tokens / row_tile + (p->tokens % row_tile != 0);
    size_t tiles = p->outputs / task.columns + (p->outputs % task.columns != 0);
    bool large_fhm = p->input_dtype == TRILLIM_FP32 ? p->outputs >= 32768 : p->outputs > 1024;
    if (operation == project_factored_int4 && p->width <= MAX_FHM_WIDTH && large_fhm) {
        bool handled;
        TrillimStatus status = run_fhm_int4(&task, tiles, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
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
