#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <dlfcn.h>
#include <limits.h>
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


#if defined(__APPLE__)
enum {
    PACKED_PREFILL_M = 64,
    PACKED_PREFILL_N = 96,
    PACKED_PREFILL_K = 1536,
    PACKED_PREFILL_LANES = 32,
    PACKED_PREFILL_ACTIVATIONS = 1 << 20,
};

typedef void (*PackedPrefillSgemm)(int, int, int, int, int, int, float, const float *, int,
                                   const float *, int, float, float *, int);
typedef unsigned (*PackedPrefillGetThreading)(void);
typedef int (*PackedPrefillSetThreading)(unsigned);

static PackedPrefillSgemm packed_prefill_sgemm;
static PackedPrefillGetThreading packed_prefill_get_threading;
static PackedPrefillSetThreading packed_prefill_set_threading;

static bool initialize_packed_prefill_blas(void) {
    static atomic_uint state;
    if (atomic_load_explicit(&state, memory_order_acquire) != 2) {
        unsigned expected = 0;
        if (atomic_compare_exchange_strong_explicit(&state, &expected, 1,
                                                    memory_order_acq_rel, memory_order_acquire)) {
            void *library = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                                  RTLD_LOCAL | RTLD_LAZY);
            if (library != NULL) {
                void *gemm = dlsym(library, "cblas_sgemm$NEWLAPACK");
                if (gemm == NULL) {
                    gemm = dlsym(library, "cblas_sgemm");
                }
                void *getter = dlsym(library, "BLASGetThreading");
                void *setter = dlsym(library, "BLASSetThreading");
                if (gemm != NULL && getter != NULL && setter != NULL) {
                    _Static_assert(sizeof(packed_prefill_sgemm) == sizeof(gemm), "function pointer size");
                    memcpy(&packed_prefill_sgemm, &gemm, sizeof(packed_prefill_sgemm));
                    memcpy(&packed_prefill_get_threading, &getter, sizeof(packed_prefill_get_threading));
                    memcpy(&packed_prefill_set_threading, &setter, sizeof(packed_prefill_set_threading));
                } else {
                    dlclose(library);
                }
            }
            atomic_store_explicit(&state, 2, memory_order_release);
        } else {
            while (atomic_load_explicit(&state, memory_order_acquire) != 2) {
                atomic_signal_fence(memory_order_acquire);
            }
        }
    }
    return packed_prefill_sgemm != NULL;
}

typedef struct {
    const Task *task;
    const float *activation;
    float *weights;
    size_t columns, depth, tiles, lanes;
    atomic_bool ordinary;
} PackedPrefillTask;

static inline __attribute__((always_inline)) TrillimStatus project_packed_prefill_panel(
    PackedPrefillTask *job, float *weights, size_t tile, uint32_t dtype
) {
    const Task *task = job->task;
    const Parameters *p = task->p;
    size_t first = tile * job->columns;
    size_t columns = p->outputs - first < job->columns ? p->outputs - first : job->columns;
    size_t group = group_width(dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    PackedGroupRow row[PACKED_PREFILL_N];
    row[0] = packed_group_row(task->weight, p->width, p->outputs, first, dtype);
    bool ordinary;
    TrillimStatus status = bounded_group_scales(row[0].scales, columns * groups, task->context, &ordinary);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!ordinary) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    for (size_t c = 1; c < columns; ++c) {
        row[c] = packed_group_row(task->weight, p->width, p->outputs, first + c, dtype);
    }
    for (size_t begin = 0; begin < p->width; begin += job->depth) {
        size_t depth = p->width - begin < job->depth ? p->width - begin : job->depth;
        for (size_t c = 0; c < columns; ++c) {
            if ((c & 7) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float *destination = weights + c * depth;
            for (size_t k = 0; k < depth; k += 32) {
                float32x4_t values[8];
                packed_group_decode32(&row[c], begin + k, dtype, false, values);
                for (size_t j = 0; j < 8; ++j) {
                    vst1q_f32(destination + k + j * 4, values[j]);
                }
            }
        }
        for (size_t row_index = 0; row_index < p->tokens; row_index += PACKED_PREFILL_M) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t batch = p->tokens - row_index < PACKED_PREFILL_M ? p->tokens - row_index : PACKED_PREFILL_M;
            packed_prefill_sgemm(101, 111, 112, (int)batch, (int)columns, (int)depth, 1.0f,
                job->activation + row_index * p->width + begin, (int)p->width,
                weights, (int)depth, begin == 0 ? 0.0f : 1.0f,
                (float *)task->output + row_index * p->outputs + first, (int)p->outputs);
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#define DEFINE_PACKED_PREFILL(NAME, DTYPE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    PackedPrefillTask *job = opaque; \
    if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) { \
        return TRILLIM_OK; \
    } \
    if (!fhm_default_fpcr()) { \
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed); \
        return TRILLIM_OK; \
    } \
    unsigned previous = packed_prefill_get_threading(); \
    if (previous != 1 && packed_prefill_set_threading(1) != 0) { \
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed); \
        return TRILLIM_OK; \
    } \
    TrillimStatus status = TRILLIM_OK; \
    float *weights = job->weights + tile * job->columns * job->depth; \
    for (size_t next = tile; next < job->tiles; next += job->lanes) { \
        status = project_packed_prefill_panel(job, weights, next, DTYPE); \
        if (status != TRILLIM_OK || !atomic_load_explicit(&job->ordinary, memory_order_relaxed)) { \
            break; \
        } \
    } \
    if (previous != 1) { \
        (void)packed_prefill_set_threading(previous); \
    } \
    return status; \
}

DEFINE_PACKED_PREFILL(project_prefill_int4, TRILLIM_GRP_INT4)
DEFINE_PACKED_PREFILL(project_prefill_int5, TRILLIM_GRP_INT5)
DEFINE_PACKED_PREFILL(project_prefill_int6, TRILLIM_DUBGRP_INT6)
DEFINE_PACKED_PREFILL(project_prefill_int8, TRILLIM_GRP_INT8)

#undef DEFINE_PACKED_PREFILL

static TrillimTask packed_prefill_operation(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return project_prefill_int4;
    case TRILLIM_GRP_INT5: return project_prefill_int5;
    case TRILLIM_DUBGRP_INT6: return project_prefill_int6;
    case TRILLIM_GRP_INT8: return project_prefill_int8;
    default: return NULL;
    }
}

static TrillimStatus pack_prefill_activations(const Task *task, float *packed, bool *ordinary) {
    const Parameters *p = task->p;
    const uint32x4_t minimum = vdupq_n_u32((uint32_t)MIN_GROUP_EXPONENT << 23);
    const uint32x4_t span = vdupq_n_u32(((uint32_t)(MAX_GROUP_EXPONENT - MIN_GROUP_EXPONENT + 1) << 23) - 1);
    uint32x4_t rejected = vdupq_n_u32(0);
    size_t elements = p->tokens * p->width;
    for (size_t i = 0; i < elements; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t values = load4(task->input, i, p->input_dtype);
        uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(values), vdupq_n_u32(UINT32_C(0x7fffffff)));
        uint32x4_t outside = vcgtq_u32(vsubq_u32(magnitude, minimum), span);
        rejected = vorrq_u32(rejected, vbicq_u32(outside, vceqzq_u32(magnitude)));
        if (packed != NULL) {
            vst1q_f32(packed + i, values);
        }
    }
    *ordinary = vmaxvq_u32(rejected) == 0;
    return TRILLIM_OK;
}

static TrillimStatus try_packed_prefill(const Task *task, bool *handled) {
    const Parameters *p = task->p;
    const TrillimThreadPool *pool = task->context->thread_pool;
    *handled = false;
    if (pool->worker_count == 0 || p->tokens < 32 || p->outputs < 1024 || p->width < 256 || p->width % 32 != 0 ||
        p->width > INT_MAX || p->outputs > INT_MAX || p->tokens > PACKED_PREFILL_ACTIVATIONS / p->width ||
        p->width < ((1 << 21) + p->outputs - 1) / p->outputs ||
        (uintptr_t)task->output % _Alignof(float) != 0 || p->output_dtype != TRILLIM_FP32) {
        return TRILLIM_OK;
    }
    TrillimTask operation = packed_prefill_operation(p->weight_dtype);
    if (operation == NULL || !fhm_default_fpcr() || !initialize_packed_prefill_blas()) {
        return TRILLIM_OK;
    }
    size_t columns = p->outputs / pool->worker_count / 16 * 16;
    columns = columns < 16 ? 16 : (columns > PACKED_PREFILL_N ? PACKED_PREFILL_N : columns);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    size_t lanes = pool->worker_count < tiles ? pool->worker_count : tiles;
    lanes = lanes < PACKED_PREFILL_LANES ? lanes : PACKED_PREFILL_LANES;
    /* Each lane reuses one panel while activations stay widened for the entire projection. */
    size_t depth = p->width < PACKED_PREFILL_K ? p->width : PACKED_PREFILL_K;
    bool direct = p->input_dtype == TRILLIM_FP32 && (uintptr_t)task->input % _Alignof(float) == 0;
    size_t activations = direct ? 0 : p->tokens * p->width;
    float *workspace = malloc((activations + lanes * columns * depth) * sizeof(float));
    if (workspace == NULL) {
        return TRILLIM_OK;
    }
    bool ordinary;
    TrillimStatus status = pack_prefill_activations(task, direct ? NULL : workspace, &ordinary);
    if (status == TRILLIM_OK && ordinary) {
        PackedPrefillTask job = {task, direct ? task->input : workspace, workspace + activations,
            columns, depth, tiles, lanes, ATOMIC_VAR_INIT(true)};
        status = pool->run(pool->context, lanes, operation, &job);
        *handled = atomic_load_explicit(&job.ordinary, memory_order_relaxed);
    }
    free(workspace);
    return status;
}

/* Link the FFN component without exporting another kernel entry point. */
__asm__(".private_extern _trillim_neon_try_packed_linear_prefill");
static __attribute__((used)) TrillimStatus trillim_neon_try_packed_linear_prefill(
    void *output, const void *input, const void *weight, const void *parameters,
    const TrillimKernelContext *context, bool *handled
) {
    const Task task = {.output = output, .input = input, .weight = weight, .p = parameters, .context = context};
    return try_packed_prefill(&task, handled);
}
#endif

enum {
    INT8_ACTIVATION_MAX_WIDTH = 65536,
    INT8_ACTIVATION_Q8_0 = 32,
    INT8_ACTIVATION_Q8_K = 256,
};

typedef struct {
    uint16_t scale;
    int8_t values[INT8_ACTIVATION_Q8_0];
} Q8ActivationBlock;

typedef struct {
    float scale;
    int8_t values[INT8_ACTIVATION_Q8_K];
    int16_t sums[INT8_ACTIVATION_Q8_K / 16];
} Q8KActivationBlock;

typedef struct {
    const Task *projection;
    const void *activation;
    const uint8_t *gate;
    float (*silu)(float);
} Int8ProjectionTask;

__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_sdot_v\\reg,\\reg\n"
    ".endr\n"
);

static bool available_dotprod(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value != 0) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_DotProd", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP_ASIMDDP)
    available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

static inline int32x4_t int8_dot_widen(int8x16_t weight, int8x16_t input) {
    const int16x8_t low = vmull_s8(vget_low_s8(weight), vget_low_s8(input));
    const int16x8_t high = vmull_high_s8(weight, input);
    return vpaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high));
}

static inline int32x4_t int8_sdot_add(int32x4_t accum, int8x16_t weight, int8x16_t input) {
    __asm__(".inst 0x4e809400 | trillim_sdot_%0 | (trillim_sdot_%1 << 5) | (trillim_sdot_%2 << 16)"
            : "+w"(accum) : "w"(weight), "w"(input));
    return accum;
}

static inline int32x4_t int8_sdot(int8x16_t weight, int8x16_t input) {
    return int8_sdot_add(vdupq_n_s32(0), weight, input);
}

static inline int8x16_t quantize_activation_16(
    float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3, float inverse
) {
    const int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(x0, inverse));
    const int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(x1, inverse));
    const int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(x2, inverse));
    const int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(x3, inverse));
    return vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(q0), vmovn_s32(q1))),
                        vmovn_s16(vcombine_s16(vmovn_s32(q2), vmovn_s32(q3))));
}

static bool quantize_activation_q8(const void *input, size_t k, uint32_t dtype, Q8ActivationBlock *out) {
    float32x4_t x[8];
    float32x4_t max_vector = vdupq_n_f32(0);
    for (size_t i = 0; i < 8; ++i) {
        x[i] = load4(input, k + i * 4, dtype);
        max_vector = vmaxq_f32(max_vector, vabsq_f32(x[i]));
    }
    const float maximum = vmaxvq_f32(max_vector);
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0 ? 1.0f / scale : 0;
    if (!isfinite(inverse) || !isfinite(maximum) || scale > 65504.0f) {
        return false;
    }
    out->scale = trillim_to_fp16(scale);
    if (maximum != 0 && out->scale == 0) {
        return false;
    }
    vst1q_s8(out->values, quantize_activation_16(x[0], x[1], x[2], x[3], inverse));
    vst1q_s8(out->values + 16, quantize_activation_16(x[4], x[5], x[6], x[7], inverse));
    return true;
}

static inline __attribute__((always_inline)) bool quantize_activation_q8k_typed(
    const void *input, size_t k, uint32_t dtype, Q8KActivationBlock *out, bool keep_sums
) {
    float values[INT8_ACTIVATION_Q8_K];
    float32x4_t max0 = vdupq_n_f32(0);
    float32x4_t max1 = max0, max2 = max0, max3 = max0;
    for (size_t i = 0; i < INT8_ACTIVATION_Q8_K; i += 16) {
        const float32x4_t x0 = load4(input, k + i, dtype);
        const float32x4_t x1 = load4(input, k + i + 4, dtype);
        const float32x4_t x2 = load4(input, k + i + 8, dtype);
        const float32x4_t x3 = load4(input, k + i + 12, dtype);
        vst1q_f32(values + i, x0);
        vst1q_f32(values + i + 4, x1);
        vst1q_f32(values + i + 8, x2);
        vst1q_f32(values + i + 12, x3);
        max0 = vmaxq_f32(max0, vabsq_f32(x0));
        max1 = vmaxq_f32(max1, vabsq_f32(x1));
        max2 = vmaxq_f32(max2, vabsq_f32(x2));
        max3 = vmaxq_f32(max3, vabsq_f32(x3));
    }
    const float maximum = vmaxvq_f32(vmaxq_f32(vmaxq_f32(max0, max1), vmaxq_f32(max2, max3)));
    if (maximum == 0) {
        memset(out, 0, sizeof(*out));
        return true;
    }
    if (!isfinite(maximum)) {
        return false;
    }
    /* Q8_K uses the sign of the first value attaining the maximum magnitude. */
    size_t peak = 0;
    const float32x4_t maximum4 = vdupq_n_f32(maximum);
    uint32x4_t matches;
    do {
        matches = vceqq_f32(vabsq_f32(vld1q_f32(values + peak)), maximum4);
        if (vmaxvq_u32(matches) != 0) {
            break;
        }
        peak += 4;
    } while (peak < INT8_ACTIVATION_Q8_K);
    if (peak == INT8_ACTIVATION_Q8_K) {
        return false;
    }
    const uint32x4_t lane = {0, 1, 2, 3};
    peak += vminvq_u32(vbslq_u32(matches, lane, vdupq_n_u32(4)));
    const float inverse = -127.0f / values[peak];
    if (!isfinite(inverse) || inverse == 0) {
        return false;
    }
    out->scale = 1.0f / inverse;
    for (size_t i = 0; i < INT8_ACTIVATION_Q8_K; i += 16) {
        const int8x16_t x = quantize_activation_16(vld1q_f32(values + i), vld1q_f32(values + i + 4),
            vld1q_f32(values + i + 8), vld1q_f32(values + i + 12), inverse);
        vst1q_s8(out->values + i, x);
        if (keep_sums) {
            out->sums[i / 16] = vaddlvq_s8(x);
        }
    }
    return true;
}

static __attribute__((noinline)) bool quantize_activation_q8k(
    const void *input, size_t k, uint32_t dtype, Q8KActivationBlock *out, bool keep_sums
) {
    switch (dtype) {
    case TRILLIM_FP32:
        return quantize_activation_q8k_typed(input, k, TRILLIM_FP32, out, keep_sums);
    case TRILLIM_BF16:
        return quantize_activation_q8k_typed(input, k, TRILLIM_BF16, out, keep_sums);
    case TRILLIM_FP16:
        return quantize_activation_q8k_typed(input, k, TRILLIM_FP16, out, keep_sums);
    default:
        return false;
    }
}

static inline float int8_weight_scale(const uint8_t *scales, size_t block) {
    uint16_t bits;
    memcpy(&bits, scales + block * sizeof(bits), sizeof(bits));
    return trillim_bf16(bits);
}

static inline int8x16x2_t int8_decode_q5(const uint8_t *low, const uint8_t *high) {
    uint32_t packed_high;
    memcpy(&packed_high, high, sizeof(packed_high));
    const uint8x16_t packed_low = vld1q_u8(low);
    const uint8x16_t bits = vreinterpretq_u8_u32(vsetq_lane_u32(packed_high, vdupq_n_u32(0), 0));
    const uint8x16_t indexes0 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    const uint8x16_t indexes1 = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    const uint8x16_t masks = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t center = vdupq_n_u8(16);
    const uint8x16_t center0 = vbicq_u8(center, vtstq_u8(vqtbl1q_u8(bits, indexes0), masks));
    const uint8x16_t center1 = vbicq_u8(center, vtstq_u8(vqtbl1q_u8(bits, indexes1), masks));
    return (int8x16x2_t){{
        vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed_low, vdupq_n_u8(15)), center0)),
        vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed_low, 4), center1)),
    }};
}

static inline int8x16x2_t int8_decode_q4(const uint8_t *low) {
    const uint8x16_t bits = vld1q_u8(low);
    return (int8x16x2_t){{
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bits, vdupq_n_u8(15))), vdupq_n_s8(8)),
        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bits, 4)), vdupq_n_s8(8)),
    }};
}

static inline __attribute__((always_inline)) float dot_q8_activation(
    const uint8_t *low, const uint8_t *scales, const uint8_t *high, const Q8ActivationBlock *x,
    size_t groups, uint32_t dtype, bool widening
) {
    /* Match the four lane sums for even and odd blocks in the reference DOT kernel. */
    float32x4_t even = vdupq_n_f32(0), odd = vdupq_n_f32(0);
    size_t g = 0;
    for (; g + 1 < groups; g += 2) {
        int32x4_t partial[2];
        float scale[2];
        for (size_t p = 0; p < 2; ++p) {
            const size_t block = g + p;
            int8x16x2_t w;
            if (dtype == TRILLIM_GRP_INT5) {
                w = int8_decode_q5(low + block * 16, high + block * 4);
            } else if (dtype == TRILLIM_GRP_INT4) {
                w = int8_decode_q4(low + block * 16);
            } else {
                w.val[0] = vld1q_s8((const int8_t *) low + block * 32);
                w.val[1] = vld1q_s8((const int8_t *) low + block * 32 + 16);
            }
            const int8x16_t a0 = vld1q_s8(x[block].values);
            const int8x16_t a1 = vld1q_s8(x[block].values + 16);
            partial[p] = widening ? vaddq_s32(int8_dot_widen(w.val[0], a0), int8_dot_widen(w.val[1], a1))
                                  : vaddq_s32(int8_sdot(w.val[0], a0), int8_sdot(w.val[1], a1));
            scale[p] = int8_weight_scale(scales, block) * (float) vreinterpret_f16_u16(vdup_n_u16(x[block].scale))[0];
        }
        even = vfmaq_n_f32(even, vcvtq_f32_s32(partial[0]), scale[0]);
        odd = vfmaq_n_f32(odd, vcvtq_f32_s32(partial[1]), scale[1]);
    }
    float result = vaddvq_f32(even) + vaddvq_f32(odd);
    if (g < groups) {
        int8x16x2_t w;
        if (dtype == TRILLIM_GRP_INT5) {
            w = int8_decode_q5(low + g * 16, high + g * 4);
        } else if (dtype == TRILLIM_GRP_INT4) {
            w = int8_decode_q4(low + g * 16);
        } else {
            w.val[0] = vld1q_s8((const int8_t *) low + g * 32);
            w.val[1] = vld1q_s8((const int8_t *) low + g * 32 + 16);
        }
        const int32x4_t partial = vaddq_s32(int8_dot_widen(w.val[0], vld1q_s8(x[g].values)),
            int8_dot_widen(w.val[1], vld1q_s8(x[g].values + 16)));
        result += (int8_weight_scale(scales, g) * trillim_fp16(x[g].scale)) * vaddvq_s32(partial);
    }
    return result;
}

static inline __attribute__((always_inline)) float dot_q8k_activation(
    const uint8_t *restrict low, const uint8_t *restrict scales, const uint8_t *restrict high,
    const int8_t *restrict subscales, const Q8KActivationBlock *restrict x, size_t groups, bool widening
) {
    float output = 0;
    const uint8x16_t low_mask = vdupq_n_u8(15);
    const uint8x16_t high_mask = vdupq_n_u8(48);
    for (size_t g = 0; g < groups; ++g) {
        int32x4_t sum = vdupq_n_s32(0);
        for (size_t half = 0; half < 2; ++half) {
            const uint8_t *l = low + g * 128 + half * 64;
            const uint8_t *h = high + g * 64 + half * 32;
            const int8_t *a = x[g].values + half * 128;
            const int8x16_t s = vld1q_s8(subscales + g * 16);
            const int16x8_t half_scales = half == 0 ? vmovl_s8(vget_low_s8(s)) : vmovl_high_s8(s);
            const int16x8_t sums = vld1q_s16(x[g].sums + half * 8);
            for (size_t lane = 0; lane < 4; ++lane) {
                const int8x16_t shift_l = vdupq_n_s8(-(int8_t)((lane % 2) * 4));
                const int8x16_t shift_h = vdupq_n_s8((int8_t)(4 - 2 * (int)lane));
                const uint8_t *l32 = l + (lane / 2) * 32;
                const uint8x16_t q0 = vorrq_u8(vandq_u8(vshlq_u8(vld1q_u8(l32), shift_l), low_mask),
                    vandq_u8(vshlq_u8(vld1q_u8(h), shift_h), high_mask));
                const uint8x16_t q1 = vorrq_u8(vandq_u8(vshlq_u8(vld1q_u8(l32 + 16), shift_l), low_mask),
                    vandq_u8(vshlq_u8(vld1q_u8(h + 16), shift_h), high_mask));
                const int8x16_t a0 = vld1q_s8(a + lane * 32);
                const int8x16_t a1 = vld1q_s8(a + lane * 32 + 16);
                int32x4_t d0 = widening ? int8_dot_widen(vreinterpretq_s8_u8(q0), a0) : int8_sdot(vreinterpretq_s8_u8(q0), a0);
                int32x4_t d1 = widening ? int8_dot_widen(vreinterpretq_s8_u8(q1), a1) : int8_sdot(vreinterpretq_s8_u8(q1), a1);
                const int16_t s0 = subscales[g * 16 + half * 8 + lane * 2];
                const int16_t s1 = subscales[g * 16 + half * 8 + lane * 2 + 1];
                sum = vmlaq_n_s32(sum, d0, s0);
                sum = vmlaq_n_s32(sum, d1, s1);
            }
            const int32x4_t correction = vaddq_s32(
                vmull_s16(vget_low_s16(sums), vget_low_s16(half_scales)),
                vmull_high_s16(sums, half_scales));
            sum = vmlsq_n_s32(sum, correction, 32);
        }
        output += (int8_weight_scale(scales, g) * x[g].scale) * vaddvq_s32(sum);
    }
    return output;
}

typedef struct {
    const uint8_t *low;
    const uint8_t *scales;
    const uint8_t *high;
    const int8_t *subscales;
} Int8PackedWeight;

static inline Int8PackedWeight int8_packed_weight(const uint8_t *weight, size_t total_groups, uint32_t dtype) {
    const size_t low_size = dtype == TRILLIM_DUBGRP_INT6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *scales = weight + total_groups * low_size;
    const uint8_t *high = scales + total_groups * 2;
    const int8_t *subscales = (const int8_t *) (high + (dtype == TRILLIM_DUBGRP_INT6 ? total_groups * 64 : 0));
    return (Int8PackedWeight){weight, scales, high, subscales};
}

static inline __attribute__((always_inline)) float int8_dot_row(
    const Int8PackedWeight *weight, size_t offset, const void *activation, size_t groups,
    uint32_t dtype, bool widening
) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return dot_q8k_activation(weight->low + offset * 128, weight->scales + offset * 2,
            weight->high + offset * 64, weight->subscales + offset * 16, activation, groups, widening);
    }
    const size_t low_size = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    return dot_q8_activation(weight->low + offset * low_size, weight->scales + offset * 2,
        weight->high + (dtype == TRILLIM_GRP_INT5 ? offset * 4 : 0), activation, groups, dtype, widening);
}

static inline Int8PackedWeight q8_activation_weight(Int8PackedWeight weight, size_t offset, uint32_t dtype) {
    const size_t bytes = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    weight.low += offset * bytes;
    weight.scales += offset * 2;
    if (dtype == TRILLIM_GRP_INT5) {
        weight.high += offset * 4;
    }
    return weight;
}

static inline __attribute__((always_inline)) int32x4_t q8_activation_dot(
    Int8PackedWeight weight, size_t block, int8x16x2_t x, uint32_t dtype, bool widening
) {
    int8x16x2_t w;
    if (dtype == TRILLIM_GRP_INT5) {
        w = int8_decode_q5(weight.low + block * 16, weight.high + block * 4);
    } else if (dtype == TRILLIM_GRP_INT4) {
        w = int8_decode_q4(weight.low + block * 16);
    } else {
        w = vld1q_s8_x2((const int8_t *)weight.low + block * 32);
    }
    if (widening) {
        /* Both products fit in a signed 16-bit lane because activations use [-127, 127]. */
        const int16x8_t low = vmlal_s8(vmull_s8(vget_low_s8(w.val[0]), vget_low_s8(x.val[0])),
            vget_low_s8(w.val[1]), vget_low_s8(x.val[1]));
        const int16x8_t high = vmlal_high_s8(vmull_high_s8(w.val[0], x.val[0]), w.val[1], x.val[1]);
        return vpaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high));
    }
    return int8_sdot_add(int8_sdot(w.val[0], x.val[0]), w.val[1], x.val[1]);
}

static inline __attribute__((always_inline)) void dot_q8_activation_pair(
    Int8PackedWeight w0, Int8PackedWeight w1, const Q8ActivationBlock *x, size_t groups,
    uint32_t dtype, bool widening, float result[2]
) {
    /* Match the four lane sums for even and odd blocks in the reference DOT kernel. */
    float32x4_t even0 = vdupq_n_f32(0), odd0 = even0;
    float32x4_t even1 = even0, odd1 = even0;
    size_t g = 0;
    for (; g + 1 < groups; g += 2) {
        const int8x16x2_t x0 = vld1q_s8_x2(x[g].values);
        const float scale0 = (float)vreinterpret_f16_u16(vdup_n_u16(x[g].scale))[0];
        const int32x4_t d00 = q8_activation_dot(w0, g, x0, dtype, widening);
        const int32x4_t d10 = q8_activation_dot(w1, g, x0, dtype, widening);
        even0 = vfmaq_n_f32(even0, vcvtq_f32_s32(d00), int8_weight_scale(w0.scales, g) * scale0);
        even1 = vfmaq_n_f32(even1, vcvtq_f32_s32(d10), int8_weight_scale(w1.scales, g) * scale0);

        const int8x16x2_t x1 = vld1q_s8_x2(x[g + 1].values);
        const float scale1 = (float)vreinterpret_f16_u16(vdup_n_u16(x[g + 1].scale))[0];
        const int32x4_t d01 = q8_activation_dot(w0, g + 1, x1, dtype, widening);
        const int32x4_t d11 = q8_activation_dot(w1, g + 1, x1, dtype, widening);
        odd0 = vfmaq_n_f32(odd0, vcvtq_f32_s32(d01), int8_weight_scale(w0.scales, g + 1) * scale1);
        odd1 = vfmaq_n_f32(odd1, vcvtq_f32_s32(d11), int8_weight_scale(w1.scales, g + 1) * scale1);
    }
    result[0] = vaddvq_f32(even0) + vaddvq_f32(odd0);
    result[1] = vaddvq_f32(even1) + vaddvq_f32(odd1);
    if (g < groups) {
        const int8x16x2_t values = vld1q_s8_x2(x[g].values);
        const float scale = trillim_fp16(x[g].scale);
        const int32x4_t d0 = q8_activation_dot(w0, g, values, dtype, widening);
        const int32x4_t d1 = q8_activation_dot(w1, g, values, dtype, widening);
        result[0] += (int8_weight_scale(w0.scales, g) * scale) * vaddvq_s32(d0);
        result[1] += (int8_weight_scale(w1.scales, g) * scale) * vaddvq_s32(d1);
    }
}

static __attribute__((noinline)) TrillimStatus project_int8_nonfinite(
    const Task *task, const uint8_t *weight, size_t row, float *result
) {
    const Parameters *p = task->p;
    const size_t start = row * p->width;
    float sum = 0.0f;
    for (size_t k = 0; k < p->width; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const float x = trillim_load(task->input, k, p->input_dtype).f;
        const float w = trillim_packed_read_float(weight, start + k, p->weight_dtype, p->outputs, p->width);
        sum += x * w;
    }
    *result = sum;
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_int8(
    void *opaque, size_t index, bool share_activation, bool widening, uint32_t dtype
) {
    const Int8ProjectionTask *quantized = opaque;
    const Task *task = quantized->projection;
    const Parameters *p = task->p;
    const bool paired = quantized->gate != NULL;
    const size_t group_width = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    const size_t groups = p->width / group_width;
    const size_t total_groups = groups * p->outputs;
    const Int8PackedWeight weight = int8_packed_weight(task->weight, total_groups, dtype);
    const Int8PackedWeight gate = paired ? int8_packed_weight(quantized->gate, total_groups, dtype) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    const size_t stride = share_activation && !paired ? 2 : 1;
    for (size_t row = begin; row < end; row += stride) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t offset = row * groups;
        const bool next = stride == 2 && row + 1 < end;
        float result[2];
        if (share_activation) {
            const Int8PackedWeight first = q8_activation_weight(weight, offset, dtype);
            const Int8PackedWeight second = q8_activation_weight(gate, offset + (next ? groups : 0), dtype);
            dot_q8_activation_pair(first, second, quantized->activation, groups, dtype, widening, result);
        } else {
            result[0] = int8_dot_row(&weight, offset, quantized->activation, groups, dtype, widening);
        }
        const size_t count = next ? 2 : 1;
        for (size_t c = 0; c < count; ++c) {
            float value = result[c];
            if (!isfinite(value)) {
                const TrillimStatus status = project_int8_nonfinite(task, task->weight, row + c, &value);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (paired) {
                float gate_value = share_activation ? result[1] :
                    int8_dot_row(&gate, offset, quantized->activation, groups, dtype, widening);
                if (!isfinite(gate_value)) {
                    const TrillimStatus status = project_int8_nonfinite(task, quantized->gate, row, &gate_value);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                value *= quantized->silu(gate_value);
            }
            store_result(task->output, row + c, TRILLIM_FP32, (TrillimValue){.f = value});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#define DEFINE_INT8_PROJECTION(name, share, widening, dtype) \
    static TrillimStatus name(void *opaque, size_t index, size_t worker) { \
        (void) worker; \
        return project_int8(opaque, index, share, widening, dtype); \
    }

DEFINE_INT8_PROJECTION(project_int8_q4_neon, false, true, TRILLIM_GRP_INT4)
DEFINE_INT8_PROJECTION(project_int8_q4_neon_pair, true, true, TRILLIM_GRP_INT4)
DEFINE_INT8_PROJECTION(project_int8_q5_neon, true, true, TRILLIM_GRP_INT5)
DEFINE_INT8_PROJECTION(project_int8_q6_neon, false, true, TRILLIM_DUBGRP_INT6)
DEFINE_INT8_PROJECTION(project_int8_q8_neon, false, true, TRILLIM_GRP_INT8)
DEFINE_INT8_PROJECTION(project_int8_q8_neon_pair, true, true, TRILLIM_GRP_INT8)

DEFINE_INT8_PROJECTION(project_int8_q4_sdot, false, false, TRILLIM_GRP_INT4)
DEFINE_INT8_PROJECTION(project_int8_q4_sdot_pair, true, false, TRILLIM_GRP_INT4)
DEFINE_INT8_PROJECTION(project_int8_q5_sdot, true, false, TRILLIM_GRP_INT5)
enum { Q6_AQ_COLUMNS = 2 };

#if defined(__clang__)
#define Q6_AQ_UNROLL _Pragma("clang loop unroll(full)")
#else
#define Q6_AQ_UNROLL
#endif

static inline int8x16_t q6_signed4(uint8x16_t packed, uint8x16_t upper, bool high_nibble) {
    const uint8x16_t lower = high_nibble
        ? vshlq_n_u8(vshrq_n_u8(packed, 4), 2)
        : vshlq_n_u8(packed, 2);
    return vreinterpretq_s8_u8(vsliq_n_u8(lower, upper, 6));
}

static inline int32x4_t q6_aq_quarter(
    const uint8_t *low, uint8x16x2_t upper, const int8x16x4_t activation
) {
    const uint8x16x2_t packed = vld1q_u8_x2(low);
    const int32x4_t d0 = int8_sdot(q6_signed4(packed.val[0], upper.val[0], false), activation.val[0]);
    const int32x4_t d1 = int8_sdot(q6_signed4(packed.val[1], upper.val[1], false), activation.val[1]);
    const uint8x16_t h0 = vshrq_n_u8(upper.val[0], 2);
    const uint8x16_t h1 = vshrq_n_u8(upper.val[1], 2);
    const int32x4_t d2 = int8_sdot(q6_signed4(packed.val[0], h0, true), activation.val[2]);
    const int32x4_t d3 = int8_sdot(q6_signed4(packed.val[1], h1, true), activation.val[3]);
    return vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
}

static void q6_aq_tile(
    const Int8PackedWeight weight[Q6_AQ_COLUMNS], const Q8KActivationBlock *x, size_t groups,
    float result[Q6_AQ_COLUMNS]
) {
    float sum[Q6_AQ_COLUMNS] = {0};
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *input = x[g].values;
        int32x4_t block_dot[Q6_AQ_COLUMNS] = {0};
        for (size_t h = 0; h < 2; ++h) {
            const int8x16x4_t x0 = vld1q_s8_x4(input + h * 128);
            const int8x16x4_t x1 = vld1q_s8_x4(input + h * 128 + 64);
            Q6_AQ_UNROLL
            for (size_t c = 0; c < Q6_AQ_COLUMNS; ++c) {
                const uint8_t *low = weight[c].low + g * 128 + h * 64;
                uint8x16x2_t upper = vld1q_u8_x2(weight[c].high + g * 64 + h * 32);
                /* Flip all four sign bits. Insert the selected two high bits at
                   the top of each byte to get four times the centered code. */
                upper.val[0] = veorq_u8(upper.val[0], vdupq_n_u8(0xaa));
                upper.val[1] = veorq_u8(upper.val[1], vdupq_n_u8(0xaa));
                const int32x4_t first = q6_aq_quarter(low, upper, x0);
                upper.val[0] = vshrq_n_u8(upper.val[0], 4);
                upper.val[1] = vshrq_n_u8(upper.val[1], 4);
                const int32x4_t second = q6_aq_quarter(low + 32, upper, x1);
                const int16x8_t scales = vmovl_s8(vld1_s8(weight[c].subscales + g * 16 + h * 8));
                block_dot[c] = vmlaq_s32(block_dot[c], first, vmovl_s16(vget_low_s16(scales)));
                block_dot[c] = vmlaq_s32(block_dot[c], second, vmovl_high_s16(scales));
            }
        }
        Q6_AQ_UNROLL
        for (size_t c = 0; c < Q6_AQ_COLUMNS; ++c) {
            const int32_t dot = vaddvq_s32(vshrq_n_s32(block_dot[c], 2));
            sum[c] += (int8_weight_scale(weight[c].scales, g) * x[g].scale) * dot;
        }
    }
    Q6_AQ_UNROLL
    for (size_t c = 0; c < Q6_AQ_COLUMNS; ++c) {
        result[c] = sum[c];
    }
}

static TrillimStatus project_int8_q6_sdot(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Int8ProjectionTask *quantized = opaque;
    const Task *task = quantized->projection;
    const Parameters *p = task->p;
    const size_t groups = p->width / 256;
    const size_t total = groups * p->outputs;
    const bool paired = quantized->gate != NULL;
    const size_t stride = paired ? Q6_AQ_COLUMNS / 2 : Q6_AQ_COLUMNS;
    const Int8PackedWeight weight = int8_packed_weight(task->weight, total, TRILLIM_DUBGRP_INT6);
    const Int8PackedWeight gate = paired
        ? int8_packed_weight(quantized->gate, total, TRILLIM_DUBGRP_INT6) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    for (size_t row = begin; row < end; row += stride) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t count = end - row < stride ? end - row : stride;
        Int8PackedWeight tile[Q6_AQ_COLUMNS];
        for (size_t c = 0; c < Q6_AQ_COLUMNS; ++c) {
            const bool use_gate = paired && (c & 1);
            const Int8PackedWeight *operand = use_gate ? &gate : &weight;
            const size_t position = paired ? c / 2 : c;
            const size_t offset = (row + (position < count ? position : 0)) * groups;
            tile[c] = (Int8PackedWeight){
                operand->low + offset * 128, operand->scales + offset * 2,
                operand->high + offset * 64, operand->subscales + offset * 16
            };
        }
        float result[Q6_AQ_COLUMNS];
        q6_aq_tile(tile, quantized->activation, groups, result);
        for (size_t c = 0; c < count; ++c) {
            float value = result[paired ? c * 2 : c];
            if (!isfinite(value)) {
                const TrillimStatus status = project_int8_nonfinite(task, task->weight, row + c, &value);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (paired) {
                float gate_value = result[c * 2 + 1];
                if (!isfinite(gate_value)) {
                    const TrillimStatus status = project_int8_nonfinite(task, quantized->gate, row + c, &gate_value);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                value *= quantized->silu(gate_value);
            }
            store_result(task->output, row + c, TRILLIM_FP32, (TrillimValue){.f = value});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#undef Q6_AQ_UNROLL

enum { Q6_PREFILL_ROWS = 4, Q6_PREFILL_PAIRS = Q6_PREFILL_ROWS / 2, Q6_PREFILL_MAX_TOKENS = 128 };

typedef struct {
    int8x16_t values[32][Q6_PREFILL_PAIRS];
    float32x4_t scales[Q6_PREFILL_PAIRS];
} Q6PrefillActivation;

typedef struct {
    int8x16_t values[32];
    int32x4_t subscales[16];
    float32x4_t scale;
} Q6PrefillWeight;

typedef struct {
    const Task *projection;
    const uint8_t *gate;
    float (*silu)(float);
    const Q6PrefillActivation *activation;
    Q6PrefillWeight *workspace;
    atomic_bool ordinary;
} Q6PrefillTask;

static bool q6_prefill_i8mm_available(void) {
    static atomic_int cached;
    const int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value != 0) {
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

/* SMMLA multiplies two token rows by two weight columns, each eight elements wide. */
static inline int32x4_t q6_prefill_i8mm(int32x4_t accum, int8x16_t input, int8x16_t weight) {
    __asm__(".inst 0x4e80a400 | trillim_sdot_%0 | (trillim_sdot_%1 << 5) | (trillim_sdot_%2 << 16)"
            : "+w"(accum) : "w"(input), "w"(weight));
    return accum;
}

static TrillimStatus pack_q6_prefill_activation(
    const Task *task, Q6PrefillActivation *packed, size_t tiles, bool *ordinary
) {
    const Parameters *p = task->p;
    const size_t groups = p->width / INT8_ACTIVATION_Q8_K;
    *ordinary = false;
    for (size_t t = 0; t < tiles; ++t) {
        for (size_t g = 0; g < groups; ++g) {
            if ((g & 3) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            Q6PrefillActivation *destination = packed + t * groups + g;
            for (size_t pair = 0; pair < Q6_PREFILL_PAIRS; ++pair) {
                const size_t row = t * Q6_PREFILL_ROWS + 2 * pair;
                Q8KActivationBlock first = {0}, second = {0};
                if ((row < p->tokens && !quantize_activation_q8k(
                        task->input, row * p->width + g * INT8_ACTIVATION_Q8_K, p->input_dtype, &first, false)) ||
                    (row + 1 < p->tokens && !quantize_activation_q8k(
                        task->input, (row + 1) * p->width + g * INT8_ACTIVATION_Q8_K, p->input_dtype, &second, false))) {
                    return TRILLIM_OK;
                }
                destination->scales[pair] = (float32x4_t){first.scale, first.scale, second.scale, second.scale};
                for (size_t k = 0; k < 32; ++k) {
                    destination->values[k][pair] = vcombine_s8(
                        vld1_s8(first.values + k * 8), vld1_s8(second.values + k * 8));
                }
            }
        }
    }
    *ordinary = true;
    return TRILLIM_OK;
}

/* The expanded pair stays in worker-local scratch across all token tiles. */
static void pack_q6_prefill_weight(
    Int8PackedWeight first, Int8PackedWeight second, size_t a, size_t b, size_t groups, Q6PrefillWeight *packed
) {
    for (size_t g = 0; g < groups; ++g) {
        Q6PrefillWeight *destination = packed + g;
        const float scale0 = int8_weight_scale(first.scales, a + g);
        const float scale1 = int8_weight_scale(second.scales, b + g);
        destination->scale = (float32x4_t){scale0, scale1, scale0, scale1};
        const uint8_t *low0 = first.low + (a + g) * 128;
        const uint8_t *low1 = second.low + (b + g) * 128;
        const uint8_t *high0 = first.high + (a + g) * 64;
        const uint8_t *high1 = second.high + (b + g) * 64;
        const int8_t *sub0 = first.subscales + (a + g) * 16;
        const int8_t *sub1 = second.subscales + (b + g) * 16;
        for (size_t j = 0; j < 16; ++j) {
            const size_t block = j / 2;
            const size_t low = block / 2 * 32 + j % 2 * 16;
            const size_t high = block / 4 * 32 + j % 2 * 16;
            const int8x16_t shift = vdupq_n_s8(-2 * (int)(block % 4));
            const uint8x16_t h0 = vshlq_u8(veorq_u8(vld1q_u8(high0 + high), vdupq_n_u8(0xaa)), shift);
            const uint8x16_t h1 = vshlq_u8(veorq_u8(vld1q_u8(high1 + high), vdupq_n_u8(0xaa)), shift);
            const int8x16_t w0 = q6_signed4(vld1q_u8(low0 + low), h0, block % 2 != 0);
            const int8x16_t w1 = q6_signed4(vld1q_u8(low1 + low), h1, block % 2 != 0);
            destination->values[j * 2] = vcombine_s8(vget_low_s8(w0), vget_low_s8(w1));
            destination->values[j * 2 + 1] = vcombine_s8(vget_high_s8(w0), vget_high_s8(w1));
            destination->subscales[j] = (int32x4_t){sub0[j], sub1[j], sub0[j], sub1[j]};
        }
    }
}

#if defined(__clang__)
#define Q6_PREFILL_UNROLL _Pragma("clang loop unroll(full)")
#else
#define Q6_PREFILL_UNROLL
#endif

static void dot_q6_prefill(
    const Q6PrefillWeight *restrict weight, const Q6PrefillActivation *restrict input,
    size_t groups, float result[Q6_PREFILL_PAIRS][4]
) {
    float32x4_t total[Q6_PREFILL_PAIRS] = {0};
    for (size_t g = 0; g < groups; ++g) {
        const Q6PrefillWeight *w = weight + g;
        const Q6PrefillActivation *x = input + g;
        int32x4_t accum[Q6_PREFILL_PAIRS] = {0};
        for (size_t j = 0; j < 16; ++j) {
            const int8x16_t w0 = w->values[2 * j], w1 = w->values[2 * j + 1];
            const int32x4_t s = w->subscales[j];
            Q6_PREFILL_UNROLL
            for (size_t pair = 0; pair < Q6_PREFILL_PAIRS; ++pair) {
                int32x4_t dot = q6_prefill_i8mm(vdupq_n_s32(0), x->values[2 * j][pair], w0);
                dot = q6_prefill_i8mm(dot, x->values[2 * j + 1][pair], w1);
                accum[pair] = vmlaq_s32(accum[pair], dot, s);
            }
        }
        Q6_PREFILL_UNROLL
        for (size_t pair = 0; pair < Q6_PREFILL_PAIRS; ++pair) {
            const float32x4_t scale = vmulq_f32(w->scale, x->scales[pair]);
            /* q6_signed4 scales each centered code by four for byte assembly. */
            const float32x4_t dot = vcvtq_f32_s32(vshrq_n_s32(accum[pair], 2));
            total[pair] = vfmaq_f32(total[pair], dot, scale);
        }
    }
    Q6_PREFILL_UNROLL
    for (size_t pair = 0; pair < Q6_PREFILL_PAIRS; ++pair) {
        vst1q_f32(result[pair], total[pair]);
    }
}

#undef Q6_PREFILL_UNROLL

static TrillimStatus project_q6_prefill(void *opaque, size_t index, size_t worker) {
    Q6PrefillTask *job = opaque;
    const Task *task = job->projection;
    const Parameters *p = task->p;
    const size_t groups = p->width / INT8_ACTIVATION_Q8_K;
    const size_t total = groups * p->outputs;
    const bool paired = job->gate != NULL;
    const size_t stride = paired ? 1 : 2;
    const Int8PackedWeight weight = int8_packed_weight(task->weight, total, TRILLIM_DUBGRP_INT6);
    const Int8PackedWeight second = paired
        ? int8_packed_weight(job->gate, total, TRILLIM_DUBGRP_INT6) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    Q6PrefillWeight *packed = job->workspace + worker * groups;
    if (!fhm_default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    for (size_t column = begin; column < end; column += stride) {
        if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) {
            return TRILLIM_OK;
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const bool has_next = !paired && column + 1 < end;
        pack_q6_prefill_weight(weight, second, column * groups, (column + has_next) * groups, groups, packed);
        for (size_t row = 0; row < p->tokens; row += Q6_PREFILL_ROWS) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float result[Q6_PREFILL_PAIRS][4];
            dot_q6_prefill(packed, job->activation + row / Q6_PREFILL_ROWS * groups, groups, result);
            const size_t count = p->tokens - row < Q6_PREFILL_ROWS ? p->tokens - row : Q6_PREFILL_ROWS;
            for (size_t r = 0; r < count; ++r) {
                const float a = result[r / 2][2 * (r % 2)], b = result[r / 2][2 * (r % 2) + 1];
                if (!isfinite(a) || !isfinite(b)) {
                    atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                    return TRILLIM_OK;
                }
                const size_t destination = (row + r) * p->outputs + column;
                ((float *)task->output)[destination] = paired ? a * job->silu(b) : a;
                if (has_next) {
                    ((float *)task->output)[destination + 1] = b;
                }
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus try_q6_prefill(
    const Task *task, const uint8_t *gate, float (*silu)(float), size_t workers, bool *handled
) {
    const Parameters *p = task->p;
    if (p->weight_dtype != TRILLIM_DUBGRP_INT6 || p->tokens < 4 || p->tokens > Q6_PREFILL_MAX_TOKENS ||
        p->width < INT8_ACTIVATION_Q8_K || p->width % INT8_ACTIVATION_Q8_K != 0 ||
        p->width > INT8_ACTIVATION_MAX_WIDTH || p->outputs < 16 || p->output_dtype != TRILLIM_FP32 ||
        (uintptr_t)task->output % _Alignof(float) != 0 || !trillim_float(p->input_dtype) ||
        !fhm_default_fpcr() || !q6_prefill_i8mm_available()) {
        return TRILLIM_OK;
    }
    const size_t groups = p->width / INT8_ACTIVATION_Q8_K;
    const size_t rows = p->tokens / Q6_PREFILL_ROWS + (p->tokens % Q6_PREFILL_ROWS != 0);
    const size_t activation_bytes = rows * groups * sizeof(Q6PrefillActivation);
    const size_t weight_stride = groups * sizeof(Q6PrefillWeight);
    if (activation_bytes > 4 * 1024 * 1024 || workers > (8 * 1024 * 1024 - activation_bytes) / weight_stride) {
        return TRILLIM_OK;
    }
    Q6PrefillActivation *activation = malloc(activation_bytes + workers * weight_stride);
    if (activation == NULL) {
        return TRILLIM_OK;
    }
    bool ordinary;
    TrillimStatus status = pack_q6_prefill_activation(task, activation, rows, &ordinary);
    if (status == TRILLIM_OK && ordinary) {
        Task scheduled = *task;
        const size_t chunk = p->outputs / workers / 2;
        scheduled.columns = chunk < 16 ? 16 : (chunk + 15) / 16 * 16;
        const size_t tiles = p->outputs / scheduled.columns + (p->outputs % scheduled.columns != 0);
        Q6PrefillWeight *workspace = (Q6PrefillWeight *)((uint8_t *)activation + activation_bytes);
        Q6PrefillTask job = {&scheduled, gate, silu, activation, workspace, ATOMIC_VAR_INIT(true)};
        const TrillimThreadPool *pool = task->context->thread_pool;
        status = tiles == 1 ? project_q6_prefill(&job, 0, 0) :
            pool->run(pool->context, tiles, project_q6_prefill, &job);
        *handled = atomic_load_explicit(&job.ordinary, memory_order_relaxed);
    }
    free(activation);
    return status;
}

DEFINE_INT8_PROJECTION(project_int8_q8_sdot, false, false, TRILLIM_GRP_INT8)
DEFINE_INT8_PROJECTION(project_int8_q8_sdot_pair, true, false, TRILLIM_GRP_INT8)
static TrillimTask int8_projection(uint32_t dtype, bool paired, size_t width) {
    const bool dotprod = available_dotprod();
    switch (dtype) {
    case TRILLIM_GRP_INT4:
        if (paired && width > 2048) {
            return dotprod ? project_int8_q4_sdot : project_int8_q4_neon;
        }
        return dotprod ? project_int8_q4_sdot_pair : project_int8_q4_neon_pair;
    case TRILLIM_GRP_INT5: return dotprod ? project_int8_q5_sdot : project_int8_q5_neon;
    case TRILLIM_DUBGRP_INT6: return dotprod ? project_int8_q6_sdot : project_int8_q6_neon;
    case TRILLIM_GRP_INT8:
        if (paired) {
            return dotprod ? project_int8_q8_sdot : project_int8_q8_neon;
        }
        return dotprod ? project_int8_q8_sdot_pair : project_int8_q8_neon_pair;
    default: return NULL;
    }
}

static __attribute__((noinline)) TrillimStatus try_int8_projection(
    const Task *task, const uint8_t *gate, float (*silu)(float), size_t workers, bool *handled
) {
    *handled = false;
    const Parameters *p = task->p;
    if (p->tokens != 1) {
        return try_q6_prefill(task, gate, silu, workers, handled);
    }
    if (p->output_dtype != TRILLIM_FP32 || p->width < 32 ||
        p->width > INT8_ACTIVATION_MAX_WIDTH || p->outputs < 16 ||
        !trillim_float(p->input_dtype) || !fhm_default_fpcr()) {
        return TRILLIM_OK;
    }
    const size_t width = p->weight_dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    const TrillimTask operation = int8_projection(p->weight_dtype, gate != NULL, p->width);
    if (operation == NULL || p->width % width != 0) {
        return TRILLIM_OK;
    }
    const size_t groups = p->width / width;
    const size_t storage = groups * (width == 256 ? sizeof(Q8KActivationBlock) : sizeof(Q8ActivationBlock));
    void *activation = malloc(storage);
    if (activation == NULL) {
        return TRILLIM_OK;
    }
    TrillimStatus status = TRILLIM_OK;
    for (size_t g = 0; g < groups; ++g) {
        if (g % (1024 / width) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            status = TRILLIM_CANCELLED;
            break;
        }
        const bool valid = width == 256
            ? quantize_activation_q8k(task->input, g * width, p->input_dtype,
                (Q8KActivationBlock *)activation + g, operation != project_int8_q6_sdot)
            : quantize_activation_q8(task->input, g * width, p->input_dtype, (Q8ActivationBlock *)activation + g);
        if (!valid) {
            free(activation);
            return TRILLIM_OK;
        }
    }
    if (status == TRILLIM_OK) {
        Task scheduled;
        if (operation == project_int8_q6_sdot && workers > 1) {
            scheduled = *task;
            const size_t columns = p->outputs / workers + (p->outputs % workers != 0);
            scheduled.columns = columns + (columns % 2 != 0);
            task = &scheduled;
        }
        Int8ProjectionTask int8 = {task, activation, gate, silu};
        const size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0);
        if (p->width <= 65536 / p->outputs || workers == 1) {
            Task serial = *task;
            serial.columns = p->outputs;
            Int8ProjectionTask serial_int8 = {&serial, activation, gate, silu};
            status = operation(&serial_int8, 0, 0);
        } else {
            status = task->context->thread_pool->run(task->context->thread_pool->context, tiles, operation, &int8);
        }
        *handled = true;
    }
    free(activation);
    return status;
}

#if defined(__APPLE__) || defined(__linux__)
#if defined(__APPLE__)
__asm__(".private_extern _trillim_neon_try_packed_ffn_int8");
#else
/* The leading dot keeps the helper outside the catalog's kernel symbol namespace. */
__asm__(".hidden .trillim_neon_try_packed_ffn_int8\n.globl .trillim_neon_try_packed_ffn_int8");
static TrillimStatus trillim_neon_try_packed_ffn_int8(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    const TrillimKernelContext *, bool *
) __asm__(".trillim_neon_try_packed_ffn_int8");
#endif
static __attribute__((used)) TrillimStatus trillim_neon_try_packed_ffn_int8(
    void *output, const void *input, const void *up, const void *gate, const void *parameters,
    float (*silu)(float), const TrillimKernelContext *context, bool *handled
) {
    const size_t workers = context->thread_pool->worker_count;
    if (workers == 0) {
        *handled = false;
        return TRILLIM_OK;
    }
    const Parameters *p = parameters;
    const size_t chunk = p->outputs / workers / 4;
    const size_t columns = chunk < 16 ? 16 : chunk;
    const Task task = {output, input, up, p, columns, context};
    return try_int8_projection(&task, gate, silu, workers, handled);
}
#endif

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
    bool int8_handled;
    TrillimStatus int8_status = try_int8_projection(&task, NULL, NULL, workers, &int8_handled);
    if (int8_status != TRILLIM_OK || int8_handled) {
        return int8_status;
    }

#if defined(__APPLE__)
    bool prefill_handled;
    TrillimStatus prefill = try_packed_prefill(&task, &prefill_handled);
    if (prefill != TRILLIM_OK || prefill_handled) {
        return prefill;
    }
#endif
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
