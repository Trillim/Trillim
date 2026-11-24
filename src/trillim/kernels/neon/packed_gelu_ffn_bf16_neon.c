#include "common.h"
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

/* Gelu. */
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} GeluParameters;

typedef struct {
    void *output;
    const void *input;
    const GeluParameters *p;
    const TrillimKernelContext *context;
} GeluTask;

static float gelu_round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float gelu_gelu_tanh(float x, uint32_t dtype) {
    float square = gelu_round_dtype(x * x, dtype);
    float cube = gelu_round_dtype(square * x, dtype);
    float correction = gelu_round_dtype(cube * gelu_round_dtype(0.044715f, dtype), dtype);
    float sum = gelu_round_dtype(x + correction, dtype);
    float argument = gelu_round_dtype(sum * gelu_round_dtype(0.7978845608028654f, dtype), dtype);
    float scaled = gelu_round_dtype(argument * 2.0f, dtype);
    float e = gelu_round_dtype(expf(-fabsf(scaled)), dtype);
    float denominator = gelu_round_dtype(e + 1.0f, dtype);
    float probability = scaled >= 0.0f ? 1.0f / denominator : gelu_round_dtype(e / denominator, dtype);
    float cdf = probability * 2.0f - 1.0f;
    float half = gelu_round_dtype(x * 0.5f, dtype);
    return half * (cdf + 1.0f);
}

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
#define gelu_TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define gelu_TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

static float32x4_t gelu_exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

static float32x4_t gelu_gelu_tanh4(float32x4_t x) {
    float32x4_t cube = vmulq_f32(vmulq_f32(x, x), x);
    float32x4_t sum = vaddq_f32(x, vmulq_n_f32(cube, 0.044715f));
    float32x4_t argument = vmulq_n_f32(sum, 0.7978845608028654f);
    float32x4_t scaled = vmulq_n_f32(argument, 2.0f);
    float32x4_t e = gelu_exp4(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0.0f)), positive, negative);
    float32x4_t cdf = vsubq_f32(vmulq_n_f32(probability, 2.0f), vdupq_n_f32(1.0f));
    return vmulq_f32(vmulq_n_f32(x, 0.5f), vaddq_f32(cdf, vdupq_n_f32(1.0f)));
}

#if defined(__APPLE__) || defined(__linux__)
static float32x4_t gelu_tanh4_prefill(float32x4_t x) {
    return gelu_gelu_tanh4(x);
}
#endif

static TrillimStatus gelu_activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const GeluTask *task = opaque;
    const GeluParameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;
    if (p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_FP32) {
        for (; end - i >= 4; i += 4) {
            float32x4_t input = vld1q_f32((const float *)task->input + i);
            vst1q_f32((float *)task->output + i, gelu_gelu_tanh4(input));
        }
    }
    for (; i < end; ++i) {
        float input = trillim_load(task->input, i, p->input_dtype).f;
        TrillimValue value = {.f = gelu_gelu_tanh(input, p->input_dtype)};
        trillim_store(task->output, i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_gelu_tanh(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const GeluParameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    GeluTask task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return gelu_activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, gelu_activate, &task);
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

/* Gelu mul. */

/* One tile of the GELU kernel keeps its call on the current worker. */
enum { gelu_mul_TILE = 1024 };

typedef struct {
    size_t count;
} GeluMulParameters;

typedef struct {
    float *output;
    const float *up;
    const float *gate;
    size_t count;
    const TrillimKernelContext *context;
} GeluMulTask;

static TrillimStatus gelu_mul_activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const GeluMulTask *task = opaque;
    size_t begin = tile * gelu_mul_TILE;
    size_t count = task->count - begin < gelu_mul_TILE ? task->count - begin : gelu_mul_TILE;
    float activation[gelu_mul_TILE];
    const void *inputs[] = {task->gate + begin};
    void *outputs[] = {activation};
    GeluParameters gelu = {count, TRILLIM_FP32, TRILLIM_FP32};
    TrillimStatus status = apply_gelu_tanh(outputs, inputs, NULL, &gelu, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    const float *up = task->up + begin;
    float *output = task->output + begin;
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        vst1q_f32(output + i, vmulq_f32(vld1q_f32(up + i), vld1q_f32(activation + i)));
    }
    for (; i < count; ++i) {
        output[i] = up[i] * activation[i];
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_gelu_tanh_mul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const GeluMulParameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    GeluMulTask task = {outputs[0], inputs[0], inputs[1], p->count, context};
    size_t tiles = p->count / gelu_mul_TILE + (p->count % gelu_mul_TILE != 0);
    if (tiles == 1) {
        return gelu_mul_activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, gelu_mul_activate, &task);
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

/* Linear. */
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
#define linear_FHM
static inline float32x4_t linear_fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x4e20ec00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline float32x4_t linear_fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x6e20cc00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
/* Function targets disappear before the builder's separate assembly step. */
__asm__(".arch armv8.6-a+bf16+fp16fml");
#if defined(__clang__)
#define linear_FHM __attribute__((target("fp16fml")))
#else
#define linear_FHM __attribute__((target("+fp16fml")))
#endif
linear_FHM static inline float32x4_t linear_fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_low_f16(sum, a, b);
}
linear_FHM static inline float32x4_t linear_fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_high_f16(sum, a, b);
}
#endif

/* Apple Clang 17 crashes while optimizing the BFDOT loop. */
#if !defined(__apple_build_version__) || __clang_major__ > 17
#define linear_TRILLIM_PACKED_BFDOT 1
#if defined(__clang__)
#define linear_BF16 __attribute__((target("bf16")))
#else
#define linear_BF16 __attribute__((target("+bf16")))
#endif
linear_BF16 static inline float32x4_t dot_bf16(float32x4_t sum, uint16x8_t a, uint16x8_t b) {
    return vbfdotq_f32(sum, vreinterpretq_bf16_u16(a), vreinterpretq_bf16_u16(b));
}
#endif

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} PackedLinearParameters;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *weight;
    const PackedLinearParameters *p;
    size_t columns;
    const TrillimKernelContext *context;
} LinearTask;

static size_t linear_group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
}

static float32x4_t linear_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_BF16) {
        return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(bits));
}

static float32x4_t linear_convert4(int16x4_t codes, float scale) {
    return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(codes)), scale);
}

static inline __attribute__((always_inline)) void linear_decode16(
    const LinearTask *task, size_t column, size_t k, uint32_t dtype, float32x4_t result[4]
) {
    const PackedLinearParameters *p = task->p;
    const uint8_t *values = task->weight;
    size_t group = linear_group_width(dtype);
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
            result[0] = linear_convert4(vget_low_s16(a), scale);
            result[1] = linear_convert4(vget_high_s16(a), scale);
            result[2] = linear_convert4(vget_low_s16(b), scale);
            result[3] = linear_convert4(vget_high_s16(b), scale);
            return;
        }
    }
    int16x8_t low = vmovl_s8(vget_low_s8(codes)), high = vmovl_s8(vget_high_s8(codes));
    result[0] = linear_convert4(vget_low_s16(low), scale);
    result[1] = linear_convert4(vget_high_s16(low), scale);
    result[2] = linear_convert4(vget_low_s16(high), scale);
    result[3] = linear_convert4(vget_high_s16(high), scale);
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        int32x4_t code[4] = {vmovl_s16(vget_low_s16(low)), vmovl_s16(vget_high_s16(low)),
            vmovl_s16(vget_low_s16(high)), vmovl_s16(vget_high_s16(high))};
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vbslq_f32(vceqq_s32(code[i], vdupq_n_s32(2)), vdupq_n_f32(NAN), result[i]);
        }
    }
}

static inline float linear_trillim_read_float(
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

static inline __attribute__((always_inline)) void linear_store_result(
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
} linear_PackedGroupRow;

static inline linear_PackedGroupRow linear_packed_group_row(
    const uint8_t *data, size_t width, size_t outputs, size_t column, uint32_t dtype
) {
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    size_t groups = width / group + (width % group != 0);
    size_t block = column * groups, total = outputs * groups;
    size_t code_bytes = dtype == TRILLIM_DUBGRP_INT6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    linear_PackedGroupRow row = {
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

static inline float32x4_t linear_packed_group_scaled(int16x4_t codes, float scale, bool factored) {
    float32x4_t value = vcvtq_f32_s32(vmovl_s16(codes));
    return factored ? value : vmulq_n_f32(value, scale);
}

static inline void linear_packed_group_widen(
    int8x16_t codes, bool scaled, int8_t subscale, float scale, bool factored, float32x4_t result[4]
) {
    int16x8_t low = vmovl_s8(vget_low_s8(codes));
    int16x8_t high = vmovl_high_s8(codes);
    if (scaled) {
        low = vmulq_n_s16(low, subscale);
        high = vmulq_n_s16(high, subscale);
    }
    result[0] = linear_packed_group_scaled(vget_low_s16(low), scale, factored);
    result[1] = linear_packed_group_scaled(vget_high_s16(low), scale, factored);
    result[2] = linear_packed_group_scaled(vget_low_s16(high), scale, factored);
    result[3] = linear_packed_group_scaled(vget_high_s16(high), scale, factored);
}

static inline float16x8x2_t linear_packed_group_q4_half(uint8x16_t codes) {
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

static inline void linear_packed_group_q4(uint8x16_t codes, float scale, bool factored, float32x4_t result[4]) {
    float16x8x2_t values = linear_packed_group_q4_half(codes);
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

static inline uint8x16_t linear_packed_group_q5_high(const uint8_t *high) {
    static const uint8_t masks[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
    return vandq_u8(vtstq_u8(bytes, vld1q_u8(masks)), vdupq_n_u8(16));
}

static inline __attribute__((always_inline)) uint8x16x2_t linear_packed_group_codes(
    const linear_PackedGroupRow *row, size_t k, uint32_t dtype
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
        result.val[0] = vorrq_u8(result.val[0], linear_packed_group_q5_high(high));
        result.val[1] = vorrq_u8(result.val[1], linear_packed_group_q5_high(high + 2));
    }
    return result;
}

static inline float linear_packed_group_scale(const linear_PackedGroupRow *row, size_t k, uint32_t dtype) {
    size_t index = dtype == TRILLIM_DUBGRP_INT6 ? k / 256 : k / 32;
    uint16_t bits;
    memcpy(&bits, row->scales + index * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline __attribute__((always_inline)) void linear_packed_group_decode32(
    const linear_PackedGroupRow *row, size_t k, uint32_t dtype, bool factored, float32x4_t result[8]
) {
    uint8x16x2_t codes = linear_packed_group_codes(row, k, dtype);
    float scale = factored ? 0 : linear_packed_group_scale(row, k, dtype);
    if (dtype == TRILLIM_GRP_INT4) {
        linear_packed_group_q4(codes.val[0], scale, factored, result);
        linear_packed_group_q4(codes.val[1], scale, factored, result + 4);
        return;
    }
    bool has_subscale = dtype == TRILLIM_DUBGRP_INT6;
    uint8_t zero = has_subscale ? 32 : (dtype == TRILLIM_GRP_INT5 ? 16 : 0);
    const int8_t *subscales = has_subscale ? (const int8_t *)row->subscales + k / 16 : NULL;
    for (size_t i = 0; i < 2; ++i) {
        int8x16_t centered = vreinterpretq_s8_u8(vsubq_u8(codes.val[i], vdupq_n_u8(zero)));
        linear_packed_group_widen(centered, has_subscale, has_subscale ? subscales[i] : 1, scale, factored, result + i * 4);
    }
}

#define linear_DEFINE_PROJECT(NAME, DTYPE, MTILE, KSTEP) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const LinearTask *task = opaque; \
    const PackedLinearParameters *p = task->p; \
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
        linear_PackedGroupRow group0, group1; \
        if (KSTEP == 32) { \
            group0 = linear_packed_group_row(task->weight, p->width, p->outputs, column, DTYPE); \
            group1 = linear_packed_group_row(task->weight, p->width, p->outputs, second, DTYPE); \
        } \
        size_t k = 0; \
        for (; p->width - k >= KSTEP; k += KSTEP) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t w0[KSTEP / 4], w1[KSTEP / 4]; \
            if (KSTEP == 32) { \
                linear_packed_group_decode32(&group0, k, DTYPE, false, w0); \
                linear_packed_group_decode32(&group1, k, DTYPE, false, w1); \
            } else { \
                linear_decode16(task, column, k, DTYPE, w0); \
                linear_decode16(task, second, k, DTYPE, w1); \
            } \
            for (size_t j = 0; j < KSTEP / 4; ++j) { \
                float32x4_t x = linear_load4(task->input, row * p->width + k + j * 4, p->input_dtype); \
                s00 = vfmaq_f32(s00, x, w0[j]); \
                s01 = vfmaq_f32(s01, x, w1[j]); \
                if (rows > 1) { \
                    x = linear_load4(task->input, (row + 1) * p->width + k + j * 4, p->input_dtype); \
                    s10 = vfmaq_f32(s10, x, w0[j]); \
                    s11 = vfmaq_f32(s11, x, w1[j]); \
                } \
                if (rows > 2) { \
                    x = linear_load4(task->input, (row + 2) * p->width + k + j * 4, p->input_dtype); \
                    s20 = vfmaq_f32(s20, x, w0[j]); \
                    s21 = vfmaq_f32(s21, x, w1[j]); \
                } \
                if (rows > 3) { \
                    x = linear_load4(task->input, (row + 3) * p->width + k + j * 4, p->input_dtype); \
                    s30 = vfmaq_f32(s30, x, w0[j]); \
                    s31 = vfmaq_f32(s31, x, w1[j]); \
                } \
                if (rows > 4) { \
                    x = linear_load4(task->input, (row + 4) * p->width + k + j * 4, p->input_dtype); \
                    s40 = vfmaq_f32(s40, x, w0[j]); \
                    s41 = vfmaq_f32(s41, x, w1[j]); \
                } \
                if (rows > 5) { \
                    x = linear_load4(task->input, (row + 5) * p->width + k + j * 4, p->input_dtype); \
                    s50 = vfmaq_f32(s50, x, w0[j]); \
                    s51 = vfmaq_f32(s51, x, w1[j]); \
                } \
                if (rows > 6) { \
                    x = linear_load4(task->input, (row + 6) * p->width + k + j * 4, p->input_dtype); \
                    s60 = vfmaq_f32(s60, x, w0[j]); \
                    s61 = vfmaq_f32(s61, x, w1[j]); \
                } \
                if (rows > 7) { \
                    x = linear_load4(task->input, (row + 7) * p->width + k + j * 4, p->input_dtype); \
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
            float w0 = linear_trillim_read_float(task->weight, column * p->width + k, p->weight_dtype, p->outputs, p->width); \
            float w1 = linear_trillim_read_float(task->weight, second * p->width + k, p->weight_dtype, p->outputs, p->width); \
            for (size_t r = 0; r < rows; ++r) { \
                float x = trillim_load(task->input, (row + r) * p->width + k, p->input_dtype).f; \
                sums[r][0] += x * w0; \
                sums[r][1] += x * w1; \
            } \
        } \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue value = {.f = sums[r][0]}; \
            linear_store_result(task->output, (row + r) * p->outputs + column, p->output_dtype, value); \
            if (second != column) { \
                value.f = sums[r][1]; \
                linear_store_result(task->output, (row + r) * p->outputs + second, p->output_dtype, value); \
            } \
        } \
    } \
    return TRILLIM_OK; \
}

linear_DEFINE_PROJECT(linear_project_row_int8_4, TRILLIM_ROW_INT8, 4, 16)
linear_DEFINE_PROJECT(linear_project_mat_ternary_4, TRILLIM_MAT_TERNARY, 4, 16)
linear_DEFINE_PROJECT(linear_project_grp_int8_4, TRILLIM_GRP_INT8, 4, 16)
linear_DEFINE_PROJECT(linear_project_grp_int4_4, TRILLIM_GRP_INT4, 4, 16)
linear_DEFINE_PROJECT(linear_project_grp_int5_4, TRILLIM_GRP_INT5, 4, 16)
linear_DEFINE_PROJECT(linear_project_dubgrp_int6_4, TRILLIM_DUBGRP_INT6, 4, 16)
linear_DEFINE_PROJECT(linear_project_grp_binary_4, TRILLIM_GRP_BINARY, 4, 16)
linear_DEFINE_PROJECT(linear_project_grp_ternary_4, TRILLIM_GRP_TERNARY, 4, 16)
linear_DEFINE_PROJECT(linear_project_mxfp4_4, TRILLIM_MXFP4, 4, 16)
linear_DEFINE_PROJECT(linear_project_mat_ternary_8, TRILLIM_MAT_TERNARY, 8, 16)
linear_DEFINE_PROJECT(linear_project_grp_int5_8, TRILLIM_GRP_INT5, 8, 16)
linear_DEFINE_PROJECT(linear_project_dubgrp_int6_8, TRILLIM_DUBGRP_INT6, 8, 16)
linear_DEFINE_PROJECT(linear_project_grp_binary_8, TRILLIM_GRP_BINARY, 8, 16)
linear_DEFINE_PROJECT(linear_project_grp_ternary_8, TRILLIM_GRP_TERNARY, 8, 16)
linear_DEFINE_PROJECT(linear_project_mxfp4_8, TRILLIM_MXFP4, 8, 16)

linear_DEFINE_PROJECT(linear_project_grp_int4_group, TRILLIM_GRP_INT4, 4, 32)
linear_DEFINE_PROJECT(linear_project_grp_int5_group, TRILLIM_GRP_INT5, 8, 32)
linear_DEFINE_PROJECT(linear_project_dubgrp_int6_group, TRILLIM_DUBGRP_INT6, 8, 32)

#undef linear_DEFINE_PROJECT

#define linear_DEFINE_GROUP_DECODE(NAME, DTYPE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const LinearTask *task = opaque; \
    const PackedLinearParameters *p = task->p; \
    size_t first = tile * task->columns; \
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns; \
    for (size_t column = first; column < end; column += 4) { \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        size_t count = end - column < 4 ? end - column : 4; \
        linear_PackedGroupRow group[4]; \
        float32x4_t sums[4]; \
        for (size_t c = 0; c < 4; ++c) { \
            group[c] = linear_packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), DTYPE); \
            sums[c] = vdupq_n_f32(0); \
        } \
        for (size_t k = 0; k < p->width; k += 32) { \
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t x[8]; \
            for (size_t j = 0; j < 8; ++j) { \
                x[j] = linear_load4(task->input, k + j * 4, p->input_dtype); \
            } \
            for (size_t c = 0; c < 4; ++c) { \
                float32x4_t weights[8], sum = sums[c]; \
                linear_packed_group_decode32(&group[c], k, DTYPE, false, weights); \
                for (size_t j = 0; j < 8; ++j) { \
                    sum = vfmaq_f32(sum, x[j], weights[j]); \
                } \
                sums[c] = sum; \
            } \
        } \
        for (size_t c = 0; c < count; ++c) { \
            linear_store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])}); \
        } \
    } \
    return TRILLIM_OK; \
}

linear_DEFINE_GROUP_DECODE(linear_project_grp_int4_decode, TRILLIM_GRP_INT4)
linear_DEFINE_GROUP_DECODE(linear_project_grp_int5_decode, TRILLIM_GRP_INT5)
linear_DEFINE_GROUP_DECODE(linear_project_dubgrp_int6_decode, TRILLIM_DUBGRP_INT6)
linear_DEFINE_GROUP_DECODE(linear_project_grp_int8_decode, TRILLIM_GRP_INT8)

#undef linear_DEFINE_GROUP_DECODE

/* Nonzero inputs and scales in [2^-40, 2^41) keep factored dots in the normal FP32 range. */
enum { linear_MIN_GROUP_EXPONENT = 87, linear_MAX_GROUP_EXPONENT = 167, linear_MAX_FACTORED_WIDTH = 1 << 16 };

static TrillimStatus linear_bounded_group_scales(
    const uint8_t *scales, size_t count, const TrillimKernelContext *context, bool *valid
) {
    const unsigned minimum = linear_MIN_GROUP_EXPONENT << 7;
    const unsigned span = ((linear_MAX_GROUP_EXPONENT - linear_MIN_GROUP_EXPONENT + 1) << 7) - 1;
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

#if defined(linear_TRILLIM_PACKED_BFDOT)
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

static TrillimStatus linear_check_factored_input(const LinearTask *task, bool *bounded) {
    const PackedLinearParameters *p = task->p;
    const uint32x4_t minimum = vdupq_n_u32((uint32_t)linear_MIN_GROUP_EXPONENT << 23);
    const uint32x4_t span = vdupq_n_u32(((uint32_t)(linear_MAX_GROUP_EXPONENT - linear_MIN_GROUP_EXPONENT + 1) << 23) - 1);
    uint32x4_t rejected = vdupq_n_u32(0);
    for (size_t k = 0; k < p->width; k += 4) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(linear_load4(task->input, k, p->input_dtype)),
            vdupq_n_u32(UINT32_C(0x7fffffff)));
        uint32x4_t outside = vcgtq_u32(vsubq_u32(magnitude, minimum), span);
        rejected = vorrq_u32(rejected, vbicq_u32(outside, vceqzq_u32(magnitude)));
    }
    *bounded = vmaxvq_u32(rejected) == 0;
    return TRILLIM_OK;
}

#if defined(linear_TRILLIM_PACKED_BFDOT)
linear_BF16 static TrillimStatus project_bf16_int5(void *opaque, size_t tile, size_t worker) {
    const LinearTask *task = opaque;
    const PackedLinearParameters *p = task->p;
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t groups = p->width / 32;
    linear_PackedGroupRow first_row = linear_packed_group_row(task->weight, p->width, p->outputs, first, TRILLIM_GRP_INT5);
    bool valid;
    TrillimStatus status = linear_bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!valid) {
        return linear_project_grp_int5_decode(opaque, tile, worker);
    }
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        linear_PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = linear_packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), TRILLIM_GRP_INT5);
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
                uint8x16x2_t codes = linear_packed_group_codes(&row[c], k, TRILLIM_GRP_INT5);
                uint16x8x2_t lo = bf16_from_q5(codes.val[0]);
                uint16x8x2_t hi = bf16_from_q5(codes.val[1]);
                float32x4_t sum = dot_bf16(vdupq_n_f32(0), x[0], lo.val[0]);
                sum = dot_bf16(sum, x[1], lo.val[1]);
                sum = dot_bf16(sum, x[2], hi.val[0]);
                sum = dot_bf16(sum, x[3], hi.val[1]);
                partial[c] = sum;
            }
            for (size_t c = 0; c < 4; ++c) {
                float scale = linear_packed_group_scale(&row[c], k, TRILLIM_GRP_INT5);
                sums[c] = vfmaq_n_f32(sums[c], partial[c], scale);
            }
        }
        for (size_t c = 0; c < count; ++c) {
            linear_store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

#endif

/* Keep activation and code products in FP32 and apply each scale after its block dot. */
static inline __attribute__((always_inline)) TrillimStatus linear_project_factored_group(
    void *opaque, size_t tile, uint32_t dtype, TrillimTask fallback
) {
    const LinearTask *task = opaque;
    const PackedLinearParameters *p = task->p;
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    size_t groups = p->width / group + (p->width % group != 0);
    linear_PackedGroupRow first_row = linear_packed_group_row(task->weight, p->width, p->outputs, first, dtype);
    bool valid;
    TrillimStatus status = linear_bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
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
        linear_PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = linear_packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), dtype);
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
                    x[j] = linear_load4(task->input, k + offset + j * 4, p->input_dtype);
                }
                for (size_t c = 0; c < 4; ++c) {
                    float32x4_t weights[8];
                    linear_packed_group_decode32(&row[c], k + offset, dtype, true, weights);
                    float32x4_t lo = partial[c], hi = vdupq_n_f32(0);
                    for (size_t j = 0; j < 4; ++j) {
                        lo = vfmaq_f32(lo, x[j], weights[j]);
                        hi = vfmaq_f32(hi, x[j + 4], weights[j + 4]);
                    }
                    partial[c] = vaddq_f32(lo, hi);
                }
            }
            for (size_t c = 0; c < 4; ++c) {
                sums[c] = vfmaq_n_f32(sums[c], partial[c], linear_packed_group_scale(&row[c], k, dtype));
            }
        }
        for (size_t c = 0; c < count; ++c) {
            linear_store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

#define linear_DEFINE_FACTORED_PROJECT(NAME, DTYPE, FALLBACK) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    return linear_project_factored_group(opaque, tile, DTYPE, FALLBACK); \
}

linear_DEFINE_FACTORED_PROJECT(linear_project_factored_int4, TRILLIM_GRP_INT4, linear_project_grp_int4_decode)
linear_DEFINE_FACTORED_PROJECT(linear_project_factored_int5, TRILLIM_GRP_INT5, linear_project_grp_int5_decode)
linear_DEFINE_FACTORED_PROJECT(linear_project_factored_int6, TRILLIM_DUBGRP_INT6, linear_project_dubgrp_int6_decode)
linear_DEFINE_FACTORED_PROJECT(linear_project_factored_int8, TRILLIM_GRP_INT8, linear_project_grp_int8_decode)

#undef linear_DEFINE_FACTORED_PROJECT

static TrillimTask linear_factored_projection(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return linear_project_factored_int4;
    case TRILLIM_GRP_INT5: return linear_project_factored_int5;
    case TRILLIM_DUBGRP_INT6: return linear_project_factored_int6;
    case TRILLIM_GRP_INT8: return linear_project_factored_int8;
    default: return NULL;
    }
}

enum { linear_MAX_FHM_WIDTH = 8192 };

typedef struct {
    LinearTask projection;
    const uint8_t *input;
} linear_FhmTask;

static bool linear_available_fhm(void) {
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

static bool linear_fhm_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus linear_prepare_fhm_input(const LinearTask *task, uint16_t *input, bool *valid) {
    const PackedLinearParameters *p = task->p;
    uint32x4_t changed = vdupq_n_u32(0);
    for (size_t k = 0; k < p->width; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t low = linear_load4(task->input, k, p->input_dtype);
        float32x4_t high = linear_load4(task->input, k + 4, p->input_dtype);
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

linear_FHM static TrillimStatus linear_project_fhm_int4(void *opaque, size_t tile, size_t worker) {
    linear_FhmTask *fhm = opaque;
    const LinearTask *task = &fhm->projection;
    const PackedLinearParameters *p = task->p;
    if (!linear_fhm_default_fpcr()) {
        return linear_project_factored_int4(&fhm->projection, tile, worker);
    }
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t groups = p->width / 32;
    linear_PackedGroupRow first_row = linear_packed_group_row(task->weight, p->width, p->outputs, first, TRILLIM_GRP_INT4);
    bool valid;
    TrillimStatus status = linear_bounded_group_scales(first_row.scales, (end - first) * groups, task->context, &valid);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!valid) {
        return linear_project_grp_int4_decode(&fhm->projection, tile, worker);
    }
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        linear_PackedGroupRow row[4];
        float32x4_t sums[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            row[c] = linear_packed_group_row(task->weight, p->width, p->outputs, column + (c < count ? c : 0), TRILLIM_GRP_INT4);
        }
        for (size_t k = 0; k < p->width; k += 32) {
            if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8x16x4_t input = vld1q_u8_x4(fhm->input + k * sizeof(uint16_t));
            float16x8_t x0 = vreinterpretq_f16_u8(input.val[0]), x1 = vreinterpretq_f16_u8(input.val[1]);
            float16x8_t x2 = vreinterpretq_f16_u8(input.val[2]), x3 = vreinterpretq_f16_u8(input.val[3]);
            for (size_t c = 0; c < 4; ++c) {
                uint8x16x2_t codes = linear_packed_group_codes(&row[c], k, TRILLIM_GRP_INT4);
                float16x8x2_t low = linear_packed_group_q4_half(codes.val[0]);
                float16x8x2_t high = linear_packed_group_q4_half(codes.val[1]);
                float32x4_t lo = vdupq_n_f32(0), hi = lo;
                lo = linear_fhm_low(lo, x0, low.val[0]);
                hi = linear_fhm_low(hi, x2, high.val[0]);
                lo = linear_fhm_high(lo, x0, low.val[0]);
                hi = linear_fhm_high(hi, x2, high.val[0]);
                lo = linear_fhm_low(lo, x1, low.val[1]);
                hi = linear_fhm_low(hi, x3, high.val[1]);
                lo = linear_fhm_high(lo, x1, low.val[1]);
                hi = linear_fhm_high(hi, x3, high.val[1]);
                float scale = linear_packed_group_scale(&row[c], k, TRILLIM_GRP_INT4);
                sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), scale);
            }
        }
        for (size_t c = 0; c < count; ++c) {
            linear_store_result(task->output, column + c, p->output_dtype, (TrillimValue){.f = vaddvq_f32(sums[c])});
        }
    }
    return TRILLIM_OK;
}

static __attribute__((noinline)) TrillimStatus linear_run_fhm_int4(const LinearTask *task, size_t tiles, bool *handled) {
    *handled = false;
    if (!linear_available_fhm() || !linear_fhm_default_fpcr()) {
        return TRILLIM_OK;
    }
    /* Widening FMA uses the half-width activations only after a bit-exact round trip. */
    uint16_t input[linear_MAX_FHM_WIDTH];
    linear_FhmTask fhm = {*task, task->input};
    if (task->p->input_dtype != TRILLIM_FP16) {
        TrillimStatus status = linear_prepare_fhm_input(task, input, handled);
        if (status != TRILLIM_OK || !*handled) {
            return status;
        }
        fhm.input = (const uint8_t *)input;
    }
    *handled = true;
    const TrillimThreadPool *pool = task->context->thread_pool;
    return pool->run(pool->context, tiles, linear_project_fhm_int4, &fhm);
}

static TrillimTask linear_group_projection(uint32_t dtype, size_t rows) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return rows == 1 ? linear_project_grp_int4_decode : linear_project_grp_int4_group;
    case TRILLIM_GRP_INT5:
        if (rows == 1) {
            return linear_project_grp_int5_decode;
        }
        return rows == 8 ? linear_project_grp_int5_group : NULL;
    case TRILLIM_DUBGRP_INT6:
        if (rows == 1) {
            return linear_project_dubgrp_int6_decode;
        }
        return rows == 8 ? linear_project_dubgrp_int6_group : NULL;
    case TRILLIM_GRP_INT8: return rows == 1 ? linear_project_grp_int8_decode : NULL;
    default: return NULL;
    }
}

static TrillimTask linear_projection(uint32_t dtype, size_t rows) {
    if (rows == 8) {
        switch (dtype) {
        case TRILLIM_MAT_TERNARY: return linear_project_mat_ternary_8;
        case TRILLIM_GRP_INT5: return linear_project_grp_int5_8;
        case TRILLIM_DUBGRP_INT6: return linear_project_dubgrp_int6_8;
        case TRILLIM_GRP_BINARY: return linear_project_grp_binary_8;
        case TRILLIM_GRP_TERNARY: return linear_project_grp_ternary_8;
        case TRILLIM_MXFP4: return linear_project_mxfp4_8;
        default: return NULL;
        }
    }
    switch (dtype) {
    case TRILLIM_ROW_INT8: return linear_project_row_int8_4;
    case TRILLIM_MAT_TERNARY: return linear_project_mat_ternary_4;
    case TRILLIM_GRP_INT8: return linear_project_grp_int8_4;
    case TRILLIM_GRP_INT4: return linear_project_grp_int4_4;
    case TRILLIM_GRP_INT5: return linear_project_grp_int5_4;
    case TRILLIM_DUBGRP_INT6: return linear_project_dubgrp_int6_4;
    case TRILLIM_GRP_BINARY: return linear_project_grp_binary_4;
    case TRILLIM_GRP_TERNARY: return linear_project_grp_ternary_4;
    case TRILLIM_MXFP4: return linear_project_mxfp4_4;
    default: return NULL;
    }
}

static TrillimStatus linear_project_ordered(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const PackedLinearParameters *p = task->p;
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
            linear_decode16(task, column, k, p->weight_dtype, a);
            linear_decode16(task, second, k, p->weight_dtype, b);
            for (size_t j = 0; j < 4; ++j) {
                vst1q_f32(w0 + j * 4, a[j]);
                vst1q_f32(w1 + j * 4, b[j]);
            }
            for (size_t r = 0; r < rows; ++r) {
                float input[16];
                for (size_t j = 0; j < 4; ++j) {
                    vst1q_f32(input + j * 4,
                        linear_load4(task->input, (row + r) * p->width + k + j * 4, p->input_dtype));
                }
                for (size_t j = 0; j < 16; ++j) {
                    sums[r][0] = fmaf(input[j], w0[j], sums[r][0]);
                    sums[r][1] = fmaf(input[j], w1[j], sums[r][1]);
                }
            }
        }
        for (; k < p->width; ++k) {
            float w0 = linear_trillim_read_float(task->weight, column * p->width + k, p->weight_dtype, p->outputs, p->width);
            float w1 = linear_trillim_read_float(task->weight, second * p->width + k, p->weight_dtype, p->outputs, p->width);
            for (size_t r = 0; r < rows; ++r) {
                float x = trillim_load(task->input, (row + r) * p->width + k, p->input_dtype).f;
                sums[r][0] = fmaf(x, w0, sums[r][0]);
                sums[r][1] = fmaf(x, w1, sums[r][1]);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue value = {.f = sums[r][0]};
            linear_store_result(task->output, (row + r) * p->outputs + column, p->output_dtype, value);
            if (second != column) {
                value.f = sums[r][1];
                linear_store_result(task->output, (row + r) * p->outputs + second, p->output_dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

#if defined(__APPLE__)
enum {
    linear_PACKED_PREFILL_M = 64,
    linear_PACKED_PREFILL_N = 96,
    linear_PACKED_PREFILL_K = 1536,
    linear_PACKED_PREFILL_LANES = 32,
    linear_PACKED_PREFILL_ACTIVATION_ELEMENTS = 1 << 20,
};

typedef void (*linear_PackedPrefillSgemm)(int, int, int, int, int, int, float, const float *, int,
                                   const float *, int, float, float *, int);
typedef unsigned (*linear_PackedPrefillGetThreading)(void);
typedef int (*linear_PackedPrefillSetThreading)(unsigned);

static linear_PackedPrefillSgemm linear_packed_prefill_sgemm;
static linear_PackedPrefillGetThreading linear_packed_prefill_get_threading;
static linear_PackedPrefillSetThreading linear_packed_prefill_set_threading;

static bool linear_initialize_packed_prefill_blas(void) {
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
                    _Static_assert(sizeof(linear_packed_prefill_sgemm) == sizeof(gemm), "function pointer size");
                    memcpy(&linear_packed_prefill_sgemm, &gemm, sizeof(linear_packed_prefill_sgemm));
                    memcpy(&linear_packed_prefill_get_threading, &getter, sizeof(linear_packed_prefill_get_threading));
                    memcpy(&linear_packed_prefill_set_threading, &setter, sizeof(linear_packed_prefill_set_threading));
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
    return linear_packed_prefill_sgemm != NULL;
}

typedef struct {
    const LinearTask *task;
    const float *activation;
    float *weights;
    size_t columns, depth, tiles, lanes;
    atomic_bool ordinary;
} linear_PackedPrefillTask;

static inline __attribute__((always_inline)) TrillimStatus linear_project_packed_prefill_panel(
    linear_PackedPrefillTask *job, float *weights, size_t tile, uint32_t dtype
) {
    const LinearTask *task = job->task;
    const PackedLinearParameters *p = task->p;
    size_t first = tile * job->columns;
    size_t columns = p->outputs - first < job->columns ? p->outputs - first : job->columns;
    size_t group = linear_group_width(dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    linear_PackedGroupRow row[linear_PACKED_PREFILL_N];
    row[0] = linear_packed_group_row(task->weight, p->width, p->outputs, first, dtype);
    bool ordinary;
    TrillimStatus status = linear_bounded_group_scales(row[0].scales, columns * groups, task->context, &ordinary);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!ordinary) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    for (size_t c = 1; c < columns; ++c) {
        row[c] = linear_packed_group_row(task->weight, p->width, p->outputs, first + c, dtype);
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
                linear_packed_group_decode32(&row[c], begin + k, dtype, false, values);
                for (size_t j = 0; j < 8; ++j) {
                    vst1q_f32(destination + k + j * 4, values[j]);
                }
            }
        }
        for (size_t row_index = 0; row_index < p->tokens; row_index += linear_PACKED_PREFILL_M) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t batch = p->tokens - row_index < linear_PACKED_PREFILL_M ? p->tokens - row_index : linear_PACKED_PREFILL_M;
            linear_packed_prefill_sgemm(101, 111, 112, (int)batch, (int)columns, (int)depth, 1.0f,
                job->activation + row_index * p->width + begin, (int)p->width,
                weights, (int)depth, begin == 0 ? 0.0f : 1.0f,
                (float *)task->output + row_index * p->outputs + first, (int)p->outputs);
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#define linear_DEFINE_PACKED_PREFILL(NAME, DTYPE) \
static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    linear_PackedPrefillTask *job = opaque; \
    if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) { \
        return TRILLIM_OK; \
    } \
    if (!linear_fhm_default_fpcr()) { \
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed); \
        return TRILLIM_OK; \
    } \
    unsigned previous = linear_packed_prefill_get_threading(); \
    if (previous != 1 && linear_packed_prefill_set_threading(1) != 0) { \
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed); \
        return TRILLIM_OK; \
    } \
    TrillimStatus status = TRILLIM_OK; \
    float *weights = job->weights + tile * job->columns * job->depth; \
    for (size_t next = tile; next < job->tiles; next += job->lanes) { \
        status = linear_project_packed_prefill_panel(job, weights, next, DTYPE); \
        if (status != TRILLIM_OK || !atomic_load_explicit(&job->ordinary, memory_order_relaxed)) { \
            break; \
        } \
    } \
    if (previous != 1) { \
        (void)linear_packed_prefill_set_threading(previous); \
    } \
    return status; \
}

linear_DEFINE_PACKED_PREFILL(linear_project_prefill_int4, TRILLIM_GRP_INT4)
linear_DEFINE_PACKED_PREFILL(linear_project_prefill_int5, TRILLIM_GRP_INT5)
linear_DEFINE_PACKED_PREFILL(linear_project_prefill_int6, TRILLIM_DUBGRP_INT6)
linear_DEFINE_PACKED_PREFILL(linear_project_prefill_int8, TRILLIM_GRP_INT8)

#undef linear_DEFINE_PACKED_PREFILL

static TrillimTask linear_packed_prefill_operation(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4: return linear_project_prefill_int4;
    case TRILLIM_GRP_INT5: return linear_project_prefill_int5;
    case TRILLIM_DUBGRP_INT6: return linear_project_prefill_int6;
    case TRILLIM_GRP_INT8: return linear_project_prefill_int8;
    default: return NULL;
    }
}

static TrillimStatus linear_pack_prefill_activations(const LinearTask *task, float *packed, bool *ordinary) {
    const PackedLinearParameters *p = task->p;
    const uint32x4_t minimum = vdupq_n_u32((uint32_t)linear_MIN_GROUP_EXPONENT << 23);
    const uint32x4_t span = vdupq_n_u32(((uint32_t)(linear_MAX_GROUP_EXPONENT - linear_MIN_GROUP_EXPONENT + 1) << 23) - 1);
    uint32x4_t rejected = vdupq_n_u32(0);
    size_t elements = p->tokens * p->width;
    for (size_t i = 0; i < elements; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t values = linear_load4(task->input, i, p->input_dtype);
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

static TrillimStatus linear_try_packed_prefill(const LinearTask *task, bool *handled) {
    const PackedLinearParameters *p = task->p;
    const TrillimThreadPool *pool = task->context->thread_pool;
    *handled = false;
    if (pool->worker_count == 0 || p->tokens == 0 || p->outputs < 1024 || p->width < 256 || p->width % 32 != 0 ||
        p->width > INT_MAX || p->outputs > INT_MAX ||
        p->width < ((1 << 21) + p->outputs - 1) / p->outputs ||
        (uintptr_t)task->output % _Alignof(float) != 0 || p->output_dtype != TRILLIM_FP32) {
        return TRILLIM_OK;
    }
    TrillimTask operation = linear_packed_prefill_operation(p->weight_dtype);
    if (operation == NULL || !linear_fhm_default_fpcr() || !linear_initialize_packed_prefill_blas()) {
        return TRILLIM_OK;
    }
    size_t columns = p->outputs / pool->worker_count / 16 * 16;
    columns = columns < 16 ? 16 : (columns > linear_PACKED_PREFILL_N ? linear_PACKED_PREFILL_N : columns);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    size_t lanes = pool->worker_count < tiles ? pool->worker_count : tiles;
    lanes = lanes < linear_PACKED_PREFILL_LANES ? lanes : linear_PACKED_PREFILL_LANES;
    size_t depth = p->width < linear_PACKED_PREFILL_K ? p->width : linear_PACKED_PREFILL_K;
    bool direct = p->input_dtype == TRILLIM_FP32 && (uintptr_t)task->input % _Alignof(float) == 0;
    size_t capacity = direct ? p->tokens : linear_PACKED_PREFILL_ACTIVATION_ELEMENTS / p->width;
    if (capacity == 0) {
        return TRILLIM_OK;
    }
    if (capacity > p->tokens) {
        capacity = p->tokens;
    }
    if (p->tokens > capacity && capacity >= linear_PACKED_PREFILL_M) {
        capacity = capacity / linear_PACKED_PREFILL_M * linear_PACKED_PREFILL_M;
    }
    size_t activations = direct ? 0 : capacity * p->width;
    float *workspace = malloc((activations + lanes * columns * depth) * sizeof(float));
    if (workspace == NULL) {
        return TRILLIM_OK;
    }
    TrillimStatus status = TRILLIM_OK;
    for (size_t first = 0; first < p->tokens;) {
        PackedLinearParameters panel_parameters = *p;
        panel_parameters.tokens = p->tokens - first < capacity ? p->tokens - first : capacity;
        LinearTask panel = *task;
        panel.p = &panel_parameters;
        panel.input = (const uint8_t *)task->input + first * p->width * trillim_itemsize(p->input_dtype);
        panel.output = (float *)task->output + first * p->outputs;
        bool ordinary;
        status = linear_pack_prefill_activations(&panel, direct ? NULL : workspace, &ordinary);
        if (status != TRILLIM_OK || !ordinary) {
            free(workspace);
            return status;
        }
        linear_PackedPrefillTask job = {&panel, direct ? panel.input : workspace, workspace + activations,
            columns, depth, tiles, lanes, ATOMIC_VAR_INIT(true)};
        status = pool->run(pool->context, lanes, operation, &job);
        if (status != TRILLIM_OK || !atomic_load_explicit(&job.ordinary, memory_order_relaxed)) {
            free(workspace);
            return status;
        }
        first += panel_parameters.tokens;
    }
    free(workspace);
    *handled = true;
    return status;
}

#endif

enum {
    linear_INT8_ACTIVATION_MAX_WIDTH = 65536,
    linear_INT8_ACTIVATION_Q8_0 = 32,
    linear_INT8_ACTIVATION_Q8_K = 256,
};

typedef struct {
    uint16_t scale;
    int8_t values[linear_INT8_ACTIVATION_Q8_0];
} linear_Q8ActivationBlock;

typedef struct {
    float scale;
    int8_t values[linear_INT8_ACTIVATION_Q8_K];
    int16_t sums[linear_INT8_ACTIVATION_Q8_K / 16];
} linear_Q8KActivationBlock;

typedef struct {
    const LinearTask *projection;
    const void *activation;
    const uint8_t *gate;
    float (*silu)(float);
} linear_Int8ProjectionTask;

__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_sdot_v\\reg,\\reg\n"
    ".endr\n"
);

static bool linear_available_dotprod(void) {
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

static inline int32x4_t linear_int8_dot_widen(int8x16_t weight, int8x16_t input) {
    const int16x8_t low = vmull_s8(vget_low_s8(weight), vget_low_s8(input));
    const int16x8_t high = vmull_high_s8(weight, input);
    return vpaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high));
}

static inline int32x4_t linear_int8_sdot_add(int32x4_t accum, int8x16_t weight, int8x16_t input) {
    __asm__(".inst 0x4e809400 | trillim_sdot_%0 | (trillim_sdot_%1 << 5) | (trillim_sdot_%2 << 16)"
            : "+w"(accum) : "w"(weight), "w"(input));
    return accum;
}

static inline int32x4_t linear_int8_sdot(int8x16_t weight, int8x16_t input) {
    return linear_int8_sdot_add(vdupq_n_s32(0), weight, input);
}

static inline int8x16_t linear_quantize_activation_16(
    float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3, float inverse
) {
    const int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(x0, inverse));
    const int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(x1, inverse));
    const int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(x2, inverse));
    const int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(x3, inverse));
    return vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(q0), vmovn_s32(q1))),
                        vmovn_s16(vcombine_s16(vmovn_s32(q2), vmovn_s32(q3))));
}

static bool linear_quantize_activation_q8(const void *input, size_t k, uint32_t dtype, linear_Q8ActivationBlock *out) {
    float32x4_t x[8];
    float32x4_t max_vector = vdupq_n_f32(0);
    for (size_t i = 0; i < 8; ++i) {
        x[i] = linear_load4(input, k + i * 4, dtype);
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
    vst1q_s8(out->values, linear_quantize_activation_16(x[0], x[1], x[2], x[3], inverse));
    vst1q_s8(out->values + 16, linear_quantize_activation_16(x[4], x[5], x[6], x[7], inverse));
    return true;
}

static inline __attribute__((always_inline)) bool linear_quantize_activation_q8k_typed(
    const void *input, size_t k, uint32_t dtype, linear_Q8KActivationBlock *out, bool keep_sums
) {
    float values[linear_INT8_ACTIVATION_Q8_K];
    float32x4_t max0 = vdupq_n_f32(0);
    float32x4_t max1 = max0, max2 = max0, max3 = max0;
    for (size_t i = 0; i < linear_INT8_ACTIVATION_Q8_K; i += 16) {
        const float32x4_t x0 = linear_load4(input, k + i, dtype);
        const float32x4_t x1 = linear_load4(input, k + i + 4, dtype);
        const float32x4_t x2 = linear_load4(input, k + i + 8, dtype);
        const float32x4_t x3 = linear_load4(input, k + i + 12, dtype);
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
    } while (peak < linear_INT8_ACTIVATION_Q8_K);
    if (peak == linear_INT8_ACTIVATION_Q8_K) {
        return false;
    }
    const uint32x4_t lane = {0, 1, 2, 3};
    peak += vminvq_u32(vbslq_u32(matches, lane, vdupq_n_u32(4)));
    const float inverse = -127.0f / values[peak];
    if (!isfinite(inverse) || inverse == 0) {
        return false;
    }
    out->scale = 1.0f / inverse;
    for (size_t i = 0; i < linear_INT8_ACTIVATION_Q8_K; i += 16) {
        const int8x16_t x = linear_quantize_activation_16(vld1q_f32(values + i), vld1q_f32(values + i + 4),
            vld1q_f32(values + i + 8), vld1q_f32(values + i + 12), inverse);
        vst1q_s8(out->values + i, x);
        if (keep_sums) {
            out->sums[i / 16] = vaddlvq_s8(x);
        }
    }
    return true;
}

static __attribute__((noinline)) bool linear_quantize_activation_q8k(
    const void *input, size_t k, uint32_t dtype, linear_Q8KActivationBlock *out, bool keep_sums
) {
    switch (dtype) {
    case TRILLIM_FP32:
        return linear_quantize_activation_q8k_typed(input, k, TRILLIM_FP32, out, keep_sums);
    case TRILLIM_BF16:
        return linear_quantize_activation_q8k_typed(input, k, TRILLIM_BF16, out, keep_sums);
    case TRILLIM_FP16:
        return linear_quantize_activation_q8k_typed(input, k, TRILLIM_FP16, out, keep_sums);
    default:
        return false;
    }
}

static inline float linear_int8_weight_scale(const uint8_t *scales, size_t block) {
    uint16_t bits;
    memcpy(&bits, scales + block * sizeof(bits), sizeof(bits));
    return trillim_bf16(bits);
}

static inline int8x16x2_t linear_int8_decode_q5(const uint8_t *low, const uint8_t *high) {
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

static inline int8x16x2_t linear_int8_decode_q4(const uint8_t *low) {
    const uint8x16_t bits = vld1q_u8(low);
    return (int8x16x2_t){{
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bits, vdupq_n_u8(15))), vdupq_n_s8(8)),
        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bits, 4)), vdupq_n_s8(8)),
    }};
}

static inline __attribute__((always_inline)) float linear_dot_q8_activation(
    const uint8_t *low, const uint8_t *scales, const uint8_t *high, const linear_Q8ActivationBlock *x,
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
                w = linear_int8_decode_q5(low + block * 16, high + block * 4);
            } else if (dtype == TRILLIM_GRP_INT4) {
                w = linear_int8_decode_q4(low + block * 16);
            } else {
                w.val[0] = vld1q_s8((const int8_t *) low + block * 32);
                w.val[1] = vld1q_s8((const int8_t *) low + block * 32 + 16);
            }
            const int8x16_t a0 = vld1q_s8(x[block].values);
            const int8x16_t a1 = vld1q_s8(x[block].values + 16);
            partial[p] = widening ? vaddq_s32(linear_int8_dot_widen(w.val[0], a0), linear_int8_dot_widen(w.val[1], a1))
                                  : vaddq_s32(linear_int8_sdot(w.val[0], a0), linear_int8_sdot(w.val[1], a1));
            scale[p] = linear_int8_weight_scale(scales, block) * (float) vreinterpret_f16_u16(vdup_n_u16(x[block].scale))[0];
        }
        even = vfmaq_n_f32(even, vcvtq_f32_s32(partial[0]), scale[0]);
        odd = vfmaq_n_f32(odd, vcvtq_f32_s32(partial[1]), scale[1]);
    }
    float result = vaddvq_f32(even) + vaddvq_f32(odd);
    if (g < groups) {
        int8x16x2_t w;
        if (dtype == TRILLIM_GRP_INT5) {
            w = linear_int8_decode_q5(low + g * 16, high + g * 4);
        } else if (dtype == TRILLIM_GRP_INT4) {
            w = linear_int8_decode_q4(low + g * 16);
        } else {
            w.val[0] = vld1q_s8((const int8_t *) low + g * 32);
            w.val[1] = vld1q_s8((const int8_t *) low + g * 32 + 16);
        }
        const int32x4_t partial = vaddq_s32(linear_int8_dot_widen(w.val[0], vld1q_s8(x[g].values)),
            linear_int8_dot_widen(w.val[1], vld1q_s8(x[g].values + 16)));
        result += (linear_int8_weight_scale(scales, g) * trillim_fp16(x[g].scale)) * vaddvq_s32(partial);
    }
    return result;
}

static inline __attribute__((always_inline)) float linear_dot_q8k_activation(
    const uint8_t *restrict low, const uint8_t *restrict scales, const uint8_t *restrict high,
    const int8_t *restrict subscales, const linear_Q8KActivationBlock *restrict x, size_t groups, bool widening
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
                int32x4_t d0 = widening ? linear_int8_dot_widen(vreinterpretq_s8_u8(q0), a0) : linear_int8_sdot(vreinterpretq_s8_u8(q0), a0);
                int32x4_t d1 = widening ? linear_int8_dot_widen(vreinterpretq_s8_u8(q1), a1) : linear_int8_sdot(vreinterpretq_s8_u8(q1), a1);
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
        output += (linear_int8_weight_scale(scales, g) * x[g].scale) * vaddvq_s32(sum);
    }
    return output;
}

typedef struct {
    const uint8_t *low;
    const uint8_t *scales;
    const uint8_t *high;
    const int8_t *subscales;
} linear_Int8PackedWeight;

static inline linear_Int8PackedWeight linear_int8_packed_weight(const uint8_t *weight, size_t total_groups, uint32_t dtype) {
    const size_t low_size = dtype == TRILLIM_DUBGRP_INT6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *scales = weight + total_groups * low_size;
    const uint8_t *high = scales + total_groups * 2;
    const int8_t *subscales = (const int8_t *) (high + (dtype == TRILLIM_DUBGRP_INT6 ? total_groups * 64 : 0));
    return (linear_Int8PackedWeight){weight, scales, high, subscales};
}

static inline __attribute__((always_inline)) float linear_int8_dot_row(
    const linear_Int8PackedWeight *weight, size_t offset, const void *activation, size_t groups,
    uint32_t dtype, bool widening
) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return linear_dot_q8k_activation(weight->low + offset * 128, weight->scales + offset * 2,
            weight->high + offset * 64, weight->subscales + offset * 16, activation, groups, widening);
    }
    const size_t low_size = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    return linear_dot_q8_activation(weight->low + offset * low_size, weight->scales + offset * 2,
        weight->high + (dtype == TRILLIM_GRP_INT5 ? offset * 4 : 0), activation, groups, dtype, widening);
}

static inline linear_Int8PackedWeight linear_q8_activation_weight(linear_Int8PackedWeight weight, size_t offset, uint32_t dtype) {
    const size_t bytes = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    weight.low += offset * bytes;
    weight.scales += offset * 2;
    if (dtype == TRILLIM_GRP_INT5) {
        weight.high += offset * 4;
    }
    return weight;
}

static inline __attribute__((always_inline)) int32x4_t linear_q8_activation_dot(
    linear_Int8PackedWeight weight, size_t block, int8x16x2_t x, uint32_t dtype, bool widening
) {
    int8x16x2_t w;
    if (dtype == TRILLIM_GRP_INT5) {
        w = linear_int8_decode_q5(weight.low + block * 16, weight.high + block * 4);
    } else if (dtype == TRILLIM_GRP_INT4) {
        w = linear_int8_decode_q4(weight.low + block * 16);
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
    return linear_int8_sdot_add(linear_int8_sdot(w.val[0], x.val[0]), w.val[1], x.val[1]);
}

static inline __attribute__((always_inline)) void linear_dot_q8_activation_pair(
    linear_Int8PackedWeight w0, linear_Int8PackedWeight w1, const linear_Q8ActivationBlock *x, size_t groups,
    uint32_t dtype, bool widening, float result[2]
) {
    /* Match the four lane sums for even and odd blocks in the reference DOT kernel. */
    float32x4_t even0 = vdupq_n_f32(0), odd0 = even0;
    float32x4_t even1 = even0, odd1 = even0;
    size_t g = 0;
    for (; g + 1 < groups; g += 2) {
        const int8x16x2_t x0 = vld1q_s8_x2(x[g].values);
        const float scale0 = (float)vreinterpret_f16_u16(vdup_n_u16(x[g].scale))[0];
        const int32x4_t d00 = linear_q8_activation_dot(w0, g, x0, dtype, widening);
        const int32x4_t d10 = linear_q8_activation_dot(w1, g, x0, dtype, widening);
        even0 = vfmaq_n_f32(even0, vcvtq_f32_s32(d00), linear_int8_weight_scale(w0.scales, g) * scale0);
        even1 = vfmaq_n_f32(even1, vcvtq_f32_s32(d10), linear_int8_weight_scale(w1.scales, g) * scale0);

        const int8x16x2_t x1 = vld1q_s8_x2(x[g + 1].values);
        const float scale1 = (float)vreinterpret_f16_u16(vdup_n_u16(x[g + 1].scale))[0];
        const int32x4_t d01 = linear_q8_activation_dot(w0, g + 1, x1, dtype, widening);
        const int32x4_t d11 = linear_q8_activation_dot(w1, g + 1, x1, dtype, widening);
        odd0 = vfmaq_n_f32(odd0, vcvtq_f32_s32(d01), linear_int8_weight_scale(w0.scales, g + 1) * scale1);
        odd1 = vfmaq_n_f32(odd1, vcvtq_f32_s32(d11), linear_int8_weight_scale(w1.scales, g + 1) * scale1);
    }
    result[0] = vaddvq_f32(even0) + vaddvq_f32(odd0);
    result[1] = vaddvq_f32(even1) + vaddvq_f32(odd1);
    if (g < groups) {
        const int8x16x2_t values = vld1q_s8_x2(x[g].values);
        const float scale = trillim_fp16(x[g].scale);
        const int32x4_t d0 = linear_q8_activation_dot(w0, g, values, dtype, widening);
        const int32x4_t d1 = linear_q8_activation_dot(w1, g, values, dtype, widening);
        result[0] += (linear_int8_weight_scale(w0.scales, g) * scale) * vaddvq_s32(d0);
        result[1] += (linear_int8_weight_scale(w1.scales, g) * scale) * vaddvq_s32(d1);
    }
}

static __attribute__((noinline)) TrillimStatus linear_project_int8_nonfinite(
    const LinearTask *task, const uint8_t *weight, size_t row, float *result
) {
    const PackedLinearParameters *p = task->p;
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

static inline __attribute__((always_inline)) TrillimStatus linear_project_int8(
    void *opaque, size_t index, bool share_activation, bool widening, uint32_t dtype, uint32_t output_dtype
) {
    const linear_Int8ProjectionTask *quantized = opaque;
    const LinearTask *task = quantized->projection;
    const PackedLinearParameters *p = task->p;
    const bool paired = quantized->gate != NULL;
    const size_t linear_group_width = dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    const size_t groups = p->width / linear_group_width;
    const size_t total_groups = groups * p->outputs;
    const linear_Int8PackedWeight weight = linear_int8_packed_weight(task->weight, total_groups, dtype);
    const linear_Int8PackedWeight gate = paired ? linear_int8_packed_weight(quantized->gate, total_groups, dtype) : weight;
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
            const linear_Int8PackedWeight first = linear_q8_activation_weight(weight, offset, dtype);
            const linear_Int8PackedWeight second = linear_q8_activation_weight(gate, offset + (next ? groups : 0), dtype);
            linear_dot_q8_activation_pair(first, second, quantized->activation, groups, dtype, widening, result);
        } else {
            result[0] = linear_int8_dot_row(&weight, offset, quantized->activation, groups, dtype, widening);
        }
        const size_t count = next ? 2 : 1;
        for (size_t c = 0; c < count; ++c) {
            float value = result[c];
            if (!isfinite(value)) {
                const TrillimStatus status = linear_project_int8_nonfinite(task, task->weight, row + c, &value);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (paired) {
                float gate_value = share_activation ? result[1] :
                    linear_int8_dot_row(&gate, offset, quantized->activation, groups, dtype, widening);
                if (!isfinite(gate_value)) {
                    const TrillimStatus status = linear_project_int8_nonfinite(task, quantized->gate, row, &gate_value);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                value *= quantized->silu(gate_value);
            }
            linear_store_result(task->output, row + c, output_dtype, (TrillimValue){.f = value});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#define linear_DEFINE_INT8_PROJECTION(name, share, widening, dtype) \
    static TrillimStatus name(void *opaque, size_t index, size_t worker) { \
        (void) worker; \
        return linear_project_int8(opaque, index, share, widening, dtype, TRILLIM_FP32); \
    }

linear_DEFINE_INT8_PROJECTION(linear_project_int8_q4_neon, false, true, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q4_neon_pair, true, true, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q5_neon, true, true, TRILLIM_GRP_INT5)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q6_neon, false, true, TRILLIM_DUBGRP_INT6)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q8_neon, false, true, TRILLIM_GRP_INT8)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q8_neon_pair, true, true, TRILLIM_GRP_INT8)

linear_DEFINE_INT8_PROJECTION(linear_project_int8_q4_sdot, false, false, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q4_sdot_pair, true, false, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q5_sdot, true, false, TRILLIM_GRP_INT5)
enum { linear_Q6_AQ_COLUMNS = 2 };

#if defined(__clang__)
#define linear_Q6_AQ_UNROLL _Pragma("clang loop unroll(full)")
#else
#define linear_Q6_AQ_UNROLL
#endif

static inline int8x16_t linear_q6_signed4(uint8x16_t packed, uint8x16_t upper, bool high_nibble) {
    const uint8x16_t lower = high_nibble
        ? vshlq_n_u8(vshrq_n_u8(packed, 4), 2)
        : vshlq_n_u8(packed, 2);
    return vreinterpretq_s8_u8(vsliq_n_u8(lower, upper, 6));
}

static inline int32x4_t linear_q6_aq_quarter(
    const uint8_t *low, uint8x16x2_t upper, const int8x16x4_t activation
) {
    const uint8x16x2_t packed = vld1q_u8_x2(low);
    const int32x4_t d0 = linear_int8_sdot(linear_q6_signed4(packed.val[0], upper.val[0], false), activation.val[0]);
    const int32x4_t d1 = linear_int8_sdot(linear_q6_signed4(packed.val[1], upper.val[1], false), activation.val[1]);
    const uint8x16_t h0 = vshrq_n_u8(upper.val[0], 2);
    const uint8x16_t h1 = vshrq_n_u8(upper.val[1], 2);
    const int32x4_t d2 = linear_int8_sdot(linear_q6_signed4(packed.val[0], h0, true), activation.val[2]);
    const int32x4_t d3 = linear_int8_sdot(linear_q6_signed4(packed.val[1], h1, true), activation.val[3]);
    return vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
}

static void linear_q6_aq_tile(
    const linear_Int8PackedWeight weight[linear_Q6_AQ_COLUMNS], const linear_Q8KActivationBlock *x, size_t groups,
    float result[linear_Q6_AQ_COLUMNS]
) {
    float sum[linear_Q6_AQ_COLUMNS] = {0};
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *input = x[g].values;
        int32x4_t block_dot[linear_Q6_AQ_COLUMNS] = {0};
        for (size_t h = 0; h < 2; ++h) {
            const int8x16x4_t x0 = vld1q_s8_x4(input + h * 128);
            const int8x16x4_t x1 = vld1q_s8_x4(input + h * 128 + 64);
            linear_Q6_AQ_UNROLL
            for (size_t c = 0; c < linear_Q6_AQ_COLUMNS; ++c) {
                const uint8_t *low = weight[c].low + g * 128 + h * 64;
                uint8x16x2_t upper = vld1q_u8_x2(weight[c].high + g * 64 + h * 32);
                /* Flip all four sign bits. Insert the selected two high bits at
                   the top of each byte to get four times the centered code. */
                upper.val[0] = veorq_u8(upper.val[0], vdupq_n_u8(0xaa));
                upper.val[1] = veorq_u8(upper.val[1], vdupq_n_u8(0xaa));
                const int32x4_t first = linear_q6_aq_quarter(low, upper, x0);
                upper.val[0] = vshrq_n_u8(upper.val[0], 4);
                upper.val[1] = vshrq_n_u8(upper.val[1], 4);
                const int32x4_t second = linear_q6_aq_quarter(low + 32, upper, x1);
                const int16x8_t scales = vmovl_s8(vld1_s8(weight[c].subscales + g * 16 + h * 8));
                block_dot[c] = vmlaq_s32(block_dot[c], first, vmovl_s16(vget_low_s16(scales)));
                block_dot[c] = vmlaq_s32(block_dot[c], second, vmovl_high_s16(scales));
            }
        }
        linear_Q6_AQ_UNROLL
        for (size_t c = 0; c < linear_Q6_AQ_COLUMNS; ++c) {
            const int32_t dot = vaddvq_s32(vshrq_n_s32(block_dot[c], 2));
            sum[c] += (linear_int8_weight_scale(weight[c].scales, g) * x[g].scale) * dot;
        }
    }
    linear_Q6_AQ_UNROLL
    for (size_t c = 0; c < linear_Q6_AQ_COLUMNS; ++c) {
        result[c] = sum[c];
    }
}

static TrillimStatus linear_project_int8_q6_sdot(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const linear_Int8ProjectionTask *quantized = opaque;
    const LinearTask *task = quantized->projection;
    const PackedLinearParameters *p = task->p;
    const size_t groups = p->width / 256;
    const size_t total = groups * p->outputs;
    const bool paired = quantized->gate != NULL;
    const size_t stride = paired ? linear_Q6_AQ_COLUMNS / 2 : linear_Q6_AQ_COLUMNS;
    const linear_Int8PackedWeight weight = linear_int8_packed_weight(task->weight, total, TRILLIM_DUBGRP_INT6);
    const linear_Int8PackedWeight gate = paired
        ? linear_int8_packed_weight(quantized->gate, total, TRILLIM_DUBGRP_INT6) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    for (size_t row = begin; row < end; row += stride) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t count = end - row < stride ? end - row : stride;
        linear_Int8PackedWeight tile[linear_Q6_AQ_COLUMNS];
        for (size_t c = 0; c < linear_Q6_AQ_COLUMNS; ++c) {
            const bool use_gate = paired && (c & 1);
            const linear_Int8PackedWeight *operand = use_gate ? &gate : &weight;
            const size_t position = paired ? c / 2 : c;
            const size_t offset = (row + (position < count ? position : 0)) * groups;
            tile[c] = (linear_Int8PackedWeight){
                operand->low + offset * 128, operand->scales + offset * 2,
                operand->high + offset * 64, operand->subscales + offset * 16
            };
        }
        float result[linear_Q6_AQ_COLUMNS];
        linear_q6_aq_tile(tile, quantized->activation, groups, result);
        for (size_t c = 0; c < count; ++c) {
            float value = result[paired ? c * 2 : c];
            if (!isfinite(value)) {
                const TrillimStatus status = linear_project_int8_nonfinite(task, task->weight, row + c, &value);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (paired) {
                float gate_value = result[c * 2 + 1];
                if (!isfinite(gate_value)) {
                    const TrillimStatus status = linear_project_int8_nonfinite(task, quantized->gate, row + c, &gate_value);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                value *= quantized->silu(gate_value);
            }
            linear_store_result(task->output, row + c, TRILLIM_FP32, (TrillimValue){.f = value});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#undef linear_Q6_AQ_UNROLL

enum { linear_INT8_PREFILL_ROWS = 4, linear_INT8_PREFILL_PAIRS = linear_INT8_PREFILL_ROWS / 2, linear_INT8_PREFILL_PANEL_TOKENS = 128 };

typedef struct {
    int8x16_t values[32][linear_INT8_PREFILL_PAIRS];
    float32x4_t scales[linear_INT8_PREFILL_PAIRS];
} linear_Q6PrefillActivation;

typedef struct {
    int8x16_t values[32];
    int32x4_t subscales[16];
    float32x4_t scale;
} linear_Q6PrefillWeight;

typedef struct {
    const LinearTask *projection;
    const uint8_t *gate;
    float (*silu)(float);
    float32x4_t (*activate4)(float32x4_t);
    const void *activation;
    atomic_bool ordinary;
    bool output_bf16;
    const uint8_t *bias;
} linear_Int8PrefillTask;

static bool linear_int8_prefill_i8mm_available(void) {
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
static inline int32x4_t linear_int8_prefill_i8mm(int32x4_t accum, int8x16_t input, int8x16_t weight) {
    __asm__(".inst 0x4e80a400 | trillim_sdot_%0 | (trillim_sdot_%1 << 5) | (trillim_sdot_%2 << 16)"
            : "+w"(accum) : "w"(input), "w"(weight));
    return accum;
}

typedef struct {
    const LinearTask *projection;
    linear_Q6PrefillActivation *packed;
    size_t tiles, cells, chunk;
    atomic_bool ordinary;
} linear_Q6PrefillPackingTask;

static bool linear_pack_q6_prefill_cell(
    const LinearTask *task, linear_Q6PrefillActivation *destination, size_t tile, size_t group
) {
    const PackedLinearParameters *p = task->p;
    for (size_t pair = 0; pair < linear_INT8_PREFILL_PAIRS; ++pair) {
        const size_t row = tile * linear_INT8_PREFILL_ROWS + 2 * pair;
        const size_t start = row * p->width + group * linear_INT8_ACTIVATION_Q8_K;
        linear_Q8KActivationBlock first = {0}, second = {0};
        if ((row < p->tokens && !linear_quantize_activation_q8k(
                task->input, start, p->input_dtype, &first, false)) ||
            (row + 1 < p->tokens && !linear_quantize_activation_q8k(
                task->input, start + p->width, p->input_dtype, &second, false))) {
            return false;
        }
        destination->scales[pair] = (float32x4_t){first.scale, first.scale, second.scale, second.scale};
        for (size_t k = 0; k < 32; ++k) {
            destination->values[k][pair] = vcombine_s8(
                vld1_s8(first.values + k * 8), vld1_s8(second.values + k * 8));
        }
    }
    return true;
}

static TrillimStatus linear_pack_q6_prefill_tile(void *opaque, size_t index, size_t worker) {
    (void)worker;
    linear_Q6PrefillPackingTask *job = opaque;
    if (!linear_fhm_default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    const LinearTask *task = job->projection;
    size_t begin = index * job->chunk;
    const size_t end = job->cells - begin < job->chunk ? job->cells : begin + job->chunk;
    size_t group = begin / job->tiles, first = begin % job->tiles;
    while (begin < end) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) {
            return TRILLIM_OK;
        }
        const size_t available = job->tiles - first;
        const size_t count = end - begin < available ? end - begin : available;
        for (size_t i = 0; i < count; ++i) {
            if (i != 0 && (i & 3) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            if (!linear_pack_q6_prefill_cell(task, job->packed + begin + i, first + i, group)) {
                atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
        begin += count;
        ++group;
        first = 0;
    }
    return TRILLIM_OK;
}

static TrillimStatus linear_pack_q6_prefill_activation(
    const LinearTask *task, linear_Q6PrefillActivation *packed, size_t tiles, bool *ordinary
) {
    const PackedLinearParameters *p = task->p;
    const TrillimThreadPool *pool = task->context->thread_pool;
    const size_t groups = p->width / linear_INT8_ACTIVATION_Q8_K;
    const size_t cells = tiles * groups;
    const size_t tasks = p->tokens * p->width < 131072 ? 1 : pool->worker_count;
    const size_t chunk = cells / tasks + (cells % tasks != 0);
    linear_Q6PrefillPackingTask job = {task, packed, tiles, cells, chunk, ATOMIC_VAR_INIT(true)};
    const size_t count = cells / chunk + (cells % chunk != 0);
    const TrillimStatus status = count == 1 ? linear_pack_q6_prefill_tile(&job, 0, 0) :
        pool->run(pool->context, count, linear_pack_q6_prefill_tile, &job);
    *ordinary = atomic_load_explicit(&job.ordinary, memory_order_relaxed);
    return status;
}

static inline uint8x16x2_t linear_q6_prefill_interleave(const uint8_t *a, const uint8_t *b) {
    const uint64x2_t x = vreinterpretq_u64_u8(vld1q_u8(a));
    const uint64x2_t y = vreinterpretq_u64_u8(vld1q_u8(b));
    return (uint8x16x2_t){{
        vreinterpretq_u8_u64(vzip1q_u64(x, y)),
        vreinterpretq_u8_u64(vzip2q_u64(x, y)),
    }};
}

static inline void linear_q6_prefill_pack_scales(
    int32x4_t *destination, int8x16_t a, int8x16_t b
) {
    const int8x16x2_t interleaved = vzipq_s8(a, b);
    for (size_t i = 0; i < 2; ++i) {
        const int16x8_t low = vmovl_s8(vget_low_s8(interleaved.val[i]));
        const int16x8_t high = vmovl_high_s8(interleaved.val[i]);
        const int32x4_t scales[4] = {
            vmovl_s16(vget_low_s16(low)), vmovl_high_s16(low),
            vmovl_s16(vget_low_s16(high)), vmovl_high_s16(high),
        };
        for (size_t j = 0; j < 4; ++j) {
            const int64x2_t pairs = vreinterpretq_s64_s32(scales[j]);
            destination[i * 8 + j * 2] = vreinterpretq_s32_s64(vdupq_laneq_s64(pairs, 0));
            destination[i * 8 + j * 2 + 1] = vreinterpretq_s32_s64(vdupq_laneq_s64(pairs, 1));
        }
    }
}

static inline void linear_pack_q6_prefill_group(
    linear_Int8PackedWeight first, linear_Int8PackedWeight second, size_t a, size_t b, linear_Q6PrefillWeight *packed
) {
    const float scale0 = linear_int8_weight_scale(first.scales, a);
    const float scale1 = linear_int8_weight_scale(second.scales, b);
    packed->scale = (float32x4_t){scale0, scale1, scale0, scale1};
    linear_q6_prefill_pack_scales(packed->subscales, vld1q_s8(first.subscales + a * 16),
        vld1q_s8(second.subscales + b * 16));
    for (size_t half = 0; half < 2; ++half) {
        const uint8_t *low0 = first.low + a * 128 + half * 64;
        const uint8_t *low1 = second.low + b * 128 + half * 64;
        const uint8_t *high0 = first.high + a * 64 + half * 32;
        const uint8_t *high1 = second.high + b * 64 + half * 32;
        for (size_t j = 0; j < 2; ++j) {
            const size_t offset = j * 16;
            const uint8x16x2_t a0 = linear_q6_prefill_interleave(low0 + offset, low1 + offset);
            const uint8x16x2_t a1 = linear_q6_prefill_interleave(low0 + offset + 32, low1 + offset + 32);
            const uint8x16x2_t upper = linear_q6_prefill_interleave(high0 + offset, high1 + offset);
            for (size_t part = 0; part < 2; ++part) {
                const size_t k = half * 16 + j * 2 + part;
                const uint8x16_t h = veorq_u8(upper.val[part], vdupq_n_u8(0xaa));
                packed->values[k] = linear_q6_signed4(a0.val[part], h, false);
                packed->values[k + 4] = linear_q6_signed4(a0.val[part], vshrq_n_u8(h, 2), true);
                packed->values[k + 8] = linear_q6_signed4(a1.val[part], vshrq_n_u8(h, 4), false);
                packed->values[k + 12] = linear_q6_signed4(a1.val[part], vshrq_n_u8(h, 6), true);
            }
        }
    }
}

#if defined(__clang__)
#define linear_INT8_PREFILL_UNROLL _Pragma("clang loop unroll(full)")
#else
#define linear_INT8_PREFILL_UNROLL
#endif

enum { linear_Q8_PREFILL_ROWS = 2, linear_Q8_PREFILL_PAIRS = linear_Q8_PREFILL_ROWS / 2 };

typedef struct {
    int8x16_t values[4][linear_Q8_PREFILL_PAIRS];
    float32x4_t scales[linear_Q8_PREFILL_PAIRS];
} linear_Q8PrefillActivation;

enum { linear_Q8_PREFILL_COLUMNS = 8, linear_Q8_PREFILL_COLUMN_PAIRS = linear_Q8_PREFILL_COLUMNS / 2 };

typedef struct {
    int8x16_t values[4][linear_Q8_PREFILL_COLUMN_PAIRS];
    float32x2_t scales[linear_Q8_PREFILL_COLUMN_PAIRS];
} linear_Q8PrefillWeight;

typedef struct {
    const LinearTask *projection;
    linear_Q8PrefillActivation *packed;
    size_t rows, cells, chunk;
    atomic_bool ordinary;
} linear_Q8PrefillPackingTask;

static inline __attribute__((always_inline)) bool linear_pack_q8_activation_cell(
    const LinearTask *task, linear_Q8PrefillActivation *destination, size_t t, size_t g, uint32_t dtype
) {
    const PackedLinearParameters *p = task->p;
    for (size_t pair = 0; pair < linear_Q8_PREFILL_PAIRS; ++pair) {
        const size_t row = t * linear_Q8_PREFILL_ROWS + 2 * pair;
        const size_t start = row * p->width + g * linear_INT8_ACTIVATION_Q8_0;
        linear_Q8ActivationBlock first, second;
        if (row < p->tokens) {
            if (!linear_quantize_activation_q8(task->input, start, dtype, &first)) {
                return false;
            }
        } else {
            memset(&first, 0, sizeof(first));
        }
        if (row + 1 < p->tokens) {
            if (!linear_quantize_activation_q8(task->input, start + p->width, dtype, &second)) {
                return false;
            }
        } else {
            memset(&second, 0, sizeof(second));
        }
        const float scale0 = (float)vreinterpret_f16_u16(vdup_n_u16(first.scale))[0];
        const float scale1 = (float)vreinterpret_f16_u16(vdup_n_u16(second.scale))[0];
        destination->scales[pair] = (float32x4_t){scale0, scale0, scale1, scale1};
        for (size_t k = 0; k < 4; ++k) {
            destination->values[k][pair] = vcombine_s8(
                vld1_s8(first.values + k * 8), vld1_s8(second.values + k * 8));
        }
    }
    return true;
}

static inline __attribute__((always_inline)) TrillimStatus linear_pack_q8_prefill_range(
    linear_Q8PrefillPackingTask *job, size_t index, uint32_t dtype
) {
    const LinearTask *task = job->projection;
    size_t begin = index * job->chunk;
    const size_t end = job->cells - begin < job->chunk ? job->cells : begin + job->chunk;
    size_t g = begin / job->rows;
    size_t first = begin % job->rows;
    while (begin < end) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) {
            return TRILLIM_OK;
        }
        const size_t remaining = job->rows - first;
        const size_t count = end - begin < remaining ? end - begin : remaining;
        for (size_t i = 0; i < count; ++i) {
            if (!linear_pack_q8_activation_cell(task, job->packed + begin + i, first + i, g, dtype)) {
                atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
        begin += count;
        ++g;
        first = 0;
    }
    return TRILLIM_OK;
}

static TrillimStatus linear_pack_q8_prefill_tile(void *opaque, size_t index, size_t worker) {
    (void)worker;
    linear_Q8PrefillPackingTask *job = opaque;
    if (!linear_fhm_default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    const uint32_t dtype = job->projection->p->input_dtype;
    return linear_pack_q8_prefill_range(job, index, dtype);
}

static TrillimStatus linear_pack_q8_prefill_activation(
    const LinearTask *task, linear_Q8PrefillActivation *activation, size_t rows, bool *ordinary
) {
    const PackedLinearParameters *p = task->p;
    const TrillimThreadPool *pool = task->context->thread_pool;
    const size_t groups = p->width / linear_INT8_ACTIVATION_Q8_0;
    const size_t cells = rows * groups;
    const size_t tasks = p->tokens * p->width < 32768 ? 1 : pool->worker_count;
    const size_t chunk = cells / tasks + (cells % tasks != 0);
    linear_Q8PrefillPackingTask job = {task, activation, rows, cells, chunk, ATOMIC_VAR_INIT(true)};
    const size_t count = cells / chunk + (cells % chunk != 0);
    TrillimStatus status = count == 1 ? linear_pack_q8_prefill_tile(&job, 0, 0) :
        pool->run(pool->context, count, linear_pack_q8_prefill_tile, &job);
    *ordinary = atomic_load_explicit(&job.ordinary, memory_order_relaxed);
    return status;
}

static inline __attribute__((always_inline)) void linear_pack_q8_prefill_group(
    linear_Int8PackedWeight first, linear_Int8PackedWeight second, size_t offsets[linear_Q8_PREFILL_COLUMN_PAIRS][2],
    size_t g, linear_Q8PrefillWeight *packed, uint32_t dtype
) {
    float scales[linear_Q8_PREFILL_COLUMNS];
    linear_INT8_PREFILL_UNROLL
    for (size_t c = 0; c < linear_Q8_PREFILL_COLUMN_PAIRS; ++c) {
        const size_t a = offsets[c][0] + g, b = offsets[c][1] + g;
        scales[2 * c] = linear_int8_weight_scale(first.scales, a);
        scales[2 * c + 1] = linear_int8_weight_scale(second.scales, b);
        const int8x16x2_t w0 = dtype == TRILLIM_GRP_INT4 ? linear_int8_decode_q4(first.low + a * 16) :
            vld1q_s8_x2((const int8_t *)first.low + a * 32);
        const int8x16x2_t w1 = dtype == TRILLIM_GRP_INT4 ? linear_int8_decode_q4(second.low + b * 16) :
            vld1q_s8_x2((const int8_t *)second.low + b * 32);
        packed->values[0][c] = vcombine_s8(vget_low_s8(w0.val[0]), vget_low_s8(w1.val[0]));
        packed->values[1][c] = vcombine_s8(vget_high_s8(w0.val[0]), vget_high_s8(w1.val[0]));
        packed->values[2][c] = vcombine_s8(vget_low_s8(w0.val[1]), vget_low_s8(w1.val[1]));
        packed->values[3][c] = vcombine_s8(vget_high_s8(w0.val[1]), vget_high_s8(w1.val[1]));
    }
    linear_INT8_PREFILL_UNROLL
    for (size_t c = 0; c < linear_Q8_PREFILL_COLUMN_PAIRS; ++c) {
        packed->scales[c] = vld1_f32(scales + 2 * c);
    }
}

static inline __attribute__((always_inline)) float32x4_t linear_q8_prefill_row(
    float32x4_t a, float32x4_t b, size_t row
) {
    return row & 1 ? vcombine_f32(vget_high_f32(a), vget_high_f32(b)) :
        vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}

static inline bool linear_finite_prefill_result(float32x4_t value) {
    const uint32x4_t absolute = vandq_u32(vreinterpretq_u32_f32(value), vdupq_n_u32(0x7fffffff));
    return vmaxvq_u32(absolute) < 0x7f800000;
}

static inline uint16x4_t linear_round_finite_bf16(float32x4_t value) {
    const uint32x4_t bits = vreinterpretq_u32_f32(value);
    const uint32x4_t upper = vshrq_n_u32(bits, 16);
    const uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    return vshrn_n_u32(vaddq_u32(bits, bias), 16);
}

static inline void linear_store_prefill_bf16(void *base, size_t offset, const float32x4_t values[2], size_t count) {
    const uint16x8_t bits = vcombine_u16(linear_round_finite_bf16(values[0]), linear_round_finite_bf16(values[1]));
    uint8_t *output = (uint8_t *)base + offset * sizeof(uint16_t);
    if (count == 8) {
        memcpy(output, &bits, sizeof(bits));
    } else {
        memcpy(output, &bits, count * sizeof(uint16_t));
    }
}

static inline __attribute__((always_inline)) void linear_accumulate_q8_prefill_group(
    const linear_Q8PrefillWeight *w, const linear_Q8PrefillActivation *x,
    float32x4_t sum[linear_Q8_PREFILL_PAIRS][linear_Q8_PREFILL_COLUMN_PAIRS]
) {
    int32x4_t accum[linear_Q8_PREFILL_PAIRS][linear_Q8_PREFILL_COLUMN_PAIRS] = {0};
    linear_INT8_PREFILL_UNROLL
    for (size_t k = 0; k < 4; ++k) {
        const int8x16_t a = x->values[k][0];
        linear_INT8_PREFILL_UNROLL
        for (size_t c = 0; c < linear_Q8_PREFILL_COLUMN_PAIRS; ++c) {
            accum[0][c] = linear_int8_prefill_i8mm(accum[0][c], a, w->values[k][c]);
        }
    }
    linear_INT8_PREFILL_UNROLL
    for (size_t c = 0; c < linear_Q8_PREFILL_COLUMN_PAIRS; ++c) {
        const float32x2_t weight_scale = w->scales[c];
        const float32x4_t scale = vcombine_f32(weight_scale, weight_scale);
        linear_INT8_PREFILL_UNROLL
        for (size_t pair = 0; pair < linear_Q8_PREFILL_PAIRS; ++pair) {
            const float32x4_t dot = vcvtq_f32_s32(accum[pair][c]);
            sum[pair][c] = vfmaq_f32(sum[pair][c], dot, vmulq_f32(scale, x->scales[pair]));
        }
    }
}

enum {
    linear_Q6_PREFILL_COLUMN_PAIRS = 4,
    linear_Q6_PREFILL_MAX_TILES = linear_INT8_PREFILL_PANEL_TOKENS / linear_INT8_PREFILL_ROWS,
};

static inline __attribute__((always_inline)) void linear_accumulate_q6_prefill_group(
    const linear_Q6PrefillWeight weight[linear_Q6_PREFILL_COLUMN_PAIRS], const linear_Q6PrefillActivation *x,
    float32x4_t total[linear_Q6_PREFILL_COLUMN_PAIRS][linear_INT8_PREFILL_PAIRS]
) {
    int32x4_t accum[linear_Q6_PREFILL_COLUMN_PAIRS][linear_INT8_PREFILL_PAIRS] = {0};
    for (size_t j = 0; j < 16; ++j) {
        int8x16_t a0[linear_INT8_PREFILL_PAIRS], a1[linear_INT8_PREFILL_PAIRS];
        linear_INT8_PREFILL_UNROLL
        for (size_t pair = 0; pair < linear_INT8_PREFILL_PAIRS; ++pair) {
            a0[pair] = x->values[2 * j][pair];
            a1[pair] = x->values[2 * j + 1][pair];
        }
        linear_INT8_PREFILL_UNROLL
        for (size_t c = 0; c < linear_Q6_PREFILL_COLUMN_PAIRS; ++c) {
            const int8x16_t w0 = weight[c].values[2 * j], w1 = weight[c].values[2 * j + 1];
            const int32x4_t scale = weight[c].subscales[j];
            linear_INT8_PREFILL_UNROLL
            for (size_t pair = 0; pair < linear_INT8_PREFILL_PAIRS; ++pair) {
                int32x4_t dot = linear_int8_prefill_i8mm(vdupq_n_s32(0), a0[pair], w0);
                dot = linear_int8_prefill_i8mm(dot, a1[pair], w1);
                accum[c][pair] = vmlaq_s32(accum[c][pair], dot, scale);
            }
        }
    }
    linear_INT8_PREFILL_UNROLL
    for (size_t c = 0; c < linear_Q6_PREFILL_COLUMN_PAIRS; ++c) {
        linear_INT8_PREFILL_UNROLL
        for (size_t pair = 0; pair < linear_INT8_PREFILL_PAIRS; ++pair) {
            const float32x4_t scale = vmulq_f32(weight[c].scale, x->scales[pair]);
            const float32x4_t dot = vcvtq_n_f32_s32(accum[c][pair], 2);
            total[c][pair] = vfmaq_f32(total[c][pair], dot, scale);
        }
    }
}

static TrillimStatus linear_project_q6_prefill(void *opaque, size_t index, size_t worker) {
    (void)worker;
    linear_Int8PrefillTask *job = opaque;
    const LinearTask *task = job->projection;
    const PackedLinearParameters *p = task->p;
    const size_t groups = p->width / linear_INT8_ACTIVATION_Q8_K;
    const size_t row_tiles = p->tokens / linear_INT8_PREFILL_ROWS + (p->tokens % linear_INT8_PREFILL_ROWS != 0);
    const size_t total = groups * p->outputs;
    const bool paired = job->gate != NULL;
    const size_t stride = paired ? 1 : 2;
    const linear_Int8PackedWeight weight = linear_int8_packed_weight(task->weight, total, TRILLIM_DUBGRP_INT6);
    const linear_Int8PackedWeight second = paired ? linear_int8_packed_weight(job->gate, total, TRILLIM_DUBGRP_INT6) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    if (!linear_fhm_default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    const size_t scale_offset = begin * groups * 2;
    const size_t scale_count = (end - begin) * groups;
    bool ordinary;
    TrillimStatus status = linear_bounded_group_scales(weight.scales + scale_offset, scale_count, task->context, &ordinary);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (ordinary && paired) {
        status = linear_bounded_group_scales(second.scales + scale_offset, scale_count, task->context, &ordinary);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (!ordinary) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    const linear_Q6PrefillActivation *activations = job->activation;
    for (size_t column = begin; column < end; column += stride * linear_Q6_PREFILL_COLUMN_PAIRS) {
        if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) {
            return TRILLIM_OK;
        }
        size_t first[linear_Q6_PREFILL_COLUMN_PAIRS], next[linear_Q6_PREFILL_COLUMN_PAIRS];
        for (size_t c = 0; c < linear_Q6_PREFILL_COLUMN_PAIRS; ++c) {
            const size_t candidate = column + stride * c;
            first[c] = candidate < end ? candidate : column;
            next[c] = first[c] + (!paired && first[c] + 1 < end);
        }
        float32x4_t totals[linear_Q6_PREFILL_MAX_TILES][linear_Q6_PREFILL_COLUMN_PAIRS][linear_INT8_PREFILL_PAIRS];
        memset(totals, 0, row_tiles * sizeof(totals[0]));
        for (size_t g = 0; g < groups; ++g) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            linear_Q6PrefillWeight packed[linear_Q6_PREFILL_COLUMN_PAIRS];
            for (size_t c = 0; c < linear_Q6_PREFILL_COLUMN_PAIRS; ++c) {
                linear_pack_q6_prefill_group(weight, second, first[c] * groups + g, next[c] * groups + g, packed + c);
            }
            for (size_t t = 0; t < row_tiles; ++t) {
                linear_accumulate_q6_prefill_group(packed, activations + g * row_tiles + t, totals[t]);
            }
        }
        for (size_t row = 0; row < p->tokens; ++row) {
            if ((row % linear_INT8_PREFILL_ROWS) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const size_t inner = row % linear_INT8_PREFILL_ROWS;
            if (paired && job->activate4 != NULL && end - column >= linear_Q6_PREFILL_COLUMN_PAIRS) {
                const size_t pair = inner / 2;
                float32x4_t (*rows)[linear_INT8_PREFILL_PAIRS] = totals[row / linear_INT8_PREFILL_ROWS];
                const float32x4_t a = linear_q8_prefill_row(rows[0][pair], rows[1][pair], inner);
                const float32x4_t b = linear_q8_prefill_row(rows[2][pair], rows[3][pair], inner);
                const uint32x4_t magnitude = vmaxq_u32(
                    vreinterpretq_u32_f32(vabsq_f32(a)), vreinterpretq_u32_f32(vabsq_f32(b)));
                if (vmaxvq_u32(magnitude) >= 0x7f800000) {
                    atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                    return TRILLIM_OK;
                }
                const float32x4_t up = vuzp1q_f32(a, b), gate = vuzp2q_f32(a, b);
                float *output = (float *)task->output + row * p->outputs + column;
                vst1q_f32(output, vmulq_f32(up, job->activate4(gate)));
                continue;
            }
            for (size_t c = 0; c < linear_Q6_PREFILL_COLUMN_PAIRS && column + c * stride < end; ++c) {
                const float32x2_t pair = (inner & 1)
                    ? vget_high_f32(totals[row / linear_INT8_PREFILL_ROWS][c][inner / 2])
                    : vget_low_f32(totals[row / linear_INT8_PREFILL_ROWS][c][inner / 2]);
                const float a = vget_lane_f32(pair, 0), b = vget_lane_f32(pair, 1);
                if (!isfinite(a) || !isfinite(b)) {
                    atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                    return TRILLIM_OK;
                }
                const size_t destination = row * p->outputs + first[c];
                ((float *)task->output)[destination] = paired ? a * job->silu(b) : a;
                if (next[c] != first[c]) {
                    ((float *)task->output)[destination + 1] = b;
                }
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#undef linear_INT8_PREFILL_UNROLL

static inline __attribute__((always_inline)) TrillimStatus linear_project_q8_prefill_wide(
    linear_Int8PrefillTask *job, linear_Int8PackedWeight weight, linear_Int8PackedWeight second,
    size_t begin, size_t end, size_t groups, uint32_t dtype, bool paired, bool biased
) {
    const LinearTask *task = job->projection;
    const PackedLinearParameters *p = task->p;
    const size_t step = paired ? linear_Q8_PREFILL_COLUMN_PAIRS : linear_Q8_PREFILL_COLUMNS;
    const size_t row_tiles = p->tokens / linear_Q8_PREFILL_ROWS + (p->tokens % linear_Q8_PREFILL_ROWS != 0);
    const linear_Q8PrefillActivation *activation = job->activation;
    for (size_t column = begin; column < end; column += step) {
        if (!atomic_load_explicit(&job->ordinary, memory_order_relaxed)) {
            return TRILLIM_OK;
        }
        const size_t count = end - column < step ? end - column : step;
        float32x4_t addends[2];
        if (biased) {
            uint16x8_t bits = vdupq_n_u16(0);
            memcpy(&bits, job->bias + column * sizeof(uint16_t), count * sizeof(uint16_t));
            addends[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16));
            addends[1] = vreinterpretq_f32_u32(vshll_high_n_u16(bits, 16));
        }
        size_t offsets[linear_Q8_PREFILL_COLUMN_PAIRS][2];
        for (size_t c = 0; c < linear_Q8_PREFILL_COLUMN_PAIRS; ++c) {
            const size_t a = column + (paired ? c : 2 * c);
            const size_t b = a + !paired;
            offsets[c][0] = (a < end ? a : column) * groups;
            offsets[c][1] = (b < end ? b : column) * groups;
        }
        /* Keep the previous even and odd accumulation order for FP32 rounding. */
        float32x4_t totals[linear_INT8_PREFILL_PANEL_TOKENS / linear_Q8_PREFILL_ROWS][2]
            [linear_Q8_PREFILL_PAIRS][linear_Q8_PREFILL_COLUMN_PAIRS];
        memset(totals, 0, row_tiles * sizeof(totals[0]));
        for (size_t g = 0; g < groups; ++g) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            linear_Q8PrefillWeight packed;
            linear_pack_q8_prefill_group(weight, second, offsets, g, &packed, dtype);
            for (size_t t = 0; t < row_tiles; ++t) {
                linear_accumulate_q8_prefill_group(&packed, activation + g * row_tiles + t, totals[t][g & 1]);
            }
        }
        for (size_t row = 0; row < p->tokens; row += linear_Q8_PREFILL_ROWS) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t (*sum)[linear_Q8_PREFILL_PAIRS][linear_Q8_PREFILL_COLUMN_PAIRS] =
                totals[row / linear_Q8_PREFILL_ROWS];
            const size_t rows = p->tokens - row < linear_Q8_PREFILL_ROWS ? p->tokens - row : linear_Q8_PREFILL_ROWS;
            for (size_t r = 0; r < rows; ++r) {
                const size_t offset = (row + r) * p->outputs + column;
                float32x4_t result[linear_Q8_PREFILL_COLUMNS / 4];
                for (size_t c = 0; c < linear_Q8_PREFILL_COLUMNS / 4; ++c) {
                    const size_t pair = r / 2;
                    const float32x4_t a = vaddq_f32(sum[0][pair][2 * c], sum[1][pair][2 * c]);
                    const float32x4_t b = vaddq_f32(sum[0][pair][2 * c + 1], sum[1][pair][2 * c + 1]);
                    result[c] = linear_q8_prefill_row(a, b, r);
                    if (!linear_finite_prefill_result(result[c])) {
                        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                        return TRILLIM_OK;
                    }
                }
                if (biased || job->output_bf16) {
                    if (biased) {
                        result[0] = vaddq_f32(result[0], addends[0]);
                        result[1] = vaddq_f32(result[1], addends[1]);
                        if (!linear_finite_prefill_result(result[0]) || !linear_finite_prefill_result(result[1])) {
                            atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
                            return TRILLIM_OK;
                        }
                    }
                    linear_store_prefill_bf16(task->output, offset, result, count);
                    continue;
                }
                float *output = (float *)task->output + offset;
                if (!paired && count == linear_Q8_PREFILL_COLUMNS) {
                    for (size_t c = 0; c < linear_Q8_PREFILL_COLUMNS / 4; ++c) {
                        vst1q_f32(output + 4 * c, result[c]);
                    }
                } else if (paired && job->activate4 != NULL && count == linear_Q8_PREFILL_COLUMN_PAIRS) {
                    const float32x4_t up = vuzp1q_f32(result[0], result[1]);
                    const float32x4_t gate = vuzp2q_f32(result[0], result[1]);
                    vst1q_f32(output, vmulq_f32(up, job->activate4(gate)));
                } else {
                    float values[linear_Q8_PREFILL_COLUMNS];
                    memcpy(values, result, sizeof(values));
                    for (size_t c = 0; c < count; ++c) {
                        output[c] = paired ? values[2 * c] * job->silu(values[2 * c + 1]) : values[c];
                    }
                }
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus linear_project_int8_prefill(
    void *opaque, size_t index, size_t worker, uint32_t dtype, bool biased
) {
    (void)worker;
    linear_Int8PrefillTask *job = opaque;
    const LinearTask *task = job->projection;
    const PackedLinearParameters *p = task->p;
    const size_t groups = p->width / linear_INT8_ACTIVATION_Q8_0;
    const size_t total = groups * p->outputs;
    const bool paired = !biased && job->gate != NULL;
    const linear_Int8PackedWeight weight = linear_int8_packed_weight(task->weight, total, dtype);
    const linear_Int8PackedWeight second = paired
        ? linear_int8_packed_weight(job->gate, total, dtype) : weight;
    const size_t begin = index * task->columns;
    const size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    if (!linear_fhm_default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    const size_t scale_offset = begin * groups * 2;
    const size_t scale_count = (end - begin) * groups;
    bool ordinary;
    TrillimStatus status = linear_bounded_group_scales(weight.scales + scale_offset, scale_count, task->context, &ordinary);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (ordinary && paired) {
        status = linear_bounded_group_scales(second.scales + scale_offset, scale_count, task->context, &ordinary);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (!ordinary) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    return linear_project_q8_prefill_wide(job, weight, second, begin, end, groups, dtype, paired, biased);
}

static TrillimStatus linear_project_q8_prefill(void *opaque, size_t index, size_t worker) {
    return linear_project_int8_prefill(opaque, index, worker, TRILLIM_GRP_INT8, false);
}

static TrillimStatus linear_project_q4_prefill(void *opaque, size_t index, size_t worker) {
    return linear_project_int8_prefill(opaque, index, worker, TRILLIM_GRP_INT4, false);
}

static TrillimStatus linear_project_q4_bias_prefill(void *opaque, size_t index, size_t worker) {
    return linear_project_int8_prefill(opaque, index, worker, TRILLIM_GRP_INT4, true);
}

static TrillimStatus linear_project_q8_bias_prefill(void *opaque, size_t index, size_t worker) {
    return linear_project_int8_prefill(opaque, index, worker, TRILLIM_GRP_INT8, true);
}

static TrillimStatus linear_try_int8_prefill(
    const LinearTask *task, const uint8_t *gate, float (*silu)(float),
    float32x4_t (*activate4)(float32x4_t), bool output_bf16, const uint8_t *bias,
    size_t workers, bool *handled
) {
    *handled = false;
    const PackedLinearParameters *p = task->p;
    const bool q6 = p->weight_dtype == TRILLIM_DUBGRP_INT6;
    const size_t group = q6 ? linear_INT8_ACTIVATION_Q8_K : linear_INT8_ACTIVATION_Q8_0;
    if ((!q6 && p->weight_dtype != TRILLIM_GRP_INT8 &&
        p->weight_dtype != TRILLIM_GRP_INT4) ||
        p->tokens < 4 || p->width < group || p->width % group != 0 ||
        p->width > linear_INT8_ACTIVATION_MAX_WIDTH || p->outputs < 16 || p->output_dtype != TRILLIM_FP32 ||
        (!output_bf16 && (uintptr_t)task->output % _Alignof(float) != 0) || !trillim_float(p->input_dtype) ||
        !linear_fhm_default_fpcr() || !linear_int8_prefill_i8mm_available()) {
        return TRILLIM_OK;
    }
    const size_t groups = p->width / group;
    const size_t tile_rows = q6 ? linear_INT8_PREFILL_ROWS : linear_Q8_PREFILL_ROWS;
    const size_t record_size = q6 ? sizeof(linear_Q6PrefillActivation) : sizeof(linear_Q8PrefillActivation);
    size_t capacity = (4 * 1024 * 1024) / record_size / groups;
    if (capacity > linear_INT8_PREFILL_PANEL_TOKENS / tile_rows) {
        capacity = linear_INT8_PREFILL_PANEL_TOKENS / tile_rows;
    }
    size_t needed = p->tokens / tile_rows + (p->tokens % tile_rows != 0);
    if (capacity > needed) {
        capacity = needed;
    }
    if (capacity == 0) {
        return TRILLIM_OK;
    }
    void *activation = malloc(capacity * groups * record_size);
    if (activation == NULL) {
        return TRILLIM_OK;
    }
    capacity *= tile_rows;
    const size_t chunk = q6 ? p->outputs / workers / 2 : p->outputs / workers;
    const size_t columns = chunk < 16 ? 16 : (chunk + 15) / 16 * 16;
    const size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    const TrillimThreadPool *pool = task->context->thread_pool;
    TrillimTask project = q6 ? linear_project_q6_prefill : (p->weight_dtype == TRILLIM_GRP_INT4 ? linear_project_q4_prefill : linear_project_q8_prefill);
    if (bias != NULL) {
        project = p->weight_dtype == TRILLIM_GRP_INT4 ? linear_project_q4_bias_prefill : linear_project_q8_bias_prefill;
    }
    TrillimStatus status = TRILLIM_OK;
    for (size_t first = 0; first < p->tokens;) {
        PackedLinearParameters panel_parameters = *p;
        panel_parameters.tokens = p->tokens - first < capacity ? p->tokens - first : capacity;
        LinearTask panel = *task;
        panel.p = &panel_parameters;
        panel.columns = columns;
        panel.input = (const uint8_t *)task->input + first * p->width * trillim_itemsize(p->input_dtype);
        panel.output = (uint8_t *)task->output + first * p->outputs * (output_bf16 ? sizeof(uint16_t) : sizeof(float));
        const size_t rows = panel_parameters.tokens / tile_rows + (panel_parameters.tokens % tile_rows != 0);
        bool ordinary;
        status = q6 ? linear_pack_q6_prefill_activation(&panel, activation, rows, &ordinary) :
            linear_pack_q8_prefill_activation(&panel, activation, rows, &ordinary);
        if (status != TRILLIM_OK || !ordinary) {
            free(activation);
            return status;
        }
        linear_Int8PrefillTask job = {&panel, gate, silu, activate4, activation, ATOMIC_VAR_INIT(true),
            output_bf16, bias};
        status = tiles == 1 ? project(&job, 0, 0) : pool->run(pool->context, tiles, project, &job);
        if (status != TRILLIM_OK || !atomic_load_explicit(&job.ordinary, memory_order_relaxed)) {
            free(activation);
            return status;
        }
        first += panel_parameters.tokens;
    }
    free(activation);
    *handled = true;
    return status;
}

linear_DEFINE_INT8_PROJECTION(linear_project_int8_q8_sdot, false, false, TRILLIM_GRP_INT8)
linear_DEFINE_INT8_PROJECTION(linear_project_int8_q8_sdot_pair, true, false, TRILLIM_GRP_INT8)
static TrillimTask linear_int8_projection(uint32_t dtype, bool paired, size_t width) {
    const bool dotprod = linear_available_dotprod();
    switch (dtype) {
    case TRILLIM_GRP_INT4:
        if (paired && width > 2048) {
            return dotprod ? linear_project_int8_q4_sdot : linear_project_int8_q4_neon;
        }
        return dotprod ? linear_project_int8_q4_sdot_pair : linear_project_int8_q4_neon_pair;
    case TRILLIM_GRP_INT5: return dotprod ? linear_project_int8_q5_sdot : linear_project_int8_q5_neon;
    case TRILLIM_DUBGRP_INT6: return dotprod ? linear_project_int8_q6_sdot : linear_project_int8_q6_neon;
    case TRILLIM_GRP_INT8:
        if (paired) {
            return dotprod ? linear_project_int8_q8_sdot : linear_project_int8_q8_neon;
        }
        return dotprod ? linear_project_int8_q8_sdot_pair : linear_project_int8_q8_neon_pair;
    default: return NULL;
    }
}

#define linear_DEFINE_INT8_BF16_PROJECTION(name, widening, dtype) \
    static TrillimStatus name(void *opaque, size_t index, size_t worker) { \
        (void)worker; \
        return linear_project_int8(opaque, index, true, widening, dtype, TRILLIM_BF16); \
    }

linear_DEFINE_INT8_BF16_PROJECTION(linear_project_int8_q4_neon_bf16, true, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_BF16_PROJECTION(linear_project_int8_q4_sdot_bf16, false, TRILLIM_GRP_INT4)
linear_DEFINE_INT8_BF16_PROJECTION(linear_project_int8_q8_neon_bf16, true, TRILLIM_GRP_INT8)
linear_DEFINE_INT8_BF16_PROJECTION(linear_project_int8_q8_sdot_bf16, false, TRILLIM_GRP_INT8)

static TrillimTask linear_int8_projection_bf16(uint32_t dtype) {
    const bool dotprod = linear_available_dotprod();
    return dtype == TRILLIM_GRP_INT4 ?
        (dotprod ? linear_project_int8_q4_sdot_bf16 : linear_project_int8_q4_neon_bf16) :
        (dotprod ? linear_project_int8_q8_sdot_bf16 : linear_project_int8_q8_neon_bf16);
}

static __attribute__((noinline)) TrillimStatus linear_try_int8_projection(
    const LinearTask *task, const uint8_t *gate, float (*silu)(float), bool output_bf16, size_t workers, bool *handled
) {
    *handled = false;
    const PackedLinearParameters *p = task->p;
    if (p->tokens != 1) {
        return linear_try_int8_prefill(task, gate, silu, NULL, output_bf16, NULL, workers, handled);
    }
    if (p->output_dtype != TRILLIM_FP32 || p->width < 32 ||
        p->width > linear_INT8_ACTIVATION_MAX_WIDTH || p->outputs < 16 ||
        !trillim_float(p->input_dtype) || !linear_fhm_default_fpcr()) {
        return TRILLIM_OK;
    }
    const size_t width = p->weight_dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    const TrillimTask operation = output_bf16 ? linear_int8_projection_bf16(p->weight_dtype) :
        linear_int8_projection(p->weight_dtype, gate != NULL, p->width);
    if (operation == NULL || p->width % width != 0) {
        return TRILLIM_OK;
    }
    const size_t groups = p->width / width;
    const size_t storage = groups * (width == 256 ? sizeof(linear_Q8KActivationBlock) : sizeof(linear_Q8ActivationBlock));
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
            ? linear_quantize_activation_q8k(task->input, g * width, p->input_dtype,
                (linear_Q8KActivationBlock *)activation + g, operation != linear_project_int8_q6_sdot)
            : linear_quantize_activation_q8(task->input, g * width, p->input_dtype, (linear_Q8ActivationBlock *)activation + g);
        if (!valid) {
            free(activation);
            return TRILLIM_OK;
        }
    }
    if (status == TRILLIM_OK) {
        LinearTask scheduled;
        const bool wide_q8_square = p->weight_dtype == TRILLIM_GRP_INT8 &&
            p->width >= 3072 && p->width == p->outputs;
        const bool one_tile_per_worker = width == 32 && !wide_q8_square;
        if (workers > 1 && (operation == linear_project_int8_q6_sdot || one_tile_per_worker)) {
            scheduled = *task;
            const size_t columns = p->outputs / workers + (p->outputs % workers != 0);
            scheduled.columns = columns + (columns % 2 != 0);
            if (one_tile_per_worker && scheduled.columns < task->columns) {
                scheduled.columns = task->columns;
            }
            task = &scheduled;
        }
        linear_Int8ProjectionTask int8 = {task, activation, gate, silu};
        const size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0);
        if (p->width <= 65536 / p->outputs || workers == 1) {
            LinearTask serial = *task;
            serial.columns = p->outputs;
            linear_Int8ProjectionTask serial_int8 = {&serial, activation, gate, silu};
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
static TrillimStatus try_ffn_int8(
    void *output, const void *input, const void *up, const void *gate, const void *parameters,
    float (*silu)(float), const TrillimKernelContext *context, bool *handled
) {
    const size_t workers = context->thread_pool->worker_count;
    if (workers == 0) {
        *handled = false;
        return TRILLIM_OK;
    }
    const PackedLinearParameters *p = parameters;
    const size_t chunk = p->outputs / workers / 4;
    const size_t columns = chunk < 16 ? 16 : chunk;
    const LinearTask task = {output, input, up, p, columns, context};
    return linear_try_int8_projection(&task, gate, silu, false, workers, handled);
}
#endif

#if defined(__APPLE__) || defined(__linux__)
static TrillimStatus try_ffn_int8_prefill(
    void *output, const void *input, const void *up, const void *gate, const void *parameters,
    float (*scalar_gate)(float), float32x4_t (*vector_gate)(float32x4_t),
    const TrillimKernelContext *context, bool *handled
) {
    *handled = false;
    const size_t workers = context->thread_pool->worker_count;
    if (workers == 0) {
        return TRILLIM_OK;
    }
    const LinearTask task = {output, input, up, parameters, 0, context};
    return linear_try_int8_prefill(&task, gate, scalar_gate, vector_gate, false, NULL, workers, handled);
}
#endif

#if defined(__APPLE__) || defined(__linux__)
static TrillimStatus try_linear_bf16(
    void *output, const void *input, const void *weight, const void *parameters,
    const TrillimKernelContext *context, bool *handled
) {
    *handled = false;
    const PackedLinearParameters *p = parameters;
    const size_t workers = context->thread_pool->worker_count;
    if (workers == 0 || (p->weight_dtype != TRILLIM_GRP_INT4 && p->weight_dtype != TRILLIM_GRP_INT8)) {
        return TRILLIM_OK;
    }
    const size_t chunk = p->outputs / workers / 4 / 2 * 2;
    const LinearTask task = {output, input, weight, p, chunk < 16 ? 16 : chunk, context};
    return linear_try_int8_projection(&task, NULL, NULL, true, workers, handled);
}
#endif

static TrillimStatus project_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const PackedLinearParameters *p = parameters;
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
    LinearTask task = {outputs[0], inputs[0], inputs[1], p, columns, context};
    bool int8_handled;
    TrillimStatus int8_status = linear_try_int8_projection(&task, NULL, NULL, false, workers, &int8_handled);
    if (int8_status != TRILLIM_OK || int8_handled) {
        return int8_status;
    }

#if defined(__APPLE__)
    if (p->tokens >= 32) {
        bool prefill_handled;
        TrillimStatus prefill = linear_try_packed_prefill(&task, &prefill_handled);
        if (prefill != TRILLIM_OK || prefill_handled) {
            return prefill;
        }
    }
#endif
    bool large_tile = p->tokens >= 16 && p->output_dtype == TRILLIM_FP32 &&
        (p->weight_dtype == TRILLIM_MAT_TERNARY || p->weight_dtype >= TRILLIM_GRP_INT5);
    size_t row_tile = large_tile ? 8 : 4;
    TrillimTask operation = p->output_dtype == TRILLIM_FP32 ? linear_projection(p->weight_dtype, row_tile) : linear_project_ordered;
    if (p->output_dtype == TRILLIM_FP32 && p->width != 0 && p->width % 32 == 0) {
        size_t group_rows = p->tokens == 1 ? 1 : row_tile;
        TrillimTask groups = linear_group_projection(p->weight_dtype, group_rows);
        if (groups != NULL) {
            operation = groups;
            row_tile = group_rows;
            if (row_tile == 1) {
                task.columns = columns / 4 * 4;
            }
        }
    }
    if (row_tile == 1 && p->width >= 1024 && p->width <= linear_MAX_FACTORED_WIDTH && p->outputs >= 16) {
        TrillimTask factored = linear_factored_projection(p->weight_dtype);
        if (factored != NULL) {
            bool bounded;
            TrillimStatus status = linear_check_factored_input(&task, &bounded);
            if (status != TRILLIM_OK) {
                return status;
            }
            if (bounded) {
                operation = factored;
#if defined(linear_TRILLIM_PACKED_BFDOT)
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
    if (operation == linear_project_factored_int4 && p->width <= linear_MAX_FHM_WIDTH && large_fhm) {
        bool handled;
        TrillimStatus status = linear_run_fhm_int4(&task, tiles, &handled);
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

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

/* Gelu ffn. */
#include <stdatomic.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_gelu_ffn_fhm_v\\reg,\\reg\n"
    ".endr\n"
);
#define gelu_ffn_FHM
static inline float32x4_t gelu_ffn_fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x4e20ec00 | trillim_gelu_ffn_fhm_%0 | (trillim_gelu_ffn_fhm_%1 << 5) | (trillim_gelu_ffn_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline float32x4_t gelu_ffn_fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x6e20cc00 | trillim_gelu_ffn_fhm_%0 | (trillim_gelu_ffn_fhm_%1 << 5) | (trillim_gelu_ffn_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
#if defined(__clang__)
#define gelu_ffn_FHM __attribute__((target("fp16fml")))
#else
#define gelu_ffn_FHM __attribute__((target("+fp16fml")))
#endif
gelu_ffn_FHM static inline float32x4_t gelu_ffn_fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_low_f16(sum, a, b);
}
gelu_ffn_FHM static inline float32x4_t gelu_ffn_fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_high_f16(sum, a, b);
}
#endif

static bool gelu_ffn_available_fhm(void) {
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

static bool gelu_ffn_fhm_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

enum { gelu_ffn_MAX_FHM_WIDTH = 8192 };

#if defined(__APPLE__) || defined(__linux__)

static float packed_gelu_tanh(float x) {
    const float cube = (x * x) * x;
    const float correction = cube * 0.044715f;
    const float argument = (x + correction) * 0.7978845608028654f;
    const float scaled = argument * 2.0f;
    const float e = expf(-fabsf(scaled));
    const float denominator = e + 1.0f;
    const float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    const float doubled = probability * 2.0f;
    const float cdf = doubled - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}
#endif

enum { gelu_ffn_MAX_TILE = 64, gelu_ffn_MAX_FACTORED_WIDTH = 1 << 16 };

typedef struct {
    size_t tokens, width, hidden, outputs;
} PackedGeluFfnParameters;

typedef struct {
    float *hidden;
    const uint8_t *input;
    const uint8_t *up, *gate;
    const PackedGeluFfnParameters *p;
    const TrillimKernelContext *context;
    size_t columns;
    const uint16_t *half;
} GeluFfnTask;

typedef struct {
    const uint8_t *codes, *scales;
} gelu_ffn_WeightRow;

static TrillimStatus gelu_ffn_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const PackedGeluFfnParameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width < 1024 || p->width > gelu_ffn_MAX_FACTORED_WIDTH || p->width % 32 != 0 ||
        p->hidden < 16 || p->hidden % 4 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, TRILLIM_GRP_INT4, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, TRILLIM_GRP_INT4, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch) ||
        *scratch > SIZE_MAX / 2) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch *= 2;
    return TRILLIM_OK;
}

static TrillimStatus gelu_ffn_check_bounds(
    const uint8_t *values, size_t count, const TrillimKernelContext *context, bool *bounded
) {
    /* These exponent bounds and a width of at most 65536 leave room for the grouped dot. */
    const uint16x8_t minimum = vdupq_n_u16(87 << 7);
    const uint16x8_t span = vdupq_n_u16((81 << 7) - 1);
    uint16x8_t rejected = vdupq_n_u16(0);
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t raw = vreinterpretq_u16_u8(vld1q_u8(values + i * 2));
        uint16x8_t magnitude = vandq_u16(raw, vdupq_n_u16(0x7fff));
        uint16x8_t outside = vcgtq_u16(vsubq_u16(magnitude, minimum), span);
        rejected = vorrq_u16(rejected, vbicq_u16(outside, vceqzq_u16(magnitude)));
    }
    *bounded = vmaxvq_u16(rejected) == 0;
    for (; i < count && *bounded; ++i) {
        uint16_t raw;
        memcpy(&raw, values + i * 2, sizeof(raw));
        unsigned magnitude = raw & 0x7fff;
        *bounded = magnitude == 0 || (magnitude - (87u << 7)) <= (81u << 7) - 1;
    }
    return TRILLIM_OK;
}

static inline float16x8x2_t gelu_ffn_decode_half_codes(uint8x16_t codes) {
    /* Each centered Q4 code has a zero low byte in FP16. */
    static const uint8_t fp16_high[16] = {
        200, 199, 198, 197, 196, 194, 192, 188, 0, 60, 64, 66, 68, 69, 70, 71
    };
    uint8x16_t bits = vqtbl1q_u8(vld1q_u8(fp16_high), codes);
    float16x8x2_t halves = {{
        vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(bits), 8)),
        vreinterpretq_f16_u16(vshll_high_n_u8(bits, 8))
    }};
    return halves;
}

static inline void gelu_ffn_decode_codes(uint8x16_t codes, float32x4_t result[4]) {
    float16x8x2_t halves = gelu_ffn_decode_half_codes(codes);
    result[0] = vcvt_f32_f16(vget_low_f16(halves.val[0]));
    result[1] = vcvt_high_f32_f16(halves.val[0]);
    result[2] = vcvt_f32_f16(vget_low_f16(halves.val[1]));
    result[3] = vcvt_high_f32_f16(halves.val[1]);
}

static inline __attribute__((always_inline)) TrillimStatus gelu_ffn_project_pair(
    const GeluFfnTask *task, const gelu_ffn_WeightRow weights[4], bool factored, float result[4]
) {
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < task->p->width; k += 32) {
        if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t input[8];
        for (size_t i = 0; i < 8; ++i) {
            uint16x4_t bits = vreinterpret_u16_u8(vld1_u8(task->input + k * 2 + i * 8));
            input[i] = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
        }
        for (size_t c = 0; c < 4; ++c) {
            uint8x16_t packed = vld1q_u8(weights[c].codes + k / 2);
            float32x4_t codes[8];
            gelu_ffn_decode_codes(vandq_u8(packed, vdupq_n_u8(15)), codes);
            gelu_ffn_decode_codes(vshrq_n_u8(packed, 4), codes + 4);
            uint16_t bits;
            memcpy(&bits, weights[c].scales + k / 16, sizeof(bits));
            float scale = trillim_bf16(bits);
            if (factored) {
                float32x4_t lo = vdupq_n_f32(0), hi = lo;
                for (size_t i = 0; i < 4; ++i) {
                    lo = vfmaq_f32(lo, input[i], codes[i]);
                    hi = vfmaq_f32(hi, input[i + 4], codes[i + 4]);
                }
                sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), scale);
            } else {
                for (size_t i = 0; i < 8; ++i) {
                    sums[c] = vfmaq_f32(sums[c], input[i], vmulq_n_f32(codes[i], scale));
                }
            }
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

gelu_ffn_FHM static TrillimStatus gelu_ffn_project_fhm_pair(const GeluFfnTask *task, const gelu_ffn_WeightRow weights[4], float result[4]) {
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < task->p->width; k += 32) {
        if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8x4_t raw = vld1q_u16_x4(task->half + k);
        float16x8_t x0 = vreinterpretq_f16_u16(raw.val[0]), x1 = vreinterpretq_f16_u16(raw.val[1]);
        float16x8_t x2 = vreinterpretq_f16_u16(raw.val[2]), x3 = vreinterpretq_f16_u16(raw.val[3]);
        for (size_t c = 0; c < 4; ++c) {
            uint8x16_t packed = vld1q_u8(weights[c].codes + k / 2);
            float16x8x2_t low = gelu_ffn_decode_half_codes(vandq_u8(packed, vdupq_n_u8(15)));
            float16x8x2_t high = gelu_ffn_decode_half_codes(vshrq_n_u8(packed, 4));
            float32x4_t lo = vdupq_n_f32(0), hi = lo;
            lo = gelu_ffn_fhm_low(lo, x0, low.val[0]);
            hi = gelu_ffn_fhm_low(hi, x2, high.val[0]);
            lo = gelu_ffn_fhm_high(lo, x0, low.val[0]);
            hi = gelu_ffn_fhm_high(hi, x2, high.val[0]);
            lo = gelu_ffn_fhm_low(lo, x1, low.val[1]);
            hi = gelu_ffn_fhm_low(hi, x3, high.val[1]);
            lo = gelu_ffn_fhm_high(lo, x1, low.val[1]);
            hi = gelu_ffn_fhm_high(hi, x3, high.val[1]);
            uint16_t bits;
            memcpy(&bits, weights[c].scales + k / 16, sizeof(bits));
            sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), trillim_bf16(bits));
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

static TrillimStatus gelu_ffn_project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const GeluFfnTask *task = opaque;
    const PackedGeluFfnParameters *p = task->p;
    size_t groups = p->width / 32;
    size_t first = tile * task->columns;
    size_t count = p->hidden - first < task->columns ? p->hidden - first : task->columns;
    const uint8_t *up_scales = task->up + (p->hidden * 16 + first * 2) * groups;
    const uint8_t *gate_scales = task->gate + (p->hidden * 16 + first * 2) * groups;
    bool up_bounded, gate_bounded;
    TrillimStatus status = gelu_ffn_check_bounds(up_scales, count * groups, task->context, &up_bounded);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = gelu_ffn_check_bounds(gate_scales, count * groups, task->context, &gate_bounded);
    if (status != TRILLIM_OK) {
        return status;
    }
    bool factored = up_bounded && gate_bounded;
    bool fhm = factored && task->half != NULL && gelu_ffn_fhm_default_fpcr();
    float up[gelu_ffn_MAX_TILE], gate[gelu_ffn_MAX_TILE];
    for (size_t column = 0; column < count; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = column * groups;
        const uint8_t *up_codes = task->up + (first + column) * groups * 16;
        const uint8_t *gate_codes = task->gate + (first + column) * groups * 16;
        gelu_ffn_WeightRow weights[] = {
            {up_codes, up_scales + offset * 2},
            {gate_codes, gate_scales + offset * 2},
            {up_codes + groups * 16, up_scales + (offset + groups) * 2},
            {gate_codes + groups * 16, gate_scales + (offset + groups) * 2},
        };
        float values[4];
        if (fhm) {
            status = gelu_ffn_project_fhm_pair(task, weights, values);
        } else if (factored) {
            status = gelu_ffn_project_pair(task, weights, true, values);
        } else {
            status = gelu_ffn_project_pair(task, weights, false, values);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
        up[column] = values[0];
        gate[column] = values[1];
        up[column + 1] = values[2];
        gate[column + 1] = values[3];
    }
    void *outputs[] = {gate};
    const void *inputs[] = {gate};
    GeluParameters parameters = {count, TRILLIM_FP32, TRILLIM_FP32};
    status = apply_gelu_tanh(outputs, inputs, NULL, &parameters, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    for (size_t i = 0; i < count; i += 4) {
        vst1q_f32(task->hidden + first + i, vmulq_f32(vld1q_f32(up + i), vld1q_f32(gate + i)));
    }
    return TRILLIM_OK;
}

static TrillimStatus gelu_ffn_run_hidden(GeluFfnTask *task, size_t tiles) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    return tiles == 1 ? gelu_ffn_project_hidden(task, 0, 0) : pool->run(pool->context, tiles, gelu_ffn_project_hidden, task);
}

static TrillimStatus gelu_ffn_run_hidden_fhm(GeluFfnTask *task, size_t tiles) {
    uint16_t input[gelu_ffn_MAX_FHM_WIDTH];
    uint32x4_t error = vdupq_n_u32(0);
    for (size_t k = 0; k < task->p->width; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t raw = vreinterpretq_u16_u8(vld1q_u8(task->input + k * 2));
        uint32x4_t lo = vshll_n_u16(vget_low_u16(raw), 16);
        uint32x4_t hi = vshll_high_n_u16(raw, 16);
        float16x8_t half = vcvt_high_f16_f32(vcvt_f16_f32(vreinterpretq_f32_u32(lo)), vreinterpretq_f32_u32(hi));
        error = vorrq_u32(error, veorq_u32(lo, vreinterpretq_u32_f32(vcvt_f32_f16(vget_low_f16(half)))));
        error = vorrq_u32(error, veorq_u32(hi, vreinterpretq_u32_f32(vcvt_high_f32_f16(half))));
        vst1q_u16(input + k, vreinterpretq_u16_f16(half));
    }
    if (vmaxvq_u32(error) == 0) {
        task->half = input;
    }
    return gelu_ffn_run_hidden(task, tiles);
}

static TrillimStatus project_gelu_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const PackedGeluFfnParameters *p = parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    float *hidden = scratch;
    PackedLinearParameters projection = {p->tokens, p->width, p->hidden, TRILLIM_BF16, TRILLIM_GRP_INT4, TRILLIM_FP32};
    bool hidden_handled = false;
#if defined(__APPLE__) || defined(__linux__)
    if (p->tokens == 1 || p->tokens >= 4) {
        TrillimStatus status = p->tokens == 1
            ? try_ffn_int8(
                hidden, inputs[0], inputs[1], inputs[2], &projection, packed_gelu_tanh, context, &hidden_handled)
            : try_ffn_int8_prefill(
                hidden, inputs[0], inputs[1], inputs[2], &projection, packed_gelu_tanh,
                gelu_tanh4_prefill, context, &hidden_handled);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
#endif
    bool bounded = false;
    if (p->tokens == 1 && !hidden_handled) {
        TrillimStatus status = gelu_ffn_check_bounds(inputs[0], p->width, context, &bounded);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (bounded) {
        size_t workers = context->thread_pool->worker_count;
        if (workers == 0) {
            return TRILLIM_ERROR;
        }
        size_t columns = p->hidden / workers / 4;
        columns = columns < 16 ? 16 : columns > gelu_ffn_MAX_TILE ? gelu_ffn_MAX_TILE : columns / 4 * 4;
        size_t tiles = p->hidden / columns + (p->hidden % columns != 0);
        GeluFfnTask task = {hidden, inputs[0], inputs[1], inputs[2], p, context, columns, NULL};
        TrillimStatus status = p->width <= gelu_ffn_MAX_FHM_WIDTH && p->hidden >= 1152 && gelu_ffn_available_fhm() && gelu_ffn_fhm_default_fpcr()
            ? gelu_ffn_run_hidden_fhm(&task, tiles) : gelu_ffn_run_hidden(&task, tiles);
        if (status != TRILLIM_OK) {
            return status;
        }
    } else if (!hidden_handled) {
        size_t count = p->tokens * p->hidden;
        float *gate = hidden + count;
        void *up_outputs[] = {hidden}, *gate_outputs[] = {gate};
        const void *up_inputs[] = {inputs[0], inputs[1]}, *gate_inputs[] = {inputs[0], inputs[2]};
        TrillimStatus status = project_linear(up_outputs, up_inputs, NULL, &projection, context);
        if (status == TRILLIM_OK) {
            status = project_linear(gate_outputs, gate_inputs, NULL, &projection, context);
        }
        if (status == TRILLIM_OK) {
            const void *glu_inputs[] = {hidden, gate};
            const GeluMulParameters activation = {count};
            status = apply_gelu_tanh_mul(up_outputs, glu_inputs, NULL, &activation, context);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    const void *down_inputs[] = {hidden, inputs[3]};
    PackedLinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, TRILLIM_GRP_INT4, TRILLIM_FP32};
    return project_linear(outputs, down_inputs, NULL, &down, context);
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

/* Quantized gelu ffn. */

#if defined(__APPLE__) || defined(__linux__)

static float quantized_gelu_ffn_gelu_tanh(float x) {
    const float cube = (x * x) * x;
    const float correction = cube * 0.044715f;
    const float argument = (x + correction) * 0.7978845608028654f;
    const float scaled = argument * 2.0f;
    const float e = expf(-fabsf(scaled));
    const float denominator = e + 1.0f;
    const float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    const float doubled = probability * 2.0f;
    const float cdf = doubled - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}
#endif

enum { quantized_gelu_ffn_MAX_WIDTH = 65536 };

typedef struct {
    size_t tokens, width, hidden, outputs;
    uint32_t up_dtype, down_dtype;
} PackedQuantizedGeluFfnParameters;

static TrillimStatus quantized_gelu_ffn_requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const PackedQuantizedGeluFfnParameters *p = opaque;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    const size_t group = p->up_dtype == TRILLIM_DUBGRP_INT6 ? 256 : 32;
    if (p->width < 1024 || p->width > quantized_gelu_ffn_MAX_WIDTH || p->width % group != 0 || p->hidden < 16 ||
        (p->up_dtype != TRILLIM_GRP_INT5 && p->up_dtype != TRILLIM_DUBGRP_INT6 && p->up_dtype != TRILLIM_GRP_INT8) ||
        p->down_dtype < TRILLIM_GRP_INT8 || p->down_dtype > TRILLIM_DUBGRP_INT6 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, p->up_dtype, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, p->down_dtype, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch) ||
        *scratch > SIZE_MAX / 2) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch *= 2;
    return TRILLIM_OK;
}

static TrillimStatus project_quantized_gelu_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const PackedQuantizedGeluFfnParameters *p = parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    float *hidden = scratch;
    const PackedLinearParameters expand = {p->tokens, p->width, p->hidden, TRILLIM_BF16, p->up_dtype, TRILLIM_FP32};
    bool hidden_handled = false;
#if defined(__APPLE__) || defined(__linux__)
    if (p->tokens == 1 || (p->tokens >= 4 && p->up_dtype == TRILLIM_GRP_INT8)) {
        TrillimStatus status = p->tokens == 1
            ? try_ffn_int8(
                hidden, inputs[0], inputs[1], inputs[2], &expand, quantized_gelu_ffn_gelu_tanh, context, &hidden_handled)
            : try_ffn_int8_prefill(
                hidden, inputs[0], inputs[1], inputs[2], &expand, quantized_gelu_ffn_gelu_tanh,
                gelu_tanh4_prefill, context, &hidden_handled);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
#endif
    if (!hidden_handled) {
        const size_t count = p->tokens * p->hidden;
        float *gate = hidden + count;
        void *up_outputs[] = {hidden}, *gate_outputs[] = {gate};
        const void *up_inputs[] = {inputs[0], inputs[1]}, *gate_inputs[] = {inputs[0], inputs[2]};
        TrillimStatus status = project_linear(up_outputs, up_inputs, NULL, &expand, context);
        if (status == TRILLIM_OK) {
            status = project_linear(gate_outputs, gate_inputs, NULL, &expand, context);
        }
        if (status == TRILLIM_OK) {
            const void *glu_inputs[] = {hidden, gate};
            const GeluMulParameters activation = {count};
            status = apply_gelu_tanh_mul(up_outputs, glu_inputs, NULL, &activation, context);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    const void *down_inputs[] = {hidden, inputs[3]};
    const PackedLinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return project_linear(outputs, down_inputs, NULL, &down, context);
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC reset_options
#endif

enum { CAST_TILE_ELEMENTS = 32768, CAST_CHUNK_ELEMENTS = 4096 };

typedef struct {
    uint8_t *output;
    const uint8_t *input;
    size_t count;
    const TrillimKernelContext *context;
} Bf16CastTask;

static inline uint16x4_t narrow_bf16(uint32x4_t bits) {
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
    uint32x4_t nan = vcgtq_u32(magnitude, vdupq_n_u32(0x7f800000));
    uint16x4_t rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
    uint16x4_t quiet = vorr_u16(vmovn_u32(upper), vdup_n_u16(0x40));
    return vbsl_u16(vmovn_u32(nan), quiet, rounded);
}

static TrillimStatus cast_bf16_tile(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Bf16CastTask *task = opaque;
    size_t index = tile * CAST_TILE_ELEMENTS;
    const size_t limit = task->count - index < CAST_TILE_ELEMENTS ? task->count : index + CAST_TILE_ELEMENTS;
    while (index < limit) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t end = limit - index < CAST_CHUNK_ELEMENTS ? limit : index + CAST_CHUNK_ELEMENTS;
        for (; end - index >= 8; index += 8) {
            const uint8_t *input = task->input + index * sizeof(float);
            uint32x4_t lo, hi;
            memcpy(&lo, input, sizeof(lo));
            memcpy(&hi, input + sizeof(lo), sizeof(hi));
            const uint16x8_t bits = vcombine_u16(narrow_bf16(lo), narrow_bf16(hi));
            memcpy(task->output + index * sizeof(uint16_t), &bits, sizeof(bits));
        }
        for (; index < end; ++index) {
            const float value = trillim_load(task->input, index, TRILLIM_FP32).f;
            const uint16_t bits = trillim_to_bf16(value);
            memcpy(task->output + index * sizeof(uint16_t), &bits, sizeof(bits));
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus cast_fp32_to_bf16(
    void *output, const void *input, size_t count, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Bf16CastTask task = {output, input, count, context};
    size_t tiles = count / CAST_TILE_ELEMENTS + (count % CAST_TILE_ELEMENTS != 0);
    if (tiles > 1 && context->thread_pool->worker_count > 1) {
        return context->thread_pool->run(context->thread_pool->context, tiles, cast_bf16_tile, &task);
    }
    for (size_t tile = 0; tile < tiles; ++tile) {
        const TrillimStatus status = cast_bf16_tile(&task, tile, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static bool can_reuse_gate(const PackedQuantizedGeluFfnParameters *p) {
    return p->outputs <= p->hidden &&
           (p->up_dtype == TRILLIM_GRP_INT4 || p->up_dtype == TRILLIM_GRP_INT8) &&
           (p->down_dtype == TRILLIM_GRP_INT4 || p->down_dtype == TRILLIM_GRP_INT8);
}

#if defined(__APPLE__) || defined(__linux__)
static TrillimStatus try_direct_bf16(
    void *output, const void *const *inputs, void *hidden, const PackedQuantizedGeluFfnParameters *p,
    const TrillimKernelContext *context, bool *handled
) {
    *handled = false;
    if (p->tokens != 1 && p->tokens < 4) {
        return TRILLIM_OK;
    }
    const PackedLinearParameters expand = {p->tokens, p->width, p->hidden, TRILLIM_BF16, p->up_dtype, TRILLIM_FP32};
    bool expanded;
    const TrillimStatus status = p->tokens == 1
        ? try_ffn_int8(
            hidden, inputs[0], inputs[1], inputs[2], &expand, packed_gelu_tanh, context, &expanded)
        : try_ffn_int8_prefill(
            hidden, inputs[0], inputs[1], inputs[2], &expand, packed_gelu_tanh,
            gelu_tanh4_prefill, context, &expanded);
    if (status != TRILLIM_OK || !expanded) {
        return status;
    }
    const PackedLinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return try_linear_bf16(output, hidden, inputs[3], &down, context, handled);
}
#endif

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const PackedQuantizedGeluFfnParameters *p = opaque;
    size_t projected;
    TrillimStatus status;
    if (p->up_dtype == TRILLIM_GRP_INT4) {
        if (p->down_dtype != TRILLIM_GRP_INT4) {
            return TRILLIM_ERROR;
        }
        const PackedGeluFfnParameters projection = {p->tokens, p->width, p->hidden, p->outputs};
        status = gelu_ffn_requirements(&projection, inputs, &projected, scratch);
    } else {
        status = quantized_gelu_ffn_requirements(p, inputs, &projected, scratch);
    }
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!can_reuse_gate(p)) {
        if (*scratch > SIZE_MAX - projected) {
            return TRILLIM_ERROR;
        }
        *scratch += projected;
    }
    outputs[0] = projected / 2;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_gelu_ffn_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const PackedQuantizedGeluFfnParameters *p = opaque;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    const bool reuse_gate = can_reuse_gate(p);
#if defined(__APPLE__) || defined(__linux__)
    if (reuse_gate) {
        bool handled;
        const TrillimStatus direct = try_direct_bf16(outputs[0], inputs, scratch, p, context, &handled);
        if (direct != TRILLIM_OK || handled) {
            return direct;
        }
    }
#endif
    /* The child consumes the gate projection before writing the down projection. */
    const size_t hidden_bytes = p->tokens * p->hidden * sizeof(float);
    void *temporary = (uint8_t *)scratch + (reuse_gate ? hidden_bytes : 2 * hidden_bytes);
    void *projected[] = {temporary};
    TrillimStatus status;
    if (p->up_dtype == TRILLIM_GRP_INT4) {
        const PackedGeluFfnParameters projection = {p->tokens, p->width, p->hidden, p->outputs};
        status = project_gelu_ffn(projected, inputs, scratch, &projection, context);
    } else {
        status = project_quantized_gelu_ffn(projected, inputs, scratch, p, context);
    }
    if (status != TRILLIM_OK) {
        return status;
    }

    return cast_fp32_to_bf16(outputs[0], temporary, p->tokens * p->outputs, context);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_gelu_ffn_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(PackedQuantizedGeluFfnParameters, tokens), NULL},
        {"width", "size_t", offsetof(PackedQuantizedGeluFfnParameters, width), NULL},
        {"hidden", "size_t", offsetof(PackedQuantizedGeluFfnParameters, hidden), NULL},
        {"outputs", "size_t", offsetof(PackedQuantizedGeluFfnParameters, outputs), NULL},
        {"up_dtype", "uint32_t", offsetof(PackedQuantizedGeluFfnParameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(PackedQuantizedGeluFfnParameters, down_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(PackedQuantizedGeluFfnParameters), TRILLIM_ALIGNOF(PackedQuantizedGeluFfnParameters), 6, fields, 4, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_gelu_ffn_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}

TRILLIM_KERNEL const char *trillim_neon_packed_gelu_ffn_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "\n"
           "physical_inputs = FFN_INPUTS = (\"x\", \"up\", \"gate\", \"down\")\n"
           "GELU_FORMATS = {\"grp_int5\", \"dubgrp_int6\", \"grp_int8\"}\n"
           "\n"
           "def contiguous(*bindings) -> None:\n"
           "    for binding in bindings:\n"
           "        stride = _contiguous_strides(binding.shape)\n"
           "        if (binding.strides or stride) != stride:\n"
           "            raise ValueError(\"requires contiguous operands\")\n"
           "\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in FFN_INPUTS)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != \"bf16\" or output.dtype != \"bf16\":\n"
           "        raise ValueError(\"requires BF16 input and output\")\n"
           "    contiguous(x, up, gate, down, output)\n"
           "    allowed_down = {\"grp_int4\"} if up.dtype == \"grp_int4\" else GELU_FORMATS | {\"grp_int4\"}\n"
           "    if up.dtype not in GELU_FORMATS | {\"grp_int4\"} or gate.dtype != up.dtype or down.dtype not in allowed_down:\n"
           "        raise ValueError(\"unsupported GELU FFN formats\")\n"
           "    if any(len(weight.shape) != 2 or _storage_shape(weight) != weight.shape\n"
           "           for weight in (up, gate, down)):\n"
           "        raise ValueError(\"requires complete packed matrices\")\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width or output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError(\"incompatible GELU FFN dimensions\")\n"
           "    up_dtype = _DTYPE_IDS[up.dtype]\n"
           "    group = 256 if up.dtype == \"dubgrp_int6\" else 32\n"
           "    if (width.value is None or not 1024 <= width.value <= 65536\n"
           "        or width.value % group or hidden.value is None or hidden.value < 16\n"
           "        or up.dtype == \"grp_int4\" and hidden.value % 4):\n"
           "        raise ValueError(\"requires aligned fixed-width GELU projections\")\n"
           "    return {\"tokens\": _product(x.shape[:-1]), \"width\": width, \"hidden\": hidden,\n"
           "            \"outputs\": down.shape[0], \"up_dtype\": up_dtype, \"down_dtype\": _DTYPE_IDS[down.dtype]}\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, activation,\n"
           "                 up_bias=None, gate_bias=None, down_bias=None, dtype: DType = \"bf16\") -> tuple[Tensor, ...]:\n"
           "    if activation is Silu or up_bias is not None or gate_bias is not None or down_bias is not None or dtype != \"bf16\":\n"
           "        raise ValueError(\"requires unbiased BF16 GELU FFN\")\n"
           "    return (Cast(FFN(x, up=up, gate=gate, down=down,\n"
           "                     activation=lambda value: Gelu(value, approximate=\"tanh\")), \"bf16\"),)\n";
}
