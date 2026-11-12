#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Apple's older assembly printer drops operands on optional ARM instructions. */
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_asm_v\\reg,\\reg\n"
    ".endr\n"
);
#define DOTPROD
#define I8MM
static inline int32x4_t dot_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__ volatile(".inst 0x4e809400 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}

#else
/* Assembly is rebuilt with baseline flags after function target attributes vanish. */
__asm__(".arch armv8.6-a+dotprod+i8mm");
#if defined(__clang__)
#define DOTPROD __attribute__((target("dotprod")))
#define I8MM __attribute__((target("i8mm")))
#else
#define DOTPROD __attribute__((target("+dotprod")))
#define I8MM __attribute__((target("+i8mm")))
#endif
DOTPROD static inline int32x4_t dot_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    return vdotq_s32(sum, a, b);
}

#endif

typedef struct {
    size_t tokens, width, outputs;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    void *output;
    const int8_t *input, *weight;
    const float *scales;
    const Parameters *p;
    size_t columns;
    float gamma;
    const TrillimKernelContext *context;
} Task;

static unsigned features(void) {
    static atomic_uint cached;
    unsigned value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
        return value;
    }
    value = 1;
#if defined(__APPLE__)
    int enabled = 0;
    size_t size = sizeof(enabled);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &enabled, &size, NULL, 0) == 0 && enabled) {
        value |= 2;
    }
    enabled = 0;
    size = sizeof(enabled);
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &enabled, &size, NULL, 0) == 0 && enabled) {
        value |= 4;
    }
#elif defined(__linux__) && defined(__aarch64__)
    if (getauxval(AT_HWCAP) & (1UL << 20)) {
        value |= 2;
    }
    if (getauxval(AT_HWCAP2) & (1UL << 13)) {
        value |= 4;
    }
#elif defined(__ARM_FEATURE_DOTPROD)
    value |= 2;
#endif
    atomic_store_explicit(&cached, value, memory_order_relaxed);
    return value;
}

typedef struct {
    size_t tokens, width, outputs;
} BitLinearParameters;

typedef struct {
    const float *input;
    int8_t *codes;
    float *scales;
    const BitLinearParameters *p;
    const TrillimKernelContext *context;
} QuantizeTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const BitLinearParameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    if (!p->width || p->width > INT32_MAX / 256 || groups > SIZE_MAX / 32 ||
        p->tokens > SIZE_MAX / sizeof(float) / p->width ||
        p->outputs > (SIZE_MAX - sizeof(float)) / (groups * 32) ||
        p->tokens > SIZE_MAX / sizeof(float) / p->outputs ||
        p->tokens * p->width > SIZE_MAX - 3 ||
        p->tokens > (SIZE_MAX - ((p->tokens * p->width + 3) & ~(size_t)3)) / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * sizeof(float);
    inputs[1] = p->outputs * groups * 32 + sizeof(float);
    outputs[0] = p->tokens * p->outputs * sizeof(float);
    *scratch = ((p->tokens * p->width + 3) & ~(size_t)3) + p->tokens * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus quantize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const QuantizeTask *task = opaque;
    const BitLinearParameters *p = task->p;
    const float *source = task->input + row * p->width;
    int8_t *target = task->codes + row * p->width;
    float32x4_t maximum = vdupq_n_f32(1e-5f);
    uint32x4_t valid = vdupq_n_u32(UINT32_MAX);
    size_t k = 0;
    for (; k + 4 <= p->width; k += 4) {
        if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vabsq_f32(vld1q_f32(source + k));
        valid = vandq_u32(valid, vcltq_f32(value, vdupq_n_f32(INFINITY)));
        maximum = vmaxq_f32(maximum, value);
    }
    if (!vminvq_u32(valid)) {
        return TRILLIM_ERROR;
    }
    float largest = vmaxvq_f32(maximum);
    for (; k < p->width; ++k) {
        if (!isfinite(source[k])) {
            return TRILLIM_ERROR;
        }
        largest = fmaxf(largest, fabsf(source[k]));
    }
    float multiplier = 127.0f / largest;
    task->scales[row] = multiplier;
    k = 0;
    for (; k + 16 <= p->width; k += 16) {
        if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(source + k), multiplier));
        int32x4_t b = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(source + k + 4), multiplier));
        int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(source + k + 8), multiplier));
        int32x4_t d = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(source + k + 12), multiplier));
        int16x8_t low = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        int16x8_t high = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
        vst1q_s8(target + k, vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
    }
    for (; k < p->width; ++k) {
        float value = fminf(127.0f, fmaxf(-128.0f, source[k] * multiplier));
        target[k] = (int8_t)vcvtns_s32_f32(value);
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) void store(const Task *task, size_t row, size_t column, int32_t sum) {
    TrillimValue value = {.f = ((float)sum * task->gamma) / task->scales[row]};
    size_t index = row * task->p->outputs + column;
    if (task->p->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + index * 4, &value.f, sizeof(value.f));
    } else {
        uint16_t encoded = task->p->output_dtype == TRILLIM_BF16 ?
            trillim_to_bf16(value.f) : trillim_to_fp16(value.f);
        memcpy((uint8_t *)task->output + index * 2, &encoded, sizeof(encoded));
    }
}

static int32x4_t multiply(int32x4_t sum, int8x16_t a, int8x16_t b) {
    sum = vpadalq_s16(sum, vmull_s8(vget_low_s8(a), vget_low_s8(b)));
    return vpadalq_s16(sum, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
}

static TrillimStatus project_neon(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = (p->outputs + task->columns - 1) / task->columns;
    size_t row = tile / tiles * 4;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t first = tile % tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t stride = (p->width / 128 + (p->width % 128 != 0)) * 32;
    const int8_t *x0 = task->input + row * p->width;
    const int8_t *x1 = rows > 1 ? x0 + p->width : x0;
    const int8_t *x2 = rows > 2 ? x1 + p->width : x0;
    const int8_t *x3 = rows > 3 ? x2 + p->width : x0;
    const uint8x16_t mask = vdupq_n_u8(3);
    const int8x16_t one = vdupq_n_s8(1);
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        const uint8_t *w0 = (const uint8_t *)task->weight + column * stride;
        const uint8_t *w1 = (const uint8_t *)task->weight + second * stride;
        int32x4_t s00 = vdupq_n_s32(0), s01 = s00, s10 = s00, s11 = s00;
        int32x4_t s20 = s00, s21 = s00, s30 = s00, s31 = s00;
        uint8x16_t invalid = vdupq_n_u8(0);
        size_t k = 0;
        for (; k + 128 <= p->width; k += 128) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            for (size_t half = 0; half < 32; half += 16) {
                uint8x16_t p0 = vld1q_u8(w0 + k / 4 + half);
                uint8x16_t p1 = vld1q_u8(w1 + k / 4 + half);
                invalid = vorrq_u8(invalid, vandq_u8(p0, vshrq_n_u8(p0, 1)));
                invalid = vorrq_u8(invalid, vandq_u8(p1, vshrq_n_u8(p1, 1)));
                int8x16_t b00 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 6), mask)), one);
                int8x16_t b01 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 6), mask)), one);
                int8x16_t b10 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 4), mask)), one);
                int8x16_t b11 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 4), mask)), one);
                int8x16_t b20 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 2), mask)), one);
                int8x16_t b21 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 2), mask)), one);
                int8x16_t b30 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), one);
                int8x16_t b31 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask)), one);
                int8x16_t a0 = vld1q_s8(x0 + k + half + 0);
                s00 = multiply(s00, a0, b00);
                s01 = multiply(s01, a0, b01);
                int8x16_t a1 = vld1q_s8(x0 + k + half + 32);
                s00 = multiply(s00, a1, b10);
                s01 = multiply(s01, a1, b11);
                int8x16_t a2 = vld1q_s8(x0 + k + half + 64);
                s00 = multiply(s00, a2, b20);
                s01 = multiply(s01, a2, b21);
                int8x16_t a3 = vld1q_s8(x0 + k + half + 96);
                s00 = multiply(s00, a3, b30);
                s01 = multiply(s01, a3, b31);
                if (rows > 1) {
                    int8x16_t a0 = vld1q_s8(x1 + k + half + 0);
                    s10 = multiply(s10, a0, b00);
                    s11 = multiply(s11, a0, b01);
                    int8x16_t a1 = vld1q_s8(x1 + k + half + 32);
                    s10 = multiply(s10, a1, b10);
                    s11 = multiply(s11, a1, b11);
                    int8x16_t a2 = vld1q_s8(x1 + k + half + 64);
                    s10 = multiply(s10, a2, b20);
                    s11 = multiply(s11, a2, b21);
                    int8x16_t a3 = vld1q_s8(x1 + k + half + 96);
                    s10 = multiply(s10, a3, b30);
                    s11 = multiply(s11, a3, b31);
                }
                if (rows > 2) {
                    int8x16_t a0 = vld1q_s8(x2 + k + half + 0);
                    s20 = multiply(s20, a0, b00);
                    s21 = multiply(s21, a0, b01);
                    int8x16_t a1 = vld1q_s8(x2 + k + half + 32);
                    s20 = multiply(s20, a1, b10);
                    s21 = multiply(s21, a1, b11);
                    int8x16_t a2 = vld1q_s8(x2 + k + half + 64);
                    s20 = multiply(s20, a2, b20);
                    s21 = multiply(s21, a2, b21);
                    int8x16_t a3 = vld1q_s8(x2 + k + half + 96);
                    s20 = multiply(s20, a3, b30);
                    s21 = multiply(s21, a3, b31);
                }
                if (rows > 3) {
                    int8x16_t a0 = vld1q_s8(x3 + k + half + 0);
                    s30 = multiply(s30, a0, b00);
                    s31 = multiply(s31, a0, b01);
                    int8x16_t a1 = vld1q_s8(x3 + k + half + 32);
                    s30 = multiply(s30, a1, b10);
                    s31 = multiply(s31, a1, b11);
                    int8x16_t a2 = vld1q_s8(x3 + k + half + 64);
                    s30 = multiply(s30, a2, b20);
                    s31 = multiply(s31, a2, b21);
                    int8x16_t a3 = vld1q_s8(x3 + k + half + 96);
                    s30 = multiply(s30, a3, b30);
                    s31 = multiply(s31, a3, b31);
                }
            }
        }
        if (vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            return TRILLIM_ERROR;
        }
        int32_t total0 = vaddvq_s32(s00);
        for (size_t j = k; j < p->width; ++j) {
            uint8_t packed = w0[(j / 128) * 32 + j % 32];
            int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            total0 += (int32_t)x0[j] * (code - 1);
        }
        store(task, row + 0, column, total0);
        int32_t total1 = vaddvq_s32(s01);
        for (size_t j = k; j < p->width; ++j) {
            uint8_t packed = w1[(j / 128) * 32 + j % 32];
            int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            total1 += (int32_t)x0[j] * (code - 1);
        }
        if (second != column) {
            store(task, row + 0, second, total1);
        }
        if (rows > 1) {
            int32_t total0 = vaddvq_s32(s10);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x1[j] * (code - 1);
            }
            store(task, row + 1, column, total0);
            int32_t total1 = vaddvq_s32(s11);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x1[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 1, second, total1);
            }
        }
        if (rows > 2) {
            int32_t total0 = vaddvq_s32(s20);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x2[j] * (code - 1);
            }
            store(task, row + 2, column, total0);
            int32_t total1 = vaddvq_s32(s21);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x2[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 2, second, total1);
            }
        }
        if (rows > 3) {
            int32_t total0 = vaddvq_s32(s30);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x3[j] * (code - 1);
            }
            store(task, row + 3, column, total0);
            int32_t total1 = vaddvq_s32(s31);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x3[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 3, second, total1);
            }
        }
    }
    return TRILLIM_OK;
}

DOTPROD static TrillimStatus project_dotprod(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = (p->outputs + task->columns - 1) / task->columns;
    size_t row = tile / tiles * 4;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t first = tile % tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t stride = (p->width / 128 + (p->width % 128 != 0)) * 32;
    const int8_t *x0 = task->input + row * p->width;
    const int8_t *x1 = rows > 1 ? x0 + p->width : x0;
    const int8_t *x2 = rows > 2 ? x1 + p->width : x0;
    const int8_t *x3 = rows > 3 ? x2 + p->width : x0;
    const uint8x16_t mask = vdupq_n_u8(3);
    const int8x16_t one = vdupq_n_s8(1);
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        const uint8_t *w0 = (const uint8_t *)task->weight + column * stride;
        const uint8_t *w1 = (const uint8_t *)task->weight + second * stride;
        int32x4_t s00 = vdupq_n_s32(0), s01 = s00, s10 = s00, s11 = s00;
        int32x4_t s20 = s00, s21 = s00, s30 = s00, s31 = s00;
        uint8x16_t invalid = vdupq_n_u8(0);
        size_t k = 0;
        for (; k + 128 <= p->width; k += 128) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            for (size_t half = 0; half < 32; half += 16) {
                uint8x16_t p0 = vld1q_u8(w0 + k / 4 + half);
                uint8x16_t p1 = vld1q_u8(w1 + k / 4 + half);
                invalid = vorrq_u8(invalid, vandq_u8(p0, vshrq_n_u8(p0, 1)));
                invalid = vorrq_u8(invalid, vandq_u8(p1, vshrq_n_u8(p1, 1)));
                int8x16_t b00 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 6), mask)), one);
                int8x16_t b01 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 6), mask)), one);
                int8x16_t b10 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 4), mask)), one);
                int8x16_t b11 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 4), mask)), one);
                int8x16_t b20 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p0, 2), mask)), one);
                int8x16_t b21 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(p1, 2), mask)), one);
                int8x16_t b30 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), one);
                int8x16_t b31 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask)), one);
                int8x16_t a0 = vld1q_s8(x0 + k + half + 0);
                s00 = dot_product(s00, a0, b00);
                s01 = dot_product(s01, a0, b01);
                int8x16_t a1 = vld1q_s8(x0 + k + half + 32);
                s00 = dot_product(s00, a1, b10);
                s01 = dot_product(s01, a1, b11);
                int8x16_t a2 = vld1q_s8(x0 + k + half + 64);
                s00 = dot_product(s00, a2, b20);
                s01 = dot_product(s01, a2, b21);
                int8x16_t a3 = vld1q_s8(x0 + k + half + 96);
                s00 = dot_product(s00, a3, b30);
                s01 = dot_product(s01, a3, b31);
                if (rows > 1) {
                    int8x16_t a0 = vld1q_s8(x1 + k + half + 0);
                    s10 = dot_product(s10, a0, b00);
                    s11 = dot_product(s11, a0, b01);
                    int8x16_t a1 = vld1q_s8(x1 + k + half + 32);
                    s10 = dot_product(s10, a1, b10);
                    s11 = dot_product(s11, a1, b11);
                    int8x16_t a2 = vld1q_s8(x1 + k + half + 64);
                    s10 = dot_product(s10, a2, b20);
                    s11 = dot_product(s11, a2, b21);
                    int8x16_t a3 = vld1q_s8(x1 + k + half + 96);
                    s10 = dot_product(s10, a3, b30);
                    s11 = dot_product(s11, a3, b31);
                }
                if (rows > 2) {
                    int8x16_t a0 = vld1q_s8(x2 + k + half + 0);
                    s20 = dot_product(s20, a0, b00);
                    s21 = dot_product(s21, a0, b01);
                    int8x16_t a1 = vld1q_s8(x2 + k + half + 32);
                    s20 = dot_product(s20, a1, b10);
                    s21 = dot_product(s21, a1, b11);
                    int8x16_t a2 = vld1q_s8(x2 + k + half + 64);
                    s20 = dot_product(s20, a2, b20);
                    s21 = dot_product(s21, a2, b21);
                    int8x16_t a3 = vld1q_s8(x2 + k + half + 96);
                    s20 = dot_product(s20, a3, b30);
                    s21 = dot_product(s21, a3, b31);
                }
                if (rows > 3) {
                    int8x16_t a0 = vld1q_s8(x3 + k + half + 0);
                    s30 = dot_product(s30, a0, b00);
                    s31 = dot_product(s31, a0, b01);
                    int8x16_t a1 = vld1q_s8(x3 + k + half + 32);
                    s30 = dot_product(s30, a1, b10);
                    s31 = dot_product(s31, a1, b11);
                    int8x16_t a2 = vld1q_s8(x3 + k + half + 64);
                    s30 = dot_product(s30, a2, b20);
                    s31 = dot_product(s31, a2, b21);
                    int8x16_t a3 = vld1q_s8(x3 + k + half + 96);
                    s30 = dot_product(s30, a3, b30);
                    s31 = dot_product(s31, a3, b31);
                }
            }
        }
        if (vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            return TRILLIM_ERROR;
        }
        int32_t total0 = vaddvq_s32(s00);
        for (size_t j = k; j < p->width; ++j) {
            uint8_t packed = w0[(j / 128) * 32 + j % 32];
            int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            total0 += (int32_t)x0[j] * (code - 1);
        }
        store(task, row + 0, column, total0);
        int32_t total1 = vaddvq_s32(s01);
        for (size_t j = k; j < p->width; ++j) {
            uint8_t packed = w1[(j / 128) * 32 + j % 32];
            int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            total1 += (int32_t)x0[j] * (code - 1);
        }
        if (second != column) {
            store(task, row + 0, second, total1);
        }
        if (rows > 1) {
            int32_t total0 = vaddvq_s32(s10);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x1[j] * (code - 1);
            }
            store(task, row + 1, column, total0);
            int32_t total1 = vaddvq_s32(s11);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x1[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 1, second, total1);
            }
        }
        if (rows > 2) {
            int32_t total0 = vaddvq_s32(s20);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x2[j] * (code - 1);
            }
            store(task, row + 2, column, total0);
            int32_t total1 = vaddvq_s32(s21);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x2[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 2, second, total1);
            }
        }
        if (rows > 3) {
            int32_t total0 = vaddvq_s32(s30);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w0[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total0 += (int32_t)x3[j] * (code - 1);
            }
            store(task, row + 3, column, total0);
            int32_t total1 = vaddvq_s32(s31);
            for (size_t j = k; j < p->width; ++j) {
                uint8_t packed = w1[(j / 128) * 32 + j % 32];
                int32_t code = (packed >> (6 - ((j % 128) / 32) * 2)) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                total1 += (int32_t)x3[j] * (code - 1);
            }
            if (second != column) {
                store(task, row + 3, second, total1);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bitlinear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const BitLinearParameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t scale_offset = (p->tokens * p->width + 3) & ~(size_t)3;
    float *scales = (float *)((uint8_t *)scratch + scale_offset);
    QuantizeTask quantization = {inputs[0], scratch, scales, p, context};
    TrillimStatus status;
    if (p->width <= 65536 / p->tokens) {
        status = TRILLIM_OK;
        for (size_t row = 0; row < p->tokens; ++row) {
            status = quantize(&quantization, row, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    } else {
        status = context->thread_pool->run(context->thread_pool->context, p->tokens, quantize, &quantization);
    }
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t columns = p->outputs / workers;
    columns = columns / 2 * 2;
    if (columns < 16) {
        columns = 16;
    }
    size_t stride = (p->width / 128 + (p->width % 128 != 0)) * 32;
    float gamma;
    memcpy(&gamma, (const uint8_t *)inputs[1] + p->outputs * stride, sizeof(gamma));
    Parameters projection = {p->tokens, p->width, p->outputs, TRILLIM_FP32};
    Task task = {outputs[0], scratch, inputs[1], scales, &projection, columns, gamma, context};
    size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    TrillimTask project = (features() & 2) ? project_dotprod : project_neon;
    if (p->width <= 65536 / (p->tokens * p->outputs)) {
        task.columns = p->outputs;
        for (size_t tile = 0; tile < rows; ++tile) {
            TrillimStatus status = project(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_bitlinear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bitlinear as _bind, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return _bind(context)\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (BitLinear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bitlinear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(BitLinearParameters, tokens), NULL},
        {"width", "size_t", offsetof(BitLinearParameters, width), NULL},
        {"outputs", "size_t", offsetof(BitLinearParameters, outputs), NULL}
    };
    static const size_t inputs[] = {4, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(BitLinearParameters), TRILLIM_ALIGNOF(BitLinearParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bitlinear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
