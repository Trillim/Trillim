#include "common.h"
#include <arm_neon.h>
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
    __asm__(".inst 0x4e809400 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}

static inline int32x4_t matrix_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e80a400 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
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

I8MM static inline int32x4_t matrix_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    return vmmlaq_s32(sum, a, b);
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
    int32_t input_sum;
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

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!p->width || p->width > INT32_MAX / 256 || groups > SIZE_MAX / 32 ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        p->outputs > (SIZE_MAX - sizeof(float)) / (groups * 32) ||
        p->tokens > SIZE_MAX / bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * 32 + sizeof(float);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
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
    int16x8_t products = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    products = vmlal_high_s8(products, a, b);
    return vpadalq_s16(sum, products);
}

DOTPROD static inline __attribute__((always_inline)) int32x4_t dot_codes(
    int32x4_t sum, uint8x16_t packed, int8x16_t x0, int8x16_t x1, int8x16_t x2, int8x16_t x3
) {
    const uint8x16_t mask = vdupq_n_u8(3);
    sum = dot_product(sum, x0, vreinterpretq_s8_u8(vshrq_n_u8(packed, 6)));
    sum = dot_product(sum, x1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, 4), mask)));
    sum = dot_product(sum, x2, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, 2), mask)));
    return dot_product(sum, x3, vreinterpretq_s8_u8(vandq_u8(packed, mask)));
}

DOTPROD static TrillimStatus project_decode(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t first = tile * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    size_t stride = p->width / 4;
    const int32x4_t zero = vdupq_n_s32(0);
    for (size_t column = first; column < end; column += 4) {
        const uint8_t *weight = (const uint8_t *)task->weight + column * stride;
        int32x4_t s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        uint8x16_t invalid = vdupq_n_u8(0);
        for (size_t k = 0; k < p->width; k += 128) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            for (size_t half = 0; half < 32; half += 16) {
                const int8_t *x = task->input + k + half;
                const uint8_t *w = weight + k / 4 + half;
                const int8x16_t x0 = vld1q_s8(x);
                const int8x16_t x1 = vld1q_s8(x + 32);
                const int8x16_t x2 = vld1q_s8(x + 64);
                const int8x16_t x3 = vld1q_s8(x + 96);
                const uint8x16_t w0 = vld1q_u8(w);
                const uint8x16_t w1 = vld1q_u8(w + stride);
                const uint8x16_t w2 = vld1q_u8(w + 2 * stride);
                const uint8x16_t w3 = vld1q_u8(w + 3 * stride);
                s0 = dot_codes(s0, w0, x0, x1, x2, x3);
                s1 = dot_codes(s1, w1, x0, x1, x2, x3);
                s2 = dot_codes(s2, w2, x0, x1, x2, x3);
                s3 = dot_codes(s3, w3, x0, x1, x2, x3);
                invalid = vorrq_u8(invalid, vandq_u8(w0, vshrq_n_u8(w0, 1)));
                invalid = vorrq_u8(invalid, vandq_u8(w1, vshrq_n_u8(w1, 1)));
                invalid = vorrq_u8(invalid, vandq_u8(w2, vshrq_n_u8(w2, 1)));
                invalid = vorrq_u8(invalid, vandq_u8(w3, vshrq_n_u8(w3, 1)));
            }
        }
        if (vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            return TRILLIM_ERROR;
        }
        store(task, 0, column, vaddvq_s32(s0) - task->input_sum);
        store(task, 0, column + 1, vaddvq_s32(s1) - task->input_sum);
        store(task, 0, column + 2, vaddvq_s32(s2) - task->input_sum);
        store(task, 0, column + 3, vaddvq_s32(s3) - task->input_sum);
    }
    return TRILLIM_OK;
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

enum { TERNARY_WIDTH = 128, TERNARY_BLOCK_BYTES = 32, DOT_ROWS = 4, MATRIX_ROWS = 8, FAST_COLUMNS = 4 };

static inline __attribute__((always_inline)) uint8x16_t ternary_codes(uint8x16_t packed, size_t plane) {
    switch (plane) {
    case 0: return vshrq_n_u8(packed, 6);
    case 1: return vandq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(3));
    case 2: return vandq_u8(vshrq_n_u8(packed, 2), vdupq_n_u8(3));
    default: return vandq_u8(packed, vdupq_n_u8(3));
    }
}

static inline uint8x16_t invalid_ternary_codes(uint8x16_t invalid, uint8x16_t packed) {
    return vorrq_u8(invalid, vandq_u8(packed, vshrq_n_u8(packed, 1)));
}

static TrillimStatus store_projection(const Task *task, size_t row, size_t rows, size_t column,
                                      size_t columns, size_t start, const uint8_t *const *weights,
                                      const int32_t *row_sums, int32_t sums[][FAST_COLUMNS]) {
    const size_t width = task->p->width;
    if (start < width) {
        for (size_t c = 0; c < columns; ++c) {
            for (size_t k = start; k < width; ++k) {
                const unsigned shift = 6 - (unsigned)(k % TERNARY_WIDTH / 32) * 2;
                const int32_t code = (weights[c][start / 4 + k % 32] >> shift) & 3;
                if (code == 3) {
                    return TRILLIM_ERROR;
                }
                for (size_t r = 0; r < rows; ++r) {
                    sums[r][c] += (int32_t)task->input[(row + r) * width + k] * code;
                }
            }
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < columns; ++c) {
            store(task, row + r, column + c, sums[r][c] - row_sums[r]);
        }
    }
    return TRILLIM_OK;
}

static inline int8x16_t interleave_pair(int8x16_t a, int8x16_t b, size_t part) {
    const int64x2_t first = vreinterpretq_s64_s8(a);
    const int64x2_t second = vreinterpretq_s64_s8(b);
    return vreinterpretq_s8_s64(part == 0 ? vzip1q_s64(first, second) : vzip2q_s64(first, second));
}

I8MM static TrillimStatus project_i8mm(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    const size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0);
    const size_t row = tile / tiles * MATRIX_ROWS;
    const size_t rows = p->tokens - row < MATRIX_ROWS ? p->tokens - row : MATRIX_ROWS;
    size_t column = tile % tiles * task->columns;
    const size_t end = p->outputs - column < task->columns ? p->outputs : column + task->columns;
    const size_t stride = (p->width / TERNARY_WIDTH + (p->width % TERNARY_WIDTH != 0)) * TERNARY_BLOCK_BYTES;
    const int8_t *input = task->input + row * p->width;
    if (rows < MATRIX_ROWS) {
        size_t dot_tile = (tile / tiles * (MATRIX_ROWS / DOT_ROWS)) * tiles + tile % tiles;
        TrillimStatus status = project_dotprod(opaque, dot_tile, worker);
        if (status == TRILLIM_OK && rows > DOT_ROWS) {
            status = project_dotprod(opaque, dot_tile + tiles, worker);
        }
        return status;
    }

    int32_t row_sums[MATRIX_ROWS];
    for (size_t r = 0; r < MATRIX_ROWS; ++r) {
        const int8_t *x = input + r * p->width;
        int32x4_t sum = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= p->width; k += 16) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            sum = vpadalq_s16(sum, vpaddlq_s8(vld1q_s8(x + k)));
        }
        row_sums[r] = vaddvq_s32(sum);
        for (; k < p->width; ++k) {
            row_sums[r] += x[k];
        }
    }
    for (; column < end; column += FAST_COLUMNS) {
        const size_t columns = end - column < FAST_COLUMNS ? end - column : FAST_COLUMNS;
        const uint8_t *weights[FAST_COLUMNS];
        for (size_t c = 0; c < FAST_COLUMNS; ++c) {
            weights[c] = (const uint8_t *)task->weight + (column + (c < columns ? c : 0)) * stride;
        }
        int32x4_t dots[MATRIX_ROWS / 2][FAST_COLUMNS / 2] = {0};
        uint8x16_t invalid = vdupq_n_u8(0);
        size_t k = 0;
        for (; k + TERNARY_WIDTH <= p->width; k += TERNARY_WIDTH) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            for (size_t half = 0; half < TERNARY_BLOCK_BYTES; half += 16) {
                uint8x16_t packed[2][FAST_COLUMNS / 2];
                for (size_t c = 0; c < FAST_COLUMNS / 2; ++c) {
                    int8x16_t a = vld1q_s8((const int8_t *)weights[c * 2] + k / 4 + half);
                    int8x16_t b = vld1q_s8((const int8_t *)weights[c * 2 + 1] + k / 4 + half);
                    invalid = invalid_ternary_codes(invalid, vreinterpretq_u8_s8(a));
                    invalid = invalid_ternary_codes(invalid, vreinterpretq_u8_s8(b));
                    packed[0][c] = vreinterpretq_u8_s8(interleave_pair(a, b, 0));
                    packed[1][c] = vreinterpretq_u8_s8(interleave_pair(a, b, 1));
                }
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
                for (size_t plane = 0; plane < 4; ++plane) {
                    int8x16_t values[2][MATRIX_ROWS / 2];
                    for (size_t r = 0; r < MATRIX_ROWS / 2; ++r) {
                        const int8_t *a = input + r * 2 * p->width + k + half + plane * 32;
                        int8x16_t x0 = vld1q_s8(a);
                        int8x16_t x1 = vld1q_s8(a + p->width);
                        values[0][r] = interleave_pair(x0, x1, 0);
                        values[1][r] = interleave_pair(x0, x1, 1);
                    }
                    for (size_t part = 0; part < 2; ++part) {
                        for (size_t c = 0; c < FAST_COLUMNS / 2; ++c) {
                            int8x16_t codes = vreinterpretq_s8_u8(ternary_codes(packed[part][c], plane));
                            for (size_t r = 0; r < MATRIX_ROWS / 2; ++r) {
                                dots[r][c] = matrix_product(dots[r][c], values[part][r], codes);
                            }
                        }
                    }
                }
            }
        }
        if (vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            return TRILLIM_ERROR;
        }
        int32_t sums[MATRIX_ROWS][FAST_COLUMNS];
        for (size_t r = 0; r < MATRIX_ROWS / 2; ++r) {
            const int64x2_t first = vreinterpretq_s64_s32(dots[r][0]);
            const int64x2_t second = vreinterpretq_s64_s32(dots[r][1]);
            vst1q_s32(sums[r * 2], vreinterpretq_s32_s64(vzip1q_s64(first, second)));
            vst1q_s32(sums[r * 2 + 1], vreinterpretq_s32_s64(vzip2q_s64(first, second)));
        }
        TrillimStatus status = store_projection(task, row, rows, column, columns, k, weights, row_sums, sums);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_quantized_linear(
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
    size_t columns = p->outputs / workers;
    unsigned cpu_features = features();
    bool decode = (cpu_features & 2) && p->tokens == 1 &&
        p->width % 128 == 0 && p->outputs % 4 == 0;
    const bool matrix = (cpu_features & 6) == 6 && p->tokens >= MATRIX_ROWS && p->width >= TERNARY_WIDTH;
    size_t column_step = decode || matrix ? FAST_COLUMNS : 2;
    columns = columns / column_step * column_step;
    if (columns < 16) {
        columns = 16;
    }
    size_t stride = (p->width / 128 + (p->width % 128 != 0)) * 32;
    float gamma;
    memcpy(&gamma, (const uint8_t *)inputs[1] + p->outputs * stride, sizeof(gamma));
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, columns, gamma, 0, context};
    const size_t row_tile = matrix ? MATRIX_ROWS : DOT_ROWS;
    size_t rows = p->tokens / row_tile + (p->tokens % row_tile != 0);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    TrillimTask project = matrix ? project_i8mm : (cpu_features & 2) ? project_dotprod : project_neon;
    if (decode) {
        // The packed codes are weight + 1. Correct every dot with the same input sum.
        int32x4_t sum = vdupq_n_s32(0);
        for (size_t k = 0; k < p->width; k += 16) {
            if ((k & 4095) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            sum = vpadalq_s16(sum, vpaddlq_s8(vld1q_s8(task.input + k)));
        }
        task.input_sum = vaddvq_s32(sum);
        project = project_decode;
    }
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

TRILLIM_KERNEL const char *trillim_neon_quantized_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantized_linear as _bind, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return _bind(context)\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_quantized_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters),
        4, fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_quantized_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
