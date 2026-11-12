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
    __asm__ volatile(".inst 0x4e809400 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline int32x4_t matrix_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__ volatile(".inst 0x4e80a400 | trillim_asm_%0 | (trillim_asm_%1 << 5) | (trillim_asm_%2 << 16)"
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
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype) || !p->width || p->width > INT32_MAX / (128 * 128) ||
        p->width > SIZE_MAX - 2 || p->outputs > SIZE_MAX / (p->width + 2) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * (p->width + 2);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) void store(const Task *task, size_t row, size_t column, int32_t sum) {
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)task->weight + task->p->outputs * task->p->width + column * 2, sizeof(bits));
    TrillimValue value = {.f = (float)sum * (trillim_bf16(bits) * (1.0f / task->scales[row]))};
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
    const int8_t *x0 = task->input + row * p->width;
    const int8_t *x1 = rows > 1 ? x0 + p->width : x0;
    const int8_t *x2 = rows > 2 ? x1 + p->width : x0;
    const int8_t *x3 = rows > 3 ? x2 + p->width : x0;
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = count > 1 ? w0 + p->width : w0;
        const int8_t *w2 = count > 2 ? w1 + p->width : w0;
        const int8_t *w3 = count > 3 ? w2 + p->width : w0;
        int32x4_t s00 = vdupq_n_s32(0), s01 = vdupq_n_s32(0), s02 = vdupq_n_s32(0), s03 = vdupq_n_s32(0);
        int32x4_t s10 = vdupq_n_s32(0), s11 = vdupq_n_s32(0), s12 = vdupq_n_s32(0), s13 = vdupq_n_s32(0);
        int32x4_t s20 = vdupq_n_s32(0), s21 = vdupq_n_s32(0), s22 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
        int32x4_t s30 = vdupq_n_s32(0), s31 = vdupq_n_s32(0), s32 = vdupq_n_s32(0), s33 = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= p->width; k += 16) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t b0 = vld1q_s8(w0 + k);
            int8x16_t b1 = vld1q_s8(w1 + k);
            int8x16_t b2 = vld1q_s8(w2 + k);
            int8x16_t b3 = vld1q_s8(w3 + k);
            int8x16_t a0 = vld1q_s8(x0 + k);
            s00 = multiply(s00, a0, b0);
            s01 = multiply(s01, a0, b1);
            s02 = multiply(s02, a0, b2);
            s03 = multiply(s03, a0, b3);
            if (rows > 1) {
                int8x16_t a1 = vld1q_s8(x1 + k);
                s10 = multiply(s10, a1, b0);
                s11 = multiply(s11, a1, b1);
                s12 = multiply(s12, a1, b2);
                s13 = multiply(s13, a1, b3);
            }
            if (rows > 2) {
                int8x16_t a2 = vld1q_s8(x2 + k);
                s20 = multiply(s20, a2, b0);
                s21 = multiply(s21, a2, b1);
                s22 = multiply(s22, a2, b2);
                s23 = multiply(s23, a2, b3);
            }
            if (rows > 3) {
                int8x16_t a3 = vld1q_s8(x3 + k);
                s30 = multiply(s30, a3, b0);
                s31 = multiply(s31, a3, b1);
                s32 = multiply(s32, a3, b2);
                s33 = multiply(s33, a3, b3);
            }
        }
        if (count > 0) {
            int32_t sum = vaddvq_s32(s00);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w0[j];
            }
            store(task, row + 0, column + 0, sum);
        }
        if (count > 1) {
            int32_t sum = vaddvq_s32(s01);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w1[j];
            }
            store(task, row + 0, column + 1, sum);
        }
        if (count > 2) {
            int32_t sum = vaddvq_s32(s02);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w2[j];
            }
            store(task, row + 0, column + 2, sum);
        }
        if (count > 3) {
            int32_t sum = vaddvq_s32(s03);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w3[j];
            }
            store(task, row + 0, column + 3, sum);
        }
        if (rows > 1) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s10);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w0[j];
                }
                store(task, row + 1, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s11);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w1[j];
                }
                store(task, row + 1, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s12);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w2[j];
                }
                store(task, row + 1, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s13);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w3[j];
                }
                store(task, row + 1, column + 3, sum);
            }
        }
        if (rows > 2) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s20);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w0[j];
                }
                store(task, row + 2, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s21);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w1[j];
                }
                store(task, row + 2, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s22);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w2[j];
                }
                store(task, row + 2, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s23);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w3[j];
                }
                store(task, row + 2, column + 3, sum);
            }
        }
        if (rows > 3) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s30);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w0[j];
                }
                store(task, row + 3, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s31);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w1[j];
                }
                store(task, row + 3, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s32);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w2[j];
                }
                store(task, row + 3, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s33);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w3[j];
                }
                store(task, row + 3, column + 3, sum);
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
    const int8_t *x0 = task->input + row * p->width;
    const int8_t *x1 = rows > 1 ? x0 + p->width : x0;
    const int8_t *x2 = rows > 2 ? x1 + p->width : x0;
    const int8_t *x3 = rows > 3 ? x2 + p->width : x0;
    for (size_t column = first; column < end; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 4 ? end - column : 4;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = count > 1 ? w0 + p->width : w0;
        const int8_t *w2 = count > 2 ? w1 + p->width : w0;
        const int8_t *w3 = count > 3 ? w2 + p->width : w0;
        int32x4_t s00 = vdupq_n_s32(0), s01 = vdupq_n_s32(0), s02 = vdupq_n_s32(0), s03 = vdupq_n_s32(0);
        int32x4_t s10 = vdupq_n_s32(0), s11 = vdupq_n_s32(0), s12 = vdupq_n_s32(0), s13 = vdupq_n_s32(0);
        int32x4_t s20 = vdupq_n_s32(0), s21 = vdupq_n_s32(0), s22 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
        int32x4_t s30 = vdupq_n_s32(0), s31 = vdupq_n_s32(0), s32 = vdupq_n_s32(0), s33 = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= p->width; k += 16) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t b0 = vld1q_s8(w0 + k);
            int8x16_t b1 = vld1q_s8(w1 + k);
            int8x16_t b2 = vld1q_s8(w2 + k);
            int8x16_t b3 = vld1q_s8(w3 + k);
            int8x16_t a0 = vld1q_s8(x0 + k);
            s00 = dot_product(s00, a0, b0);
            s01 = dot_product(s01, a0, b1);
            s02 = dot_product(s02, a0, b2);
            s03 = dot_product(s03, a0, b3);
            if (rows > 1) {
                int8x16_t a1 = vld1q_s8(x1 + k);
                s10 = dot_product(s10, a1, b0);
                s11 = dot_product(s11, a1, b1);
                s12 = dot_product(s12, a1, b2);
                s13 = dot_product(s13, a1, b3);
            }
            if (rows > 2) {
                int8x16_t a2 = vld1q_s8(x2 + k);
                s20 = dot_product(s20, a2, b0);
                s21 = dot_product(s21, a2, b1);
                s22 = dot_product(s22, a2, b2);
                s23 = dot_product(s23, a2, b3);
            }
            if (rows > 3) {
                int8x16_t a3 = vld1q_s8(x3 + k);
                s30 = dot_product(s30, a3, b0);
                s31 = dot_product(s31, a3, b1);
                s32 = dot_product(s32, a3, b2);
                s33 = dot_product(s33, a3, b3);
            }
        }
        if (count > 0) {
            int32_t sum = vaddvq_s32(s00);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w0[j];
            }
            store(task, row + 0, column + 0, sum);
        }
        if (count > 1) {
            int32_t sum = vaddvq_s32(s01);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w1[j];
            }
            store(task, row + 0, column + 1, sum);
        }
        if (count > 2) {
            int32_t sum = vaddvq_s32(s02);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w2[j];
            }
            store(task, row + 0, column + 2, sum);
        }
        if (count > 3) {
            int32_t sum = vaddvq_s32(s03);
            for (size_t j = k; j < p->width; ++j) {
                sum += (int32_t)x0[j] * w3[j];
            }
            store(task, row + 0, column + 3, sum);
        }
        if (rows > 1) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s10);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w0[j];
                }
                store(task, row + 1, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s11);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w1[j];
                }
                store(task, row + 1, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s12);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w2[j];
                }
                store(task, row + 1, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s13);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x1[j] * w3[j];
                }
                store(task, row + 1, column + 3, sum);
            }
        }
        if (rows > 2) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s20);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w0[j];
                }
                store(task, row + 2, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s21);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w1[j];
                }
                store(task, row + 2, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s22);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w2[j];
                }
                store(task, row + 2, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s23);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x2[j] * w3[j];
                }
                store(task, row + 2, column + 3, sum);
            }
        }
        if (rows > 3) {
            if (count > 0) {
                int32_t sum = vaddvq_s32(s30);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w0[j];
                }
                store(task, row + 3, column + 0, sum);
            }
            if (count > 1) {
                int32_t sum = vaddvq_s32(s31);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w1[j];
                }
                store(task, row + 3, column + 1, sum);
            }
            if (count > 2) {
                int32_t sum = vaddvq_s32(s32);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w2[j];
                }
                store(task, row + 3, column + 2, sum);
            }
            if (count > 3) {
                int32_t sum = vaddvq_s32(s33);
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)x3[j] * w3[j];
                }
                store(task, row + 3, column + 3, sum);
            }
        }
    }
    return TRILLIM_OK;
}

I8MM static TrillimStatus project_i8mm(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = (p->outputs + task->columns - 1) / task->columns;
    size_t row = tile / tiles * 8;
    size_t rows = p->tokens - row < 8 ? p->tokens - row : 8;
    size_t first = tile % tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    const int8_t *x0 = task->input + (row + (rows > 0 ? 0 : 0)) * p->width;
    const int8_t *x1 = task->input + (row + (rows > 1 ? 1 : 0)) * p->width;
    const int8_t *x2 = task->input + (row + (rows > 2 ? 2 : 0)) * p->width;
    const int8_t *x3 = task->input + (row + (rows > 3 ? 3 : 0)) * p->width;
    const int8_t *x4 = task->input + (row + (rows > 4 ? 4 : 0)) * p->width;
    const int8_t *x5 = task->input + (row + (rows > 5 ? 5 : 0)) * p->width;
    const int8_t *x6 = task->input + (row + (rows > 6 ? 6 : 0)) * p->width;
    const int8_t *x7 = task->input + (row + (rows > 7 ? 7 : 0)) * p->width;
    for (size_t column = first; column < end; column += 8) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - column < 8 ? end - column : 8;
        const int8_t *w0 = task->weight + (column + (count > 0 ? 0 : 0)) * p->width;
        const int8_t *w1 = task->weight + (column + (count > 1 ? 1 : 0)) * p->width;
        const int8_t *w2 = task->weight + (column + (count > 2 ? 2 : 0)) * p->width;
        const int8_t *w3 = task->weight + (column + (count > 3 ? 3 : 0)) * p->width;
        const int8_t *w4 = task->weight + (column + (count > 4 ? 4 : 0)) * p->width;
        const int8_t *w5 = task->weight + (column + (count > 5 ? 5 : 0)) * p->width;
        const int8_t *w6 = task->weight + (column + (count > 6 ? 6 : 0)) * p->width;
        const int8_t *w7 = task->weight + (column + (count > 7 ? 7 : 0)) * p->width;
        int32x4_t s00 = vdupq_n_s32(0), s01 = vdupq_n_s32(0), s02 = vdupq_n_s32(0), s03 = vdupq_n_s32(0);
        int32x4_t s10 = vdupq_n_s32(0), s11 = vdupq_n_s32(0), s12 = vdupq_n_s32(0), s13 = vdupq_n_s32(0);
        int32x4_t s20 = vdupq_n_s32(0), s21 = vdupq_n_s32(0), s22 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
        int32x4_t s30 = vdupq_n_s32(0), s31 = vdupq_n_s32(0), s32 = vdupq_n_s32(0), s33 = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 8 <= p->width; k += 8) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t a0 = vcombine_s8(vld1_s8(x0 + k), vld1_s8(x1 + k));
            int8x16_t a1 = vcombine_s8(vld1_s8(x2 + k), vld1_s8(x3 + k));
            int8x16_t a2 = vcombine_s8(vld1_s8(x4 + k), vld1_s8(x5 + k));
            int8x16_t a3 = vcombine_s8(vld1_s8(x6 + k), vld1_s8(x7 + k));
            int8x16_t b0 = vcombine_s8(vld1_s8(w0 + k), vld1_s8(w1 + k));
            s00 = matrix_product(s00, a0, b0);
            s10 = matrix_product(s10, a1, b0);
            s20 = matrix_product(s20, a2, b0);
            s30 = matrix_product(s30, a3, b0);
            int8x16_t b1 = vcombine_s8(vld1_s8(w2 + k), vld1_s8(w3 + k));
            s01 = matrix_product(s01, a0, b1);
            s11 = matrix_product(s11, a1, b1);
            s21 = matrix_product(s21, a2, b1);
            s31 = matrix_product(s31, a3, b1);
            int8x16_t b2 = vcombine_s8(vld1_s8(w4 + k), vld1_s8(w5 + k));
            s02 = matrix_product(s02, a0, b2);
            s12 = matrix_product(s12, a1, b2);
            s22 = matrix_product(s22, a2, b2);
            s32 = matrix_product(s32, a3, b2);
            int8x16_t b3 = vcombine_s8(vld1_s8(w6 + k), vld1_s8(w7 + k));
            s03 = matrix_product(s03, a0, b3);
            s13 = matrix_product(s13, a1, b3);
            s23 = matrix_product(s23, a2, b3);
            s33 = matrix_product(s33, a3, b3);
        }
        int32_t sums[8][8];
        sums[0][0] = vgetq_lane_s32(s00, 0);
        sums[0][1] = vgetq_lane_s32(s00, 1);
        sums[0][2] = vgetq_lane_s32(s01, 0);
        sums[0][3] = vgetq_lane_s32(s01, 1);
        sums[0][4] = vgetq_lane_s32(s02, 0);
        sums[0][5] = vgetq_lane_s32(s02, 1);
        sums[0][6] = vgetq_lane_s32(s03, 0);
        sums[0][7] = vgetq_lane_s32(s03, 1);
        sums[1][0] = vgetq_lane_s32(s00, 2);
        sums[1][1] = vgetq_lane_s32(s00, 3);
        sums[1][2] = vgetq_lane_s32(s01, 2);
        sums[1][3] = vgetq_lane_s32(s01, 3);
        sums[1][4] = vgetq_lane_s32(s02, 2);
        sums[1][5] = vgetq_lane_s32(s02, 3);
        sums[1][6] = vgetq_lane_s32(s03, 2);
        sums[1][7] = vgetq_lane_s32(s03, 3);
        sums[2][0] = vgetq_lane_s32(s10, 0);
        sums[2][1] = vgetq_lane_s32(s10, 1);
        sums[2][2] = vgetq_lane_s32(s11, 0);
        sums[2][3] = vgetq_lane_s32(s11, 1);
        sums[2][4] = vgetq_lane_s32(s12, 0);
        sums[2][5] = vgetq_lane_s32(s12, 1);
        sums[2][6] = vgetq_lane_s32(s13, 0);
        sums[2][7] = vgetq_lane_s32(s13, 1);
        sums[3][0] = vgetq_lane_s32(s10, 2);
        sums[3][1] = vgetq_lane_s32(s10, 3);
        sums[3][2] = vgetq_lane_s32(s11, 2);
        sums[3][3] = vgetq_lane_s32(s11, 3);
        sums[3][4] = vgetq_lane_s32(s12, 2);
        sums[3][5] = vgetq_lane_s32(s12, 3);
        sums[3][6] = vgetq_lane_s32(s13, 2);
        sums[3][7] = vgetq_lane_s32(s13, 3);
        sums[4][0] = vgetq_lane_s32(s20, 0);
        sums[4][1] = vgetq_lane_s32(s20, 1);
        sums[4][2] = vgetq_lane_s32(s21, 0);
        sums[4][3] = vgetq_lane_s32(s21, 1);
        sums[4][4] = vgetq_lane_s32(s22, 0);
        sums[4][5] = vgetq_lane_s32(s22, 1);
        sums[4][6] = vgetq_lane_s32(s23, 0);
        sums[4][7] = vgetq_lane_s32(s23, 1);
        sums[5][0] = vgetq_lane_s32(s20, 2);
        sums[5][1] = vgetq_lane_s32(s20, 3);
        sums[5][2] = vgetq_lane_s32(s21, 2);
        sums[5][3] = vgetq_lane_s32(s21, 3);
        sums[5][4] = vgetq_lane_s32(s22, 2);
        sums[5][5] = vgetq_lane_s32(s22, 3);
        sums[5][6] = vgetq_lane_s32(s23, 2);
        sums[5][7] = vgetq_lane_s32(s23, 3);
        sums[6][0] = vgetq_lane_s32(s30, 0);
        sums[6][1] = vgetq_lane_s32(s30, 1);
        sums[6][2] = vgetq_lane_s32(s31, 0);
        sums[6][3] = vgetq_lane_s32(s31, 1);
        sums[6][4] = vgetq_lane_s32(s32, 0);
        sums[6][5] = vgetq_lane_s32(s32, 1);
        sums[6][6] = vgetq_lane_s32(s33, 0);
        sums[6][7] = vgetq_lane_s32(s33, 1);
        sums[7][0] = vgetq_lane_s32(s30, 2);
        sums[7][1] = vgetq_lane_s32(s30, 3);
        sums[7][2] = vgetq_lane_s32(s31, 2);
        sums[7][3] = vgetq_lane_s32(s31, 3);
        sums[7][4] = vgetq_lane_s32(s32, 2);
        sums[7][5] = vgetq_lane_s32(s32, 3);
        sums[7][6] = vgetq_lane_s32(s33, 2);
        sums[7][7] = vgetq_lane_s32(s33, 3);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < count; ++c) {
                int32_t sum = sums[r][c];
                for (size_t j = k; j < p->width; ++j) {
                    sum += (int32_t)task->input[(row + r) * p->width + j] *
                        task->weight[(column + c) * p->width + j];
                }
                store(task, row + r, column + c, sum);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_row_int8(
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
    size_t task_factor = p->tokens >= 8 ? 1 : 2;
    size_t columns = p->outputs / workers / task_factor;
    columns = columns / 8 * 8;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, columns, context};
    size_t row_tile = 4;
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    unsigned available = features();
    TrillimTask project = project_neon;
    if ((available & 4) && p->tokens >= 8) {
        project = project_i8mm;
        row_tile = 8;
    } else if (available & 2) {
        project = project_dotprod;
    }
    size_t rows = p->tokens / row_tile + (p->tokens % row_tile != 0);
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

TRILLIM_KERNEL const char *trillim_neon_row_int8_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _row_int8_linear as _bind, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return _bind(context)\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_row_int8_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_row_int8_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
