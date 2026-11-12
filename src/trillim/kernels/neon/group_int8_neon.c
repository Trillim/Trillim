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
    if (!trillim_float(p->output_dtype) || !p->width || p->width % 32 || p->width / 32 > SIZE_MAX / 34 ||
        p->outputs > SIZE_MAX / (p->width / 32 * 34) || p->tokens > SIZE_MAX / p->width ||
        p->tokens > SIZE_MAX / sizeof(float) || (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs))
        return TRILLIM_ERROR;
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * (p->width / 32 * 34);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static float weight_scale(const Task *task, size_t column, size_t k) {
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)task->weight + task->p->outputs * task->p->width +
        (column * (task->p->width / 32) + k / 32) * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline __attribute__((always_inline)) void store(const Task *task, size_t row, size_t column, float total) {
    TrillimValue value = {.f = total * (1.0f / task->scales[row])};
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
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = task->weight + second * p->width;
        float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
        float s20 = 0, s21 = 0, s30 = 0, s31 = 0;
        for (size_t k = 0; k < p->width; k += 32) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t b00 = vld1q_s8(w0 + k), b01 = vld1q_s8(w0 + k + 16);
            int8x16_t b10 = vld1q_s8(w1 + k), b11 = vld1q_s8(w1 + k + 16);
            float scale0 = weight_scale(task, column, k);
            float scale1 = weight_scale(task, second, k);
            int8x16_t a0 = vld1q_s8(x0 + k), a1 = vld1q_s8(x0 + k + 16);
            int32x4_t t0 = multiply(vdupq_n_s32(0), a0, b00);
            t0 = multiply(t0, a1, b01);
            s00 += (float)vaddvq_s32(t0) * scale0;
            int32x4_t t1 = multiply(vdupq_n_s32(0), a0, b10);
            t1 = multiply(t1, a1, b11);
            s01 += (float)vaddvq_s32(t1) * scale1;
            if (rows > 1) {
                int8x16_t a0 = vld1q_s8(x1 + k), a1 = vld1q_s8(x1 + k + 16);
                int32x4_t t0 = multiply(vdupq_n_s32(0), a0, b00);
                t0 = multiply(t0, a1, b01);
                s10 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = multiply(vdupq_n_s32(0), a0, b10);
                t1 = multiply(t1, a1, b11);
                s11 += (float)vaddvq_s32(t1) * scale1;
            }
            if (rows > 2) {
                int8x16_t a0 = vld1q_s8(x2 + k), a1 = vld1q_s8(x2 + k + 16);
                int32x4_t t0 = multiply(vdupq_n_s32(0), a0, b00);
                t0 = multiply(t0, a1, b01);
                s20 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = multiply(vdupq_n_s32(0), a0, b10);
                t1 = multiply(t1, a1, b11);
                s21 += (float)vaddvq_s32(t1) * scale1;
            }
            if (rows > 3) {
                int8x16_t a0 = vld1q_s8(x3 + k), a1 = vld1q_s8(x3 + k + 16);
                int32x4_t t0 = multiply(vdupq_n_s32(0), a0, b00);
                t0 = multiply(t0, a1, b01);
                s30 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = multiply(vdupq_n_s32(0), a0, b10);
                t1 = multiply(t1, a1, b11);
                s31 += (float)vaddvq_s32(t1) * scale1;
            }
        }
        store(task, row + 0, column, s00);
        if (second != column) {
            store(task, row + 0, second, s01);
        }
        if (rows > 1) {
            store(task, row + 1, column, s10);
            if (second != column) {
                store(task, row + 1, second, s11);
            }
        }
        if (rows > 2) {
            store(task, row + 2, column, s20);
            if (second != column) {
                store(task, row + 2, second, s21);
            }
        }
        if (rows > 3) {
            store(task, row + 3, column, s30);
            if (second != column) {
                store(task, row + 3, second, s31);
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
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t second = column + 1 < end ? column + 1 : column;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = task->weight + second * p->width;
        float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
        float s20 = 0, s21 = 0, s30 = 0, s31 = 0;
        for (size_t k = 0; k < p->width; k += 32) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t b00 = vld1q_s8(w0 + k), b01 = vld1q_s8(w0 + k + 16);
            int8x16_t b10 = vld1q_s8(w1 + k), b11 = vld1q_s8(w1 + k + 16);
            float scale0 = weight_scale(task, column, k);
            float scale1 = weight_scale(task, second, k);
            int8x16_t a0 = vld1q_s8(x0 + k), a1 = vld1q_s8(x0 + k + 16);
            int32x4_t t0 = dot_product(vdupq_n_s32(0), a0, b00);
            t0 = dot_product(t0, a1, b01);
            s00 += (float)vaddvq_s32(t0) * scale0;
            int32x4_t t1 = dot_product(vdupq_n_s32(0), a0, b10);
            t1 = dot_product(t1, a1, b11);
            s01 += (float)vaddvq_s32(t1) * scale1;
            if (rows > 1) {
                int8x16_t a0 = vld1q_s8(x1 + k), a1 = vld1q_s8(x1 + k + 16);
                int32x4_t t0 = dot_product(vdupq_n_s32(0), a0, b00);
                t0 = dot_product(t0, a1, b01);
                s10 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = dot_product(vdupq_n_s32(0), a0, b10);
                t1 = dot_product(t1, a1, b11);
                s11 += (float)vaddvq_s32(t1) * scale1;
            }
            if (rows > 2) {
                int8x16_t a0 = vld1q_s8(x2 + k), a1 = vld1q_s8(x2 + k + 16);
                int32x4_t t0 = dot_product(vdupq_n_s32(0), a0, b00);
                t0 = dot_product(t0, a1, b01);
                s20 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = dot_product(vdupq_n_s32(0), a0, b10);
                t1 = dot_product(t1, a1, b11);
                s21 += (float)vaddvq_s32(t1) * scale1;
            }
            if (rows > 3) {
                int8x16_t a0 = vld1q_s8(x3 + k), a1 = vld1q_s8(x3 + k + 16);
                int32x4_t t0 = dot_product(vdupq_n_s32(0), a0, b00);
                t0 = dot_product(t0, a1, b01);
                s30 += (float)vaddvq_s32(t0) * scale0;
                int32x4_t t1 = dot_product(vdupq_n_s32(0), a0, b10);
                t1 = dot_product(t1, a1, b11);
                s31 += (float)vaddvq_s32(t1) * scale1;
            }
        }
        store(task, row + 0, column, s00);
        if (second != column) {
            store(task, row + 0, second, s01);
        }
        if (rows > 1) {
            store(task, row + 1, column, s10);
            if (second != column) {
                store(task, row + 1, second, s11);
            }
        }
        if (rows > 2) {
            store(task, row + 2, column, s20);
            if (second != column) {
                store(task, row + 2, second, s21);
            }
        }
        if (rows > 3) {
            store(task, row + 3, column, s30);
            if (second != column) {
                store(task, row + 3, second, s31);
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
        size_t c1 = count > 1 ? column + 1 : column;
        size_t c2 = count > 2 ? column + 2 : column;
        size_t c3 = count > 3 ? column + 3 : column;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = task->weight + c1 * p->width;
        const int8_t *w2 = task->weight + c2 * p->width;
        const int8_t *w3 = task->weight + c3 * p->width;
        float32x4_t s00 = vdupq_n_f32(0), s01 = s00, s10 = s00, s11 = s00;
        for (size_t k = 0; k < p->width; k += 32) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int32x4_t t00 = vdupq_n_s32(0), t01 = t00, t10 = t00, t11 = t00;
            for (size_t j = 0; j < 32; j += 8) {
                int8x16_t a0 = vcombine_s8(vld1_s8(x0 + k + j), vld1_s8(x1 + k + j));
                int8x16_t a1 = vcombine_s8(vld1_s8(x2 + k + j), vld1_s8(x3 + k + j));
                int8x16_t b0 = vcombine_s8(vld1_s8(w0 + k + j), vld1_s8(w1 + k + j));
                int8x16_t b1 = vcombine_s8(vld1_s8(w2 + k + j), vld1_s8(w3 + k + j));
                t00 = matrix_product(t00, a0, b0);
                t01 = matrix_product(t01, a0, b1);
                t10 = matrix_product(t10, a1, b0);
                t11 = matrix_product(t11, a1, b1);
            }
            float scale0 = weight_scale(task, column, k), scale1 = weight_scale(task, c1, k);
            float scale2 = weight_scale(task, c2, k), scale3 = weight_scale(task, c3, k);
            float32x2_t scale01 = vdup_n_f32(scale0), scale23 = vdup_n_f32(scale2);
            scale01 = vset_lane_f32(scale1, scale01, 1);
            scale23 = vset_lane_f32(scale3, scale23, 1);
            float32x4_t a = vcombine_f32(scale01, scale01), b = vcombine_f32(scale23, scale23);
            s00 = vfmaq_f32(s00, vcvtq_f32_s32(t00), a);
            s01 = vfmaq_f32(s01, vcvtq_f32_s32(t01), b);
            s10 = vfmaq_f32(s10, vcvtq_f32_s32(t10), a);
            s11 = vfmaq_f32(s11, vcvtq_f32_s32(t11), b);
        }
        if (count > 0) {
            store(task, row + 0, column + 0, vgetq_lane_f32(s00, 0));
        }
        if (count > 1) {
            store(task, row + 0, column + 1, vgetq_lane_f32(s00, 1));
        }
        if (count > 2) {
            store(task, row + 0, column + 2, vgetq_lane_f32(s01, 0));
        }
        if (count > 3) {
            store(task, row + 0, column + 3, vgetq_lane_f32(s01, 1));
        }
        if (rows > 1) {
            if (count > 0) {
                store(task, row + 1, column + 0, vgetq_lane_f32(s00, 2));
            }
            if (count > 1) {
                store(task, row + 1, column + 1, vgetq_lane_f32(s00, 3));
            }
            if (count > 2) {
                store(task, row + 1, column + 2, vgetq_lane_f32(s01, 2));
            }
            if (count > 3) {
                store(task, row + 1, column + 3, vgetq_lane_f32(s01, 3));
            }
        }
        if (rows > 2) {
            if (count > 0) {
                store(task, row + 2, column + 0, vgetq_lane_f32(s10, 0));
            }
            if (count > 1) {
                store(task, row + 2, column + 1, vgetq_lane_f32(s10, 1));
            }
            if (count > 2) {
                store(task, row + 2, column + 2, vgetq_lane_f32(s11, 0));
            }
            if (count > 3) {
                store(task, row + 2, column + 3, vgetq_lane_f32(s11, 1));
            }
        }
        if (rows > 3) {
            if (count > 0) {
                store(task, row + 3, column + 0, vgetq_lane_f32(s10, 2));
            }
            if (count > 1) {
                store(task, row + 3, column + 1, vgetq_lane_f32(s10, 3));
            }
            if (count > 2) {
                store(task, row + 3, column + 2, vgetq_lane_f32(s11, 2));
            }
            if (count > 3) {
                store(task, row + 3, column + 3, vgetq_lane_f32(s11, 3));
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_group_int8(
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
    size_t task_factor = p->tokens >= 4 ? 1 : 4;
    size_t columns = p->outputs / workers / task_factor;
    columns = columns / 4 * 4;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, columns, context};
    size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    unsigned available = features();
    TrillimTask project = project_neon;
    if ((available & 4) && p->tokens >= 4) {
        project = project_i8mm;
    } else if (available & 2) {
        project = project_dotprod;
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

TRILLIM_KERNEL const char *trillim_neon_group_int8_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _group_int8_linear as _bind, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    source = context.inputs['x']\n"
           "    width = source.shape[-1].value if source.shape else None\n"
           "    if width is None or width <= 0 or width % 32:\n"
           "        raise ValueError('group INT8 projection requires a fixed positive width divisible by 32')\n"
           "    return _bind(context)\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_group_int8_parameters(void) {
    static const TrillimParameter fields[] = {{"tokens", "size_t", offsetof(Parameters, tokens), NULL},
                                              {"width", "size_t", offsetof(Parameters, width), NULL},
                                              {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
                                              {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}};
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1, requirements, NULL, inputs, outputs, 1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_group_int8_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
