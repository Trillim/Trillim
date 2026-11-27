#include "common.h"
#include <arm_neon.h>
#include <stdatomic.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Apple Clang 17 drops operands for optional ARM instructions in assembly output. */
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_binary_v\\reg,\\reg\n"
    ".endr\n"
);
#define DOTPROD
#define I8MM
static inline int32x4_t dot_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e809400 | trillim_binary_%0 | (trillim_binary_%1 << 5) | (trillim_binary_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline int32x4_t matrix_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e80a400 | trillim_binary_%0 | (trillim_binary_%1 << 5) | (trillim_binary_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
/* The builder reassembles compiler output with baseline flags. */
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

enum { WEIGHT_GROUP = 128, ACTIVATION_GROUP = 32, ROWS = 8, COLUMNS = 4 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} Parameters;

typedef struct {
    union {
        int8x16_t rows[ROWS][ACTIVATION_GROUP / 16];
        int8x16_t pairs[ACTIVATION_GROUP / 8][ROWS / 2];
    } codes;
    union {
        float rows[ROWS];
        float32x4_t pairs[ROWS / 2];
    } scales;
} Activation;

typedef struct {
    const Parameters *p;
    const TrillimKernelContext *context;
    const void *input;
    const uint8_t *weight, *weight_scales;
    void *output;
    Activation *activation;
    size_t groups, activation_groups, columns, column_tiles;
    bool pairwise;
    atomic_bool ordinary;
} Task;

static size_t ceil_div(size_t n, size_t d) {
    return n / d + (n % d != 0);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->weight_dtype != TRILLIM_GRP_BINARY || !trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t in_size = trillim_itemsize(p->input_dtype);
    size_t out_size = trillim_itemsize(p->output_dtype);
    size_t groups = ceil_div(p->width, ACTIVATION_GROUP);
    if (p->tokens > SIZE_MAX / p->outputs / out_size ||
        (p->width && (p->tokens > SIZE_MAX / p->width / in_size ||
                      ceil_div(p->tokens, ROWS) > SIZE_MAX / groups / sizeof(Activation))) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * in_size;
    outputs[0] = p->tokens * p->outputs * out_size;
    *scratch = ceil_div(p->tokens, ROWS) * groups * sizeof(Activation);
    return TRILLIM_OK;
}

static bool available(const char *feature, unsigned long mask, int capability) {
#if defined(__APPLE__)
    (void)mask;
    (void)capability;
    int result = 0;
    size_t size = sizeof(result);
    return sysctlbyname(feature, &result, &size, NULL, 0) == 0 && result;
#elif defined(__linux__)
    (void)feature;
    return (getauxval((unsigned long)capability) & mask) != 0;
#else
    (void)feature;
    (void)mask;
    (void)capability;
    return false;
#endif
}

static bool has_dotprod(void) {
    static atomic_int cached;
    int supported = atomic_load_explicit(&cached, memory_order_relaxed);
    if (!supported) {
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
        supported = available("", HWCAP_ASIMDDP, AT_HWCAP) ? 1 : -1;
#else
        supported = available("hw.optional.arm.FEAT_DotProd", 0, 0) ? 1 : -1;
#endif
        atomic_store_explicit(&cached, supported, memory_order_relaxed);
    }
    return supported > 0;
}

static bool has_i8mm(void) {
    static atomic_int cached;
    int supported = atomic_load_explicit(&cached, memory_order_relaxed);
    if (!supported) {
#if defined(__linux__) && defined(HWCAP2_I8MM)
        supported = available("", HWCAP2_I8MM, AT_HWCAP2) ? 1 : -1;
#else
        supported = available("hw.optional.arm.FEAT_I8MM", 0, 0) ? 1 : -1;
#endif
        atomic_store_explicit(&cached, supported, memory_order_relaxed);
    }
    return supported > 0;
}

static inline float32x4_t load4(const void *input, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)input + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)input + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline int8x16_t quantize16(const float32x4_t *values, float inverse) {
    int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(values[0], inverse));
    int32x4_t b = vcvtnq_s32_f32(vmulq_n_f32(values[1], inverse));
    int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(values[2], inverse));
    int32x4_t d = vcvtnq_s32_f32(vmulq_n_f32(values[3], inverse));
    int16x8_t low = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
    int16x8_t high = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
    return vcombine_s8(vmovn_s16(low), vmovn_s16(high));
}

static inline bool quantize_group(
    const Task *task, size_t row, size_t offset, int8x16_t *codes, float *scale
) {
    const Parameters *p = task->p;
    size_t first = row * p->width + offset;
    float32x4_t values[8];
    if (p->width - offset >= ACTIVATION_GROUP) {
        for (size_t i = 0; i < 8; ++i) {
            values[i] = load4(task->input, first + 4 * i, p->input_dtype);
        }
    } else {
        float tail[ACTIVATION_GROUP] = {0};
        for (size_t i = 0; i < p->width - offset; ++i) {
            tail[i] = trillim_load(task->input, first + i, p->input_dtype).f;
        }
        for (size_t i = 0; i < 8; ++i) {
            values[i] = vld1q_f32(tail + 4 * i);
        }
    }
    float32x4_t maximum = vabsq_f32(values[0]);
    for (size_t i = 1; i < 8; ++i) {
        maximum = vmaxq_f32(maximum, vabsq_f32(values[i]));
    }
    float magnitude = vmaxvq_f32(maximum);
    // Q8_0 rounds codes with the FP32 step and accumulates with its FP16 value.
    float step = magnitude / 127.0f;
    uint16_t bits = trillim_to_fp16(step);
    if (!isfinite(magnitude) || bits == 0x7c00 || (magnitude != 0.0f && bits == 0)) {
        return false;
    }
    float inverse = step != 0.0f ? 1.0f / step : 0.0f;
    codes[0] = quantize16(values, inverse);
    codes[1] = quantize16(values + 4, inverse);
    *scale = trillim_fp16(bits);
    return true;
}

static TrillimStatus prepare(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    Task *task = opaque;
    const size_t row = tile * ROWS;
    const size_t rows = task->p->tokens - row < ROWS ? task->p->tokens - row : ROWS;
    Activation *activation = task->activation + tile * task->activation_groups;
    for (size_t g = 0; g < task->activation_groups; ++g) {
        if ((g & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int8x16_t codes[ROWS][2];
        float scales[ROWS];
        for (size_t r = 0; r < rows; ++r) {
            if (!quantize_group(task, row + r, g * ACTIVATION_GROUP, codes[r], &scales[r])) {
                atomic_store_explicit(&task->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
        if (task->pairwise) {
            for (size_t r = rows; r < ROWS; ++r) {
                codes[r][0] = codes[r][1] = vdupq_n_s8(0);
                scales[r] = 0.0f;
            }
            for (size_t r = 0; r < ROWS / 2; ++r) {
                float32x2_t pair = vld1_f32(scales + 2 * r);
                activation[g].scales.pairs[r] = vzip1q_f32(vcombine_f32(pair, pair), vcombine_f32(pair, pair));
                for (size_t c = 0; c < 2; ++c) {
                    int64x2_t a = vreinterpretq_s64_s8(codes[2 * r][c]);
                    int64x2_t b = vreinterpretq_s64_s8(codes[2 * r + 1][c]);
                    activation[g].codes.pairs[c * 2][r] = vreinterpretq_s8_s64(vzip1q_s64(a, b));
                    activation[g].codes.pairs[c * 2 + 1][r] = vreinterpretq_s8_s64(vzip2q_s64(a, b));
                }
            }
        } else {
            for (size_t r = 0; r < rows; ++r) {
                activation[g].codes.rows[r][0] = codes[r][0];
                activation[g].codes.rows[r][1] = codes[r][1];
                activation[g].scales.rows[r] = scales[r];
            }
        }
    }
    return TRILLIM_OK;
}

static inline int8x16_t binary_pair(uint8_t first, uint8_t second) {
    const uint8x16_t mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(first), vdup_n_u8(second));
    uint8x16_t flags;
    __asm__("cmtst %0.16b, %1.16b, %2.16b" : "=w"(flags) : "w"(bytes), "w"(mask));
    return vreinterpretq_s8_u8(vornq_u8(vdupq_n_u8(1), flags));
}

static inline float32x4_t load_scales(const Task *task, size_t block, size_t columns) {
    uint16_t bits[COLUMNS];
    for (size_t c = 0; c < COLUMNS; ++c) {
        memcpy(&bits[c], task->weight_scales + (block + (c < columns ? c : 0) * task->groups) * 2, 2);
    }
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(bits), 16));
}

static inline void store_row(const Task *task, size_t row, size_t column, size_t count, float32x4_t total) {
    size_t index = row * task->p->outputs + column;
    if (task->p->output_dtype == TRILLIM_FP32 && count == COLUMNS) {
        vst1q_f32((float *)task->output + index, total);
        return;
    }
    float values[COLUMNS];
    vst1q_f32(values, total);
    for (size_t c = 0; c < count; ++c) {
        trillim_store(task->output, index + c, task->p->output_dtype, (TrillimValue){.f = values[c]});
    }
}

I8MM
static TrillimStatus project_i8mm(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t row_tile = tile / task->column_tiles;
    size_t row = row_tile * ROWS;
    size_t rows = task->p->tokens - row < ROWS ? task->p->tokens - row : ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = task->p->outputs - first < task->columns ? task->p->outputs : first + task->columns;
    const Activation *activation = task->activation + row_tile * task->activation_groups;
    for (size_t column = first; column < end; column += COLUMNS) {
        size_t columns = end - column < COLUMNS ? end - column : COLUMNS;
        const uint8_t *packed[COLUMNS];
        for (size_t c = 0; c < COLUMNS; ++c) {
            packed[c] = task->weight + (column + (c < columns ? c : 0)) * task->groups * 16;
        }
        float32x4_t total[ROWS / 2][2];
        for (size_t r = 0; r < ROWS / 2; ++r) {
            total[r][0] = total[r][1] = vdupq_n_f32(0.0f);
        }
        for (size_t g = 0; g < task->groups; ++g) {
            if ((g & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t weights = load_scales(task, column * task->groups + g, columns);
            float32x4_t group[ROWS / 2][2];
            for (size_t r = 0; r < ROWS / 2; ++r) {
                group[r][0] = group[r][1] = vdupq_n_f32(0.0f);
            }
            size_t a_first = g * (WEIGHT_GROUP / ACTIVATION_GROUP);
            size_t a_end = a_first + (WEIGHT_GROUP / ACTIVATION_GROUP);
            if (a_end > task->activation_groups) {
                a_end = task->activation_groups;
            }
            for (size_t a = a_first; a < a_end; ++a) {
                const Activation *x = activation + a;
                size_t offset = g * 16 + (a - a_first) * 4;
                int32x4_t sum[ROWS / 2][2];
                for (size_t r = 0; r < ROWS / 2; ++r) {
                    sum[r][0] = sum[r][1] = vdupq_n_s32(0);
                }
                for (size_t k = 0; k < 4; ++k) {
                    int8x16_t w01 = binary_pair(packed[0][offset + k], packed[1][offset + k]);
                    int8x16_t w23 = binary_pair(packed[2][offset + k], packed[3][offset + k]);
                    for (size_t r = 0; r < ROWS / 2; ++r) {
                        int8x16_t input = x->codes.pairs[k][r];
                        sum[r][0] = matrix_product(sum[r][0], input, w01);
                        sum[r][1] = matrix_product(sum[r][1], input, w23);
                    }
                }
                for (size_t r = 0; r < ROWS / 2; ++r) {
                    float32x4_t scale = x->scales.pairs[r];
                    group[r][0] = vfmaq_f32(group[r][0], vcvtq_f32_s32(sum[r][0]), scale);
                    group[r][1] = vfmaq_f32(group[r][1], vcvtq_f32_s32(sum[r][1]), scale);
                }
            }
            float64x2_t weight_pairs = vreinterpretq_f64_f32(weights);
            float32x4_t w01 = vreinterpretq_f32_f64(vzip1q_f64(weight_pairs, weight_pairs));
            float32x4_t w23 = vreinterpretq_f32_f64(vzip2q_f64(weight_pairs, weight_pairs));
            for (size_t r = 0; r < ROWS / 2; ++r) {
                total[r][0] = vfmaq_f32(total[r][0], group[r][0], w01);
                total[r][1] = vfmaq_f32(total[r][1], group[r][1], w23);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            float64x2_t c01 = vreinterpretq_f64_f32(total[r / 2][0]);
            float64x2_t c23 = vreinterpretq_f64_f32(total[r / 2][1]);
            float64x2_t values = r % 2 ? vzip2q_f64(c01, c23) : vzip1q_f64(c01, c23);
            store_row(task, row + r, column, columns, vreinterpretq_f32_f64(values));
        }
    }
    return TRILLIM_OK;
}

static inline int32x4_t widening_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    int16x8_t low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    return vpadalq_s16(sum, vmlal_high_s8(low, a, b));
}

#define DEFINE_PROJECT(NAME, TARGET, PRODUCT) \
TARGET static TrillimStatus NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    size_t row = tile / task->column_tiles; \
    size_t first = tile % task->column_tiles * task->columns; \
    size_t end = task->p->outputs - first < task->columns ? task->p->outputs : first + task->columns; \
    const Activation *activation = task->activation + row / ROWS * task->activation_groups; \
    size_t r = row % ROWS; \
    for (size_t column = first; column < end; column += COLUMNS) { \
        size_t columns = end - column < COLUMNS ? end - column : COLUMNS; \
        const uint8_t *packed[COLUMNS]; \
        for (size_t c = 0; c < COLUMNS; ++c) { \
            packed[c] = task->weight + (column + (c < columns ? c : 0)) * task->groups * 16; \
        } \
        float32x4_t total = vdupq_n_f32(0.0f); \
        for (size_t g = 0; g < task->groups; ++g) { \
            if ((g & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float32x4_t weights = load_scales(task, column * task->groups + g, columns); \
            float32x4_t group = vdupq_n_f32(0.0f); \
            size_t a_first = g * (WEIGHT_GROUP / ACTIVATION_GROUP); \
            size_t a_end = a_first + (WEIGHT_GROUP / ACTIVATION_GROUP); \
            if (a_end > task->activation_groups) { \
                a_end = task->activation_groups; \
            } \
            for (size_t a = a_first; a < a_end; ++a) { \
                const Activation *x = activation + a; \
                size_t offset = g * 16 + (a - a_first) * 4; \
                int32x4_t sum[COLUMNS]; \
                for (size_t c = 0; c < COLUMNS; ++c) { \
                    const uint8_t *bits = packed[c] + offset; \
                    int8x16_t w0 = binary_pair(bits[0], bits[1]); \
                    int8x16_t w1 = binary_pair(bits[2], bits[3]); \
                    int32x4_t dot = PRODUCT(vdupq_n_s32(0), x->codes.rows[r][0], w0); \
                    sum[c] = PRODUCT(dot, x->codes.rows[r][1], w1); \
                } \
                int32x4_t sums01 = vpaddq_s32(sum[0], sum[1]); \
                int32x4_t sums23 = vpaddq_s32(sum[2], sum[3]); \
                group = vfmaq_n_f32(group, vcvtq_f32_s32(vpaddq_s32(sums01, sums23)), x->scales.rows[r]); \
            } \
            total = vfmaq_f32(total, group, weights); \
        } \
        store_row(task, row, column, columns, total); \
    } \
    return TRILLIM_OK; \
}

DEFINE_PROJECT(project_dotprod, DOTPROD, dot_product)
DEFINE_PROJECT(project_neon, , widening_product)

static TrillimStatus project_float(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = tile / task->column_tiles;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    for (size_t column = first; column < end; ++column) {
        float sum = 0.0f;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f;
            float w = trillim_packed_read_float(task->weight, column * p->width + k,
                p->weight_dtype, p->outputs, p->width);
            sum = fmaf(x, w, sum);
        }
        trillim_store(task->output, row * p->outputs + column, p->output_dtype, (TrillimValue){.f = sum});
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_auto_binary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t input_sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(p, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || !p->tokens || !p->outputs) {
        return status;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    if (!pool->worker_count) {
        return TRILLIM_ERROR;
    }
    size_t groups = ceil_div(p->width, WEIGHT_GROUP);
    size_t columns = p->outputs / pool->worker_count / (COLUMNS * 4) * COLUMNS;
    if (columns < COLUMNS) {
        columns = COLUMNS;
    }
    bool pairwise = p->tokens >= ROWS / 2 && has_i8mm();
    size_t row_tiles = ceil_div(p->tokens, ROWS);
    const uint8_t *weight = inputs[1];
    const uint8_t *scales = groups ? weight + p->outputs * groups * 16 : weight;
    Task task = {
        p, context, inputs[0], weight, scales,
        outputs[0], scratch, groups, ceil_div(p->width, ACTIVATION_GROUP),
        columns, ceil_div(p->outputs, columns), pairwise, ATOMIC_VAR_INIT(true),
    };
    if (!groups) {
        return pool->run(pool->context, p->tokens * task.column_tiles, project_float, &task);
    }
    status = row_tiles == 1 ? prepare(&task, 0, 0) : pool->run(pool->context, row_tiles, prepare, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!atomic_load_explicit(&task.ordinary, memory_order_relaxed)) {
        return pool->run(pool->context, p->tokens * task.column_tiles, project_float, &task);
    }
    TrillimTask project = pairwise ? project_i8mm : (has_dotprod() ? project_dotprod : project_neon);
    size_t tiles = (pairwise ? row_tiles : p->tokens) * task.column_tiles;
    return tiles == 1 ? project(&task, 0, 0) : pool->run(pool->context, tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_auto_binary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "def bind(context):\n"
           "    source, weight = context.inputs['x'], context.inputs['weight']\n"
           "    if source.dtype not in {'fp16', 'bf16', 'fp32'} or weight.dtype != 'grp_binary' or context.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('binary linear requires floating activations and binary weights')\n"
           "    for tensor in (source, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('binary linear requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('binary linear requires a complete packed weight matrix')\n"
           "    values = _linear(context)\n"
           "    return dict(tokens=values['m'], width=values['k'], outputs=values['n'], input_dtype=values['a_dtype'], weight_dtype=values['b_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased floating linear projection')\n"
           "    result = Linear(x, weight)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_auto_binary_linear_parameters(void) {
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
        requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(Activation)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_auto_binary_linear_capabilities(void) {
#if defined(__APPLE__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
