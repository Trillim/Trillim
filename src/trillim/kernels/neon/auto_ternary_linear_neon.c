#include "common.h"
#include <arm_neon.h>
#include <float.h>
#include <stdatomic.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/sysctl.h>
#include <time.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
/* Older Apple Clang drops operands on optional ARM instructions in assembly output. */
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set packed_qlinear_v\\reg,\\reg\n"
    ".endr\n"
);
#define DOTPROD
#define I8MM
static inline int32x4_t dot_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e809400 | packed_qlinear_%0 | (packed_qlinear_%1 << 5) | (packed_qlinear_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline int32x4_t matrix_product(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e80a400 | packed_qlinear_%0 | (packed_qlinear_%1 << 5) | (packed_qlinear_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
/* The catalog reassembles compiler output with baseline flags. */
__asm__(".arch armv8.6-a+dotprod+i8mm");
#if defined(__clang__)
#define DOTPROD __attribute__((target("dotprod")))
#define I8MM __attribute__((target("i8mm")))
#elif defined(__GNUC__)
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

#ifdef DOTPROD
#include <stdatomic.h>
#endif

enum { PACKED_ROWS = 4, PACKED_COLUMNS = 16, LOWBIT_COLUMNS = 4, LOWBIT_WIDTH = 128, GROUP32_WIDTH = 32, Q6_WIDTH = 256, I8MM_ROWS = 8 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype, input_dtype;
} Parameters;

typedef struct {
    size_t width, values, scales, high, subscales;
} Layout;

typedef struct {
    const uint8_t *values, *high;
    const int8_t *subscales;
    uint32_t dtype;
} Block;

typedef struct {
    void *output;
    const int8_t *input;
    const uint8_t *values, *scales, *high;
    const int8_t *subscales;
    const float *activation_scales;
    const Parameters *p;
    const Layout *layout;
    size_t groups, columns, column_tiles;
    const TrillimKernelContext *context;
} Task;

static const Layout *packed_layout(uint32_t dtype) {
    static const Layout layouts[] = {
        [TRILLIM_GRP_INT4 - TRILLIM_GRP_INT4] = {32, 16, 2, 0, 0},
        [TRILLIM_GRP_INT5 - TRILLIM_GRP_INT4] = {32, 16, 2, 4, 0},
        [TRILLIM_DUBGRP_INT6 - TRILLIM_GRP_INT4] = {256, 128, 2, 64, 16},
        [TRILLIM_GRP_BINARY - TRILLIM_GRP_INT4] = {128, 16, 2, 0, 0},
        [TRILLIM_GRP_TERNARY - TRILLIM_GRP_INT4] = {128, 32, 2, 0, 0},
        [TRILLIM_MXFP4 - TRILLIM_GRP_INT4] = {32, 16, 1, 0, 0},
    };
    return dtype >= TRILLIM_GRP_INT4 && dtype <= TRILLIM_MXFP4 ? &layouts[dtype - TRILLIM_GRP_INT4] : NULL;
}

static TrillimStatus quantized_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t record = layout->values + layout->scales + layout->high + layout->subscales;
    if (groups > SIZE_MAX / record || p->outputs > SIZE_MAX / (groups * record) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * record;
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

static float weight_scale(const Task *task, size_t block) {
    if (task->p->weight_dtype == TRILLIM_MXFP4) {
        // The integer E2M1 palette contains twice each decoded value.
        return trillim_mxfp4_scale(task->scales[block]) * 0.5f;
    }
    uint16_t bits;
    memcpy(&bits, task->scales + block * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static inline uint8x16_t spread_bits(const uint8_t *packed, uint8x16_t mask) {
    uint8x16_t bytes = vcombine_u8(vdup_n_u8(packed[0]), vdup_n_u8(packed[1]));
    return vceqq_u8(vandq_u8(bytes, mask), mask);
}

static inline int8x16_t decode(const Block *block, size_t k) {
    const uint8_t *values = block->values;
    const uint8x16_t one = vdupq_n_u8(1);
    const uint8x16_t nibbles = vdupq_n_u8(15);
    switch (block->dtype) {
    case TRILLIM_GRP_BINARY: {
        const uint8x16_t mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
        uint8x16_t bits = spread_bits(values + k / 8, mask);
        return vreinterpretq_s8_u8(vsubq_u8(vdupq_n_u8(0), vorrq_u8(bits, one)));
    }
    case TRILLIM_GRP_TERNARY: {
        int8x16_t shift = vdupq_n_s8((int)(k / 32) * 2 - 6);
        uint8x16_t packed = vld1q_u8(values + k % 32);
        uint8x16_t codes = vandq_u8(vshlq_u8(packed, shift), vdupq_n_u8(3));
        return vreinterpretq_s8_u8(vsubq_u8(codes, one));
    }
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5: {
        uint8x16_t packed = vld1q_u8(values);
        uint8x16_t codes = k == 0 ? vandq_u8(packed, nibbles) : vshrq_n_u8(packed, 4);
        if (block->dtype == TRILLIM_GRP_INT5) {
            const uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
            uint8x16_t high = spread_bits(block->high + k / 8, mask);
            codes = vorrq_u8(codes, vandq_u8(high, vdupq_n_u8(16)));
            return vreinterpretq_s8_u8(vsubq_u8(codes, vdupq_n_u8(16)));
        }
        return vreinterpretq_s8_u8(vsubq_u8(codes, vdupq_n_u8(8)));
    }
    case TRILLIM_DUBGRP_INT6: {
        size_t half = k / 128;
        size_t lane = k % 128 / 32;
        uint8x16_t low = vld1q_u8(values + half * 64 + lane / 2 * 32 + k % 32);
        uint8x16_t high = vld1q_u8(block->high + half * 32 + k % 32);
        low = lane % 2 == 0 ? vandq_u8(low, nibbles) : vshrq_n_u8(low, 4);
        high = vshlq_u8(high, vdupq_n_s8(-(int)lane * 2));
        high = vshlq_n_u8(vandq_u8(high, vdupq_n_u8(3)), 4);
        return vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(low, high), vdupq_n_u8(32)));
    }
    case TRILLIM_MXFP4: {
        const int8x16_t palette = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        uint8x8_t packed = vld1_u8(values + k / 2);
        uint8x8_t low = vand_u8(packed, vdup_n_u8(15));
        uint8x8_t high = vshr_n_u8(packed, 4);
        uint8x8x2_t codes = vzip_u8(low, high);
        return vqtbl1q_s8(palette, vcombine_u8(codes.val[0], codes.val[1]));
    }
    default:
        return vdupq_n_s8(0);
    }
}

static inline int32x4_t products(int8x16_t x, int8x16_t w) {
    // Weight codes have magnitude at most 32, so paired products fit in int16.
    int16x8_t low = vmull_s8(vget_low_s8(x), vget_low_s8(w));
    int16x8_t pairs = vmlal_high_s8(low, x, w);
    return vpaddlq_s16(pairs);
}

static inline bool packed_dot(const Block *block, const int8_t *input, size_t stride, size_t rows,
                              size_t count, int32_t *sums) {
    int32x4_t acc[PACKED_ROWS] = {0};
    for (size_t k = 0; k < count; k += 16) {
        int8x16_t weights = decode(block, k);
        size_t lanes = count - k < 16 ? count - k : 16;
        if (block->dtype == TRILLIM_GRP_TERNARY) {
            int8x16_t valid = weights;
            if (lanes < 16) {
                const uint8x16_t indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
                uint8x16_t mask = vcltq_u8(indices, vdupq_n_u8((uint8_t)lanes));
                valid = vandq_s8(weights, vreinterpretq_s8_u8(mask));
            }
            if (vmaxvq_s8(valid) == 2) {
                return false;
            }
        }
        for (size_t row = 0; row < PACKED_ROWS; ++row) {
            if (row == rows) {
                break;
            }
            const int8_t *src = input + row * stride + k;
            int8x16_t x;
            if (lanes < 16) {
                int8_t tail[16] = {0};
                memcpy(tail, src, lanes);
                x = vld1q_s8(tail);
            } else {
                x = vld1q_s8(src);
            }
            int32x4_t dot = products(x, weights);
            if (block->dtype == TRILLIM_DUBGRP_INT6) {
                acc[row] = vmlaq_n_s32(acc[row], dot, block->subscales[k / 16]);
            } else {
                acc[row] = vaddq_s32(acc[row], dot);
            }
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = vaddvq_s32(acc[row]);
    }
    return true;
}

static TrillimStatus project_columns(const Task *task, size_t row, size_t rows, size_t first, size_t end) {
    const Parameters *p = task->p;
    const Layout *layout = task->layout;
    for (size_t column = first; column < end; ++column) {
        float totals[PACKED_ROWS] = {0};
        for (size_t g = 0; g < task->groups; ++g) {
            size_t k = g * layout->width;
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->width - k < layout->width ? p->width - k : layout->width;
            size_t index = column * task->groups + g;
            Block block = {
                task->values + index * layout->values,
                task->high + index * layout->high,
                task->subscales + index * layout->subscales,
                p->weight_dtype,
            };
            int32_t sums[PACKED_ROWS] = {0};
            if (!packed_dot(&block, task->input + row * p->width + k, p->width, rows, count, sums)) {
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = NAN;
                }
                break;
            }
            float scale = weight_scale(task, index);
            for (size_t r = 0; r < rows; ++r) {
                totals[r] += (float)sums[r] * scale;
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue result = {.f = totals[r] / task->activation_scales[row + r]};
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = tile / task->column_tiles * PACKED_ROWS;
    size_t rows = p->tokens - row < PACKED_ROWS ? p->tokens - row : PACKED_ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    return project_columns(task, row, rows, first, end);
}

#ifdef DOTPROD
static inline bool group32_dtype(uint32_t dtype) {
    return dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5 || dtype == TRILLIM_MXFP4;
}

static void add_lowbit_tail(const Task *task, size_t row, size_t rows, size_t column, float32x4_t *totals) {
    const Parameters *p = task->p;
    const size_t group = task->groups - 1;
    const size_t start = group * LOWBIT_WIDTH;
    float values[PACKED_ROWS][LOWBIT_COLUMNS];
    float scales[LOWBIT_COLUMNS];
    for (size_t c = 0; c < LOWBIT_COLUMNS; ++c) {
        const size_t index = (column + c) * task->groups + group;
        const Block block = {
            task->values + index * task->layout->values,
            NULL,
            NULL,
            p->weight_dtype,
        };
        int32_t sums[PACKED_ROWS];
        const bool valid = packed_dot(&block, task->input + row * p->width + start, p->width, rows,
                                      p->width - start, sums);
        scales[c] = weight_scale(task, index);
        for (size_t r = 0; r < rows; ++r) {
            values[r][c] = valid ? (float)sums[r] : NAN;
        }
    }
    const float32x4_t weight_scales = vld1q_f32(scales);
    for (size_t r = 0; r < rows; ++r) {
        totals[r] = vfmaq_f32(totals[r], vld1q_f32(values[r]), weight_scales);
    }
}

static inline void store_tile(const Task *task, size_t row, size_t column, float32x4_t totals) {
    const Parameters *p = task->p;
    const float32x4_t values = vdivq_f32(totals, vdupq_n_f32(task->activation_scales[row]));
    const size_t index = row * p->outputs + column;
    if (p->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + index * sizeof(float), &values, sizeof(values));
    } else if (p->output_dtype == TRILLIM_BF16) {
        const uint32x4_t bits = vreinterpretq_u32_f32(values);
        const uint32x4_t highest = vshrq_n_u32(bits, 16);
        const uint32x4_t rounded = vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(highest, vdupq_n_u32(1))));
        const uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
        const uint16x4_t nan_mask = vmovn_u32(vcgtq_u32(magnitude, vdupq_n_u32(0x7f800000)));
        const uint16x4_t nan_bits = vorr_u16(vmovn_u32(highest), vdup_n_u16(64));
        const uint16x4_t result = vbsl_u16(nan_mask, nan_bits, vshrn_n_u32(rounded, 16));
        memcpy((uint8_t *)task->output + index * sizeof(uint16_t), &result, sizeof(result));
    } else {
        float results[LOWBIT_COLUMNS];
        vst1q_f32(results, values);
        for (size_t c = 0; c < LOWBIT_COLUMNS; ++c) {
            TrillimValue result = {.f = results[c]};
            trillim_store(task->output, index + c, p->output_dtype, result);
        }
    }
}

static bool has_dotprod(void) {
    static atomic_int cached;
    int support = atomic_load_explicit(&cached, memory_order_relaxed);
    if (support == 0) {
        bool available = false;
#if defined(__APPLE__)
        int value = 0;
        size_t size = sizeof(value);
        available = sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, NULL, 0) == 0 && value;
#elif defined(__linux__) && defined(HWCAP_ASIMDDP)
        available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__ARM_FEATURE_DOTPROD)
        available = true;
#endif
        support = available ? 2 : 1;
        atomic_store_explicit(&cached, support, memory_order_relaxed);
    }
    return support == 2;
}

DOTPROD __attribute__((always_inline))
static inline TrillimStatus project_dotprod_columns(const Task *task, size_t row, size_t rows,
                                                size_t first, size_t end, uint32_t dtype,
                                                const int16_t *activation_sums) {
    const Parameters *p = task->p;
    const bool ternary = dtype == TRILLIM_GRP_TERNARY;
    const bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    const bool group32 = group32_dtype(dtype);
    const bool direct_decode = group32 || q6;
    const size_t group_width = q6 ? Q6_WIDTH : group32 ? GROUP32_WIDTH : LOWBIT_WIDTH;
    const size_t block_bytes = q6 ? 128 : ternary ? 32 : 16;
    const bool lowbit_tail = !direct_decode && p->width % LOWBIT_WIDTH != 0;
    const size_t full_groups = task->groups - lowbit_tail;
    const uint8x16_t sign_mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
    const uint8x16_t first_pair = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    const uint8x16_t one = vdupq_n_u8(1);
    size_t column = first;
    for (; end - column >= LOWBIT_COLUMNS; column += LOWBIT_COLUMNS) {
        float32x4_t totals[PACKED_ROWS] = {0};
        uint8x16_t invalid = vdupq_n_u8(0);
        for (size_t g = 0; g < full_groups; ++g) {
            if ((g & (q6 ? 0 : group32 ? 7 : 1)) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8x16_t packed[2][LOWBIT_COLUMNS];
            Block blocks[LOWBIT_COLUMNS];
            uint16_t scale_bits[LOWBIT_COLUMNS];
            float mxfp4_scales[LOWBIT_COLUMNS];
            for (size_t c = 0; c < LOWBIT_COLUMNS; ++c) {
                size_t block = (column + c) * task->groups + g;
                const uint8_t *values = task->values + block * block_bytes;
                if (dtype == TRILLIM_MXFP4) {
                    mxfp4_scales[c] = trillim_mxfp4_scale(task->scales[block]) * 0.5f;
                } else {
                    memcpy(&scale_bits[c], task->scales + block * 2, sizeof(scale_bits[c]));
                }
                if (direct_decode) {
                    blocks[c] = (Block){
                        values,
                        task->high + block * task->layout->high,
                        task->subscales + block * task->layout->subscales,
                        dtype,
                    };
                } else {
                    packed[0][c] = vld1q_u8(values);
                }
                if (ternary) {
                    packed[1][c] = vld1q_u8(values + 16);
                    uint8x16_t low = vandq_u8(packed[0][c], vshrq_n_u8(packed[0][c], 1));
                    uint8x16_t high = vandq_u8(packed[1][c], vshrq_n_u8(packed[1][c], 1));
                    invalid = vorrq_u8(invalid, vorrq_u8(low, high));
                }
            }

            int32x4_t dots[PACKED_ROWS][LOWBIT_COLUMNS] = {0};
            const int8_t *input = task->input + row * p->width + g * group_width;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (size_t chunk = 0; chunk < group_width / 16; ++chunk) {
                int8x16_t activations[PACKED_ROWS];
                for (size_t r = 0; r < rows; ++r) {
                    activations[r] = vld1q_s8(input + r * p->width + chunk * 16);
                }
                for (size_t c = 0; c < LOWBIT_COLUMNS; ++c) {
                    int8x16_t weights;
                    if (direct_decode) {
                        weights = decode(&blocks[c], chunk * 16);
                    } else if (ternary) {
                        int8x16_t shift = vdupq_n_s8((int)(chunk / 2) * 2 - 6);
                        uint8x16_t codes = vshlq_u8(packed[chunk % 2][c], shift);
                        codes = vandq_u8(codes, vdupq_n_u8(3));
                        weights = vreinterpretq_s8_u8(activation_sums ? codes : vsubq_u8(codes, one));
                    } else {
                        uint8x16_t indices = vaddq_u8(first_pair, vdupq_n_u8((uint8_t)(chunk * 2)));
                        uint8x16_t bytes = vqtbl1q_u8(packed[0][c], indices);
                        uint8x16_t bits = vtstq_u8(bytes, sign_mask);
                        weights = vreinterpretq_s8_u8(activation_sums ? bits : vornq_u8(one, bits));
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        if (q6) {
                            int32x4_t dot = dot_product(vdupq_n_s32(0), activations[r], weights);
                            dots[r][c] = vmlaq_n_s32(dots[r][c], dot, blocks[c].subscales[chunk]);
                        } else {
                            dots[r][c] = dot_product(dots[r][c], activations[r], weights);
                        }
                    }
                }
            }
            const float32x4_t scales = dtype == TRILLIM_MXFP4 ? vld1q_f32(mxfp4_scales)
                : vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(scale_bits), 16));
            for (size_t r = 0; r < rows; ++r) {
                int32x4_t first_pair_sum = vpaddq_s32(dots[r][0], dots[r][1]);
                int32x4_t second_pair_sum = vpaddq_s32(dots[r][2], dots[r][3]);
                int32x4_t sums = vpaddq_s32(first_pair_sum, second_pair_sum);
                if (activation_sums) {
                    if (!ternary) {
                        sums = vshlq_n_s32(vnegq_s32(sums), 1);
                    }
                    sums = vsubq_s32(sums, vdupq_n_s32(activation_sums[g]));
                }
                totals[r] = vfmaq_f32(totals[r], vcvtq_f32_s32(sums), scales);
            }
        }
        if (ternary && vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            TrillimStatus status = project_columns(task, row, rows, column, column + LOWBIT_COLUMNS);
            if (status != TRILLIM_OK) {
                return status;
            }
            continue;
        }
        if (lowbit_tail) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            add_lowbit_tail(task, row, rows, column, totals);
        }
        for (size_t r = 0; r < rows; ++r) {
            store_tile(task, row + r, column, totals[r]);
        }
    }
    return project_columns(task, row, rows, column, end);
}

DOTPROD __attribute__((always_inline))
static inline TrillimStatus project_dotprod_dtype(const Task *task, size_t row, size_t rows, size_t first, size_t end) {
    switch (task->p->weight_dtype) {
    case TRILLIM_GRP_INT4:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_GRP_INT4, NULL);
    case TRILLIM_GRP_INT5:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_GRP_INT5, NULL);
    case TRILLIM_DUBGRP_INT6:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_DUBGRP_INT6, NULL);
    case TRILLIM_GRP_BINARY:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_GRP_BINARY, NULL);
    case TRILLIM_GRP_TERNARY:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_GRP_TERNARY, NULL);
    case TRILLIM_MXFP4:
        return project_dotprod_columns(task, row, rows, first, end, TRILLIM_MXFP4, NULL);
    default:
        return project_columns(task, row, rows, first, end);
    }
}

enum { MAX_SUM_GROUPS = 128 };

DOTPROD
static TrillimStatus project_lowbit_decode(const Task *task, size_t row, size_t first, size_t end) {
    int16_t sums[MAX_SUM_GROUPS];
    const int8_t *input = task->input + row * task->p->width;
    const size_t groups = task->p->width / LOWBIT_WIDTH;
    for (size_t g = 0; g < groups; ++g) {
        if ((g & 15) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int16x8_t sum = vdupq_n_s16(0);
        for (size_t k = 0; k < LOWBIT_WIDTH; k += 16) {
            sum = vpadalq_s8(sum, vld1q_s8(input + g * LOWBIT_WIDTH + k));
        }
        sums[g] = vaddvq_s16(sum);
    }
    if (task->p->weight_dtype == TRILLIM_GRP_BINARY) {
        return project_dotprod_columns(task, row, 1, first, end, TRILLIM_GRP_BINARY, sums);
    }
    return project_dotprod_columns(task, row, 1, first, end, TRILLIM_GRP_TERNARY, sums);
}

DOTPROD
static TrillimStatus project_dotprod_rows(const Task *task, size_t row, size_t rows, size_t first, size_t end) {
    if (rows == PACKED_ROWS) {
        return project_dotprod_dtype(task, row, PACKED_ROWS, first, end);
    }
    if (rows == 1) {
        const uint32_t dtype = task->p->weight_dtype;
        if ((dtype == TRILLIM_GRP_BINARY || dtype == TRILLIM_GRP_TERNARY) &&
            task->groups <= MAX_SUM_GROUPS && end - first >= LOWBIT_COLUMNS * 2) {
            return project_lowbit_decode(task, row, first, end);
        }
        return project_dotprod_dtype(task, row, 1, first, end);
    }
    switch (task->p->weight_dtype) {
    case TRILLIM_GRP_BINARY:
        return rows == 2 ? project_dotprod_columns(task, row, 2, first, end, TRILLIM_GRP_BINARY, NULL)
                         : project_dotprod_columns(task, row, 3, first, end, TRILLIM_GRP_BINARY, NULL);
    case TRILLIM_GRP_TERNARY:
        return rows == 2 ? project_dotprod_columns(task, row, 2, first, end, TRILLIM_GRP_TERNARY, NULL)
                         : project_dotprod_columns(task, row, 3, first, end, TRILLIM_GRP_TERNARY, NULL);
    default:
        return project_columns(task, row, rows, first, end);
    }
}

DOTPROD
static TrillimStatus project_dotprod(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = tile / task->column_tiles * PACKED_ROWS;
    size_t rows = p->tokens - row < PACKED_ROWS ? p->tokens - row : PACKED_ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    return project_dotprod_rows(task, row, rows, first, end);
}

static bool has_i8mm(void) {
    static atomic_int cached;
    int support = atomic_load_explicit(&cached, memory_order_relaxed);
    if (support == 0) {
        bool available = false;
#if defined(__APPLE__)
        int value = 0;
        size_t size = sizeof(value);
        available = sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &size, NULL, 0) == 0 && value;
#elif defined(__linux__) && defined(HWCAP2_I8MM)
        available = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#elif defined(__ARM_FEATURE_MATMUL_INT8)
        available = true;
#endif
        support = available ? 2 : 1;
        atomic_store_explicit(&cached, support, memory_order_relaxed);
    }
    return support == 2;
}

static inline int8x16_t zip_low_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}

static inline int8x16_t zip_high_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}

static inline float32x4_t zip_low_f32(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_u64(vzip1q_u64(vreinterpretq_u64_f32(a), vreinterpretq_u64_f32(b)));
}

static inline float32x4_t zip_high_f32(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_u64(vzip2q_u64(vreinterpretq_u64_f32(a), vreinterpretq_u64_f32(b)));
}

static TrillimStatus project_i8mm_edges(const Task *task, size_t row, size_t rows, size_t first, size_t end) {
    for (size_t r = 0; r < rows; r += PACKED_ROWS) {
        size_t count = rows - r < PACKED_ROWS ? rows - r : PACKED_ROWS;
        TrillimStatus status = project_dotprod_rows(task, row + r, count, first, end);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

I8MM __attribute__((always_inline))
static inline TrillimStatus project_i8mm_columns(
    const Task *task, size_t row, size_t first, size_t end, uint32_t dtype, const int8x16_t *packed_input
) {
    const Parameters *p = task->p;
    const bool ternary = dtype == TRILLIM_GRP_TERNARY;
    const bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    const bool group32 = group32_dtype(dtype);
    const bool direct_decode = group32 || q6;
    const size_t group_width = q6 ? Q6_WIDTH : group32 ? GROUP32_WIDTH : LOWBIT_WIDTH;
    const size_t block_bytes = q6 ? 128 : ternary ? 32 : 16;
    const bool lowbit_tail = !direct_decode && p->width % LOWBIT_WIDTH != 0;
    const size_t full_groups = task->groups - lowbit_tail;
    const uint8x16_t sign_mask = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
    const uint8x16_t first_pair = {0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8};
    const uint8x16_t one = vdupq_n_u8(1);
    size_t column = first;
    for (; end - column >= LOWBIT_COLUMNS; column += LOWBIT_COLUMNS) {
        float32x4_t totals[I8MM_ROWS / 2][LOWBIT_COLUMNS / 2] = {0};
        uint8x16_t invalid = vdupq_n_u8(0);
        for (size_t g = 0; g < full_groups; ++g) {
            if ((g & (group32 ? 3 : 0)) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int8x16_t packed[2][LOWBIT_COLUMNS];
            Block blocks[LOWBIT_COLUMNS];
            uint16_t scale_bits[LOWBIT_COLUMNS];
            float mxfp4_scales[LOWBIT_COLUMNS];
            for (size_t c = 0; c < LOWBIT_COLUMNS; ++c) {
                size_t block = (column + c) * task->groups + g;
                const uint8_t *values = task->values + block * block_bytes;
                if (dtype == TRILLIM_MXFP4) {
                    mxfp4_scales[c] = trillim_mxfp4_scale(task->scales[block]) * 0.5f;
                } else {
                    memcpy(&scale_bits[c], task->scales + block * 2, sizeof(scale_bits[c]));
                }
                if (direct_decode) {
                    blocks[c] = (Block){
                        values,
                        task->high + block * task->layout->high,
                        task->subscales + block * task->layout->subscales,
                        dtype,
                    };
                } else {
                    packed[0][c] = vld1q_s8((const int8_t *)values);
                }
                if (ternary) {
                    packed[1][c] = vld1q_s8((const int8_t *)values + 16);
                    uint8x16_t lo = vreinterpretq_u8_s8(packed[0][c]);
                    uint8x16_t hi = vreinterpretq_u8_s8(packed[1][c]);
                    invalid = vorrq_u8(invalid, vandq_u8(lo, vshrq_n_u8(lo, 1)));
                    invalid = vorrq_u8(invalid, vandq_u8(hi, vshrq_n_u8(hi, 1)));
                }
            }

            uint8x16_t pairs[2][2][LOWBIT_COLUMNS / 2];
            if (!direct_decode) {
                for (size_t h = 0; h < (ternary ? 2u : 1u); ++h) {
                    for (size_t c = 0; c < LOWBIT_COLUMNS / 2; ++c) {
                        int8x16_t a = packed[h][c * 2];
                        int8x16_t b = packed[h][c * 2 + 1];
                        pairs[h][0][c] = vreinterpretq_u8_s8(zip_low_s8(a, b));
                        pairs[h][1][c] = vreinterpretq_u8_s8(zip_high_s8(a, b));
                    }
                }
            }

            // SMMLA accumulates a pair of token rows by a pair of output columns.
            int32x4_t dots[I8MM_ROWS / 2][LOWBIT_COLUMNS / 2] = {0};
            const int8_t *input = task->input + row * p->width + g * group_width;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (size_t chunk = 0; chunk < group_width / 16; ++chunk) {
                if (q6 && chunk == 8 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                int8x16_t local[I8MM_ROWS];
                const int8x16_t *activations;
                if (packed_input) {
                    activations = packed_input + (g * (group_width / 16) + chunk) * I8MM_ROWS;
                } else {
                    for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
                        const int8_t *a = input + r * 2 * p->width + chunk * 16;
                        int8x16_t first = vld1q_s8(a);
                        int8x16_t second = vld1q_s8(a + p->width);
                        local[r] = zip_low_s8(first, second);
                        local[I8MM_ROWS / 2 + r] = zip_high_s8(first, second);
                    }
                    activations = local;
                }
                if (q6) {
                    // Apply the shared subscale to each complete 16-element dot.
                    for (size_t c = 0; c < LOWBIT_COLUMNS / 2; ++c) {
                        const Block *left = &blocks[c * 2];
                        const Block *right = &blocks[c * 2 + 1];
                        int8x16_t a = decode(left, chunk * 16);
                        int8x16_t b = decode(right, chunk * 16);
                        int8x16_t lo = zip_low_s8(a, b);
                        int8x16_t hi = zip_high_s8(a, b);
                        int32x2_t subscales = {left->subscales[chunk], right->subscales[chunk]};
                        int32x4_t scales = vcombine_s32(subscales, subscales);
                        for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
                            int32x4_t dot = matrix_product(vdupq_n_s32(0), activations[r], lo);
                            dot = matrix_product(dot, activations[I8MM_ROWS / 2 + r], hi);
                            dots[r][c] = vmlaq_s32(dots[r][c], dot, scales);
                        }
                    }
                    continue;
                }
                for (size_t part = 0; part < 2; ++part) {
                    for (size_t c = 0; c < LOWBIT_COLUMNS / 2; ++c) {
                        int8x16_t weights;
                        if (group32) {
                            int8x16_t first_weights = decode(&blocks[c * 2], chunk * 16);
                            int8x16_t second_weights = decode(&blocks[c * 2 + 1], chunk * 16);
                            weights = part == 0 ? zip_low_s8(first_weights, second_weights) : zip_high_s8(first_weights, second_weights);
                        } else if (ternary) {
                            int8x16_t shift = vdupq_n_s8((int)(chunk / 2) * 2 - 6);
                            uint8x16_t codes = vshlq_u8(pairs[chunk % 2][part][c], shift);
                            weights = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(codes, vdupq_n_u8(3)), one));
                        } else {
                            uint8x16_t indices = vaddq_u8(first_pair, vdupq_n_u8((uint8_t)(chunk * 2 + part) % 8));
                            uint8x16_t bytes = vqtbl1q_u8(pairs[0][chunk / 4][c], indices);
                            weights = vreinterpretq_s8_u8(vornq_u8(one, vtstq_u8(bytes, sign_mask)));
                        }
                        for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
                            dots[r][c] = matrix_product(dots[r][c], activations[part * (I8MM_ROWS / 2) + r], weights);
                        }
                    }
                }
            }
            const float32x4_t scales = dtype == TRILLIM_MXFP4 ? vld1q_f32(mxfp4_scales)
                : vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(scale_bits), 16));
            const float32x4_t scales0 = zip_low_f32(scales, scales);
            const float32x4_t scales1 = zip_high_f32(scales, scales);
            for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
                totals[r][0] = vfmaq_f32(totals[r][0], vcvtq_f32_s32(dots[r][0]), scales0);
                totals[r][1] = vfmaq_f32(totals[r][1], vcvtq_f32_s32(dots[r][1]), scales1);
            }
        }
        if (ternary && vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55)))) {
            TrillimStatus status = project_i8mm_edges(task, row, I8MM_ROWS, column, column + LOWBIT_COLUMNS);
            if (status != TRILLIM_OK) {
                return status;
            }
            continue;
        }
        if (lowbit_tail && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
            float32x4_t row_totals[2] = {zip_low_f32(totals[r][0], totals[r][1]), zip_high_f32(totals[r][0], totals[r][1])};
            if (lowbit_tail) {
                add_lowbit_tail(task, row + r * 2, 2, column, row_totals);
            }
            for (size_t part = 0; part < 2; ++part) {
                store_tile(task, row + r * 2 + part, column, row_totals[part]);
            }
        }
    }
    return project_i8mm_edges(task, row, I8MM_ROWS, column, end);
}

enum { BINARY_PREFILL_MAX_WIDTH = 8192 };

I8MM __attribute__((noinline))
static TrillimStatus project_i8mm_binary(const Task *task, size_t row, size_t first, size_t end) {
    const size_t width = task->p->width;
    const size_t full_width = width / LOWBIT_WIDTH * LOWBIT_WIDTH;
    const int8_t *input = task->input + row * width;
    // Each group of output columns reuses the same interleaved token pairs.
    int8x16_t packed_input[BINARY_PREFILL_MAX_WIDTH / 16 * I8MM_ROWS];
    for (size_t k = 0; k < full_width; k += 16) {
        if ((k & 511) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int8x16_t *packed = packed_input + k / 16 * I8MM_ROWS;
        for (size_t r = 0; r < I8MM_ROWS / 2; ++r) {
            const int8_t *pair = input + r * 2 * width + k;
            int8x16_t first_row = vld1q_s8(pair);
            int8x16_t second_row = vld1q_s8(pair + width);
            packed[r] = zip_low_s8(first_row, second_row);
            packed[I8MM_ROWS / 2 + r] = zip_high_s8(first_row, second_row);
        }
    }
    return project_i8mm_columns(task, row, first, end, TRILLIM_GRP_BINARY, packed_input);
}

I8MM
static TrillimStatus project_i8mm(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = tile / task->column_tiles * I8MM_ROWS;
    size_t rows = p->tokens - row < I8MM_ROWS ? p->tokens - row : I8MM_ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    if (rows < I8MM_ROWS) {
        return project_i8mm_edges(task, row, rows, first, end);
    }
    switch (p->weight_dtype) {
    case TRILLIM_GRP_INT4:
        return project_i8mm_columns(task, row, first, end, TRILLIM_GRP_INT4, NULL);
    case TRILLIM_GRP_INT5:
        return project_i8mm_columns(task, row, first, end, TRILLIM_GRP_INT5, NULL);
    case TRILLIM_DUBGRP_INT6:
        return project_i8mm_columns(task, row, first, end, TRILLIM_DUBGRP_INT6, NULL);
    case TRILLIM_GRP_BINARY:
        if (p->width <= BINARY_PREFILL_MAX_WIDTH && end - first >= PACKED_COLUMNS) {
            return project_i8mm_binary(task, row, first, end);
        }
        return project_i8mm_columns(task, row, first, end, TRILLIM_GRP_BINARY, NULL);
    case TRILLIM_GRP_TERNARY:
        return project_i8mm_columns(task, row, first, end, TRILLIM_GRP_TERNARY, NULL);
    case TRILLIM_MXFP4:
        return project_i8mm_columns(task, row, first, end, TRILLIM_MXFP4, NULL);
    default:
        return project_i8mm_edges(task, row, rows, first, end);
    }
}

#endif

static TrillimStatus run_quantized(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t blocks = p->outputs * groups;
    const uint8_t *values = inputs[1];
    const uint8_t *scales = values + blocks * layout->values;
    const uint8_t *high = scales + blocks * layout->scales;
    const int8_t *subscales = (const int8_t *)(high + blocks * layout->high);
    size_t columns = p->outputs / workers / (4 * PACKED_COLUMNS) * PACKED_COLUMNS;
    if (columns < PACKED_COLUMNS) {
        columns = PACKED_COLUMNS;
    }
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    size_t tile_rows = PACKED_ROWS;
    Task task = {
        outputs[0], inputs[0], values, scales, high, subscales, inputs[2],
        p, layout, groups, columns, tiles, context,
    };
    TrillimTask project_task = project;
#ifdef DOTPROD
    bool group32 = group32_dtype(p->weight_dtype);
    const bool lowbit = p->weight_dtype == TRILLIM_GRP_BINARY || p->weight_dtype == TRILLIM_GRP_TERNARY;
    const bool supported_width = lowbit ? p->width >= LOWBIT_WIDTH : p->width % layout->width == 0;
    if (supported_width && (group32 || lowbit || p->weight_dtype == TRILLIM_DUBGRP_INT6) && has_dotprod()) {
        project_task = project_dotprod;
        bool prefer_i8mm = p->tokens >= I8MM_ROWS;
#if defined(__APPLE__)
        // SDOT is faster for 32-value groups with small Apple Silicon thread pools.
        prefer_i8mm = prefer_i8mm && (!group32 || workers > 4);
#endif
        if (prefer_i8mm && has_i8mm()) {
            project_task = project_i8mm;
            tile_rows = I8MM_ROWS;
        }
    }
#endif
    size_t rows = p->tokens / tile_rows + (p->tokens % tile_rows != 0);
    return rows * tiles == 1 ? project_task(&task, 0, 0)
                            : context->thread_pool->run(context->thread_pool->context, rows * tiles, project_task, &task);
}

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input;
    const Parameters *parameters;
    const TrillimKernelContext *context;
    atomic_bool *ordinary;
} QuantizeTask;

static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline int32x4_t round4(float32x4_t values, float32x4_t scale) {
    float32x4_t scaled = vmulq_f32(values, scale);
    scaled = vminq_f32(vmaxq_f32(scaled, vdupq_n_f32(-127.0f)), vdupq_n_f32(127.0f));
    return vcvtnq_s32_f32(scaled);
}

static TrillimStatus quantize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const QuantizeTask *task = opaque;
    const Parameters *p = task->parameters;
    size_t start = row * p->width;
    size_t column = 0;
    float32x4_t maximum4 = vdupq_n_f32(1e-5f);
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t magnitude = vabsq_f32(load4(task->input, start + column, p->input_dtype));
        if (vminvq_u32(vcleq_f32(magnitude, vdupq_n_f32(FLT_MAX))) != UINT32_MAX) {
            atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
            return TRILLIM_OK;
        }
        maximum4 = vmaxq_f32(maximum4, magnitude);
    }
    float maximum = vmaxvq_f32(maximum4);
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        if (!isfinite(value)) {
            atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
            return TRILLIM_OK;
        }
        maximum = fmaxf(maximum, fabsf(value));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    float32x4_t multiplier = vdupq_n_f32(scale);
    column = 0;
    for (; p->width - column >= 16; column += 16) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int32x4_t a = round4(load4(task->input, start + column, p->input_dtype), multiplier);
        int32x4_t b = round4(load4(task->input, start + column + 4, p->input_dtype), multiplier);
        int32x4_t c = round4(load4(task->input, start + column + 8, p->input_dtype), multiplier);
        int32x4_t d = round4(load4(task->input, start + column + 12, p->input_dtype), multiplier);
        int16x8_t low = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        int16x8_t high = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        vst1q_s8(task->codes + start + column, vcombine_s8(vmovn_s16(low), vmovn_s16(high)));
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f * scale;
        float magnitude = fminf(fabsf(value), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[start + column] = (int8_t)(value < 0 ? -code : code);
    }
    return TRILLIM_OK;
}

#if defined(__APPLE__)
enum { BLAS_ROWS = 512, BLAS_DECODE_COLUMNS = 16 };

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
typedef unsigned (*GetThreading)(void);

static Sgemm sgemm;
static GetThreading get_threading;
typedef ssize_t (*MatmulWorkspace)(bool, bool, float, const BNNSNDArrayDescriptor *,
    const BNNSNDArrayDescriptor *, const BNNSNDArrayDescriptor *, const BNNSFilterParameters *);
typedef int (*Matmul)(bool, bool, float, const BNNSNDArrayDescriptor *,
    const BNNSNDArrayDescriptor *, const BNNSNDArrayDescriptor *, void *, const BNNSFilterParameters *);
static MatmulWorkspace matmul_workspace;
static Matmul matmul;

static bool initialize_blas(void) {
    static atomic_uint state;
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&state, &expected, 1, memory_order_acq_rel, memory_order_acquire)) {
        void *library = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LOCAL | RTLD_LAZY);
        if (library != NULL) {
            void *gemm = dlsym(library, "cblas_sgemm$NEWLAPACK");
            void *getter = dlsym(library, "BLASGetThreading");
            if (gemm == NULL) {
                gemm = dlsym(library, "cblas_sgemm");
            }
            if (gemm != NULL && getter != NULL) {
                memcpy(&sgemm, &gemm, sizeof(sgemm));
                memcpy(&get_threading, &getter, sizeof(get_threading));
                void *size = dlsym(library, "BNNSMatMulWorkspaceSize");
                void *multiply = dlsym(library, "BNNSMatMul");
                if (size != NULL && multiply != NULL) {
                    memcpy(&matmul_workspace, &size, sizeof(matmul_workspace));
                    memcpy(&matmul, &multiply, sizeof(matmul));
                }
            } else {
                dlclose(library);
            }
        }
        atomic_store_explicit(&state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&state, memory_order_acquire) == 1) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

static size_t blas_rows(const Parameters *p) {
    if (p->tokens < 64 || p->outputs < 1024 || p->width < 512 ||
        p->width < (1 << 20) / p->outputs || p->width % LOWBIT_WIDTH != 0) {
        return 0;
    }
    size_t rows = (1 << 23) / p->width;
    if (rows < 32) {
        return 0;
    }
    if (rows > BLAS_ROWS) {
        rows = BLAS_ROWS;
    }
    return p->tokens < rows ? p->tokens : rows;
}

static size_t blas_workspace(const Parameters *p) {
    size_t rows = blas_rows(p);
    if (!rows || p->outputs > (1 << 27) / p->width) {
        return 0;
    }
    return (rows * p->width + p->outputs * (p->width + rows)) * sizeof(float);
}

static size_t blas_reservation(const Parameters *p) {
    if (p->tokens < 64 || p->outputs < 1024 || p->width < 512) {
        return 0;
    }
    /* Bounds also cover smaller aligned widths and shapes below the work limit. */
    size_t rows = p->tokens < BLAS_ROWS ? p->tokens : BLAS_ROWS;
    size_t weights = p->outputs > (1 << 27) / p->width ? (1 << 27) : p->outputs * p->width;
    size_t width = p->width < (1 << 18) ? p->width : (1 << 18);
    size_t columns = p->outputs < (1 << 18) ? p->outputs : (1 << 18);
    size_t activation = rows * width < (1 << 23) ? rows * width : (1 << 23);
    return (weights + activation + rows * columns) * sizeof(float);
}

typedef struct {
    const Parameters *p;
    const TrillimKernelContext *context;
    const int8_t *codes;
    const float *scales;
    const uint8_t *weights, *weight_scales;
    void *output;
    float *activation, *expanded_weights, *accumulator;
    size_t row, rows;
    atomic_bool ordinary;
    bool compact;
    void *workspace;
    size_t workspace_size;
} BlasTask;

static bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static inline void widen8(float *output, int16x8_t value) {
    vst1q_f32(output, vcvtq_f32_s32(vmovl_s16(vget_low_s16(value))));
    vst1q_f32(output + 4, vcvtq_f32_s32(vmovl_high_s16(value)));
}

static TrillimStatus widen_activation(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    size_t width = job->p->width;
    const int8_t *codes = job->codes + (job->row + row) * width;
    float *output = job->activation + row * width;
    uint16_t *halves = (uint16_t *)job->activation + row * width;
    for (size_t k = 0; k < width; k += 16) {
        if ((k & 1023) == 0 && job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int8x16_t values = vld1q_s8(codes + k);
        if (job->compact) {
            int16x8_t a = vmovl_s8(vget_low_s8(values)), b = vmovl_high_s8(values);
            uint16x4_t c0 = vshrn_n_u32(vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a)))), 16);
            uint16x4_t c1 = vshrn_n_u32(vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_high_s16(a))), 16);
            uint16x4_t c2 = vshrn_n_u32(vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(b)))), 16);
            uint16x4_t c3 = vshrn_n_u32(vreinterpretq_u32_f32(vcvtq_f32_s32(vmovl_high_s16(b))), 16);
            vst1q_u16(halves + k, vcombine_u16(c0, c1));
            vst1q_u16(halves + k + 8, vcombine_u16(c2, c3));
        } else {
            widen8(output + k, vmovl_s8(vget_low_s8(values)));
            widen8(output + k + 8, vmovl_high_s8(values));
        }
    }
    return TRILLIM_OK;
}

static inline void store_weights(void *output, uint8x16_t codes, uint8x16_t low_table, uint8x16_t high_table, bool compact) {
    uint8x16_t low = vqtbl1q_u8(low_table, codes);
    uint8x16_t high = vqtbl1q_u8(high_table, codes);
    uint16x8_t a = vreinterpretq_u16_u8(vzip1q_u8(low, high));
    uint16x8_t b = vreinterpretq_u16_u8(vzip2q_u8(low, high));
    if (compact) {
        vst1q_u16(output, a);
        vst1q_u16((uint16_t *)output + 8, b);
    } else {
        vst1q_f32(output, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16)));
        vst1q_f32((float *)output + 4, vreinterpretq_f32_u32(vshll_high_n_u16(a, 16)));
        vst1q_f32((float *)output + 8, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16)));
        vst1q_f32((float *)output + 12, vreinterpretq_f32_u32(vshll_high_n_u16(b, 16)));
    }
}

static bool widen_weight_row(const BlasTask *job, size_t block, size_t groups, void *output) {
    uint8x16_t invalid = vdupq_n_u8(0);
    for (size_t g = 0; g < groups; ++g) {
        uint16_t bits;
        memcpy(&bits, job->weight_scales + (block + g) * 2, 2);
        uint16_t magnitude = bits & 0x7fff;
        // Keep scales normal with room for FP32 accumulation.
        if (magnitude && (magnitude < 0x1000 || magnitude > 0x6b80)) {
            return false;
        }
        uint32_t low = bits & 0xff;
        uint32_t high = bits >> 8;
        // The ternary codes select -scale, zero, or scale.
        uint8x16_t low_table = vreinterpretq_u8_u32(vdupq_n_u32(low | (low << 16)));
        uint8x16_t high_table = vreinterpretq_u8_u32(vdupq_n_u32((high ^ 0x80) | (high << 16)));
        const uint8_t *packed = job->weights + (block + g) * 32;
        for (size_t h = 0; h < 2; ++h) {
            uint8x16_t data = vld1q_u8(packed + h * 16);
            invalid = vorrq_u8(invalid, vandq_u8(data, vshrq_n_u8(data, 1)));
            size_t bytes = job->compact ? sizeof(uint16_t) : sizeof(float);
            uint8_t *destination = (uint8_t *)output + (g * LOWBIT_WIDTH + h * 16) * bytes;
            store_weights(destination, vshrq_n_u8(data, 6), low_table, high_table, job->compact);
            store_weights(destination + 32 * bytes, vandq_u8(vshrq_n_u8(data, 4), vdupq_n_u8(3)), low_table, high_table, job->compact);
            store_weights(destination + 64 * bytes, vandq_u8(vshrq_n_u8(data, 2), vdupq_n_u8(3)), low_table, high_table, job->compact);
            store_weights(destination + 96 * bytes, vandq_u8(data, vdupq_n_u8(3)), low_table, high_table, job->compact);
        }
    }
    return vmaxvq_u8(vandq_u8(invalid, vdupq_n_u8(0x55))) == 0;
}

static TrillimStatus prepare_weights(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    BlasTask *job = opaque;
    size_t width = job->p->width, groups = width / LOWBIT_WIDTH;
    size_t begin = tile * BLAS_DECODE_COLUMNS;
    size_t end = job->p->outputs - begin < BLAS_DECODE_COLUMNS ? job->p->outputs : begin + BLAS_DECODE_COLUMNS;
    if (!default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    for (size_t c = begin; c < end && atomic_load_explicit(&job->ordinary, memory_order_relaxed); ++c) {
        if (job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!widen_weight_row(job, c * groups, groups, (uint8_t *)job->expanded_weights + c * width * (job->compact ? sizeof(uint16_t) : sizeof(float)))) {
            atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
            break;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus finish_output(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const size_t columns = job->p->outputs;
    if (job->context->is_cancelled(job->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const float *values = job->accumulator + row * columns;
    size_t start = (job->row + row) * columns;
    float scale = job->scales[job->row + row];
    size_t c = 0;
    if (job->p->output_dtype == TRILLIM_FP16) {
        float32x4_t divisor = vdupq_n_f32(scale);
        for (; columns - c >= 8; c += 8) {
            if ((c & 1023) == 0 && job->context->is_cancelled(job->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a = vdivq_f32(vld1q_f32(values + c), divisor);
            float32x4_t b = vdivq_f32(vld1q_f32(values + c + 4), divisor);
            float16x8_t halves = vcvt_high_f16_f32(vcvt_f16_f32(a), b);
            vst1q_u16((uint16_t *)job->output + start + c, vreinterpretq_u16_f16(halves));
        }
    }
    for (; c < columns; ++c) {
        if ((c & 1023) == 0 && job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_store(job->output, start + c, job->p->output_dtype, (TrillimValue){.f = values[c] / scale});
    }
    return TRILLIM_OK;
}

static bool blas_thread_limit(size_t workers) {
    const char *limit = getenv("VECLIB_MAXIMUM_THREADS");
    if (limit == NULL || workers < 4 || workers > 32) {
        return false;
    }
    size_t threads = 0;
    for (; *limit != '\0'; ++limit) {
        if (*limit < '0' || *limit > '9') {
            return false;
        }
        threads = threads * 10 + (size_t)(*limit - '0');
        if (threads > workers) {
            return false;
        }
    }
    return threads >= 4;
}

typedef struct {
    const BlasTask *job;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    uint64_t deadline;
    size_t workers, arrived;
    bool declined, done;
} SharedBlas;

static TrillimStatus multiply_blas(const BlasTask *job) {
    const Parameters *p = job->p;
    const TrillimKernelContext *context = job->context;
    // Bound the work between cancellation checks inside a large projection.
    size_t columns = UINT64_C(16000000000) / (job->rows * p->width);
    columns = columns / 128 * 128;
    if (columns < 128) {
        columns = 128;
    }
    for (size_t first = 0; first < p->outputs; first += columns) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->outputs - first < columns ? p->outputs - first : columns;
        if (job->compact) {
            BNNSNDArrayDescriptor a = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {p->width, job->rows}, .data = job->activation, .data_type = BNNSDataTypeBFloat16};
            BNNSNDArrayDescriptor b = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {p->width, count}, .data = (uint16_t *)job->expanded_weights + first * p->width,
                .data_type = BNNSDataTypeBFloat16};
            BNNSNDArrayDescriptor c = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {count, job->rows}, .stride = {1, p->outputs},
                .data = job->accumulator + first, .data_type = BNNSDataTypeFloat32};
            BNNSFilterParameters options = {.n_threads = context->thread_pool->worker_count};
            ssize_t needed = matmul_workspace(false, true, 1.0f, &a, &b, &c, &options);
            if (needed < 0 || (size_t)needed > job->workspace_size ||
                matmul(false, true, 1.0f, &a, &b, &c, job->workspace, &options) != 0) {
                return TRILLIM_ERROR;
            }
        } else {
            sgemm(101, 111, 112, (int)job->rows, (int)count, (int)p->width, 1.0f,
                job->activation, (int)p->width, job->expanded_weights + first * p->width, (int)p->width,
                0.0f, job->accumulator + first, (int)p->outputs);
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus run_shared_blas(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    SharedBlas *shared = opaque;
    const TrillimKernelContext *context = shared->job->context;
    pthread_mutex_lock(&shared->mutex);
    uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (!now || now >= shared->deadline || (tile == 0 && (!default_fpcr() || (!shared->job->compact && get_threading() != 0)))) {
        shared->declined = true;
    }
    if (++shared->arrived == shared->workers || shared->declined) {
        pthread_cond_broadcast(&shared->changed);
    }
    // The pool may provide fewer concurrent workers than it advertises.
    while (!shared->declined && shared->arrived < shared->workers) {
        now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (!now || now >= shared->deadline || context->is_cancelled(context->cancellation_context)) {
            shared->declined = true;
        } else {
            const struct timespec remaining = {.tv_sec = 0, .tv_nsec = (long)(shared->deadline - now)};
            if (pthread_cond_timedwait_relative_np(&shared->changed, &shared->mutex, &remaining) != 0 &&
                shared->arrived < shared->workers) {
                shared->declined = true;
            }
        }
        if (shared->declined) {
            pthread_cond_broadcast(&shared->changed);
        }
    }
    if (shared->declined) {
        pthread_mutex_unlock(&shared->mutex);
        return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
    }
    if (tile != 0) {
        while (!shared->done) {
            pthread_cond_wait(&shared->changed, &shared->mutex);
        }
        pthread_mutex_unlock(&shared->mutex);
        return TRILLIM_OK;
    }
    pthread_mutex_unlock(&shared->mutex);
    const TrillimStatus status = multiply_blas(shared->job);
    pthread_mutex_lock(&shared->mutex);
    shared->done = true;
    pthread_cond_broadcast(&shared->changed);
    pthread_mutex_unlock(&shared->mutex);
    return status;
}

static TrillimStatus try_shared_blas(const BlasTask *job, bool *completed) {
    const TrillimThreadPool *pool = job->context->thread_pool;
    uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (!now) {
        *completed = false;
        return TRILLIM_OK;
    }
    SharedBlas shared = {
        .job = job, .mutex = PTHREAD_MUTEX_INITIALIZER, .changed = PTHREAD_COND_INITIALIZER,
        .deadline = now + UINT64_C(2000000), .workers = pool->worker_count,
    };
    const TrillimStatus status = pool->run(pool->context, pool->worker_count, run_shared_blas, &shared);
    pthread_cond_destroy(&shared.changed);
    pthread_mutex_destroy(&shared.mutex);
    *completed = status == TRILLIM_OK && shared.done;
    return status;
}

static TrillimStatus try_blas(
    const Parameters *p, const TrillimKernelContext *context, void *output, const void *weights,
    const int8_t *codes, const float *scales, float *scratch, bool compact, bool *handled
) {
    *handled = false;
    const size_t rows = blas_rows(p);
    const TrillimThreadPool *pool = context->thread_pool;
    if (!blas_workspace(p) || !initialize_blas() || !default_fpcr() ||
        (compact ? (matmul == NULL || pool->worker_count < 4 || pool->worker_count > 32) :
                   !blas_thread_limit(pool->worker_count))) {
        return TRILLIM_OK;
    }
    /* INT8 codes and BF16 weight scales are exact in the compact BF16 operands. */
    float *expanded = scratch + rows * p->width;
    size_t blocks = p->outputs * (p->width / LOWBIT_WIDTH);
    BlasTask job = {
        p, context, codes, scales, weights, (const uint8_t *)weights + blocks * 32,
        output, scratch, expanded, expanded + p->outputs * p->width,
        0, 0, ATOMIC_VAR_INIT(true), compact, NULL, 0,
    };
    if (compact) {
        job.expanded_weights = (float *)((uint16_t *)scratch + rows * p->width);
        job.accumulator = (float *)((uint16_t *)job.expanded_weights + p->outputs * p->width);
        uintptr_t end = (uintptr_t)(job.accumulator + rows * p->outputs);
        uintptr_t aligned = (end + 63) & ~(uintptr_t)63;
        uintptr_t limit = (uintptr_t)scratch + blas_workspace(p);
        if (aligned >= limit) {
            return TRILLIM_OK;
        }
        job.workspace = (void *)aligned;
        job.workspace_size = limit - aligned;
    }
    size_t tiles = p->outputs / BLAS_DECODE_COLUMNS + (p->outputs % BLAS_DECODE_COLUMNS != 0);
    TrillimStatus status = pool->run(pool->context, tiles, prepare_weights, &job);
    if (status != TRILLIM_OK || !atomic_load_explicit(&job.ordinary, memory_order_relaxed)) {
        return status;
    }
    while (job.row < p->tokens) {
        job.rows = p->tokens - job.row < rows ? p->tokens - job.row : rows;
        status = pool->run(pool->context, job.rows, widen_activation, &job);
        if (status != TRILLIM_OK) {
            return status;
        }
        bool completed;
        status = try_shared_blas(&job, &completed);
        if (status != TRILLIM_OK || !completed) {
            return compact && status == TRILLIM_ERROR ? TRILLIM_OK : status;
        }
        status = pool->run(pool->context, job.rows, finish_output, &job);
        if (status != TRILLIM_OK) {
            return status;
        }
        job.row += job.rows;
    }
    *handled = true;
    return TRILLIM_OK;
}
#endif

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    size_t internal_inputs[3];
    size_t itemsize = trillim_itemsize(p->input_dtype);
    TrillimStatus status = quantized_requirements(parameters, internal_inputs, outputs, scratch);
    if (status != TRILLIM_OK || p->weight_dtype != TRILLIM_GRP_TERNARY || !trillim_float(p->input_dtype)) {
        return TRILLIM_ERROR;
    }
    size_t count = internal_inputs[0];
    if (count > SIZE_MAX / itemsize || count > SIZE_MAX - 3 ||
        p->tokens > (SIZE_MAX - ((count + 3) & ~(size_t)3)) / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count * itemsize;
    inputs[1] = internal_inputs[1];
    *scratch = ((count + 3) & ~(size_t)3) + p->tokens * sizeof(float);
#if defined(__APPLE__)
    size_t workspace = blas_reservation(p);
    if (*scratch > SIZE_MAX - workspace) {
        return TRILLIM_ERROR;
    }
    *scratch += workspace;
#endif
    return TRILLIM_OK;
}

typedef struct {
    const void *input, *weight;
    void *output;
    const Parameters *p;
    const TrillimKernelContext *context;
} FloatTask;

static TrillimStatus floating_projection(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const FloatTask *task = opaque;
    const Parameters *p = task->p;
    const size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    const size_t row = index / tiles, first = index % tiles * 16;
    const size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
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
        TrillimValue value = {.f = sum};
        trillim_store(task->output, row * p->outputs + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_auto_ternary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t count = p->tokens * p->width;
    int8_t *codes = scratch;
    float *scales = (float *)((uint8_t *)scratch + ((count + 3) & ~(size_t)3));
    atomic_bool ordinary = ATOMIC_VAR_INIT(true);
    QuantizeTask task = {codes, scales, inputs[0], p, context, &ordinary};
    status = p->tokens == 1 ? quantize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, quantize, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!atomic_load_explicit(&ordinary, memory_order_relaxed)) {
        FloatTask fallback = {inputs[0], inputs[1], outputs[0], p, context};
        size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
        return context->thread_pool->run(context->thread_pool->context,
            p->tokens * tiles, floating_projection, &fallback);
    }
#if defined(__APPLE__)
    bool handled;
    status = try_blas(p, context, outputs[0], inputs[1], codes, scales, scales + p->tokens, true, &handled);
    if (status != TRILLIM_OK || handled) {
        return status;
    }
    status = try_blas(p, context, outputs[0], inputs[1], codes, scales, scales + p->tokens, false, &handled);
    if (status != TRILLIM_OK || handled) {
        return status;
    }
#endif
    const void *quantized_inputs[] = {codes, inputs[1], scales};
    return run_quantized(outputs, quantized_inputs, NULL, parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_auto_ternary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if weight.dtype != 'grp_ternary' or x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires grouped ternary weights and floating activations')\n"
           "    for tensor in (x, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete packed matrix')\n"
           "    p = _linear(context)\n"
           "    return dict(tokens=p['m'], width=p['k'], outputs=p['n'], weight_dtype=p['b_dtype'], output_dtype=p['out_dtype'], input_dtype=p['a_dtype'])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased floating linear projection')\n"
           "    result = Linear(x, weight)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_auto_ternary_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_auto_ternary_linear_capabilities(void) {
#if defined(__APPLE__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
