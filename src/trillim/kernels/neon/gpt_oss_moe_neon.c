#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

/* Router projection. */

#if defined(__FLT16_MANT_DIG__)
__extension__ typedef _Float16 TrillimFloat16;
#endif

#define BLOCK_M 64
#define BLOCK_N 128
#define BLOCK_K 1024
#define LANES 32
#define WORKSPACE (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N + BLOCK_M * BLOCK_N)

static bool initialize_blas(void);

typedef struct {
    size_t batches;
    size_t m;
    size_t n;
    size_t k;
    size_t rank;
    const size_t *batch_shape;
    const size_t *a_batch_strides;
    const size_t *b_batch_strides;
    size_t a_row_stride;
    size_t a_col_stride;
    size_t b_row_stride;
    size_t b_col_stride;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t out_dtype;
    uint32_t linear_dtype;
    uint32_t bias_dtype;
    size_t a_storage_rows;
    size_t a_storage_columns;
    size_t b_storage_rows;
    size_t b_storage_columns;
} DenseBiasParameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const void *bias;
    const DenseBiasParameters *p;
    const TrillimKernelContext *context;
    float *packed;
} DenseBiasTask;

static bool packed_fp32(const DenseBiasParameters *p) {
    return p->batches == 1 && p->rank == 0 && p->m >= 16 && p->n >= 16 && p->k >= 32 &&
        p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32 &&
        p->linear_dtype == TRILLIM_FP32 && p->bias_dtype == TRILLIM_FP32 && p->out_dtype == TRILLIM_FP32;
}

static bool precise(const DenseBiasParameters *p) {
    return p->out_dtype != TRILLIM_FP32 || p->linear_dtype != TRILLIM_FP32;
}

static size_t block_n(const DenseBiasParameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t block_k(const DenseBiasParameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t block_m(const DenseBiasParameters *p) {
    bool direct_f32 = p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32 &&
        p->out_dtype == TRILLIM_FP32 && p->m >= 1024 && p->n >= 512 && p->n <= 2048 &&
        ((p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) ||
         (p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX));
    if (direct_f32) {
        return 256;
    }
    /* Direct A/C access permits taller tiles without larger packing buffers. */
    bool direct = p->a_dtype == TRILLIM_FP32 && p->out_dtype == TRILLIM_FP32 &&
        p->b_dtype != TRILLIM_FP32 && p->n >= 2560 &&
        ((p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) ||
         (p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX));
    return direct && p->m >= 128 ? 256 : BLOCK_M;
}

static bool blas_shape(const DenseBiasParameters *p, size_t workers) {
    if (precise(p) || p->n > INT32_MAX || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool contiguous = p->b_col_stride == 1 && p->k >= 256 && p->n > 32767 / p->k;
    bool a_direct = (p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) ||
        (p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX);
    bool b_direct = (p->b_col_stride == 1 && p->b_row_stride >= p->n && p->b_row_stride <= INT32_MAX) ||
        (p->b_row_stride == 1 && p->b_col_stride >= p->k && p->b_col_stride <= INT32_MAX);
    bool column_major = p->a_row_stride == 1 && p->a_col_stride != 1;
    bool small_direct = column_major ? p->m >= 8 && (p->m >= 32 || workers <= 8) :
        (p->m >= 8 && workers == 1) || (p->m >= 16 && workers <= 8);
    bool singleton_direct = p->m == 1 && workers == 1 && p->n >= 1024 && p->k >= 1024;
    bool direct = p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32 && a_direct && b_direct &&
        (small_direct || singleton_direct) && p->k >= 256 && p->n > 262143 / p->k;
    return large || medium || vocabulary || contiguous || direct;
}

static bool reserve_blas(const DenseBiasParameters *p) {
    if (precise(p) || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool possibly_contiguous = p->k >= 256 && p->n > 32767 / p->k;
    return large || medium || vocabulary || possibly_contiguous;
}

static bool use_blas(const DenseBiasParameters *p, const TrillimKernelContext *context) {
    return blas_shape(p, context->thread_pool->worker_count) && initialize_blas();
}

static bool matrix_span(size_t base, size_t rows, size_t columns,
                        size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (base >= capacity || (row_stride != 0 && rows - 1 > (capacity - 1 - base) / row_stride)) {
        return false;
    }
    base += (rows - 1) * row_stride;
    return col_stride == 0 || columns - 1 <= (capacity - 1 - base) / col_stride;
}

static TrillimStatus dense_bias_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const DenseBiasParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        !trillim_float(p->out_dtype) || !trillim_float(p->linear_dtype) ||
        !trillim_float(p->bias_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    size_t a_span, b_span;
    if (!trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span) ||
        p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / trillim_itemsize(p->out_dtype)) {
        return TRILLIM_ERROR;
    }
    size_t batches = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->batch_shape[axis] != 0 && batches > SIZE_MAX / p->batch_shape[axis]) {
            return TRILLIM_ERROR;
        }
        batches *= p->batch_shape[axis];
    }
    if (batches != p->batches) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if ((p->a_storage_columns != 0 && p->a_storage_rows > SIZE_MAX / trillim_itemsize(p->a_dtype) / p->a_storage_columns) ||
            (p->b_storage_columns != 0 && p->b_storage_rows > SIZE_MAX / trillim_itemsize(p->b_dtype) / p->b_storage_columns)) {
            return TRILLIM_ERROR;
        }
        size_t a_count = p->a_storage_rows * p->a_storage_columns;
        size_t b_count = p->b_storage_rows * p->b_storage_columns;
        if (!matrix_span(a_span ? a_span - 1 : 0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !matrix_span(b_span ? b_span - 1 : 0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = a_count * trillim_itemsize(p->a_dtype);
        inputs[1] = b_count * trillim_itemsize(p->b_dtype);
    }
    if (p->n > SIZE_MAX / trillim_itemsize(p->bias_dtype)) {
        return TRILLIM_ERROR;
    }
    inputs[2] = p->n * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    if (packed_fp32(p)) {
        if (p->n > SIZE_MAX - 7 || p->k > SIZE_MAX / sizeof(float) / ((p->n + 7) / 8) / 8) {
            return TRILLIM_ERROR;
        }
        *scratch = ((p->n + 7) / 8) * p->k * 8 * sizeof(float);
    }
    /* Reserve optional-library workspace independently of host availability. */
    if (reserve_blas(p)) {
        (void)initialize_blas();
        size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t count = p->batches * rows * columns;
        size_t workspace = (count < LANES ? count : LANES) * WORKSPACE * sizeof(float);
        if (workspace > *scratch) {
            *scratch = workspace;
        }
    }
    return TRILLIM_OK;
}

static inline float32x4_t dense_bias_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline float add_bias(const DenseBiasTask *t, size_t column, float value) {
    if (t->p->linear_dtype == TRILLIM_FP16) {
        value = trillim_fp16(trillim_to_fp16(value));
    } else if (t->p->linear_dtype == TRILLIM_BF16) {
        value = trillim_bf16(trillim_to_bf16(value));
    }
    return value + trillim_load(t->bias, column, t->p->bias_dtype).f;
}

static inline void store(const DenseBiasTask *t, size_t index, float value) {
    TrillimValue result = {.f = add_bias(t, index % t->p->n, value)};
    trillim_store(t->output, index, t->p->out_dtype, result);
}

static inline bool ordered_narrow_shape(const DenseBiasParameters *p) {
    return p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32 &&
        p->n % 8 == 0 && p->k >= 4 && p->k % 4 == 0;
}

static size_t tile_rows(const DenseBiasParameters *p, size_t workers) {
    if (precise(p)) {
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t tiles = p->batches * (p->m / 8) * columns;
        return p->m >= 8 && tiles >= workers && ordered_narrow_shape(p) ? 8 : 4;
    }
    return p->m >= 12 && p->a_col_stride == 1 && p->b_row_stride == 1 ? 6 : 4;
}

static size_t tile_columns(const DenseBiasParameters *p, size_t workers) {
    if (p->m == 1 && precise(p) && p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->k >= 256 && p->n >= 256 && p->batches * (p->n / 64) < workers) {
        return p->batches * (p->n / 32) >= workers ? 32 : 16;
    }
    return 64;
}

static inline float32x4_t ordered_widen(uint16x4_t value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(value));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(value, 16));
}

static inline __attribute__((always_inline)) float32x4x4_t ordered_narrow_weights(
    const uint16_t *weights, size_t stride, uint32_t dtype
) {
    uint16x4x2_t first = vzip_u16(vld1_u16(weights), vld1_u16(weights + stride));
    uint16x4x2_t second = vzip_u16(vld1_u16(weights + 2 * stride), vld1_u16(weights + 3 * stride));
    uint32x2x2_t low = vzip_u32(vreinterpret_u32_u16(first.val[0]), vreinterpret_u32_u16(second.val[0]));
    uint32x2x2_t high = vzip_u32(vreinterpret_u32_u16(first.val[1]), vreinterpret_u32_u16(second.val[1]));
    return (float32x4x4_t){{
        ordered_widen(vreinterpret_u16_u32(low.val[0]), dtype),
        ordered_widen(vreinterpret_u16_u32(low.val[1]), dtype),
        ordered_widen(vreinterpret_u16_u32(high.val[0]), dtype),
        ordered_widen(vreinterpret_u16_u32(high.val[1]), dtype),
    }};
}

static inline __attribute__((always_inline)) TrillimStatus ordered_narrow_block(
    const DenseBiasTask *t, size_t a_base, size_t b_base, size_t output,
    size_t first, size_t last, uint32_t dtype, size_t rows
) {
    const DenseBiasParameters *p = t->p;
    const uint16_t *input = (const uint16_t *)t->a + a_base;
    const uint16_t *weights = (const uint16_t *)t->b + b_base;
    size_t input_stride = p->a_row_stride;
    size_t weight_stride = p->b_col_stride;
    size_t depth = p->k;
    for (size_t column = first; column < last; column += 8) {
        float32x4_t sums[8][2] = {0};
        const uint16_t *left = weights + column * weight_stride;
        const uint16_t *right = left + 4 * weight_stride;
        /* Each lane accumulates its dot product in depth order. */
        for (size_t k = 0; k < depth; k += 4) {
            if ((k & 255) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4x4_t w0 = ordered_narrow_weights(left + k, weight_stride, dtype);
            float32x4x4_t w1 = ordered_narrow_weights(right + k, weight_stride, dtype);
            for (size_t row = 0; row < rows; ++row) {
                float32x4_t x = ordered_widen(vld1_u16(input + row * input_stride + k), dtype);
                sums[row][0] = vfmaq_laneq_f32(sums[row][0], w0.val[0], x, 0);
                sums[row][1] = vfmaq_laneq_f32(sums[row][1], w1.val[0], x, 0);
                sums[row][0] = vfmaq_laneq_f32(sums[row][0], w0.val[1], x, 1);
                sums[row][1] = vfmaq_laneq_f32(sums[row][1], w1.val[1], x, 1);
                sums[row][0] = vfmaq_laneq_f32(sums[row][0], w0.val[2], x, 2);
                sums[row][1] = vfmaq_laneq_f32(sums[row][1], w1.val[2], x, 2);
                sums[row][0] = vfmaq_laneq_f32(sums[row][0], w0.val[3], x, 3);
                sums[row][1] = vfmaq_laneq_f32(sums[row][1], w1.val[3], x, 3);
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float values[8];
            vst1q_f32(values, sums[row][0]);
            vst1q_f32(values + 4, sums[row][1]);
            size_t index = output + row * p->n + column;
            for (size_t lane = 0; lane < 8; ++lane) {
                store(t, index + lane, values[lane]);
            }
        }
    }
    return TRILLIM_OK;
}

static inline TrillimStatus ordered_narrow_tile(
    const DenseBiasTask *t, size_t a_base, size_t b_base, size_t output,
    size_t first, size_t last, size_t rows
) {
    if (t->p->a_dtype == TRILLIM_BF16) {
        return rows == 8 ?
            ordered_narrow_block(t, a_base, b_base, output, first, last, TRILLIM_BF16, 8) :
            ordered_narrow_block(t, a_base, b_base, output, first, last, TRILLIM_BF16, 4);
    }
    return rows == 8 ?
        ordered_narrow_block(t, a_base, b_base, output, first, last, TRILLIM_FP16, 8) :
        ordered_narrow_block(t, a_base, b_base, output, first, last, TRILLIM_FP16, 4);
}

typedef struct {
    float32x4_t low;
    float32x4_t high;
} VectorPair;

static inline VectorPair load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        const float *p = (const float *)data + index;
        VectorPair value = {vld1q_f32(p), vld1q_f32(p + 4)};
        return value;
    }
    uint16x8_t bits = vld1q_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        VectorPair value = {vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bits))),
                            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bits)))};
        return value;
    }
    VectorPair value = {vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16)),
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16))};
    return value;
}

static inline __attribute__((always_inline)) TrillimStatus decode_typed_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseBiasParameters *p = t->p;
    const void *activation = t->a;
    const void *weights = t->b;
    size_t depth = p->k;
    size_t stride = p->b_col_stride;
    for (size_t column = first; column < last; column += 2) {
        size_t columns = last - column < 2 ? last - column : 2;
        size_t w0 = b_base + column * stride;
        size_t w1 = columns == 2 ? w0 + stride : w0;
        float32x4_t s00 = vdupq_n_f32(0);
        float32x4_t s01 = vdupq_n_f32(0);
        float32x4_t s02 = vdupq_n_f32(0);
        float32x4_t s03 = vdupq_n_f32(0);
        float32x4_t s10 = vdupq_n_f32(0);
        float32x4_t s11 = vdupq_n_f32(0);
        float32x4_t s12 = vdupq_n_f32(0);
        float32x4_t s13 = vdupq_n_f32(0);
        size_t k = 0;
        for (; depth - k >= 16; k += 16) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            VectorPair x0 = load8(activation, a_base + k, a_dtype);
            VectorPair x1 = load8(activation, a_base + k + 8, a_dtype);
            VectorPair w0_0 = load8(weights, w0 + k, b_dtype);
            s00 = vfmaq_f32(s00, x0.low, w0_0.low);
            s01 = vfmaq_f32(s01, x0.high, w0_0.high);
            VectorPair w0_1 = load8(weights, w0 + k + 8, b_dtype);
            s02 = vfmaq_f32(s02, x1.low, w0_1.low);
            s03 = vfmaq_f32(s03, x1.high, w0_1.high);
            VectorPair w1_0 = load8(weights, w1 + k, b_dtype);
            s10 = vfmaq_f32(s10, x0.low, w1_0.low);
            s11 = vfmaq_f32(s11, x0.high, w1_0.high);
            VectorPair w1_1 = load8(weights, w1 + k + 8, b_dtype);
            s12 = vfmaq_f32(s12, x1.low, w1_1.low);
            s13 = vfmaq_f32(s13, x1.high, w1_1.high);
        }
        float sums[2] = {vaddvq_f32(vaddq_f32(vaddq_f32(s00, s01), vaddq_f32(s02, s03))), vaddvq_f32(vaddq_f32(vaddq_f32(s10, s11), vaddq_f32(s12, s13)))};
        for (; k < depth; ++k) {
            float x = trillim_load(activation, a_base + k, a_dtype).f;
            for (size_t c = 0; c < columns; ++c) {
                float w = trillim_load(weights, w0 + c * stride + k, b_dtype).f;
                sums[c] = fmaf(x, w, sums[c]);
            }
        }
        for (size_t c = 0; c < columns; ++c) {
            store(t, output + column + c, sums[c]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus decode_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_BF16) {
        return decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_BF16);
    }
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32) {
        return decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_FP32);
    }
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP16) {
        return decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_FP16);
    }
    return decode_typed_tile(t, a_base, b_base, output, first, last, p->a_dtype, p->b_dtype);
}

static inline __attribute__((always_inline)) float32x4x4_t ordered_weights(const DenseBiasTask *t, size_t base, size_t column, size_t k, uint32_t dtype) {
    size_t stride = t->p->b_col_stride;
    float32x4_t w0 = dense_bias_load4(t->b, base + column * stride + k, dtype);
    float32x4_t w1 = dense_bias_load4(t->b, base + (column + 1) * stride + k, dtype);
    float32x4_t w2 = dense_bias_load4(t->b, base + (column + 2) * stride + k, dtype);
    float32x4_t w3 = dense_bias_load4(t->b, base + (column + 3) * stride + k, dtype);
    float32x4x2_t even = vtrnq_f32(w0, w1);
    float32x4x2_t odd = vtrnq_f32(w2, w3);
    float32x4x4_t result = {{
        vcombine_f32(vget_low_f32(even.val[0]), vget_low_f32(odd.val[0])),
        vcombine_f32(vget_low_f32(even.val[1]), vget_low_f32(odd.val[1])),
        vcombine_f32(vget_high_f32(even.val[0]), vget_high_f32(odd.val[0])),
        vcombine_f32(vget_high_f32(even.val[1]), vget_high_f32(odd.val[1]))
    }};
    return result;
}

static inline __attribute__((always_inline)) TrillimStatus ordered_decode16_typed(const DenseBiasTask *t, size_t a_base, size_t b_base, size_t output, size_t column, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseBiasParameters *p = t->p;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t k = 0;
    for (; p->k - k >= 4; k += 4) {
        if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t x = dense_bias_load4(t->a, a_base + k, a_dtype);
        float32x4x4_t w0 = ordered_weights(t, b_base, column + 0, k, b_dtype);
        float32x4x4_t w1 = ordered_weights(t, b_base, column + 4, k, b_dtype);
        float32x4x4_t w2 = ordered_weights(t, b_base, column + 8, k, b_dtype);
        float32x4x4_t w3 = ordered_weights(t, b_base, column + 12, k, b_dtype);
        s0 = vfmaq_laneq_f32(s0, w0.val[0], x, 0);
        s1 = vfmaq_laneq_f32(s1, w1.val[0], x, 0);
        s2 = vfmaq_laneq_f32(s2, w2.val[0], x, 0);
        s3 = vfmaq_laneq_f32(s3, w3.val[0], x, 0);
        s0 = vfmaq_laneq_f32(s0, w0.val[1], x, 1);
        s1 = vfmaq_laneq_f32(s1, w1.val[1], x, 1);
        s2 = vfmaq_laneq_f32(s2, w2.val[1], x, 1);
        s3 = vfmaq_laneq_f32(s3, w3.val[1], x, 1);
        s0 = vfmaq_laneq_f32(s0, w0.val[2], x, 2);
        s1 = vfmaq_laneq_f32(s1, w1.val[2], x, 2);
        s2 = vfmaq_laneq_f32(s2, w2.val[2], x, 2);
        s3 = vfmaq_laneq_f32(s3, w3.val[2], x, 2);
        s0 = vfmaq_laneq_f32(s0, w0.val[3], x, 3);
        s1 = vfmaq_laneq_f32(s1, w1.val[3], x, 3);
        s2 = vfmaq_laneq_f32(s2, w2.val[3], x, 3);
        s3 = vfmaq_laneq_f32(s3, w3.val[3], x, 3);
    }
    float32x4_t sums[4] = {s0, s1, s2, s3};
    for (; k < p->k; ++k) {
        float x = trillim_load(t->a, a_base + k, a_dtype).f;
        for (size_t group = 0; group < 4; ++group) {
            float weights[4];
            for (size_t lane = 0; lane < 4; ++lane) {
                weights[lane] = trillim_load(t->b, b_base + (column + group * 4 + lane) * p->b_col_stride + k, b_dtype).f;
            }
            sums[group] = vfmaq_n_f32(sums[group], vld1q_f32(weights), x);
        }
    }
    for (size_t group = 0; group < 4; ++group) {
        float values[4];
        vst1q_f32(values, sums[group]);
        for (size_t lane = 0; lane < 4; ++lane) {
            store(t, output + column + group * 4 + lane, values[lane]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus ordered_decode16(const DenseBiasTask *t, size_t a_base, size_t b_base, size_t output, size_t column) {
    const DenseBiasParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP16 && p->b_dtype == TRILLIM_FP16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_FP16, TRILLIM_FP16);
    }
    if (p->a_dtype == TRILLIM_BF16 && p->b_dtype == TRILLIM_BF16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_BF16, TRILLIM_BF16);
    }
    return ordered_decode16_typed(t, a_base, b_base, output, column, p->a_dtype, p->b_dtype);
}

static TrillimStatus ordered_dot_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                                      size_t output, size_t rows, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    if (rows >= 4 && ordered_narrow_shape(p)) {
        size_t full_rows = rows == 8 ? 8 : 4;
        TrillimStatus status = ordered_narrow_tile(t, a_base, b_base, output, first, last, full_rows);
        if (status != TRILLIM_OK || rows == full_rows) {
            return status;
        }
        a_base += full_rows * p->a_row_stride;
        output += full_rows * p->n;
        rows -= full_rows;
    }
    if (rows == 1) {
        for (; last - first >= 16; first += 16) {
            TrillimStatus status = ordered_decode16(t, a_base, b_base, output, first);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    }
    for (size_t column = first; column < last; column += 4) {
        size_t columns = last - column < 4 ? last - column : 4;
        size_t b0 = b_base + column * p->b_col_stride;
        size_t b1 = columns > 1 ? b0 + p->b_col_stride : b0;
        size_t b2 = columns > 2 ? b1 + p->b_col_stride : b0;
        size_t b3 = columns > 3 ? b2 + p->b_col_stride : b0;
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
        size_t k = 0;
        for (; p->k - k >= 4; k += 4) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t w0 = dense_bias_load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = dense_bias_load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = dense_bias_load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = dense_bias_load4(t->b, b3 + k, p->b_dtype);
            float32x4x2_t even = vtrnq_f32(w0, w1);
            float32x4x2_t odd = vtrnq_f32(w2, w3);
            float32x4_t c0 = vcombine_f32(vget_low_f32(even.val[0]), vget_low_f32(odd.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(even.val[1]), vget_low_f32(odd.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(even.val[0]), vget_high_f32(odd.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(even.val[1]), vget_high_f32(odd.val[1]));
            float32x4_t x0 = dense_bias_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0 = vfmaq_laneq_f32(s0, c0, x0, 0);
            s0 = vfmaq_laneq_f32(s0, c1, x0, 1);
            s0 = vfmaq_laneq_f32(s0, c2, x0, 2);
            s0 = vfmaq_laneq_f32(s0, c3, x0, 3);
            if (rows > 1) {
                float32x4_t x1 = dense_bias_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1 = vfmaq_laneq_f32(s1, c0, x1, 0);
                s1 = vfmaq_laneq_f32(s1, c1, x1, 1);
                s1 = vfmaq_laneq_f32(s1, c2, x1, 2);
                s1 = vfmaq_laneq_f32(s1, c3, x1, 3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_bias_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2 = vfmaq_laneq_f32(s2, c0, x2, 0);
                s2 = vfmaq_laneq_f32(s2, c1, x2, 1);
                s2 = vfmaq_laneq_f32(s2, c2, x2, 2);
                s2 = vfmaq_laneq_f32(s2, c3, x2, 3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_bias_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                s3 = vfmaq_laneq_f32(s3, c0, x3, 0);
                s3 = vfmaq_laneq_f32(s3, c1, x3, 1);
                s3 = vfmaq_laneq_f32(s3, c2, x3, 2);
                s3 = vfmaq_laneq_f32(s3, c3, x3, 3);
            }
        }
        for (; k < p->k; ++k) {
            float weights[4] = {0};
            for (size_t c = 0; c < columns; ++c) {
                weights[c] = trillim_load(t->b, b_base + (column + c) * p->b_col_stride + k, p->b_dtype).f;
            }
            float32x4_t w = vld1q_f32(weights);
            float x0 = trillim_load(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype).f;
            s0 = vfmaq_n_f32(s0, w, x0);
            if (rows > 1) {
                float x1 = trillim_load(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype).f;
                s1 = vfmaq_n_f32(s1, w, x1);
            }
            if (rows > 2) {
                float x2 = trillim_load(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype).f;
                s2 = vfmaq_n_f32(s2, w, x2);
            }
            if (rows > 3) {
                float x3 = trillim_load(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype).f;
                s3 = vfmaq_n_f32(s3, w, x3);
            }
        }
        float sums[4][4];
        vst1q_f32(sums[0], s0);
        vst1q_f32(sums[1], s1);
        vst1q_f32(sums[2], s2);
        vst1q_f32(sums[3], s3);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t c = 0; c < columns; ++c) {
                store(t, output + row * p->n + column + c, sums[row][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dot_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    for (size_t column = first; column < last; column += 4) {
        size_t columns = last - column < 4 ? last - column : 4;
        size_t b0 = b_base + column * p->b_col_stride;
        size_t b1 = columns > 1 ? b0 + p->b_col_stride : b0;
        size_t b2 = columns > 2 ? b1 + p->b_col_stride : b0;
        size_t b3 = columns > 3 ? b2 + p->b_col_stride : b0;
        float32x4_t s00 = vdupq_n_f32(0), s01 = vdupq_n_f32(0), s02 = vdupq_n_f32(0), s03 = vdupq_n_f32(0);
        float32x4_t s10 = vdupq_n_f32(0), s11 = vdupq_n_f32(0), s12 = vdupq_n_f32(0), s13 = vdupq_n_f32(0);
        float32x4_t s20 = vdupq_n_f32(0), s21 = vdupq_n_f32(0), s22 = vdupq_n_f32(0), s23 = vdupq_n_f32(0);
        float32x4_t s30 = vdupq_n_f32(0), s31 = vdupq_n_f32(0), s32 = vdupq_n_f32(0), s33 = vdupq_n_f32(0);
        size_t k = 0;
        for (; p->k - k >= 4; k += 4) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t w0 = dense_bias_load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = dense_bias_load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = dense_bias_load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = dense_bias_load4(t->b, b3 + k, p->b_dtype);
            float32x4_t x0 = dense_bias_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s00 = vfmaq_f32(s00, x0, w0);
            s01 = vfmaq_f32(s01, x0, w1);
            s02 = vfmaq_f32(s02, x0, w2);
            s03 = vfmaq_f32(s03, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = dense_bias_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s10 = vfmaq_f32(s10, x1, w0);
                s11 = vfmaq_f32(s11, x1, w1);
                s12 = vfmaq_f32(s12, x1, w2);
                s13 = vfmaq_f32(s13, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_bias_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s20 = vfmaq_f32(s20, x2, w0);
                s21 = vfmaq_f32(s21, x2, w1);
                s22 = vfmaq_f32(s22, x2, w2);
                s23 = vfmaq_f32(s23, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_bias_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                s30 = vfmaq_f32(s30, x3, w0);
                s31 = vfmaq_f32(s31, x3, w1);
                s32 = vfmaq_f32(s32, x3, w2);
                s33 = vfmaq_f32(s33, x3, w3);
            }
        }
        float sums[4][4] = {
            {vaddvq_f32(s00), vaddvq_f32(s01), vaddvq_f32(s02), vaddvq_f32(s03)},
            {vaddvq_f32(s10), vaddvq_f32(s11), vaddvq_f32(s12), vaddvq_f32(s13)},
            {vaddvq_f32(s20), vaddvq_f32(s21), vaddvq_f32(s22), vaddvq_f32(s23)},
            {vaddvq_f32(s30), vaddvq_f32(s31), vaddvq_f32(s32), vaddvq_f32(s33)},
        };
        for (; k < p->k; ++k) {
            for (size_t r = 0; r < rows; ++r) {
                float a = trillim_load(t->a, a_base + r * p->a_row_stride + k, p->a_dtype).f;
                for (size_t c = 0; c < columns; ++c) {
                    float b = trillim_load(t->b, b_base + (column + c) * p->b_col_stride + k, p->b_dtype).f;
                    sums[r][c] = fmaf(a, b, sums[r][c]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                store(t, output + r * p->n + column + c, sums[r][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dot_tile6(const DenseBiasTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    for (size_t column = first; column < last; column += 4) {
        size_t columns = last - column < 4 ? last - column : 4;
        float32x4_t s0_0 = vdupq_n_f32(0);
        float32x4_t s0_1 = vdupq_n_f32(0);
        float32x4_t s0_2 = vdupq_n_f32(0);
        float32x4_t s0_3 = vdupq_n_f32(0);
        float32x4_t s1_0 = vdupq_n_f32(0);
        float32x4_t s1_1 = vdupq_n_f32(0);
        float32x4_t s1_2 = vdupq_n_f32(0);
        float32x4_t s1_3 = vdupq_n_f32(0);
        float32x4_t s2_0 = vdupq_n_f32(0);
        float32x4_t s2_1 = vdupq_n_f32(0);
        float32x4_t s2_2 = vdupq_n_f32(0);
        float32x4_t s2_3 = vdupq_n_f32(0);
        float32x4_t s3_0 = vdupq_n_f32(0);
        float32x4_t s3_1 = vdupq_n_f32(0);
        float32x4_t s3_2 = vdupq_n_f32(0);
        float32x4_t s3_3 = vdupq_n_f32(0);
        float32x4_t s4_0 = vdupq_n_f32(0);
        float32x4_t s4_1 = vdupq_n_f32(0);
        float32x4_t s4_2 = vdupq_n_f32(0);
        float32x4_t s4_3 = vdupq_n_f32(0);
        float32x4_t s5_0 = vdupq_n_f32(0);
        float32x4_t s5_1 = vdupq_n_f32(0);
        float32x4_t s5_2 = vdupq_n_f32(0);
        float32x4_t s5_3 = vdupq_n_f32(0);
        size_t k = 0;
        for (; p->k - k >= 4; k += 4) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t w0 = dense_bias_load4(t->b, b_base + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w1 = dense_bias_load4(t->b, b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w2 = dense_bias_load4(t->b, b_base + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w3 = dense_bias_load4(t->b, b_base + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t x0 = dense_bias_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0_0 = vfmaq_f32(s0_0, x0, w0);
            s0_1 = vfmaq_f32(s0_1, x0, w1);
            s0_2 = vfmaq_f32(s0_2, x0, w2);
            s0_3 = vfmaq_f32(s0_3, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = dense_bias_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1_0 = vfmaq_f32(s1_0, x1, w0);
                s1_1 = vfmaq_f32(s1_1, x1, w1);
                s1_2 = vfmaq_f32(s1_2, x1, w2);
                s1_3 = vfmaq_f32(s1_3, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_bias_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2_0 = vfmaq_f32(s2_0, x2, w0);
                s2_1 = vfmaq_f32(s2_1, x2, w1);
                s2_2 = vfmaq_f32(s2_2, x2, w2);
                s2_3 = vfmaq_f32(s2_3, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_bias_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                s3_0 = vfmaq_f32(s3_0, x3, w0);
                s3_1 = vfmaq_f32(s3_1, x3, w1);
                s3_2 = vfmaq_f32(s3_2, x3, w2);
                s3_3 = vfmaq_f32(s3_3, x3, w3);
            }
            if (rows > 4) {
                float32x4_t x4 = dense_bias_load4(t->a, a_base + 4 * p->a_row_stride + k, p->a_dtype);
                s4_0 = vfmaq_f32(s4_0, x4, w0);
                s4_1 = vfmaq_f32(s4_1, x4, w1);
                s4_2 = vfmaq_f32(s4_2, x4, w2);
                s4_3 = vfmaq_f32(s4_3, x4, w3);
            }
            if (rows > 5) {
                float32x4_t x5 = dense_bias_load4(t->a, a_base + 5 * p->a_row_stride + k, p->a_dtype);
                s5_0 = vfmaq_f32(s5_0, x5, w0);
                s5_1 = vfmaq_f32(s5_1, x5, w1);
                s5_2 = vfmaq_f32(s5_2, x5, w2);
                s5_3 = vfmaq_f32(s5_3, x5, w3);
            }
        }
        float sums[6][4] = {
            {vaddvq_f32(s0_0), vaddvq_f32(s0_1), vaddvq_f32(s0_2), vaddvq_f32(s0_3)},
            {vaddvq_f32(s1_0), vaddvq_f32(s1_1), vaddvq_f32(s1_2), vaddvq_f32(s1_3)},
            {vaddvq_f32(s2_0), vaddvq_f32(s2_1), vaddvq_f32(s2_2), vaddvq_f32(s2_3)},
            {vaddvq_f32(s3_0), vaddvq_f32(s3_1), vaddvq_f32(s3_2), vaddvq_f32(s3_3)},
            {vaddvq_f32(s4_0), vaddvq_f32(s4_1), vaddvq_f32(s4_2), vaddvq_f32(s4_3)},
            {vaddvq_f32(s5_0), vaddvq_f32(s5_1), vaddvq_f32(s5_2), vaddvq_f32(s5_3)},
        };
        for (; k < p->k; ++k) {
            for (size_t r = 0; r < rows; ++r) {
                float a = trillim_load(t->a, a_base + r * p->a_row_stride + k, p->a_dtype).f;
                for (size_t c = 0; c < columns; ++c) {
                    float b = trillim_load(t->b, b_base + (column + c) * p->b_col_stride + k, p->b_dtype).f;
                    sums[r][c] = fmaf(a, b, sums[r][c]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                store(t, output + r * p->n + column + c, sums[r][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus outer_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                                size_t output, size_t rows, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    size_t column = first;
    for (; last - column >= 8; column += 8) {
        float32x4_t s00 = vdupq_n_f32(0), s01 = vdupq_n_f32(0);
        float32x4_t s10 = vdupq_n_f32(0), s11 = vdupq_n_f32(0);
        float32x4_t s20 = vdupq_n_f32(0), s21 = vdupq_n_f32(0);
        float32x4_t s30 = vdupq_n_f32(0), s31 = vdupq_n_f32(0);
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 255) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t b_index = b_base + k * p->b_row_stride + column;
            float32x4_t w0 = dense_bias_load4(t->b, b_index, p->b_dtype);
            float32x4_t w1 = dense_bias_load4(t->b, b_index + 4, p->b_dtype);
            float x0 = trillim_load(t->a, a_base + 0 * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
            s00 = vfmaq_n_f32(s00, w0, x0);
            s01 = vfmaq_n_f32(s01, w1, x0);
            if (rows > 1) {
                float x1 = trillim_load(t->a, a_base + 1 * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
                s10 = vfmaq_n_f32(s10, w0, x1);
                s11 = vfmaq_n_f32(s11, w1, x1);
            }
            if (rows > 2) {
                float x2 = trillim_load(t->a, a_base + 2 * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
                s20 = vfmaq_n_f32(s20, w0, x2);
                s21 = vfmaq_n_f32(s21, w1, x2);
            }
            if (rows > 3) {
                float x3 = trillim_load(t->a, a_base + 3 * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
                s30 = vfmaq_n_f32(s30, w0, x3);
                s31 = vfmaq_n_f32(s31, w1, x3);
            }
        }
        float sums[4][8];
        vst1q_f32(sums[0] + 0, s00);
        vst1q_f32(sums[0] + 4, s01);
        vst1q_f32(sums[1] + 0, s10);
        vst1q_f32(sums[1] + 4, s11);
        vst1q_f32(sums[2] + 0, s20);
        vst1q_f32(sums[2] + 4, s21);
        vst1q_f32(sums[3] + 0, s30);
        vst1q_f32(sums[3] + 4, s31);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < 8; ++c) {
                store(t, output + r * p->n + column + c, sums[r][c]);
            }
        }
    }
    for (; column < last; ++column) {
        float sums[4] = {0};
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float b = trillim_load(t->b, b_base + k * p->b_row_stride + column * p->b_col_stride, p->b_dtype).f;
            for (size_t r = 0; r < rows; ++r) {
                float a = trillim_load(t->a, a_base + r * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
                sums[r] = fmaf(a, b, sums[r]);
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            store(t, output + r * p->n + column, sums[r]);
        }
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus row_vector_typed_tile(
    const DenseBiasTask *t, size_t a_base, size_t b_base, size_t output, size_t rows,
    size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseBiasParameters *p = t->p;
    for (size_t column = first; column < last; column += 16) {
        size_t columns = last - column < 16 ? last - column : 16;
        float32x4_t s0 = vdupq_n_f32(0);
        float32x4_t s1 = vdupq_n_f32(0);
        float32x4_t s2 = vdupq_n_f32(0);
        float32x4_t s3 = vdupq_n_f32(0);
        float32x4_t s4 = vdupq_n_f32(0);
        float32x4_t s5 = vdupq_n_f32(0);
        float32x4_t s6 = vdupq_n_f32(0);
        float32x4_t s7 = vdupq_n_f32(0);
        float32x4_t s8 = vdupq_n_f32(0);
        float32x4_t s9 = vdupq_n_f32(0);
        float32x4_t s10 = vdupq_n_f32(0);
        float32x4_t s11 = vdupq_n_f32(0);
        float32x4_t s12 = vdupq_n_f32(0);
        float32x4_t s13 = vdupq_n_f32(0);
        float32x4_t s14 = vdupq_n_f32(0);
        float32x4_t s15 = vdupq_n_f32(0);
        size_t k = 0;
        if (p->b_row_stride == 1) {
            for (; p->k - k >= 4; k += 4) {
                if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float32x4_t x0;
                if (rows == 4) {
                    x0 = dense_bias_load4(t->a, a_base + (k + 0) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 0) * p->a_col_stride + r, a_dtype).f;
                    }
                    x0 = vld1q_f32(values);
                }
                float32x4_t x1;
                if (rows == 4) {
                    x1 = dense_bias_load4(t->a, a_base + (k + 1) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 1) * p->a_col_stride + r, a_dtype).f;
                    }
                    x1 = vld1q_f32(values);
                }
                float32x4_t x2;
                if (rows == 4) {
                    x2 = dense_bias_load4(t->a, a_base + (k + 2) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 2) * p->a_col_stride + r, a_dtype).f;
                    }
                    x2 = vld1q_f32(values);
                }
                float32x4_t x3;
                if (rows == 4) {
                    x3 = dense_bias_load4(t->a, a_base + (k + 3) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 3) * p->a_col_stride + r, a_dtype).f;
                    }
                    x3 = vld1q_f32(values);
                }
                {
                    float32x4_t w0 = dense_bias_load4(t->b, b_base + k + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = dense_bias_load4(t->b, b_base + k + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = dense_bias_load4(t->b, b_base + k + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = dense_bias_load4(t->b, b_base + k + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = dense_bias_load4(t->b, b_base + k + (column + (columns > 4 ? 4 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = dense_bias_load4(t->b, b_base + k + (column + (columns > 5 ? 5 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = dense_bias_load4(t->b, b_base + k + (column + (columns > 6 ? 6 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = dense_bias_load4(t->b, b_base + k + (column + (columns > 7 ? 7 : 0)) * p->b_col_stride, b_dtype);
                    s0 = vfmaq_laneq_f32(s0, x0, w0, 0);
                    s1 = vfmaq_laneq_f32(s1, x0, w1, 0);
                    s2 = vfmaq_laneq_f32(s2, x0, w2, 0);
                    s3 = vfmaq_laneq_f32(s3, x0, w3, 0);
                    s4 = vfmaq_laneq_f32(s4, x0, w4, 0);
                    s5 = vfmaq_laneq_f32(s5, x0, w5, 0);
                    s6 = vfmaq_laneq_f32(s6, x0, w6, 0);
                    s7 = vfmaq_laneq_f32(s7, x0, w7, 0);
                    s0 = vfmaq_laneq_f32(s0, x1, w0, 1);
                    s1 = vfmaq_laneq_f32(s1, x1, w1, 1);
                    s2 = vfmaq_laneq_f32(s2, x1, w2, 1);
                    s3 = vfmaq_laneq_f32(s3, x1, w3, 1);
                    s4 = vfmaq_laneq_f32(s4, x1, w4, 1);
                    s5 = vfmaq_laneq_f32(s5, x1, w5, 1);
                    s6 = vfmaq_laneq_f32(s6, x1, w6, 1);
                    s7 = vfmaq_laneq_f32(s7, x1, w7, 1);
                    s0 = vfmaq_laneq_f32(s0, x2, w0, 2);
                    s1 = vfmaq_laneq_f32(s1, x2, w1, 2);
                    s2 = vfmaq_laneq_f32(s2, x2, w2, 2);
                    s3 = vfmaq_laneq_f32(s3, x2, w3, 2);
                    s4 = vfmaq_laneq_f32(s4, x2, w4, 2);
                    s5 = vfmaq_laneq_f32(s5, x2, w5, 2);
                    s6 = vfmaq_laneq_f32(s6, x2, w6, 2);
                    s7 = vfmaq_laneq_f32(s7, x2, w7, 2);
                    s0 = vfmaq_laneq_f32(s0, x3, w0, 3);
                    s1 = vfmaq_laneq_f32(s1, x3, w1, 3);
                    s2 = vfmaq_laneq_f32(s2, x3, w2, 3);
                    s3 = vfmaq_laneq_f32(s3, x3, w3, 3);
                    s4 = vfmaq_laneq_f32(s4, x3, w4, 3);
                    s5 = vfmaq_laneq_f32(s5, x3, w5, 3);
                    s6 = vfmaq_laneq_f32(s6, x3, w6, 3);
                    s7 = vfmaq_laneq_f32(s7, x3, w7, 3);
                }
                {
                    float32x4_t w0 = dense_bias_load4(t->b, b_base + k + (column + (columns > 8 ? 8 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = dense_bias_load4(t->b, b_base + k + (column + (columns > 9 ? 9 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = dense_bias_load4(t->b, b_base + k + (column + (columns > 10 ? 10 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = dense_bias_load4(t->b, b_base + k + (column + (columns > 11 ? 11 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = dense_bias_load4(t->b, b_base + k + (column + (columns > 12 ? 12 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = dense_bias_load4(t->b, b_base + k + (column + (columns > 13 ? 13 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = dense_bias_load4(t->b, b_base + k + (column + (columns > 14 ? 14 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = dense_bias_load4(t->b, b_base + k + (column + (columns > 15 ? 15 : 0)) * p->b_col_stride, b_dtype);
                    s8 = vfmaq_laneq_f32(s8, x0, w0, 0);
                    s9 = vfmaq_laneq_f32(s9, x0, w1, 0);
                    s10 = vfmaq_laneq_f32(s10, x0, w2, 0);
                    s11 = vfmaq_laneq_f32(s11, x0, w3, 0);
                    s12 = vfmaq_laneq_f32(s12, x0, w4, 0);
                    s13 = vfmaq_laneq_f32(s13, x0, w5, 0);
                    s14 = vfmaq_laneq_f32(s14, x0, w6, 0);
                    s15 = vfmaq_laneq_f32(s15, x0, w7, 0);
                    s8 = vfmaq_laneq_f32(s8, x1, w0, 1);
                    s9 = vfmaq_laneq_f32(s9, x1, w1, 1);
                    s10 = vfmaq_laneq_f32(s10, x1, w2, 1);
                    s11 = vfmaq_laneq_f32(s11, x1, w3, 1);
                    s12 = vfmaq_laneq_f32(s12, x1, w4, 1);
                    s13 = vfmaq_laneq_f32(s13, x1, w5, 1);
                    s14 = vfmaq_laneq_f32(s14, x1, w6, 1);
                    s15 = vfmaq_laneq_f32(s15, x1, w7, 1);
                    s8 = vfmaq_laneq_f32(s8, x2, w0, 2);
                    s9 = vfmaq_laneq_f32(s9, x2, w1, 2);
                    s10 = vfmaq_laneq_f32(s10, x2, w2, 2);
                    s11 = vfmaq_laneq_f32(s11, x2, w3, 2);
                    s12 = vfmaq_laneq_f32(s12, x2, w4, 2);
                    s13 = vfmaq_laneq_f32(s13, x2, w5, 2);
                    s14 = vfmaq_laneq_f32(s14, x2, w6, 2);
                    s15 = vfmaq_laneq_f32(s15, x2, w7, 2);
                    s8 = vfmaq_laneq_f32(s8, x3, w0, 3);
                    s9 = vfmaq_laneq_f32(s9, x3, w1, 3);
                    s10 = vfmaq_laneq_f32(s10, x3, w2, 3);
                    s11 = vfmaq_laneq_f32(s11, x3, w3, 3);
                    s12 = vfmaq_laneq_f32(s12, x3, w4, 3);
                    s13 = vfmaq_laneq_f32(s13, x3, w5, 3);
                    s14 = vfmaq_laneq_f32(s14, x3, w6, 3);
                    s15 = vfmaq_laneq_f32(s15, x3, w7, 3);
                }
            }
        }
        for (; k < p->k; ++k) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t x;
            if (rows == 4) {
                x = dense_bias_load4(t->a, a_base + k * p->a_col_stride, a_dtype);
            } else {
                float values[4] = {0};
                for (size_t r = 0; r < rows; ++r) {
                    values[r] = trillim_load(t->a, a_base + k * p->a_col_stride + r, a_dtype).f;
                }
                x = vld1q_f32(values);
            }
            if (columns > 0) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 0) * p->b_col_stride, b_dtype).f;
                s0 = vfmaq_n_f32(s0, x, w);
            }
            if (columns > 1) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 1) * p->b_col_stride, b_dtype).f;
                s1 = vfmaq_n_f32(s1, x, w);
            }
            if (columns > 2) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 2) * p->b_col_stride, b_dtype).f;
                s2 = vfmaq_n_f32(s2, x, w);
            }
            if (columns > 3) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 3) * p->b_col_stride, b_dtype).f;
                s3 = vfmaq_n_f32(s3, x, w);
            }
            if (columns > 4) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 4) * p->b_col_stride, b_dtype).f;
                s4 = vfmaq_n_f32(s4, x, w);
            }
            if (columns > 5) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 5) * p->b_col_stride, b_dtype).f;
                s5 = vfmaq_n_f32(s5, x, w);
            }
            if (columns > 6) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 6) * p->b_col_stride, b_dtype).f;
                s6 = vfmaq_n_f32(s6, x, w);
            }
            if (columns > 7) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 7) * p->b_col_stride, b_dtype).f;
                s7 = vfmaq_n_f32(s7, x, w);
            }
            if (columns > 8) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 8) * p->b_col_stride, b_dtype).f;
                s8 = vfmaq_n_f32(s8, x, w);
            }
            if (columns > 9) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 9) * p->b_col_stride, b_dtype).f;
                s9 = vfmaq_n_f32(s9, x, w);
            }
            if (columns > 10) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 10) * p->b_col_stride, b_dtype).f;
                s10 = vfmaq_n_f32(s10, x, w);
            }
            if (columns > 11) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 11) * p->b_col_stride, b_dtype).f;
                s11 = vfmaq_n_f32(s11, x, w);
            }
            if (columns > 12) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 12) * p->b_col_stride, b_dtype).f;
                s12 = vfmaq_n_f32(s12, x, w);
            }
            if (columns > 13) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 13) * p->b_col_stride, b_dtype).f;
                s13 = vfmaq_n_f32(s13, x, w);
            }
            if (columns > 14) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 14) * p->b_col_stride, b_dtype).f;
                s14 = vfmaq_n_f32(s14, x, w);
            }
            if (columns > 15) {
                float w = trillim_load(t->b, b_base + k * p->b_row_stride + (column + 15) * p->b_col_stride, b_dtype).f;
                s15 = vfmaq_n_f32(s15, x, w);
            }
        }
        float values[16][4];
        vst1q_f32(values[0], s0);
        vst1q_f32(values[1], s1);
        vst1q_f32(values[2], s2);
        vst1q_f32(values[3], s3);
        vst1q_f32(values[4], s4);
        vst1q_f32(values[5], s5);
        vst1q_f32(values[6], s6);
        vst1q_f32(values[7], s7);
        vst1q_f32(values[8], s8);
        vst1q_f32(values[9], s9);
        vst1q_f32(values[10], s10);
        vst1q_f32(values[11], s11);
        vst1q_f32(values[12], s12);
        vst1q_f32(values[13], s13);
        vst1q_f32(values[14], s14);
        vst1q_f32(values[15], s15);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                store(t, output + r * p->n + column + c, values[c][r]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus row_vector_tile(const DenseBiasTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseBiasParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32) {
        return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, TRILLIM_FP32, TRILLIM_FP32);
    }
    return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, p->a_dtype, p->b_dtype);
}

static TrillimStatus dense_bias_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseBiasTask *t = opaque;
    const DenseBiasParameters *p = t->p;
    size_t width = tile_columns(p, t->context->thread_pool->worker_count);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t height = tile_rows(p, t->context->thread_pool->worker_count);
    size_t row_tiles = p->m / height + (p->m % height != 0);
    size_t batch = index / columns / row_tiles;
    size_t row = index / columns % row_tiles * height;
    size_t first = index % columns * width;
    size_t last = p->n - first < width ? p->n : first + width;
    size_t rows = p->m - row < height ? p->m - row : height;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    size_t output = (batch * p->m + row) * p->n;
    if (p->a_col_stride == 1 && p->b_row_stride == 1) {
        if (precise(p)) {
            return ordered_dot_tile(t, a_base, b_base, output, rows, first, last);
        }
        if (height == 6) {
            return dot_tile6(t, a_base, b_base, output, rows, first, last);
        }
        return rows == 1 && p->k >= 64 ?
            decode_tile(t, a_base, b_base, output, first, last) :
            dot_tile(t, a_base, b_base, output, rows, first, last);
    }
    if (p->b_col_stride == 1) {
        return outer_tile(t, a_base, b_base, output, rows, first, last);
    }
    if (p->a_row_stride == 1 && rows <= 4) {
        return row_vector_tile(t, a_base, b_base, output, rows, first, last);
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = first; c < last; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < p->k; ++k) {
                if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                sum += trillim_load(t->a, a_base + r * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f *
                       trillim_load(t->b, b_base + k * p->b_row_stride + c * p->b_col_stride, p->b_dtype).f;
            }
            store(t, output + r * p->n + c, sum);
        }
    }
    return TRILLIM_OK;
}

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
#ifdef __APPLE__
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
static GetThreading get_threading;
static SetThreading set_threading;
#endif
static atomic_uint blas_state = 0;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                FreeLibrary(library);
            }
        }
#else
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
        static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
#endif
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            void *library = dlopen(names[i], RTLD_LOCAL | RTLD_LAZY);
            if (library != NULL) {
#ifdef __APPLE__
                void *address = dlsym(library, "cblas_sgemm$NEWLAPACK");
                if (address == NULL) {
                    address = dlsym(library, "cblas_sgemm");
                }
#else
                void *address = dlsym(library, "cblas_sgemm");
#endif
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

static inline float load(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)data + index * sizeof(bits), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(bits);
    }
#if defined(__FLT16_MANT_DIG__)
    TrillimFloat16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
#else
    return trillim_fp16(bits);
#endif
}

static TrillimStatus pack(const void *source, uint32_t dtype, size_t base,
                          size_t rows, size_t columns, size_t row_stride,
                          size_t col_stride, float *target,
                          const TrillimKernelContext *context) {
    for (size_t row = 0; row < rows; ++row) {
        if ((row & 7) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t index = base + row * row_stride;
        if (dtype == TRILLIM_FP32 && col_stride == 1) {
            memcpy(target + row * columns, (const uint8_t *)source + index * sizeof(float), columns * sizeof(float));
        } else if (dtype == TRILLIM_BF16 && col_stride == 1) {
            for (size_t column = 0; column < columns; ++column) {
                uint16_t bits;
                memcpy(&bits, (const uint8_t *)source + (index + column) * sizeof(bits), sizeof(bits));
                target[row * columns + column] = trillim_bf16(bits);
            }
#if defined(__FLT16_MANT_DIG__)
        } else if (dtype == TRILLIM_FP16 && col_stride == 1) {
            for (size_t column = 0; column < columns; ++column) {
                TrillimFloat16 value;
                memcpy(&value, (const uint8_t *)source + (index + column) * sizeof(value), sizeof(value));
                target[row * columns + column] = (float)value;
            }
#endif
        } else {
            for (size_t column = 0; column < columns; ++column) {
                target[row * columns + column] = load(source, index + column * col_stride, dtype);
            }
        }
    }
    return TRILLIM_OK;
}

static size_t row_reuse(const DenseBiasParameters *p, const TrillimKernelContext *context) {
    size_t workers = context->thread_pool->worker_count;
    size_t minimum_rows = workers <= 8 || p->n >= 2048 ? 128 : 512;
    if (p->m < minimum_rows || p->out_dtype != TRILLIM_FP32 || p->b_dtype == TRILLIM_FP32 || p->k == 0) {
        return 1;
    }
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
    if (columns == 0) {
        return 1;
    }
    size_t target = workers == 0 ? 1 : (workers < LANES / 2 ? workers * 2 : LANES);
    size_t groups = target / columns + (target % columns != 0);
    size_t reuse = rows / groups;
    return reuse < 2 ? 1 : (reuse < 8 ? reuse : 8);
}

typedef struct {
    DenseBiasTask task;
    float *scratch;
    size_t count;
    size_t lanes;
    size_t row_reuse;
} BlasTask;

static TrillimStatus ungrouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const DenseBiasTask *t = &job->task;
    const DenseBiasParameters *p = t->p;
    size_t width = block_n(p);
    size_t depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / block_m(p) + (p->m % block_m(p) != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    float *c_pack = b_pack + BLOCK_K * BLOCK_N;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / row_tiles;
        size_t row = tile / columns % row_tiles * block_m(p);
        size_t column = tile % columns * width;
        size_t m = p->m - row < block_m(p) ? p->m - row : block_m(p);
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        size_t output = (batch * p->m + row) * p->n + column;
        float *c = p->out_dtype == TRILLIM_FP32 ? (float *)t->output + output : c_pack;
        int ldc = p->out_dtype == TRILLIM_FP32 ? (int)p->n : (int)n;
        if (p->k == 0) {
            for (size_t r = 0; r < m; ++r) {
                memset(c + r * (size_t)ldc, 0, n * sizeof(float));
            }
        }
        for (size_t first = 0; first < p->k; first += depth) {
            if (t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t k = p->k - first < depth ? p->k - first : depth;
            size_t a_index = a_base + first * p->a_col_stride;
            size_t b_index = b_base + first * p->b_row_stride;
            const float *a = a_pack;
            const float *b = b_pack;
            int a_trans = 111, b_trans = 111, lda = (int)k, ldb = (int)n;
            TrillimStatus status;
            if (p->a_dtype == TRILLIM_FP32 && p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                lda = (int)p->a_row_stride;
            } else if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                a_trans = 112;
                lda = (int)p->a_col_stride;
            } else {
                status = pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (p->b_dtype == TRILLIM_FP32 && p->b_col_stride == 1 && p->b_row_stride >= p->n && p->b_row_stride <= INT32_MAX) {
                b = (const float *)t->b + b_index;
                ldb = (int)p->b_row_stride;
            } else if (p->b_dtype == TRILLIM_FP32 && p->b_row_stride == 1 && p->b_col_stride >= p->k && p->b_col_stride <= INT32_MAX) {
                b = (const float *)t->b + b_index;
                b_trans = 112;
                ldb = (int)p->b_col_stride;
            } else if (p->b_row_stride == 1) {
                status = pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                  a, lda, b, ldb, first == 0 ? 0.0f : 1.0f, c, ldc);
        }
        for (size_t r = 0; r < m; ++r) {
            for (size_t col = 0; col < n; ++col) {
                TrillimValue value = {.f = add_bias(t, column + col, c[r * (size_t)ldc + col])};
                trillim_store(t->output, output + r * p->n + col, p->out_dtype, value);
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus grouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const DenseBiasTask *t = &job->task;
    const DenseBiasParameters *p = t->p;
    size_t width = block_n(p), depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t groups = rows / job->row_reuse + (rows % job->row_reuse != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / groups;
        size_t group = tile / columns % groups;
        size_t first_row = group * job->row_reuse;
        size_t last_row = rows - first_row < job->row_reuse ? rows : first_row + job->row_reuse;
        size_t column = tile % columns * width;
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides);
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        for (size_t first = 0; first < p->k; first += depth) {
            if (t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t k = p->k - first < depth ? p->k - first : depth;
            /* Partial C stays in output while packed B is shared across row tiles. */
            size_t b_index = b_base + first * p->b_row_stride;
            int b_trans = 111, ldb = (int)n;
            TrillimStatus status;
            if (p->b_row_stride == 1) {
                status = pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
            }
            if (status != TRILLIM_OK) {
                return status;
            }
            for (size_t block = first_row; block < last_row; ++block) {
                if (t->context->is_cancelled(t->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t row = block * block_m(p);
                size_t m = p->m - row < block_m(p) ? p->m - row : block_m(p);
                size_t a_index = a_base + row * p->a_row_stride + first * p->a_col_stride;
                const float *a = a_pack;
                int a_trans = 111, lda = (int)k;
                if (p->a_dtype == TRILLIM_FP32 && p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                    a = (const float *)t->a + a_index;
                    lda = (int)p->a_row_stride;
                } else if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX) {
                    a = (const float *)t->a + a_index;
                    a_trans = 112;
                    lda = (int)p->a_col_stride;
                } else {
                    status = pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                size_t output = (batch * p->m + row) * p->n + column;
                float *c = (float *)t->output + output;
                sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                      a, lda, b_pack, ldb, first == 0 ? 0.0f : 1.0f, c, (int)p->n);
                if (first + k == p->k) {
                    for (size_t r = 0; r < m; ++r) {
                        for (size_t col = 0; col < n; ++col) {
                            TrillimValue value = {.f = add_bias(t, column + col, c[r * p->n + col])};
                            trillim_store(t->output, output + r * p->n + col, p->out_dtype, value);
                        }
                    }
                }
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus blas_project_impl(void *opaque, size_t index, size_t worker) {
    const BlasTask *job = opaque;
    return job->row_reuse > 1 ? grouped_project_impl(opaque, index, worker) :
        ungrouped_project_impl(opaque, index, worker);
}

static TrillimStatus blas_project(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = blas_project_impl(opaque, index, worker);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return blas_project_impl(opaque, index, worker);
}

static TrillimStatus pack_columns(void *opaque, size_t block, size_t worker) {
    (void)worker;
    const DenseBiasTask *task = opaque;
    const DenseBiasParameters *p = task->p;
    size_t first = block * 8;
    size_t count = p->n - first < 8 ? p->n - first : 8;
    float *target = task->packed + block * p->k * 8;
    const float *weight = task->b;
    for (size_t k = 0; k < p->k; ++k) {
        if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t column = 0; column < 8; ++column) {
            target[k * 8 + column] = column < count ? weight[(first + column) * p->b_col_stride + k] : 0;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus packed_project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const DenseBiasTask *task = opaque;
    const DenseBiasParameters *p = task->p;
    size_t blocks = (p->n + 7) / 8;
    size_t first = tile % blocks * 8;
    size_t row = tile / blocks * 6;
    size_t rows = p->m - row < 6 ? p->m - row : 6;
    const float *input = (const float *)task->a + row * p->a_row_stride;
    const float *a0 = input;
    const float *a1 = input + (rows > 1 ? 1 * p->a_row_stride : 0);
    const float *a2 = input + (rows > 2 ? 2 * p->a_row_stride : 0);
    const float *a3 = input + (rows > 3 ? 3 * p->a_row_stride : 0);
    const float *a4 = input + (rows > 4 ? 4 * p->a_row_stride : 0);
    const float *a5 = input + (rows > 5 ? 5 * p->a_row_stride : 0);
    const float *weights = task->packed + (tile % blocks) * p->k * 8;
    float32x4_t s00 = vdupq_n_f32(0), s01 = vdupq_n_f32(0);
    float32x4_t s10 = vdupq_n_f32(0), s11 = vdupq_n_f32(0);
    float32x4_t s20 = vdupq_n_f32(0), s21 = vdupq_n_f32(0);
    float32x4_t s30 = vdupq_n_f32(0), s31 = vdupq_n_f32(0);
    float32x4_t s40 = vdupq_n_f32(0), s41 = vdupq_n_f32(0);
    float32x4_t s50 = vdupq_n_f32(0), s51 = vdupq_n_f32(0);
    for (size_t k = 0; k < p->k; ++k) {
        if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t w0 = vld1q_f32(weights + k * 8 + 0);
        float32x4_t w1 = vld1q_f32(weights + k * 8 + 4);
        s00 = vfmaq_n_f32(s00, w0, a0[k]);
        s01 = vfmaq_n_f32(s01, w1, a0[k]);
        s10 = vfmaq_n_f32(s10, w0, a1[k]);
        s11 = vfmaq_n_f32(s11, w1, a1[k]);
        s20 = vfmaq_n_f32(s20, w0, a2[k]);
        s21 = vfmaq_n_f32(s21, w1, a2[k]);
        s30 = vfmaq_n_f32(s30, w0, a3[k]);
        s31 = vfmaq_n_f32(s31, w1, a3[k]);
        s40 = vfmaq_n_f32(s40, w0, a4[k]);
        s41 = vfmaq_n_f32(s41, w1, a4[k]);
        s50 = vfmaq_n_f32(s50, w0, a5[k]);
        s51 = vfmaq_n_f32(s51, w1, a5[k]);
    }
    size_t count = p->n - first < 8 ? p->n - first : 8;
    float bias[8] = {0};
    memcpy(bias, (const float *)task->bias + first, count * sizeof(float));
    float32x4_t biases[2] = {vld1q_f32(bias), vld1q_f32(bias + 4)};
    float32x4_t sums[6][2] = {
        {s00, s01},
        {s10, s11},
        {s20, s21},
        {s30, s31},
        {s40, s41},
        {s50, s51},
    };
    for (size_t r = 0; r < rows; ++r) {
        float *output = (float *)task->output + (row + r) * p->n + first;
        if (count == 8) {
            for (size_t c = 0; c < 2; ++c) {
                vst1q_f32(output + c * 4, vaddq_f32(sums[r][c], biases[c]));
            }
        } else {
            float tail[8];
            for (size_t c = 0; c < 2; ++c) {
                vst1q_f32(tail + c * 4, vaddq_f32(sums[r][c], biases[c]));
            }
            memcpy(output, tail, count * sizeof(float));
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dense_bias_run(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    const DenseBiasParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    DenseBiasTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context, scratch};
    if (packed_fp32(p)) {
        size_t blocks = (p->n + 7) / 8;
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, blocks, pack_columns, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        return context->thread_pool->run(context->thread_pool->context, ((p->m + 5) / 6) * blocks, packed_project, &task);
    }
    if (use_blas(p, context)) {
        size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
        size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
        size_t reuse = row_reuse(p, context);
        size_t groups = rows / reuse + (rows % reuse != 0);
        size_t count = p->batches * groups * columns;
        size_t lanes = count < LANES ? count : LANES;
        BlasTask job = {task, scratch, count, lanes, reuse};
        return context->thread_pool->run(context->thread_pool->context, lanes, blas_project, &job);
    }
    size_t height = tile_rows(p, context->thread_pool->worker_count);
    size_t rows = p->m / height + (p->m % height != 0);
    size_t width = tile_columns(p, context->thread_pool->worker_count);
    size_t columns = p->n / width + (p->n % width != 0);
    return context->thread_pool->run(context->thread_pool->context, p->batches * rows * columns, dense_bias_project, &task);
}


static const TrillimParameterSchema *dense_bias_schema(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(DenseBiasParameters, batches), NULL},
        {"m", "size_t", offsetof(DenseBiasParameters, m), NULL},
        {"n", "size_t", offsetof(DenseBiasParameters, n), NULL},
        {"k", "size_t", offsetof(DenseBiasParameters, k), NULL},
        {"rank", "size_t", offsetof(DenseBiasParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(DenseBiasParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(DenseBiasParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(DenseBiasParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(DenseBiasParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(DenseBiasParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(DenseBiasParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(DenseBiasParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(DenseBiasParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(DenseBiasParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(DenseBiasParameters, out_dtype), NULL},
        {"linear_dtype", "uint32_t", offsetof(DenseBiasParameters, linear_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(DenseBiasParameters, bias_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(DenseBiasParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(DenseBiasParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(DenseBiasParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(DenseBiasParameters, b_storage_columns), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(DenseBiasParameters), TRILLIM_ALIGNOF(DenseBiasParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, dense_bias_requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

/* Expert selection. */

enum { EXPERTS = 32, TOPK4_ROUTES = 4 };

typedef struct {
    size_t rows;
} Topk4Parameters;

static TrillimStatus topk4_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Topk4Parameters *p = parameters;
    inputs[0] = outputs[0] = outputs[1] = *scratch = 0;
    if (p->rows > SIZE_MAX / (EXPERTS * sizeof(uint16_t))) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * EXPERTS * sizeof(uint16_t);
    outputs[0] = p->rows * TOPK4_ROUTES * sizeof(uint16_t);
    outputs[1] = p->rows * TOPK4_ROUTES * sizeof(int64_t);
    return TRILLIM_OK;
}

static TrillimStatus topk4_run(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Topk4Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t row = 0; row < p->rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t source = row * EXPERTS;
        size_t target = row * TOPK4_ROUTES;
        uint32_t available = UINT32_MAX;
        for (size_t route = 0; route < TOPK4_ROUTES; ++route) {
            size_t selected = EXPERTS;
            float best = 0;
            for (size_t expert = 0; expert < EXPERTS; ++expert) {
                if (!(available & (UINT32_C(1) << expert))) {
                    continue;
                }
                float value = trillim_load(inputs[0], source + expert, TRILLIM_BF16).f;
                /* Follow the existing masked Argmax's ordered ties and NaN behavior. */
                if (selected == EXPERTS || !(best >= value)) {
                    best = value;
                    selected = expert;
                }
            }
            available &= ~(UINT32_C(1) << selected);
            trillim_store(outputs[0], target + route, TRILLIM_BF16, (TrillimValue){.f = best});
            trillim_store(outputs[1], target + route, TRILLIM_INT64, (TrillimValue){.i = (int64_t)selected});
        }
    }
    return TRILLIM_OK;
}


static const TrillimParameterSchema *topk4_schema(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Topk4Parameters, rows), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1, 1};
    static const TrillimParameterSchema schema = {
        sizeof(Topk4Parameters), TRILLIM_ALIGNOF(Topk4Parameters), 1, fields,
        1, 2, topk4_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

/* Expert score normalization. */

typedef struct {
    size_t rows, width;
    uint32_t dtype;
} SoftmaxKernelParameters;

typedef struct {
    void *output;
    const void *input;
    float *scratch;
    const SoftmaxKernelParameters *p;
    const TrillimKernelContext *context;
} SoftmaxTask;

static TrillimStatus softmax_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const SoftmaxKernelParameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || !p->width ||
        p->rows > SIZE_MAX / sizeof(float) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    *scratch = p->rows * p->width * sizeof(float);
    return TRILLIM_OK;
}

static float32x4_t softmax_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
    } else if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index,
                 vreinterpret_u16_f16(vcvt_f16_f32(value)));
    } else {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
    }
}

static float32x4_t exp4(float32x4_t x) {
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

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static TrillimStatus softmax_precise(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const SoftmaxTask *task = opaque;
    const SoftmaxKernelParameters *p = task->p;
    size_t start = row * p->width;
    float maximum = -INFINITY;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->dtype).f;
        maximum = maximum >= value ? maximum : value;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float difference = round_dtype(trillim_load(task->input, start + column, p->dtype).f - maximum, p->dtype);
        float numerator = round_dtype(expf(difference), p->dtype);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    sum = round_dtype(sum, p->dtype);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus softmax(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const SoftmaxTask *task = opaque;
    const SoftmaxKernelParameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve low-precision rounding at scalar-reference boundaries. */
        return softmax_precise(opaque, row, worker);
    }
    size_t start = row * p->width, column = 0;
    float32x4_t maximum4 = vdupq_n_f32(-INFINITY);
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = softmax_load4(task->input, start + column, p->dtype);
        if (vminvq_u32(vceqq_f32(value, value)) != UINT32_MAX) {
            return softmax_precise(opaque, row, worker);
        }
        maximum4 = vmaxnmq_f32(maximum4, value);
    }
    float maximum = vmaxnmvq_f32(maximum4);
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->dtype).f;
        if (isnan(value)) {
            return softmax_precise(opaque, row, worker);
        }
        maximum = maximum >= value ? maximum : value;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float32x4_t sum4 = vdupq_n_f32(0.0f), maximum_vector = vdupq_n_f32(maximum);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t numerator = exp4(vsubq_f32(softmax_load4(task->input, start + column, p->dtype),
                                              maximum_vector));
        vst1q_f32(task->scratch + start + column, numerator);
        sum4 = vaddq_f32(sum4, numerator);
    }
    float sum = vaddvq_f32(sum4);
    for (; column < p->width; ++column) {
        float numerator = expf(trillim_load(task->input, start + column, p->dtype).f - maximum);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    float denominator = sum > 0.0f ? sum : 1.0f;
    float32x4_t denominator_vector = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, start + column, p->dtype,
               vdivq_f32(vld1q_f32(task->scratch + start + column), denominator_vector));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus softmax_run(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const SoftmaxKernelParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    SoftmaxTask task = {outputs[0], inputs[0], scratch, p, context};
    if (p->rows == 1) {
        return softmax(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     softmax, &task);
}


static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

static const TrillimParameterSchema *softmax_schema(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(SoftmaxKernelParameters, rows), NULL},
        {"width", "size_t", offsetof(SoftmaxKernelParameters, width), NULL},
        {"dtype", "uint32_t", offsetof(SoftmaxKernelParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(SoftmaxKernelParameters), TRILLIM_ALIGNOF(SoftmaxKernelParameters), 3, fields, 1, 1,
        softmax_requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

/* Routed expert projection. */

typedef struct RoutedQ8 RoutedQ8;

typedef struct {
    size_t tokens, routes, experts, n, k;
    size_t a_token_stride, a_route_stride, index_token_stride, index_route_stride;
    uint32_t a_dtype;
} RoutedMatmulParameters;

enum { ROUTES_PER_TILE = 4 };

typedef struct {
    void *output;
    const uint8_t *activation, *codes, *scales;
    const RoutedMatmulParameters *p;
    const TrillimKernelContext *context;
    const size_t *offsets, *tiles, *slots;
    size_t columns, column_tiles;
    const RoutedQ8 *quantized;
    bool doubled_columns;
} RoutedMatmulTask;

static TrillimStatus routed_matmul_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const RoutedMatmulParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!p->experts || !p->n || !p->k || p->k % 32 || !trillim_float(p->a_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    if (p->n > SIZE_MAX / p->k || p->tokens > SIZE_MAX / p->routes) {
        return TRILLIM_ERROR;
    }
    size_t slots = p->tokens * p->routes;
    size_t index_shape[] = {p->tokens, p->routes};
    size_t index_strides[] = {p->index_token_stride, p->index_route_stride};
    size_t a_shape[] = {p->tokens, p->routes, p->k};
    size_t a_strides[] = {p->a_token_stride, p->a_route_stride, 1};
    size_t a_span, index_span;
    size_t a_size = trillim_itemsize(p->a_dtype);
    size_t scratch_limit = SIZE_MAX / sizeof(size_t);
    if (!trillim_span(3, a_shape, a_strides, &a_span) || a_span > SIZE_MAX / a_size ||
        !trillim_span(2, index_shape, index_strides, &index_span) || index_span > SIZE_MAX / sizeof(int64_t) ||
        slots > SIZE_MAX / sizeof(float) / p->n ||
        slots > scratch_limit - 2 || p->experts > (scratch_limit - 2 - slots) / 2 ||
        !trillim_packed_payload_size(p->experts, p->n * p->k, TRILLIM_MXFP4, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a_span * a_size;
    inputs[2] = index_span * sizeof(int64_t);
    outputs[0] = slots * p->n * sizeof(float);
    *scratch = (2 * (p->experts + 1) + slots) * sizeof(size_t);
    return TRILLIM_OK;
}

static inline bool cancelled(const TrillimKernelContext *context) {
    return context->is_cancelled(context->cancellation_context);
}

static inline uint64_t expert_at(const RoutedMatmulParameters *p, const void *indices, size_t slot) {
    size_t index = slot / p->routes * p->index_token_stride + slot % p->routes * p->index_route_stride;
    return (uint64_t)trillim_load(indices, index, TRILLIM_INT64).i;
}

static TrillimStatus group_routes(
    const RoutedMatmulParameters *p, const void *indices, size_t *offsets, size_t *tiles, size_t *slots,
    const TrillimKernelContext *context
) {
    size_t count = p->tokens * p->routes;
    for (size_t expert = 0; expert < p->experts; ++expert) {
        if ((expert & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        offsets[expert] = 0;
    }
    for (size_t slot = 0; slot < count; ++slot) {
        if ((slot & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        uint64_t expert = expert_at(p, indices, slot);
        if (expert >= p->experts) {
            return TRILLIM_ERROR;
        }
        ++offsets[expert];
    }
    size_t next = 0;
    tiles[0] = 0;
    for (size_t expert = 0; expert < p->experts; ++expert) {
        if ((expert & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        size_t routes = offsets[expert];
        tiles[expert + 1] = tiles[expert] + routes / ROUTES_PER_TILE + (routes % ROUTES_PER_TILE != 0);
        next += routes;
        offsets[expert] = next;
    }
    offsets[p->experts] = count;
    /* Inserting backward turns each end offset into the start of its expert's routes. */
    for (size_t slot = count; slot != 0; --slot) {
        if ((slot & 255) == 0 && cancelled(context)) {
            return TRILLIM_CANCELLED;
        }
        uint64_t expert = expert_at(p, indices, slot - 1);
        slots[--offsets[expert]] = slot - 1;
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) float32x4x4_t load16(const uint8_t *data, size_t index, uint32_t dtype) {
    float32x4x4_t result;
    if (dtype == TRILLIM_FP32) {
        memcpy(&result, data + index * sizeof(float), sizeof(result));
        return result;
    }
    uint16x8x2_t bits;
    memcpy(&bits, data + index * sizeof(uint16_t), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        result.val[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits.val[0]), 16));
        result.val[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits.val[0]), 16));
        result.val[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits.val[1]), 16));
        result.val[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits.val[1]), 16));
    } else {
        result.val[0] = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(bits.val[0])));
        result.val[1] = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(bits.val[0])));
        result.val[2] = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(bits.val[1])));
        result.val[3] = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(bits.val[1])));
    }
    return result;
}

/* Finite MXFP4 weights fit exactly in BF16. Scales 0 and 1 need subnormal encodings. */
#define MXFP4_BITS(E, BASE, MIN_SCALE) \
    ((E) == 0 ? (MIN_SCALE) : ((E) == 1 && (MIN_SCALE) == 0x0020 ? 0x0040 : (BASE) + ((E) - 127) * 128))
#define MXFP4_BYTE(E, BASE, MIN_SCALE, SHIFT) ((MXFP4_BITS(E, BASE, MIN_SCALE) >> (SHIFT)) & 255)
#define MXFP4_PALETTE(E, SHIFT, SIGN) \
    (SIGN), \
    MXFP4_BYTE(E, 0x3f00, 0x0020, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x3f80, 0x0040, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x3fc0, 0x0060, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4000, 0x0080, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4040, 0x00c0, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x4080, 0x0100, SHIFT) | (SIGN), \
    MXFP4_BYTE(E, 0x40c0, 0x0140, SHIFT) | (SIGN)
#define MXFP4_ENTRY(E) {MXFP4_PALETTE(E, 0, 0), MXFP4_PALETTE(E, 0, 0), \
                        MXFP4_PALETTE(E, 8, 0), MXFP4_PALETTE(E, 8, 128)}
#define MXFP4_ENTRIES4(E) MXFP4_ENTRY(E), MXFP4_ENTRY((E) + 1), MXFP4_ENTRY((E) + 2), MXFP4_ENTRY((E) + 3)
#define MXFP4_ENTRIES16(E) MXFP4_ENTRIES4(E), MXFP4_ENTRIES4((E) + 4), MXFP4_ENTRIES4((E) + 8), MXFP4_ENTRIES4((E) + 12)
static const uint8_t mxfp4_bf16_bits[256][32] = {
    MXFP4_ENTRIES16(0), MXFP4_ENTRIES16(16), MXFP4_ENTRIES16(32), MXFP4_ENTRIES16(48),
    MXFP4_ENTRIES16(64), MXFP4_ENTRIES16(80), MXFP4_ENTRIES16(96), MXFP4_ENTRIES16(112),
    MXFP4_ENTRIES16(128), MXFP4_ENTRIES16(144), MXFP4_ENTRIES16(160), MXFP4_ENTRIES16(176),
    MXFP4_ENTRIES16(192), MXFP4_ENTRIES16(208), MXFP4_ENTRIES16(224), MXFP4_ENTRIES16(240),
};
#undef MXFP4_ENTRIES16
#undef MXFP4_ENTRIES4
#undef MXFP4_ENTRY
#undef MXFP4_PALETTE
#undef MXFP4_BYTE
#undef MXFP4_BITS

static inline __attribute__((always_inline)) void decode16(const RoutedMatmulTask *task, size_t block, size_t k, float32x4_t weights[4]) {
    uint8_t exponent = task->scales[block + k / 32];
    uint8x8_t packed = vld1_u8(task->codes + block * 16 + k / 2);
    uint8x8x2_t split = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
    uint8x16_t codes = vcombine_u8(split.val[0], split.val[1]);
    uint8x16x2_t tables = vld1q_u8_x2(mxfp4_bf16_bits[exponent]);
    uint8x16_t low = vqtbl1q_u8(tables.val[0], codes);
    uint8x16_t high = vqtbl1q_u8(tables.val[1], codes);
    uint16x8_t a = vreinterpretq_u16_u8(vzip1q_u8(low, high));
    uint16x8_t b = vreinterpretq_u16_u8(vzip2q_u8(low, high));
    weights[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16));
    weights[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a), 16));
    weights[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
    weights[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16));
}

static TrillimStatus validate_weights(const RoutedMatmulTask *task, size_t expert, size_t first, size_t end, bool *bounded) {
    size_t groups = task->p->k / 32;
    size_t block = (expert * task->p->n + first) * groups;
    size_t count = (end - first) * groups;
    const uint8_t *scales = task->scales + block;
    uint8x16_t maximum = vdupq_n_u8(0), minimum = vdupq_n_u8(255);
    size_t i = 0;
    for (; count - i >= 16; i += 16) {
        if ((i & 8191) == 0 && cancelled(task->context)) {
            return TRILLIM_CANCELLED;
        }
        uint8x16_t scale = vld1q_u8(scales + i);
        maximum = vmaxq_u8(maximum, scale);
        minimum = vminq_u8(minimum, scale);
    }
    uint8_t bound = vmaxvq_u8(maximum), lower = vminvq_u8(minimum);
    for (; i < count; ++i) {
        if (scales[i] < lower) {
            lower = scales[i];
        }
        if (scales[i] > bound) {
            bound = scales[i];
        }
    }
    /* Keep Q8 scales, MXFP4 scales, and their summed products within FP32 range. */
    *bounded = lower >= 32 && bound <= 212;
    if (bound < 253) {
        return TRILLIM_OK;
    }

    /* Packed gather rejects NaN scales and weights that overflow FP32. */
    const uint8_t *codes = task->codes + block * 16;
    for (i = 0; i < count; ++i) {
        if ((i & 255) == 0 && cancelled(task->context)) {
            return TRILLIM_CANCELLED;
        }
        uint8_t exponent = scales[i];
        if (exponent < 253) {
            continue;
        }
        uint8x16_t packed = vld1q_u8(codes + i * 16);
        uint8x16_t mask = vdupq_n_u8(7);
        uint8x16_t low = vandq_u8(packed, mask);
        uint8x16_t high = vandq_u8(vshrq_n_u8(packed, 4), mask);
        if (exponent == 255 || vmaxvq_u8(vmaxq_u8(low, high)) >= (exponent == 253 ? 6 : 4)) {
            return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus project_fp32(
    const RoutedMatmulTask *task, size_t expert, size_t first, size_t end, const size_t *slots,
    const uint8_t *const *activation, size_t count, size_t rows, size_t columns
) {
    const RoutedMatmulParameters *p = task->p;
    size_t groups = p->k / 32;
    for (size_t column = first; column < end; column += columns) {
        size_t width = end - column < columns ? end - column : columns;
        size_t blocks[4];
        float32x4_t sums[ROUTES_PER_TILE][4];
        for (size_t c = 0; c < columns; ++c) {
            blocks[c] = (expert * p->n + column + (c < width ? c : 0)) * groups;
            for (size_t r = 0; r < rows; ++r) {
                sums[r][c] = vdupq_n_f32(0);
            }
        }
        for (size_t k = 0; k < p->k; k += 16) {
            if ((k & 1023) == 0 && cancelled(task->context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t weights[4][4];
            for (size_t c = 0; c < columns; ++c) {
                decode16(task, blocks[c], k, weights[c]);
            }
            /* Match packed matmul's four FP32 reduction lanes per output. */
            for (size_t r = 0; r < rows; ++r) {
                float32x4x4_t x = load16(activation[r], k, p->a_dtype);
                for (size_t c = 0; c < columns; ++c) {
                    float32x4_t sum = sums[r][c];
                    sum = vfmaq_f32(sum, x.val[0], weights[c][0]);
                    sum = vfmaq_f32(sum, x.val[1], weights[c][1]);
                    sum = vfmaq_f32(sum, x.val[2], weights[c][2]);
                    sums[r][c] = vfmaq_f32(sum, x.val[3], weights[c][3]);
                }
            }
        }
        for (size_t r = 0; r < count; ++r) {
            for (size_t c = 0; c < width; ++c) {
                TrillimValue value = {.f = vaddvq_f32(sums[r][c])};
                trillim_store(task->output, slots[r] * p->n + column + c, TRILLIM_FP32, value);
            }
        }
    }
    return TRILLIM_OK;
}

enum { Q8_GROUP = 32, Q8_MAX_WIDTH = 8192, Q8_MAX_BYTES = 8 * 1024 * 1024 };

struct RoutedQ8 {
    int8_t values[Q8_GROUP];
    float scale;
};

__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_routed_dot_v\\reg,\\reg\n"
    ".endr\n"
);

static inline int32x4_t routed_dot(int32x4_t sum, int8x16_t a, int8x16_t b) {
    __asm__(".inst 0x4e809400 | trillim_routed_dot_%0 | (trillim_routed_dot_%1 << 5) | (trillim_routed_dot_%2 << 16)"
            : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}

static bool available_routed_dot(void) {
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

static inline bool routed_matmul_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static inline int8x16_t quantize_16(float32x4x4_t x, float inverse) {
    int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(x.val[0], inverse));
    int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(x.val[1], inverse));
    int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(x.val[2], inverse));
    int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(x.val[3], inverse));
    return vcombine_s8(vmovn_s16(vmovn_high_s32(vmovn_s32(q0), q1)),
                       vmovn_s16(vmovn_high_s32(vmovn_s32(q2), q3)));
}

static bool quantize_q8_group(const uint8_t *a, size_t k, uint32_t dtype, RoutedQ8 *out) {
    float32x4x4_t lo = load16(a, k, dtype);
    float32x4x4_t hi = load16(a, k + 16, dtype);
    float32x4_t max_vector = vdupq_n_f32(0);
    for (size_t i = 0; i < 4; ++i) {
        max_vector = vmaxq_f32(max_vector, vabsq_f32(lo.val[i]));
        max_vector = vmaxq_f32(max_vector, vabsq_f32(hi.val[i]));
    }
    float maximum = vmaxvq_f32(max_vector);
    float scale = maximum / 127.0f;
    float inverse = scale != 0 ? 1.0f / scale : 0.0f;
    if (!isfinite(maximum) || !isfinite(inverse) || scale > 65504.0f) {
        return false;
    }
    uint16_t bits = trillim_to_fp16(scale);
    if (bits == 0 && maximum != 0.0f) {
        return false;
    }
    /* A zero group has zero codes, so a unit scale can also represent it. */
    out->scale = bits == 0 ? 1.0f : trillim_fp16(bits);
    int8x16_t x0 = quantize_16(lo, inverse), x1 = quantize_16(hi, inverse);
    /* The packed bank stores even inputs in the low nibbles and odd inputs in the high nibbles. */
    vst1q_s8(out->values, vuzp1q_s8(x0, x1));
    vst1q_s8(out->values + 16, vuzp2q_s8(x0, x1));
    return true;
}

static inline float32x4_t mxfp4_scales(const uint8_t *const scales[4], size_t g) {
    uint8_t exponent[8] = {scales[0][g], scales[1][g], scales[2][g], scales[3][g]};
    uint32x4_t wide = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(exponent))));
    /* The dot uses twice the E2M1 values. Its scale is therefore half the E8M0 scale. */
    return vreinterpretq_f32_u32(vshlq_n_u32(vsubq_u32(wide, vdupq_n_u32(1)), 23));
}

static inline __attribute__((always_inline)) TrillimStatus project_int8(
    const RoutedMatmulTask *task, size_t expert, size_t first, size_t end, const size_t *slots, size_t count, size_t rows
) {
    static const int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t palette = vld1q_s8(values);
    const RoutedMatmulParameters *p = task->p;
    size_t groups = p->k / Q8_GROUP;
    const RoutedQ8 *activation[ROUTES_PER_TILE];
    size_t route_rows = p->a_route_stride == 0 ? 1 : p->routes;
    for (size_t r = 0; r < count; ++r) {
        size_t token = p->a_token_stride == 0 ? 0 : slots[r] / p->routes;
        size_t route = p->a_route_stride == 0 ? 0 : slots[r] % p->routes;
        activation[r] = task->quantized + (token * route_rows + route) * groups;
    }
    for (size_t r = count; r < rows; ++r) {
        activation[r] = activation[0];
    }
    for (size_t column = first; column < end; column += 4) {
        size_t width = end - column < 4 ? end - column : 4;
        const uint8_t *codes[4], *scales[4];
        for (size_t c = 0; c < 4; ++c) {
            size_t block = (expert * p->n + column + (c < width ? c : 0)) * groups;
            codes[c] = task->codes + block * 16;
            scales[c] = task->scales + block;
        }
        float32x4_t sum[ROUTES_PER_TILE] = {0}, partial[ROUTES_PER_TILE] = {0};
        for (size_t g = 0; g < groups; ++g) {
            if ((g & 31) == 0 && cancelled(task->context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t weight_scale = mxfp4_scales(scales, g);
            int8x16_t w0[4], w1[4];
            for (size_t c = 0; c < 4; ++c) {
                uint8x16_t block = vld1q_u8(codes[c] + g * 16);
                w0[c] = vqtbl1q_s8(palette, vandq_u8(block, vdupq_n_u8(15)));
                w1[c] = vqtbl1q_s8(palette, vshrq_n_u8(block, 4));
            }
            for (size_t r = 0; r < rows; ++r) {
                const RoutedQ8 *x = activation[r] + g;
                int8x16_t x0 = vld1q_s8(x->values), x1 = vld1q_s8(x->values + 16);
                int32x4_t dot[4];
                for (size_t c = 0; c < 4; ++c) {
                    dot[c] = routed_dot(routed_dot(vdupq_n_s32(0), w0[c], x0), w1[c], x1);
                }
                int32x4_t reduced = vpaddq_s32(vpaddq_s32(dot[0], dot[1]), vpaddq_s32(dot[2], dot[3]));
                float32x4_t scale = vmulq_n_f32(weight_scale, x->scale);
                if ((g & 1) == 0) {
                    partial[r] = vmulq_f32(vcvtq_f32_s32(reduced), scale);
                } else {
                    sum[r] = vaddq_f32(sum[r], vfmaq_f32(partial[r], vcvtq_f32_s32(reduced), scale));
                }
            }
        }
        for (size_t r = 0; r < count; ++r) {
            float32x4_t result = groups % 2 == 0 ? sum[r] : vaddq_f32(sum[r], partial[r]);
            size_t start = slots[r] * p->n + column;
            memcpy((uint8_t *)task->output + start * sizeof(float), &result, width * sizeof(float));
        }
    }
    return TRILLIM_OK;
}

typedef struct {
    float32x4_t dot, scale;
} RoutedBlock;

static inline __attribute__((always_inline)) RoutedBlock decode_int8_block(
    const uint8_t *const codes[4], const uint8_t *const scales[4],
    const RoutedQ8 *x, size_t g, int8x16_t palette
) {
    int8x16x2_t activation = vld1q_s8_x2(x[g].values);
    int32x4_t dot[4];
    for (size_t c = 0; c < 4; ++c) {
        uint8x16_t packed = vld1q_u8(codes[c] + g * 16);
        int8x16_t w0 = vqtbl1q_s8(palette, vandq_u8(packed, vdupq_n_u8(15)));
        int8x16_t w1 = vqtbl1q_s8(palette, vshrq_n_u8(packed, 4));
        dot[c] = routed_dot(routed_dot(vdupq_n_s32(0), w0, activation.val[0]), w1, activation.val[1]);
    }
    int32x4_t combined = vpaddq_s32(vpaddq_s32(dot[0], dot[1]), vpaddq_s32(dot[2], dot[3]));
    uint8_t exponent[8] = {scales[0][g], scales[1][g], scales[2][g], scales[3][g]};
    uint32x4_t expanded = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(exponent))));
    uint32_t adjusted;
    memcpy(&adjusted, &x[g].scale, sizeof(adjusted));
    /* Bounded E8M0 scales multiply the normal FP32 activation scale by a power of two. */
    adjusted -= UINT32_C(128) << 23;
    uint32x4_t scale_bits = vaddq_u32(vshlq_n_u32(expanded, 23), vdupq_n_u32(adjusted));
    return (RoutedBlock){vcvtq_f32_s32(combined), vreinterpretq_f32_u32(scale_bits)};
}

static TrillimStatus project_int8_decode(const RoutedMatmulTask *task, size_t expert, size_t first, size_t end, size_t slot) {
    static const int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int8x16_t palette = vld1q_s8(values);
    const RoutedMatmulParameters *p = task->p;
    size_t groups = p->k / Q8_GROUP;
    const RoutedQ8 *x = task->quantized + (p->a_route_stride == 0 ? 0 : slot) * groups;
    for (size_t column = first; column < end; column += 4) {
        size_t block = (expert * p->n + column) * groups;
        const uint8_t *codes[4], *scales[4];
        for (size_t c = 0; c < 4; ++c) {
            codes[c] = task->codes + (block + c * groups) * 16;
            scales[c] = task->scales + block + c * groups;
        }
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t g = 0;
        for (; g + 1 < groups; g += 2) {
            if ((g & 31) == 0 && cancelled(task->context)) {
                return TRILLIM_CANCELLED;
            }
            RoutedBlock a = decode_int8_block(codes, scales, x, g, palette);
            RoutedBlock b = decode_int8_block(codes, scales, x, g + 1, palette);
            sum = vaddq_f32(sum, vfmaq_f32(vmulq_f32(a.dot, a.scale), b.dot, b.scale));
        }
        if (g < groups) {
            if ((g & 31) == 0 && cancelled(task->context)) {
                return TRILLIM_CANCELLED;
            }
            RoutedBlock last = decode_int8_block(codes, scales, x, g, palette);
            sum = vaddq_f32(sum, vmulq_f32(last.dot, last.scale));
        }
        memcpy((uint8_t *)task->output + (slot * p->n + column) * sizeof(float), &sum, sizeof(sum));
    }
    return TRILLIM_OK;
}

static TrillimStatus routed_matmul_project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const RoutedMatmulTask *task = opaque;
    const RoutedMatmulParameters *p = task->p;
    size_t route_tile = tile / task->column_tiles;
    size_t low = 0, high = p->experts;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (task->tiles[mid + 1] <= route_tile) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    size_t expert = low;
    size_t offset = task->offsets[expert] + (route_tile - task->tiles[expert]) * ROUTES_PER_TILE;
    size_t count = task->offsets[expert + 1] - offset;
    if (count > ROUTES_PER_TILE) {
        count = ROUTES_PER_TILE;
    }
    const size_t *slots = task->slots + offset;
    const uint8_t *activation[ROUTES_PER_TILE];
    size_t a_size = trillim_itemsize(p->a_dtype);
    for (size_t r = 0; r < count; ++r) {
        size_t index = slots[r] / p->routes * p->a_token_stride + slots[r] % p->routes * p->a_route_stride;
        activation[r] = task->activation + index * a_size;
    }
    for (size_t r = count; r < ROUTES_PER_TILE; ++r) {
        activation[r] = activation[0];
    }
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->n - first < task->columns ? p->n : first + task->columns;
    bool bounded;
    TrillimStatus status = validate_weights(task, expert, first, end, &bounded);
    if (status != TRILLIM_OK) {
        return status;
    }

    if (!bounded && task->doubled_columns && routed_matmul_default_fpcr()) {
        /* An exceptional scale only disables Q8 for its original narrower tile. */
        size_t columns = task->columns / 2;
        for (size_t column = first; column < end; column += columns) {
            size_t limit = end - column < columns ? end : column + columns;
            status = validate_weights(task, expert, column, limit, &bounded);
            if (status != TRILLIM_OK) {
                return status;
            }
            if (bounded) {
                status = project_int8_decode(task, expert, column, limit, slots[0]);
            } else {
                status = project_fp32(task, expert, column, limit, slots, activation, 1, 1, 4);
            }
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    if (task->quantized != NULL && bounded && routed_matmul_default_fpcr()) {
        if (p->tokens == 1 && count == 1 && p->n % 4 == 0) {
            return project_int8_decode(task, expert, first, end, slots[0]);
        }
        if (count == 1) {
            return project_int8(task, expert, first, end, slots, 1, 1);
        }
        if (count == 2) {
            return project_int8(task, expert, first, end, slots, 2, 2);
        }
        return project_int8(task, expert, first, end, slots, count, ROUTES_PER_TILE);
    }
    if (count == 1) {
        return project_fp32(task, expert, first, end, slots, activation, 1, 1, 4);
    }
    if (count == 2) {
        return project_fp32(task, expert, first, end, slots, activation, 2, 2, 2);
    }
    return project_fp32(task, expert, first, end, slots, activation, count, ROUTES_PER_TILE, 2);
}

static __attribute__((noinline)) TrillimStatus try_routed_int8(RoutedMatmulTask *task, bool *handled) {
    const RoutedMatmulParameters *p = task->p;
    size_t groups = p->k / Q8_GROUP;
    size_t token_rows = p->a_token_stride == 0 ? 1 : p->tokens;
    size_t route_rows = p->a_route_stride == 0 ? 1 : p->routes;
    size_t max_groups = Q8_MAX_BYTES / sizeof(RoutedQ8);
    if (token_rows > max_groups / route_rows / groups) {
        return TRILLIM_OK;
    }
    size_t total = token_rows * route_rows * groups;
    RoutedQ8 local[512];
    RoutedQ8 *quantized = total <= sizeof(local) / sizeof(local[0]) ? local : malloc(total * sizeof(*quantized));
    if (quantized == NULL) {
        return TRILLIM_OK;
    }
    size_t a_size = trillim_itemsize(p->a_dtype);
    TrillimStatus status = TRILLIM_OK;
    bool valid = true;
    for (size_t t = 0; t < token_rows && valid; ++t) {
        for (size_t r = 0; r < route_rows && valid; ++r) {
            size_t start = t * p->a_token_stride + r * p->a_route_stride;
            const uint8_t *activation = task->activation + start * a_size;
            RoutedQ8 *row = quantized + (t * route_rows + r) * groups;
            for (size_t g = 0; g < groups; ++g) {
                if ((g & 31) == 0 && cancelled(task->context)) {
                    status = TRILLIM_CANCELLED;
                    valid = false;
                    break;
                }
                if (!quantize_q8_group(activation, g * Q8_GROUP, p->a_dtype, row + g)) {
                    valid = false;
                    break;
                }
            }
        }
    }
    if (valid) {
        if (p->tokens == 1 && p->routes == 4 &&
            task->tiles[p->experts] == p->routes && p->n % 4 == 0) {
            task->doubled_columns = true;
            task->columns *= 2;
            task->column_tiles = p->n / task->columns + (p->n % task->columns != 0);
        }
        task->quantized = quantized;
        *handled = true;
        const TrillimThreadPool *pool = task->context->thread_pool;
        status = pool->run(pool->context, task->tiles[p->experts] * task->column_tiles, routed_matmul_project, task);
        task->quantized = NULL;
    }
    if (quantized != local) {
        free(quantized);
    }
    return status;
}

static TrillimStatus routed_matmul_run(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const RoutedMatmulParameters *p = parameters;
    if (cancelled(context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers || scratch == NULL) {
        return TRILLIM_ERROR;
    }
    size_t *offsets = scratch;
    size_t *tiles = offsets + p->experts + 1;
    size_t *slots = tiles + p->experts + 1;
    TrillimStatus status = group_routes(p, inputs[2], offsets, tiles, slots, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t columns = p->n / workers / 2 / 4 * 4;
    if (columns < 16) {
        columns = 16;
    }
    size_t column_tiles = p->n / columns + (p->n % columns != 0);
    const uint8_t *codes = inputs[1];
    const uint8_t *scales = codes + p->experts * p->n * (p->k / 32) * 16;
    RoutedMatmulTask task = {outputs[0], inputs[0], codes, scales, p, context, offsets, tiles, slots, columns, column_tiles, NULL, false};
    if (p->n >= 16 && p->k <= Q8_MAX_WIDTH && available_routed_dot() && routed_matmul_default_fpcr()) {
        bool handled = false;
        status = try_routed_int8(&task, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
    return context->thread_pool->run(context->thread_pool->context, tiles[p->experts] * column_tiles, routed_matmul_project, &task);
}


static const TrillimParameterSchema *routed_matmul_schema(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(RoutedMatmulParameters, tokens), NULL},
        {"routes", "size_t", offsetof(RoutedMatmulParameters, routes), NULL},
        {"experts", "size_t", offsetof(RoutedMatmulParameters, experts), NULL},
        {"n", "size_t", offsetof(RoutedMatmulParameters, n), NULL},
        {"k", "size_t", offsetof(RoutedMatmulParameters, k), NULL},
        {"a_token_stride", "size_t", offsetof(RoutedMatmulParameters, a_token_stride), NULL},
        {"a_route_stride", "size_t", offsetof(RoutedMatmulParameters, a_route_stride), NULL},
        {"index_token_stride", "size_t", offsetof(RoutedMatmulParameters, index_token_stride), NULL},
        {"index_route_stride", "size_t", offsetof(RoutedMatmulParameters, index_route_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(RoutedMatmulParameters, a_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(RoutedMatmulParameters), TRILLIM_ALIGNOF(RoutedMatmulParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, routed_matmul_requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(size_t)
    };
    return &schema;
}

/* Expert activation. */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t rows, width;
} InterleavedClippedSwigluParameters;

typedef struct {
    float *output;
    const float *input;
    const InterleavedClippedSwigluParameters *p;
    const TrillimKernelContext *context;
} InterleavedClippedSwigluTask;

static TrillimStatus interleaved_clipped_swiglu_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const InterleavedClippedSwigluParameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->width % 4) {
        return TRILLIM_ERROR;
    }
    if (!p->width || !p->rows) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / (2 * sizeof(float)) / p->width) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
    inputs[0] = outputs[0] * 2;
    return TRILLIM_OK;
}

/* Match the staged FP32 arithmetic in clipped_swiglu_neon.c. */
static float activate(float gate, float up) {
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    return (gate * probability) * (up + 1.0f);
}

static inline float32x4_t exp_negative(float32x4_t original) {
    float32x4_t x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(original, vdupq_n_f32(0)));
    float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(0.00021219444005469058f));
    float32x4_t value = vdupq_n_f32(1.0f / 5040.0f);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(0.5f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    int32x4_t exponent = vcvtq_s32_f32(n);
    int32x4_t low = vminq_s32(vaddq_s32(exponent, vdupq_n_s32(126)), vdupq_n_s32(0));
    int32x4_t high = vsubq_s32(exponent, low);
    float32x4_t main_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(high, vdupq_n_s32(127)), 23));
    float32x4_t tail_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(low, vdupq_n_s32(127)), 23));
    value = vmulq_f32(vmulq_f32(value, main_scale), tail_scale);
    return vbslq_f32(vceqq_f32(original, original), value, original);
}

static float32x4_t activate4(float32x4_t gate, float32x4_t up) {
    float32x4_t low = vdupq_n_f32(-7.0f), high = vdupq_n_f32(7.0f);
    gate = vbslq_f32(vcleq_f32(gate, high), gate, high);
    up = vbslq_f32(vcgeq_f32(up, low), up, low);
    up = vbslq_f32(vcleq_f32(up, high), up, high);
    float32x4_t scaled = vmulq_n_f32(gate, 1.702f);
    float32x4_t e = exp_negative(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t denominator = vaddq_f32(e, one);
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0)),
        vdivq_f32(one, denominator), vdivq_f32(e, denominator));
    float32x4_t result = vmulq_f32(vmulq_f32(gate, probability), vaddq_f32(up, one));
    if (vminvq_f32(scaled) < -80.0f) {
        float gates[4], ups[4], values[4], arguments[4];
        vst1q_f32(gates, gate);
        vst1q_f32(ups, up);
        vst1q_f32(values, result);
        vst1q_f32(arguments, scaled);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (arguments[lane] < -80.0f) {
                values[lane] = activate(gates[lane], ups[lane]);
            }
        }
        result = vld1q_f32(values);
    }
    return result;
}

static TrillimStatus apply(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const InterleavedClippedSwigluTask *task = opaque;
    size_t width = task->p->width;
    const float *source = task->input + row * width * 2;
    float *output = task->output + row * width;
    for (size_t i = 0; i < width; i += 4) {
        if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4x2_t pair = vld2q_f32(source + 2 * i);
        vst1q_f32(output + i, activate4(pair.val[0], pair.val[1]));
    }
    return TRILLIM_OK;
}

static inline bool interleaved_clipped_swiglu_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus interleaved_clipped_swiglu_run(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const InterleavedClippedSwigluParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    InterleavedClippedSwigluTask task = {outputs[0], inputs[0], p, context};
    bool decode_parallel = p->rows == 4 && p->width >= 2048 &&
        context->thread_pool->worker_count >= 4 && interleaved_clipped_swiglu_default_fpcr();
    if (decode_parallel || (p->rows > 4 && p->rows > 16384 / p->width)) {
        return context->thread_pool->run(context->thread_pool->context, p->rows, apply, &task);
    }
    for (size_t row = 0; row < p->rows; ++row) {
        TrillimStatus status = apply(&task, row, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}


static const TrillimParameterSchema *interleaved_clipped_swiglu_schema(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(InterleavedClippedSwigluParameters, rows), NULL},
        {"width", "size_t", offsetof(InterleavedClippedSwigluParameters, width), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(InterleavedClippedSwigluParameters), TRILLIM_ALIGNOF(InterleavedClippedSwigluParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, interleaved_clipped_swiglu_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

enum { ROUTES = 4, PANEL_ROWS = 128 };

typedef struct {
    size_t rows, width, hidden, experts;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 7 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->rows > 65536 || p->experts != 32 || p->width == 0 || p->hidden == 0 ||
        p->width > 8192 || p->hidden > 8192 || p->width % 32 || p->hidden % 32) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->experts * p->width * sizeof(uint16_t);
    inputs[2] = p->experts * sizeof(uint16_t);
    inputs[3] = p->experts * p->hidden * 2 * (p->width / 32) * 17;
    inputs[4] = p->experts * p->width * (p->hidden / 32) * 17;
    inputs[5] = p->experts * p->hidden * 2 * sizeof(uint16_t);
    inputs[6] = p->experts * p->width * sizeof(uint16_t);
    outputs[0] = p->rows * p->width * sizeof(float);
    if (p->rows != 0) {
        size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
        size_t slots = rows * ROUTES;
        size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
        *scratch = slots * (projection_width + p->hidden + 1) * sizeof(float) +
            slots * sizeof(int64_t) + (2 * (p->experts + 1) + slots) * sizeof(size_t) +
            (rows * p->experts + slots) * sizeof(uint16_t);
    }
    return TRILLIM_OK;
}

typedef struct {
    size_t batches, m, n, k, rank;
    const size_t *batch_shape, *a_batch_strides, *b_batch_strides;
    size_t a_row_stride, a_col_stride, b_row_stride, b_col_stride;
    uint32_t a_dtype, b_dtype, out_dtype, linear_dtype, bias_dtype;
    size_t a_storage_rows, a_storage_columns, b_storage_rows, b_storage_columns;
} RouterParameters;

typedef struct {
    size_t tokens, routes, experts, n, k;
    size_t a_token_stride, a_route_stride, index_token_stride, index_route_stride;
    uint32_t a_dtype;
} ProjectionParameters;

typedef struct {
    size_t rows;
} TopKParameters;

typedef struct {
    size_t rows, width;
    uint32_t dtype;
} SoftmaxParameters;

typedef struct {
    size_t rows, width;
} SwiGLUParameters;

typedef struct {
    float *projected, *hidden;
    int64_t *indices;
    size_t *routing;
    float *normalization;
    uint16_t *logits, *scores;
} Workspace;

typedef struct {
    float *projected;
    const uint16_t *bias;
    const int64_t *indices;
    size_t width;
    const TrillimKernelContext *context;
} BiasTask;

typedef struct {
    float *output;
    const float *projected;
    const uint16_t *bias, *weights;
    const int64_t *indices;
    size_t width;
    const TrillimKernelContext *context;
} OutputTask;

static inline float32x4_t bias_values(const uint16_t *bias) {
    uint16x4_t bits = vld1_u16(bias);
    uint16x4_t magnitude = vand_u16(bits, vdup_n_u16(0x7fff));
    uint16x4_t nan = vcgt_u16(magnitude, vdup_n_u16(0x7f80));
    /* Gather writes a quiet BF16 NaN before the bias is widened. */
    bits = vorr_u16(bits, vand_u16(nan, vdup_n_u16(0x0040)));
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static TrillimStatus add_up_bias(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const BiasTask *task = opaque;
    float *row = task->projected + slot * task->width;
    const uint16_t *bias = task->bias + (size_t)task->indices[slot] * task->width;
    for (size_t i = 0; i < task->width; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        vst1q_f32(row + i, vaddq_f32(vld1q_f32(row + i), bias_values(bias + i)));
    }
    return TRILLIM_OK;
}

static TrillimStatus finish_output(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const OutputTask *task = opaque;
    size_t width = task->width, slot = token * ROUTES;
    float *destination = task->output + token * width;
    const float *projected[ROUTES];
    const uint16_t *bias[ROUTES];
    float weights[ROUTES];
    for (size_t route = 0; route < ROUTES; ++route) {
        projected[route] = task->projected + (slot + route) * width;
        bias[route] = task->bias + (size_t)task->indices[slot + route] * width;
        weights[route] = trillim_bf16(task->weights[slot + route]);
    }
    for (size_t i = 0; i < width; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t sum = vdupq_n_f32(0);
        for (size_t route = 0; route < ROUTES; ++route) {
            float32x4_t projected_with_bias = vaddq_f32(vld1q_f32(projected[route] + i), bias_values(bias[route] + i));
            sum = vaddq_f32(sum, vmulq_n_f32(projected_with_bias, weights[route]));
        }
        vst1q_f32(destination + i, sum);
    }
    return TRILLIM_OK;
}

static TrillimStatus panel(
    float *output, const uint16_t *activation, const void *const *inputs,
    const Parameters *p, size_t rows, const Workspace *workspace, const TrillimKernelContext *context
) {
    size_t width = p->width, hidden = p->hidden, experts = p->experts;
    RouterParameters router = {
        .batches = 1, .m = rows, .n = experts, .k = width,
        .a_row_stride = width, .a_col_stride = 1, .b_row_stride = 1, .b_col_stride = width,
        .a_dtype = TRILLIM_BF16, .b_dtype = TRILLIM_BF16, .out_dtype = TRILLIM_BF16,
        .linear_dtype = TRILLIM_BF16, .bias_dtype = TRILLIM_BF16,
        .a_storage_rows = rows, .a_storage_columns = width,
        .b_storage_rows = experts, .b_storage_columns = width,
    };
    TrillimStatus status = dense_bias_run(
        (void *[]){workspace->logits}, (const void *[]){activation, inputs[1], inputs[2]},
        NULL, &router, context
    );
    if (status != TRILLIM_OK) return status;
    TopKParameters topk = {rows};
    status = topk4_run(
        (void *[]){workspace->scores, workspace->indices}, (const void *[]){workspace->logits},
        NULL, &topk, context
    );
    if (status != TRILLIM_OK) return status;
    SoftmaxParameters softmax = {rows, ROUTES, TRILLIM_BF16};
    status = softmax_run(
        (void *[]){workspace->scores}, (const void *[]){workspace->scores},
        workspace->normalization, &softmax, context
    );
    if (status != TRILLIM_OK) return status;
    ProjectionParameters up = {
        .tokens = rows, .routes = ROUTES, .experts = experts, .n = hidden * 2, .k = width,
        .a_token_stride = width, .a_route_stride = 0,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_BF16,
    };
    status = routed_matmul_run(
        (void *[]){workspace->projected}, (const void *[]){activation, inputs[3], workspace->indices},
        workspace->routing, &up, context
    );
    if (status != TRILLIM_OK) return status;
    size_t slots = rows * ROUTES;
    const TrillimThreadPool *pool = context->thread_pool;
    BiasTask bias = {workspace->projected, inputs[5], workspace->indices, hidden * 2, context};
    if (slots > 16 && slots > 32768 / bias.width) {
        status = pool->run(pool->context, slots, add_up_bias, &bias);
    } else {
        for (size_t slot = 0; slot < slots && status == TRILLIM_OK; ++slot) {
            status = add_up_bias(&bias, slot, 0);
        }
    }
    if (status != TRILLIM_OK) return status;
    SwiGLUParameters swiglu = {slots, hidden};
    status = interleaved_clipped_swiglu_run(
        (void *[]){workspace->hidden}, (const void *[]){workspace->projected},
        NULL, &swiglu, context
    );
    if (status != TRILLIM_OK) return status;
    ProjectionParameters down = {
        .tokens = rows, .routes = ROUTES, .experts = experts, .n = width, .k = hidden,
        .a_token_stride = ROUTES * hidden, .a_route_stride = hidden,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_FP32,
    };
    status = routed_matmul_run(
        (void *[]){workspace->projected}, (const void *[]){workspace->hidden, inputs[4], workspace->indices},
        workspace->routing, &down, context
    );
    if (status != TRILLIM_OK) return status;
    OutputTask result = {output, workspace->projected, inputs[6], workspace->scores, workspace->indices, width, context};
    return rows == 1 ? finish_output(&result, 0, 0) : pool->run(pool->context, rows, finish_output, &result);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gpt_oss_moe(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->rows == 0) {
        return TRILLIM_OK;
    }
    size_t spans[7], output_span, required;
    if (!scratch || requirements(p, spans, &output_span, &required) != TRILLIM_OK ||
        dense_bias_schema()->size != sizeof(RouterParameters) ||
        topk4_schema()->size != sizeof(TopKParameters) ||
        softmax_schema()->size != sizeof(SoftmaxParameters) ||
        routed_matmul_schema()->size != sizeof(ProjectionParameters) ||
        interleaved_clipped_swiglu_schema()->size != sizeof(SwiGLUParameters)) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
    size_t slots = rows * ROUTES;
    size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
    Workspace workspace;
    workspace.projected = scratch;
    workspace.hidden = workspace.projected + slots * projection_width;
    workspace.indices = (int64_t *)(workspace.hidden + slots * p->hidden);
    workspace.routing = (size_t *)(workspace.indices + slots);
    workspace.normalization = (float *)(workspace.routing + 2 * (p->experts + 1) + slots);
    workspace.logits = (uint16_t *)(workspace.normalization + slots);
    workspace.scores = workspace.logits + rows * p->experts;

    const uint16_t *activation = inputs[0];
    float *output = outputs[0];
    for (size_t first = 0; first < p->rows; first += rows) {
        size_t length = p->rows - first < rows ? p->rows - first : rows;
        TrillimStatus status = panel(output + first * p->width, activation + first * p->width,
                                     inputs, p, length, &workspace, context);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_neon_gpt_oss_moe_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'router', 'router_bias', 'gate_up', 'down', 'gate_up_bias', 'down_bias')\n"
           "def pattern(x: Tensor, *, router: Tensor, router_bias: Tensor, gate_up: Tensor, down: Tensor,\n"
           "            gate_up_bias: Tensor, down_bias: Tensor, top_k: int = 4, limit: float = 7.0) -> tuple[Tensor, ...]:\n"
           "    if top_k != 4 or limit != 7.0 or x.dtype != 'bf16' or router.dtype != 'bf16' or gate_up.dtype != 'mxfp4' or down.dtype != 'mxfp4':\n"
           "        raise ValueError('expected four routes, BF16 logits, and MXFP4 experts')\n"
           "    return (GptOssMoE(x, router=router, router_bias=router_bias, gate_up=gate_up, down=down,\n"
           "                     gate_up_bias=gate_up_bias, down_bias=down_bias, top_k=top_k, limit=limit),)\n"
           "def bind(c):\n"
           "    x, router, rb, gate_up, down, gb, db = (c.inputs[k] for k in physical_inputs)\n"
           "    if tuple(len(v.shape) for v in (x, router, rb, gate_up, down, gb, db)) != (2, 2, 1, 3, 3, 2, 2):\n"
           "        raise ValueError('expected contiguous activation and expert banks')\n"
           "    if any(v.strides and v.strides != _contiguous_strides(v.shape) for v in (x, router, rb, gate_up, down, gb, db)):\n"
           "        raise ValueError('noncontiguous MoE input')\n"
           "    if rb.dtype != 'bf16' or gb.dtype != 'bf16' or db.dtype != 'bf16' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('expected BF16 biases and FP32 output')\n"
           "    experts, doubled, width = gate_up.shape\n"
           "    if any(_storage_shape(w) != (experts, _product(w.shape[1:])) for w in (gate_up, down)):\n"
           "        raise ValueError('expected canonical MXFP4 expert banks')\n"
           "    if router.shape != (experts, width) or rb.shape != (experts,) or gb.shape != (experts, doubled) or down.shape[:2] != (experts, width) or db.shape != (experts, width):\n"
           "        raise ValueError('incompatible expert bank dimensions')\n"
           "    hidden = down.shape[-1]\n"
           "    if hidden.value is None or doubled.value != 2 * hidden.value or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible expert projection widths')\n"
           "    out = c.outputs[0]\n"
           "    if len(out.shape) != 2 or out.shape[-1] != width or (out.strides and out.strides != _contiguous_strides(out.shape)):\n"
           "        raise ValueError('expected contiguous FP32 output rows')\n"
           "    return dict(rows=x.shape[0], width=width, hidden=hidden, experts=experts)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gpt_oss_moe_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
    };
    static const size_t inputs[] = {2, 2, 2, 1, 1, 2, 2}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields,
        7, 1, requirements, NULL, inputs, outputs, 64,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gpt_oss_moe_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
