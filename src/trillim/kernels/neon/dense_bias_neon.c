#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
} Parameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const void *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
    float *packed;
} Task;

static bool packed_fp32(const Parameters *p) {
    return p->batches == 1 && p->rank == 0 && p->m >= 16 && p->n >= 16 && p->k >= 32 &&
        p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32 &&
        p->linear_dtype == TRILLIM_FP32 && p->bias_dtype == TRILLIM_FP32 && p->out_dtype == TRILLIM_FP32;
}

static bool precise(const Parameters *p) {
    return p->out_dtype != TRILLIM_FP32 || p->linear_dtype != TRILLIM_FP32;
}

static size_t block_n(const Parameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t block_k(const Parameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t block_m(const Parameters *p) {
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


static bool bf16_blas_shape(const Parameters *p) {
#if defined(__APPLE__)
    return p->a_dtype == TRILLIM_BF16 && p->b_dtype == TRILLIM_BF16 &&
        p->out_dtype == TRILLIM_BF16 &&
        p->linear_dtype == TRILLIM_BF16 && p->bias_dtype == TRILLIM_BF16 &&
        p->rank == 0 && p->batches == 1 &&
        p->m >= 32 && p->n >= 128 && p->k >= 256;
#else
    (void)p;
    return false;
#endif
}

static bool blas_shape(const Parameters *p, size_t workers) {
    bool bf16_contiguous = bf16_blas_shape(p) &&
        p->a_col_stride == 1 && p->a_row_stride == p->k &&
        p->b_row_stride == 1 && p->b_col_stride == p->k;
    if ((precise(p) && !bf16_contiguous) || p->n > INT32_MAX || p->n < 32 || p->k < 32) {
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

static bool reserve_blas(const Parameters *p) {
    /* Runtime dimensions can make bounded strides contiguous. */
    if ((precise(p) && !bf16_blas_shape(p)) || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool possibly_contiguous = p->k >= 256 && p->n > 32767 / p->k;
    return large || medium || vocabulary || possibly_contiguous;
}

static bool use_blas(const Parameters *p, const TrillimKernelContext *context) {
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

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
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

static inline float add_bias(const Task *t, size_t column, float value) {
    if (t->p->linear_dtype == TRILLIM_FP16) {
        value = trillim_fp16(trillim_to_fp16(value));
    } else if (t->p->linear_dtype == TRILLIM_BF16) {
        value = trillim_bf16(trillim_to_bf16(value));
    }
    return value + trillim_load(t->bias, column, t->p->bias_dtype).f;
}

static inline void store(const Task *t, size_t index, float value) {
    TrillimValue result = {.f = add_bias(t, index % t->p->n, value)};
    trillim_store(t->output, index, t->p->out_dtype, result);
}

static inline bool ordered_narrow_shape(const Parameters *p) {
    return p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32 &&
        p->n % 8 == 0 && p->k >= 4 && p->k % 4 == 0;
}

static size_t tile_rows(const Parameters *p, size_t workers) {
    if (precise(p)) {
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t tiles = p->batches * (p->m / 8) * columns;
        return p->m >= 8 && tiles >= workers && ordered_narrow_shape(p) ? 8 : 4;
    }
    return p->m >= 12 && p->a_col_stride == 1 && p->b_row_stride == 1 ? 6 : 4;
}

static size_t tile_columns(const Parameters *p, size_t workers) {
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
    const Task *t, size_t a_base, size_t b_base, size_t output,
    size_t first, size_t last, uint32_t dtype, size_t rows
) {
    const Parameters *p = t->p;
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
    const Task *t, size_t a_base, size_t b_base, size_t output,
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

static inline __attribute__((always_inline)) TrillimStatus decode_typed_tile(const Task *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const Parameters *p = t->p;
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

static TrillimStatus decode_tile(const Task *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const Parameters *p = t->p;
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

static inline __attribute__((always_inline)) float32x4x4_t ordered_weights(const Task *t, size_t base, size_t column, size_t k, uint32_t dtype) {
    size_t stride = t->p->b_col_stride;
    float32x4_t w0 = load4(t->b, base + column * stride + k, dtype);
    float32x4_t w1 = load4(t->b, base + (column + 1) * stride + k, dtype);
    float32x4_t w2 = load4(t->b, base + (column + 2) * stride + k, dtype);
    float32x4_t w3 = load4(t->b, base + (column + 3) * stride + k, dtype);
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

static inline __attribute__((always_inline)) TrillimStatus ordered_decode16_typed(const Task *t, size_t a_base, size_t b_base, size_t output, size_t column, uint32_t a_dtype, uint32_t b_dtype) {
    const Parameters *p = t->p;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t k = 0;
    for (; p->k - k >= 4; k += 4) {
        if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t x = load4(t->a, a_base + k, a_dtype);
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

static inline __attribute__((always_inline)) uint16x8x4_t ordered_bf16_weights8(const uint16_t *w, size_t stride) {
    const uint16x8_t a = vld1q_u16(w);
    const uint16x8_t b = vld1q_u16(w + stride);
    const uint16x8_t c = vld1q_u16(w + 2 * stride);
    const uint16x8_t d = vld1q_u16(w + 3 * stride);
    const uint16x8x2_t ab = vzipq_u16(a, b);
    const uint16x8x2_t cd = vzipq_u16(c, d);
    const uint32x4x2_t low = vzipq_u32(vreinterpretq_u32_u16(ab.val[0]), vreinterpretq_u32_u16(cd.val[0]));
    const uint32x4x2_t high = vzipq_u32(vreinterpretq_u32_u16(ab.val[1]), vreinterpretq_u32_u16(cd.val[1]));
    return (uint16x8x4_t){{
        vreinterpretq_u16_u32(low.val[0]), vreinterpretq_u16_u32(low.val[1]),
        vreinterpretq_u16_u32(high.val[0]), vreinterpretq_u16_u32(high.val[1])
    }};
}

static TrillimStatus ordered_bf16_decode16(
    const Task *t, size_t a_base, size_t b_base, size_t output, size_t column
) {
    const Parameters *p = t->p;
    const uint16_t *input = (const uint16_t *)t->a + a_base;
    const uint16_t *weights = (const uint16_t *)t->b + b_base + column * p->b_col_stride;
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < p->k; k += 8) {
        if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if ((k & 63) == 0 && p->k - k > 128) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
            for (size_t i = 0; i < 16; ++i) {
                __builtin_prefetch(weights + i * p->b_col_stride + k + 128, 0, 0);
            }
        }
        const uint16x8_t activation = vld1q_u16(input + k);
        const float32x4_t x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(activation), 16));
        const float32x4_t x1 = vreinterpretq_f32_u32(vshll_high_n_u16(activation, 16));
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
        for (size_t group = 0; group < 4; ++group) {
            const uint16x8x4_t w = ordered_bf16_weights8(weights + k + group * 4 * p->b_col_stride, p->b_col_stride);
            float32x4_t sum = sums[group];
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w.val[0]), 16)), x0, 0);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_high_n_u16(w.val[0], 16)), x0, 1);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w.val[1]), 16)), x0, 2);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_high_n_u16(w.val[1], 16)), x0, 3);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w.val[2]), 16)), x1, 0);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_high_n_u16(w.val[2], 16)), x1, 1);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w.val[3]), 16)), x1, 2);
            sum = vfmaq_laneq_f32(sum, vreinterpretq_f32_u32(vshll_high_n_u16(w.val[3], 16)), x1, 3);
            sums[group] = sum;
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

static TrillimStatus ordered_decode16(const Task *t, size_t a_base, size_t b_base, size_t output, size_t column) {
    const Parameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP16 && p->b_dtype == TRILLIM_FP16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_FP16, TRILLIM_FP16);
    }
    if (p->a_dtype == TRILLIM_BF16 && p->b_dtype == TRILLIM_BF16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_BF16, TRILLIM_BF16);
    }
    return ordered_decode16_typed(t, a_base, b_base, output, column, p->a_dtype, p->b_dtype);
}

static TrillimStatus ordered_dot_tile(const Task *t, size_t a_base, size_t b_base,
                                      size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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
        if (p->m == 1 && p->a_dtype == TRILLIM_BF16 && p->b_dtype == TRILLIM_BF16 &&
            p->k >= 8 && p->k % 8 == 0 && p->b_col_stride == p->k) {
            for (; last - first >= 16; first += 16) {
                const TrillimStatus status = ordered_bf16_decode16(t, a_base, b_base, output, first);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
        }
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
            float32x4_t w0 = load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = load4(t->b, b3 + k, p->b_dtype);
            float32x4x2_t even = vtrnq_f32(w0, w1);
            float32x4x2_t odd = vtrnq_f32(w2, w3);
            float32x4_t c0 = vcombine_f32(vget_low_f32(even.val[0]), vget_low_f32(odd.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(even.val[1]), vget_low_f32(odd.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(even.val[0]), vget_high_f32(odd.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(even.val[1]), vget_high_f32(odd.val[1]));
            float32x4_t x0 = load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0 = vfmaq_laneq_f32(s0, c0, x0, 0);
            s0 = vfmaq_laneq_f32(s0, c1, x0, 1);
            s0 = vfmaq_laneq_f32(s0, c2, x0, 2);
            s0 = vfmaq_laneq_f32(s0, c3, x0, 3);
            if (rows > 1) {
                float32x4_t x1 = load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1 = vfmaq_laneq_f32(s1, c0, x1, 0);
                s1 = vfmaq_laneq_f32(s1, c1, x1, 1);
                s1 = vfmaq_laneq_f32(s1, c2, x1, 2);
                s1 = vfmaq_laneq_f32(s1, c3, x1, 3);
            }
            if (rows > 2) {
                float32x4_t x2 = load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2 = vfmaq_laneq_f32(s2, c0, x2, 0);
                s2 = vfmaq_laneq_f32(s2, c1, x2, 1);
                s2 = vfmaq_laneq_f32(s2, c2, x2, 2);
                s2 = vfmaq_laneq_f32(s2, c3, x2, 3);
            }
            if (rows > 3) {
                float32x4_t x3 = load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
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

static TrillimStatus dot_tile(const Task *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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
            float32x4_t w0 = load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = load4(t->b, b3 + k, p->b_dtype);
            float32x4_t x0 = load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s00 = vfmaq_f32(s00, x0, w0);
            s01 = vfmaq_f32(s01, x0, w1);
            s02 = vfmaq_f32(s02, x0, w2);
            s03 = vfmaq_f32(s03, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s10 = vfmaq_f32(s10, x1, w0);
                s11 = vfmaq_f32(s11, x1, w1);
                s12 = vfmaq_f32(s12, x1, w2);
                s13 = vfmaq_f32(s13, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s20 = vfmaq_f32(s20, x2, w0);
                s21 = vfmaq_f32(s21, x2, w1);
                s22 = vfmaq_f32(s22, x2, w2);
                s23 = vfmaq_f32(s23, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
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

static TrillimStatus dot_tile6(const Task *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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
            float32x4_t w0 = load4(t->b, b_base + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w1 = load4(t->b, b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w2 = load4(t->b, b_base + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w3 = load4(t->b, b_base + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t x0 = load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0_0 = vfmaq_f32(s0_0, x0, w0);
            s0_1 = vfmaq_f32(s0_1, x0, w1);
            s0_2 = vfmaq_f32(s0_2, x0, w2);
            s0_3 = vfmaq_f32(s0_3, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1_0 = vfmaq_f32(s1_0, x1, w0);
                s1_1 = vfmaq_f32(s1_1, x1, w1);
                s1_2 = vfmaq_f32(s1_2, x1, w2);
                s1_3 = vfmaq_f32(s1_3, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2_0 = vfmaq_f32(s2_0, x2, w0);
                s2_1 = vfmaq_f32(s2_1, x2, w1);
                s2_2 = vfmaq_f32(s2_2, x2, w2);
                s2_3 = vfmaq_f32(s2_3, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                s3_0 = vfmaq_f32(s3_0, x3, w0);
                s3_1 = vfmaq_f32(s3_1, x3, w1);
                s3_2 = vfmaq_f32(s3_2, x3, w2);
                s3_3 = vfmaq_f32(s3_3, x3, w3);
            }
            if (rows > 4) {
                float32x4_t x4 = load4(t->a, a_base + 4 * p->a_row_stride + k, p->a_dtype);
                s4_0 = vfmaq_f32(s4_0, x4, w0);
                s4_1 = vfmaq_f32(s4_1, x4, w1);
                s4_2 = vfmaq_f32(s4_2, x4, w2);
                s4_3 = vfmaq_f32(s4_3, x4, w3);
            }
            if (rows > 5) {
                float32x4_t x5 = load4(t->a, a_base + 5 * p->a_row_stride + k, p->a_dtype);
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

static TrillimStatus outer_tile(const Task *t, size_t a_base, size_t b_base,
                                size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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
            float32x4_t w0 = load4(t->b, b_index, p->b_dtype);
            float32x4_t w1 = load4(t->b, b_index + 4, p->b_dtype);
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
    const Task *t, size_t a_base, size_t b_base, size_t output, size_t rows,
    size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const Parameters *p = t->p;
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
                    x0 = load4(t->a, a_base + (k + 0) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 0) * p->a_col_stride + r, a_dtype).f;
                    }
                    x0 = vld1q_f32(values);
                }
                float32x4_t x1;
                if (rows == 4) {
                    x1 = load4(t->a, a_base + (k + 1) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 1) * p->a_col_stride + r, a_dtype).f;
                    }
                    x1 = vld1q_f32(values);
                }
                float32x4_t x2;
                if (rows == 4) {
                    x2 = load4(t->a, a_base + (k + 2) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 2) * p->a_col_stride + r, a_dtype).f;
                    }
                    x2 = vld1q_f32(values);
                }
                float32x4_t x3;
                if (rows == 4) {
                    x3 = load4(t->a, a_base + (k + 3) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 3) * p->a_col_stride + r, a_dtype).f;
                    }
                    x3 = vld1q_f32(values);
                }
                {
                    float32x4_t w0 = load4(t->b, b_base + k + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = load4(t->b, b_base + k + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = load4(t->b, b_base + k + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = load4(t->b, b_base + k + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = load4(t->b, b_base + k + (column + (columns > 4 ? 4 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = load4(t->b, b_base + k + (column + (columns > 5 ? 5 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = load4(t->b, b_base + k + (column + (columns > 6 ? 6 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = load4(t->b, b_base + k + (column + (columns > 7 ? 7 : 0)) * p->b_col_stride, b_dtype);
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
                    float32x4_t w0 = load4(t->b, b_base + k + (column + (columns > 8 ? 8 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = load4(t->b, b_base + k + (column + (columns > 9 ? 9 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = load4(t->b, b_base + k + (column + (columns > 10 ? 10 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = load4(t->b, b_base + k + (column + (columns > 11 ? 11 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = load4(t->b, b_base + k + (column + (columns > 12 ? 12 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = load4(t->b, b_base + k + (column + (columns > 13 ? 13 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = load4(t->b, b_base + k + (column + (columns > 14 ? 14 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = load4(t->b, b_base + k + (column + (columns > 15 ? 15 : 0)) * p->b_col_stride, b_dtype);
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
                x = load4(t->a, a_base + k * p->a_col_stride, a_dtype);
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

static TrillimStatus row_vector_tile(const Task *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32) {
        return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, TRILLIM_FP32, TRILLIM_FP32);
    }
    return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, p->a_dtype, p->b_dtype);
}

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
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

static size_t row_reuse(const Parameters *p, const TrillimKernelContext *context) {
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
    Task task;
    float *scratch;
    size_t count;
    size_t lanes;
    size_t row_reuse;
} BlasTask;

static TrillimStatus ungrouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const Task *t = &job->task;
    const Parameters *p = t->p;
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
    const Task *t = &job->task;
    const Parameters *p = t->p;
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
    const Task *task = opaque;
    const Parameters *p = task->p;
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
    const Task *task = opaque;
    const Parameters *p = task->p;
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

TRILLIM_KERNEL TrillimStatus trillim_neon_dense_bias(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context, scratch};
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
    return context->thread_pool->run(context->thread_pool->context, p->batches * rows * columns, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_dense_bias_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _DTYPE_IDS\n"
           "from trillim.compiler.passes.infer import _promote\n"
           "def bind(c):\n"
           "    x, weight, bias = c.inputs['x'], c.inputs['weight'], c.inputs['bias']\n"
           "    if bias.shape != weight.shape[:1] or bias.dtype not in {'fp16', 'bf16', 'fp32'} or (bias.strides or _contiguous_strides(bias.shape)) != _contiguous_strides(bias.shape):\n"
           "        raise ValueError('linear bias requires a contiguous channel vector')\n"
           "    result = _linear(c)\n"
           "    result['linear_dtype'] = _DTYPE_IDS[_promote(x.dtype, weight.dtype)]\n"
           "    result['bias_dtype'] = _DTYPE_IDS[bias.dtype]\n"
           "    return result\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight, bias=bias),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_bias_parameters(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(Parameters, batches), NULL},
        {"m", "size_t", offsetof(Parameters, m), NULL},
        {"n", "size_t", offsetof(Parameters, n), NULL},
        {"k", "size_t", offsetof(Parameters, k), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(Parameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(Parameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(Parameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(Parameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(Parameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Parameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(Parameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(Parameters, out_dtype), NULL},
        {"linear_dtype", "uint32_t", offsetof(Parameters, linear_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(Parameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Parameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Parameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Parameters, b_storage_columns), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_bias_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
