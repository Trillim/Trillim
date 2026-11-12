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
    size_t a_storage_rows;
    size_t a_storage_columns;
    size_t b_storage_rows;
    size_t b_storage_columns;
} Parameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static bool precise(const Parameters *p) {
    return p->out_dtype != TRILLIM_FP32;
}

static size_t block_n(const Parameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t block_k(const Parameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t tile_rows(const Parameters *p) {
    return !precise(p) && p->m >= 12 && p->a_col_stride == 1 && p->b_row_stride == 1 ? 6 : 4;
}

static bool blas_shape(const Parameters *p) {
    if (precise(p) || p->n > INT32_MAX || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool contiguous = p->b_col_stride == 1 && p->k >= 256 && p->n > 32767 / p->k;
    return large || medium || vocabulary || contiguous;
}

static bool use_blas(const Parameters *p) {
    return blas_shape(p) && initialize_blas();
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
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        !trillim_float(p->out_dtype)) {
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
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    /* Reserve optional-library workspace independently of host availability. */
    if (blas_shape(p)) {
        (void)initialize_blas();
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
        size_t count = p->batches * rows * columns;
        *scratch = (count < LANES ? count : LANES) * WORKSPACE * sizeof(float);
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

static inline void store(const Task *t, size_t index, float value) {
    TrillimValue result = {.f = value};
    trillim_store(t->output, index, t->p->out_dtype, result);
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

static TrillimStatus decode_tile(const Task *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const Parameters *p = t->p;
    for (size_t column = first; column < last; column += 2) {
        size_t columns = last - column < 2 ? last - column : 2;
        float32x4_t s00 = vdupq_n_f32(0);
        float32x4_t s01 = vdupq_n_f32(0);
        float32x4_t s02 = vdupq_n_f32(0);
        float32x4_t s03 = vdupq_n_f32(0);
        float32x4_t s10 = vdupq_n_f32(0);
        float32x4_t s11 = vdupq_n_f32(0);
        float32x4_t s12 = vdupq_n_f32(0);
        float32x4_t s13 = vdupq_n_f32(0);
        size_t k = 0;
        for (; p->k - k >= 16; k += 16) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            VectorPair x0 = load8(t->a, a_base + k + 0, p->a_dtype);
            VectorPair x1 = load8(t->a, a_base + k + 8, p->a_dtype);
            size_t w0 = b_base + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride + k;
            VectorPair w0_0 = load8(t->b, w0 + 0, p->b_dtype);
            s00 = vfmaq_f32(s00, x0.low, w0_0.low);
            s01 = vfmaq_f32(s01, x0.high, w0_0.high);
            VectorPair w0_1 = load8(t->b, w0 + 8, p->b_dtype);
            s02 = vfmaq_f32(s02, x1.low, w0_1.low);
            s03 = vfmaq_f32(s03, x1.high, w0_1.high);
            size_t w1 = b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k;
            VectorPair w1_0 = load8(t->b, w1 + 0, p->b_dtype);
            s10 = vfmaq_f32(s10, x0.low, w1_0.low);
            s11 = vfmaq_f32(s11, x0.high, w1_0.high);
            VectorPair w1_1 = load8(t->b, w1 + 8, p->b_dtype);
            s12 = vfmaq_f32(s12, x1.low, w1_1.low);
            s13 = vfmaq_f32(s13, x1.high, w1_1.high);
        }
        float sums[2] = {vaddvq_f32(vaddq_f32(vaddq_f32(s00, s01), vaddq_f32(s02, s03))), vaddvq_f32(vaddq_f32(vaddq_f32(s10, s11), vaddq_f32(s12, s13)))};
        for (; k < p->k; ++k) {
            float x = trillim_load(t->a, a_base + k, p->a_dtype).f;
            for (size_t c = 0; c < columns; ++c) {
                float w = trillim_load(t->b, b_base + (column + c) * p->b_col_stride + k, p->b_dtype).f;
                sums[c] = fmaf(x, w, sums[c]);
            }
        }
        for (size_t c = 0; c < columns; ++c) {
            store(t, output + column + c, sums[c]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus ordered_dot_tile(const Task *t, size_t a_base, size_t b_base,
                                      size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    size_t height = tile_rows(p);
    size_t row_tiles = p->m / height + (p->m % height != 0);
    size_t batch = index / columns / row_tiles;
    size_t row = index / columns % row_tiles * height;
    size_t first = index % columns * 64;
    size_t last = p->n - first < 64 ? p->n : first + 64;
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

typedef struct {
    Task task;
    float *scratch;
    size_t count;
    size_t lanes;
} BlasTask;

static TrillimStatus blas_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const Task *t = &job->task;
    const Parameters *p = t->p;
    size_t width = block_n(p);
    size_t depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    float *c_pack = b_pack + BLOCK_K * BLOCK_N;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / row_tiles;
        size_t row = tile / columns % row_tiles * BLOCK_M;
        size_t column = tile % columns * width;
        size_t m = p->m - row < BLOCK_M ? p->m - row : BLOCK_M;
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
        if (p->out_dtype != TRILLIM_FP32) {
            for (size_t r = 0; r < m; ++r) {
                for (size_t col = 0; col < n; ++col) {
                    TrillimValue value = {.f = c_pack[r * n + col]};
                    trillim_store(t->output, output + r * p->n + col, p->out_dtype, value);
                }
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
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

TRILLIM_KERNEL TrillimStatus trillim_neon_dense(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    if (use_blas(p)) {
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
        size_t count = p->batches * rows * columns;
        size_t lanes = count < LANES ? count : LANES;
        BlasTask job = {task, scratch, count, lanes};
        return context->thread_pool->run(context->thread_pool->context, lanes, blas_project, &job);
    }
    size_t height = tile_rows(p);
    size_t rows = p->m / height + (p->m % height != 0);
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->batches * rows * columns, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_dense_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_parameters(void) {
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
        {"a_storage_rows", "size_t", offsetof(Parameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Parameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Parameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Parameters, b_storage_columns), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
