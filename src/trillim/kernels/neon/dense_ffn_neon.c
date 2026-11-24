#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* Gated projection. */

#if defined(__FLT16_MANT_DIG__)
__extension__ typedef _Float16 DenseGluTrillimFloat16;
#endif

#define DENSE_GLU_BLOCK_M 64
#define DENSE_GLU_BLOCK_N 128
#define DENSE_GLU_BLOCK_K 1024
#define DENSE_GLU_LANES 32
#define DENSE_GLU_WORKSPACE (DENSE_GLU_BLOCK_M * DENSE_GLU_BLOCK_K + DENSE_GLU_BLOCK_K * DENSE_GLU_BLOCK_N + DENSE_GLU_BLOCK_M * DENSE_GLU_BLOCK_N)

static bool dense_glu_initialize_blas(void);

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
} DenseGluParameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const void *up;
    const DenseGluParameters *p;
    const TrillimKernelContext *context;
} DenseGluTask;

static bool dense_glu_precise(const DenseGluParameters *p) {
    return p->out_dtype != TRILLIM_FP32;
}

static size_t dense_glu_block_n(const DenseGluParameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t dense_glu_block_k(const DenseGluParameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t dense_glu_tile_rows(const DenseGluParameters *p) {
    return !dense_glu_precise(p) && p->m >= 12 && p->a_col_stride == 1 && p->b_row_stride == 1 ? 6 : 4;
}

static bool dense_glu_blas_shape(const DenseGluParameters *p) {
    if (dense_glu_precise(p) || p->n > INT32_MAX || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool contiguous = p->b_col_stride == 1 && p->k >= 256 && p->n > 32767 / p->k;
    return large || medium || vocabulary || contiguous;
}

static bool dense_glu_reserve_blas(const DenseGluParameters *p) {
    if (dense_glu_precise(p) || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    return large || medium || vocabulary;
}

static bool dense_glu_use_blas(const DenseGluParameters *p) {
    return dense_glu_blas_shape(p) && dense_glu_initialize_blas();
}

static bool dense_glu_matrix_span(size_t base, size_t rows, size_t columns,
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

static TrillimStatus dense_glu_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const DenseGluParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        p->out_dtype != TRILLIM_FP32 || p->rank != 0 || p->batches != 1 ||
        p->a_col_stride != 1 || p->b_row_stride != 1 || p->b_col_stride != p->k ||
        (p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32)) {
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
        if (!dense_glu_matrix_span(a_span ? a_span - 1 : 0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !dense_glu_matrix_span(b_span ? b_span - 1 : 0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = a_count * trillim_itemsize(p->a_dtype);
        inputs[1] = b_count * trillim_itemsize(p->b_dtype);
    }
    inputs[2] = inputs[1];
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    /* Reserve optional-library workspace independently of host availability. */
    if (dense_glu_reserve_blas(p)) {
        (void)dense_glu_initialize_blas();
        size_t rows = p->m / DENSE_GLU_BLOCK_M + (p->m % DENSE_GLU_BLOCK_M != 0);
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t count = p->batches * rows * columns;
        *scratch = (count < DENSE_GLU_LANES ? count : DENSE_GLU_LANES) * DENSE_GLU_WORKSPACE * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline float32x4_t dense_glu_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline float activate(float gate, float up) {
    float e = expf(-fabsf(gate));
    float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return (gate * sigmoid) * up;
}

static inline float32x4_t exp_negative(float32x4_t x) {
    float32x4_t original = x;
    x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(x, vdupq_n_f32(0)));
    float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(0.00021219444005469058f));
    /* Range reduction bounds |r| by ln(2)/2 for the degree-seven polynomial. */
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

static inline float32x4_t activate4(float32x4_t gate, float32x4_t up) {
    float32x4_t e = exp_negative(vnegq_f32(vabsq_f32(gate)));
    float32x4_t denominator = vaddq_f32(vdupq_n_f32(1), e);
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    float32x4_t probability = vbslq_f32(vcgeq_f32(gate, vdupq_n_f32(0)), positive, negative);
    float32x4_t value = vmulq_f32(vmulq_f32(gate, probability), up);
    uint32x4_t sensitive = vorrq_u32(vcgtq_f32(vabsq_f32(gate), vdupq_n_f32(80)),
                                    vcgtq_f32(vabsq_f32(up), vdupq_n_f32(1e30f)));
    sensitive = vorrq_u32(sensitive, vmvnq_u32(vandq_u32(vceqq_f32(gate, gate), vceqq_f32(up, up))));
    if (vmaxvq_u32(sensitive) != 0) {
        uint32_t mask[4];
        float gates[4], ups[4], values[4];
        vst1q_u32(mask, sensitive);
        vst1q_f32(gates, gate);
        vst1q_f32(ups, up);
        vst1q_f32(values, value);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (mask[lane]) {
                values[lane] = activate(gates[lane], ups[lane]);
            }
        }
        value = vld1q_f32(values);
    }
    return value;
}

static inline void dense_glu_store(const DenseGluTask *t, size_t index, float gate, float up) {
    ((float *)t->output)[index] = activate(gate, up);
}

typedef struct {
    float32x4_t low;
    float32x4_t high;
} DenseGluVectorPair;

static inline DenseGluVectorPair dense_glu_load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        const float *p = (const float *)data + index;
        DenseGluVectorPair value = {vld1q_f32(p), vld1q_f32(p + 4)};
        return value;
    }
    uint16x8_t bits = vld1q_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        DenseGluVectorPair value = {vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bits))),
                            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bits)))};
        return value;
    }
    DenseGluVectorPair value = {vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16)),
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16))};
    return value;
}

static inline __attribute__((always_inline)) TrillimStatus dense_glu_decode_typed_tile(const DenseGluTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseGluParameters *p = t->p;
    float gates[4], ups[4];
    size_t buffered = 0;
    for (size_t column = first; column < last; ++column) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t weight = b_base + column * p->b_col_stride;
        float32x4_t g0 = vdupq_n_f32(0), g1 = g0, g2 = g0, g3 = g0;
        float32x4_t u0 = g0, u1 = g0, u2 = g0, u3 = g0;
        size_t k = 0;
        for (; p->k - k >= 16; k += 16) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            DenseGluVectorPair x0 = dense_glu_load8(t->a, a_base + k, a_dtype);
            DenseGluVectorPair x1 = dense_glu_load8(t->a, a_base + k + 8, a_dtype);
            DenseGluVectorPair gate = dense_glu_load8(t->b, weight + k, b_dtype);
            DenseGluVectorPair up = dense_glu_load8(t->up, weight + k, b_dtype);
            g0 = vfmaq_f32(g0, x0.low, gate.low);
            g1 = vfmaq_f32(g1, x0.high, gate.high);
            u0 = vfmaq_f32(u0, x0.low, up.low);
            u1 = vfmaq_f32(u1, x0.high, up.high);
            gate = dense_glu_load8(t->b, weight + k + 8, b_dtype);
            up = dense_glu_load8(t->up, weight + k + 8, b_dtype);
            g2 = vfmaq_f32(g2, x1.low, gate.low);
            g3 = vfmaq_f32(g3, x1.high, gate.high);
            u2 = vfmaq_f32(u2, x1.low, up.low);
            u3 = vfmaq_f32(u3, x1.high, up.high);
        }
        for (; p->k - k >= 4; k += 4) {
            float32x4_t x = dense_glu_load4(t->a, a_base + k, a_dtype);
            g0 = vfmaq_f32(g0, x, dense_glu_load4(t->b, weight + k, b_dtype));
            u0 = vfmaq_f32(u0, x, dense_glu_load4(t->up, weight + k, b_dtype));
        }
        float gate = vaddvq_f32(vaddq_f32(vaddq_f32(g0, g1), vaddq_f32(g2, g3)));
        float up = vaddvq_f32(vaddq_f32(vaddq_f32(u0, u1), vaddq_f32(u2, u3)));
        for (; k < p->k; ++k) {
            float x = trillim_load(t->a, a_base + k, a_dtype).f;
            gate = fmaf(x, trillim_load(t->b, weight + k, b_dtype).f, gate);
            up = fmaf(x, trillim_load(t->up, weight + k, b_dtype).f, up);
        }
        gates[buffered] = gate;
        ups[buffered] = up;
        if (++buffered == 4) {
            vst1q_f32((float *)t->output + output + column - 3,
                      activate4(vld1q_f32(gates), vld1q_f32(ups)));
            buffered = 0;
        }
    }
    for (size_t lane = 0; lane < buffered; ++lane) {
        dense_glu_store(t, output + last - buffered + lane, gates[lane], ups[lane]);
    }
    return TRILLIM_OK;
}

static TrillimStatus dense_glu_decode_tile(const DenseGluTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const DenseGluParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_BF16) {
        return dense_glu_decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_BF16);
    }
    return dense_glu_decode_typed_tile(t, a_base, b_base, output, first, last, p->a_dtype, p->b_dtype);
}

static TrillimStatus project_tile(const DenseGluTask *t, size_t a_base, size_t b_base,
                                  size_t output, size_t rows, size_t first, size_t last) {
    const DenseGluParameters *p = t->p;
    for (size_t column = first; column < last; column += 2) {
        size_t columns = last - column < 2 ? last - column : 2;
        float32x4_t g00 = vdupq_n_f32(0);
        float32x4_t g01 = vdupq_n_f32(0);
        float32x4_t g10 = vdupq_n_f32(0);
        float32x4_t g11 = vdupq_n_f32(0);
        float32x4_t g20 = vdupq_n_f32(0);
        float32x4_t g21 = vdupq_n_f32(0);
        float32x4_t g30 = vdupq_n_f32(0);
        float32x4_t g31 = vdupq_n_f32(0);
        float32x4_t g40 = vdupq_n_f32(0);
        float32x4_t g41 = vdupq_n_f32(0);
        float32x4_t g50 = vdupq_n_f32(0);
        float32x4_t g51 = vdupq_n_f32(0);
        float32x4_t u00 = vdupq_n_f32(0);
        float32x4_t u01 = vdupq_n_f32(0);
        float32x4_t u10 = vdupq_n_f32(0);
        float32x4_t u11 = vdupq_n_f32(0);
        float32x4_t u20 = vdupq_n_f32(0);
        float32x4_t u21 = vdupq_n_f32(0);
        float32x4_t u30 = vdupq_n_f32(0);
        float32x4_t u31 = vdupq_n_f32(0);
        float32x4_t u40 = vdupq_n_f32(0);
        float32x4_t u41 = vdupq_n_f32(0);
        float32x4_t u50 = vdupq_n_f32(0);
        float32x4_t u51 = vdupq_n_f32(0);
        size_t k = 0;
        for (; p->k - k >= 4; k += 4) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t index0 = b_base + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride + k;
            float32x4_t gw0 = dense_glu_load4(t->b, index0, p->b_dtype);
            float32x4_t uw0 = dense_glu_load4(t->up, index0, p->b_dtype);
            size_t index1 = b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k;
            float32x4_t gw1 = dense_glu_load4(t->b, index1, p->b_dtype);
            float32x4_t uw1 = dense_glu_load4(t->up, index1, p->b_dtype);
            float32x4_t x0 = dense_glu_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            g00 = vfmaq_f32(g00, x0, gw0);
            g01 = vfmaq_f32(g01, x0, gw1);
            u00 = vfmaq_f32(u00, x0, uw0);
            u01 = vfmaq_f32(u01, x0, uw1);
            if (rows > 1) {
                float32x4_t x1 = dense_glu_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                g10 = vfmaq_f32(g10, x1, gw0);
                g11 = vfmaq_f32(g11, x1, gw1);
                u10 = vfmaq_f32(u10, x1, uw0);
                u11 = vfmaq_f32(u11, x1, uw1);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_glu_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                g20 = vfmaq_f32(g20, x2, gw0);
                g21 = vfmaq_f32(g21, x2, gw1);
                u20 = vfmaq_f32(u20, x2, uw0);
                u21 = vfmaq_f32(u21, x2, uw1);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_glu_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                g30 = vfmaq_f32(g30, x3, gw0);
                g31 = vfmaq_f32(g31, x3, gw1);
                u30 = vfmaq_f32(u30, x3, uw0);
                u31 = vfmaq_f32(u31, x3, uw1);
            }
            if (rows > 4) {
                float32x4_t x4 = dense_glu_load4(t->a, a_base + 4 * p->a_row_stride + k, p->a_dtype);
                g40 = vfmaq_f32(g40, x4, gw0);
                g41 = vfmaq_f32(g41, x4, gw1);
                u40 = vfmaq_f32(u40, x4, uw0);
                u41 = vfmaq_f32(u41, x4, uw1);
            }
            if (rows > 5) {
                float32x4_t x5 = dense_glu_load4(t->a, a_base + 5 * p->a_row_stride + k, p->a_dtype);
                g50 = vfmaq_f32(g50, x5, gw0);
                g51 = vfmaq_f32(g51, x5, gw1);
                u50 = vfmaq_f32(u50, x5, uw0);
                u51 = vfmaq_f32(u51, x5, uw1);
            }
        }
        float gsums[12] = {vaddvq_f32(g00), vaddvq_f32(g01), vaddvq_f32(g10), vaddvq_f32(g11), vaddvq_f32(g20), vaddvq_f32(g21), vaddvq_f32(g30), vaddvq_f32(g31), vaddvq_f32(g40), vaddvq_f32(g41), vaddvq_f32(g50), vaddvq_f32(g51)};
        float usums[12] = {vaddvq_f32(u00), vaddvq_f32(u01), vaddvq_f32(u10), vaddvq_f32(u11), vaddvq_f32(u20), vaddvq_f32(u21), vaddvq_f32(u30), vaddvq_f32(u31), vaddvq_f32(u40), vaddvq_f32(u41), vaddvq_f32(u50), vaddvq_f32(u51)};
        for (; k < p->k; ++k) {
            for (size_t row = 0; row < rows; ++row) {
                float x = trillim_load(t->a, a_base + row * p->a_row_stride + k, p->a_dtype).f;
                for (size_t c = 0; c < columns; ++c) {
                    size_t index = b_base + (column + c) * p->b_col_stride + k;
                    gsums[row * 2 + c] = fmaf(x, trillim_load(t->b, index, p->b_dtype).f, gsums[row * 2 + c]);
                    usums[row * 2 + c] = fmaf(x, trillim_load(t->up, index, p->b_dtype).f, usums[row * 2 + c]);
                }
            }
        }
        size_t row = 0;
        if (columns == 2) {
            for (; rows - row >= 2; row += 2) {
                float32x4_t value = activate4(vld1q_f32(gsums + row * 2), vld1q_f32(usums + row * 2));
                vst1_f32((float *)t->output + output + row * p->n + column, vget_low_f32(value));
                vst1_f32((float *)t->output + output + (row + 1) * p->n + column, vget_high_f32(value));
            }
        }
        for (; row < rows; ++row) {
            for (size_t c = 0; c < columns; ++c) {
                dense_glu_store(t, output + row * p->n + column + c, gsums[row * 2 + c], usums[row * 2 + c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dense_glu_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseGluTask *t = opaque;
    const DenseGluParameters *p = t->p;
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    size_t height = dense_glu_tile_rows(p);
    size_t row = index / columns * height;
    size_t first = index % columns * 64;
    size_t last = p->n - first < 64 ? p->n : first + 64;
    size_t rows = p->m - row < height ? p->m - row : height;
    size_t a_base = row * p->a_row_stride;
    size_t output = row * p->n;
    return rows == 1 ? dense_glu_decode_tile(t, a_base, 0, output, first, last) :
        project_tile(t, a_base, 0, output, rows, first, last);
}

static TrillimStatus dense_glu_project_columns(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseGluTask *t = opaque;
    const DenseGluParameters *p = t->p;
    size_t first = index * 16;
    size_t last = p->n - first < 16 ? p->n : first + 16;
    size_t height = dense_glu_tile_rows(p);
    /* Reuse this weight tile in one worker before advancing to other columns. */
    for (size_t row = 0; row < p->m;) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t rows = p->m - row < height ? p->m - row : height;
        size_t a_base = row * p->a_row_stride;
        size_t output = row * p->n;
        TrillimStatus status = rows == 1 ? dense_glu_decode_tile(t, a_base, 0, output, first, last) :
            project_tile(t, a_base, 0, output, rows, first, last);
        if (status != TRILLIM_OK) {
            return status;
        }
        row += rows;
    }
    return TRILLIM_OK;
}

typedef void (*DenseGluSgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static DenseGluSgemm dense_glu_sgemm;
#ifdef __APPLE__
typedef unsigned (*DenseGluGetThreading)(void);
typedef int (*DenseGluSetThreading)(unsigned);
static DenseGluGetThreading dense_glu_get_threading;
static DenseGluSetThreading dense_glu_set_threading;
#endif
static atomic_uint dense_glu_blas_state = 0;

static bool dense_glu_initialize_blas(void) {
    if (atomic_load_explicit(&dense_glu_blas_state, memory_order_acquire) == 2) {
        return dense_glu_sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&dense_glu_blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(dense_glu_sgemm) == sizeof(address), "function pointer size");
                    memcpy(&dense_glu_sgemm, &address, sizeof(dense_glu_sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&dense_glu_get_threading, &getter, sizeof(dense_glu_get_threading));
                        memcpy(&dense_glu_set_threading, &setter, sizeof(dense_glu_set_threading));
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
                    _Static_assert(sizeof(dense_glu_sgemm) == sizeof(address), "function pointer size");
                    memcpy(&dense_glu_sgemm, &address, sizeof(dense_glu_sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&dense_glu_get_threading, &getter, sizeof(dense_glu_get_threading));
                        memcpy(&dense_glu_set_threading, &setter, sizeof(dense_glu_set_threading));
                    }
#endif
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&dense_glu_blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&dense_glu_blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return dense_glu_sgemm != NULL;
}

static inline float dense_glu_load(const void *data, size_t index, uint32_t dtype) {
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
    DenseGluTrillimFloat16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
#else
    return trillim_fp16(bits);
#endif
}

static TrillimStatus dense_glu_pack(const void *source, uint32_t dtype, size_t base,
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
                DenseGluTrillimFloat16 value;
                memcpy(&value, (const uint8_t *)source + (index + column) * sizeof(value), sizeof(value));
                target[row * columns + column] = (float)value;
            }
#endif
        } else {
            for (size_t column = 0; column < columns; ++column) {
                target[row * columns + column] = dense_glu_load(source, index + column * col_stride, dtype);
            }
        }
    }
    return TRILLIM_OK;
}

typedef struct {
    DenseGluTask task;
    float *scratch;
    size_t count;
    size_t lanes;
    size_t reuse, workspace;
    atomic_bool single_threaded;
} DenseGluBlasTask;

#if defined(__APPLE__)
enum { DENSE_GLU_MAX_ROW_REUSE = 8 };

static TrillimStatus dense_glu_blas_grouped(const DenseGluBlasTask *job, size_t index) {
    const DenseGluTask *t = &job->task;
    const DenseGluParameters *p = t->p;
    const size_t width = dense_glu_block_n(p);
    const size_t depth = dense_glu_block_k(p);
    const size_t columns = p->n / width + (p->n % width != 0);
    const size_t row_capacity = job->reuse * DENSE_GLU_BLOCK_M;
    float *a_pack = job->scratch + index * job->workspace;
    float *b_pack = a_pack + row_capacity * depth;
    float *up = b_pack + depth * width;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        const size_t row = tile / columns * row_capacity;
        const size_t column = tile % columns * width;
        const size_t rows = p->m - row < row_capacity ? p->m - row : row_capacity;
        const size_t n = p->n - column < width ? p->n - column : width;
        float *gate = (float *)t->output + row * p->n + column;
        for (size_t start = 0; start < p->k; start += depth) {
            if (t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const size_t k = p->k - start < depth ? p->k - start : depth;
            const size_t a_index = row * p->a_row_stride + start;
            const size_t b_index = column * p->b_col_stride + start;
            const float *a = a_pack;
            size_t lda = k;
            TrillimStatus status;
            if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                lda = p->a_row_stride;
            } else {
                status = dense_glu_pack(t->a, p->a_dtype, a_index, rows, k, p->a_row_stride, 1, a_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            for (size_t projection = 0; projection < 2; ++projection) {
                const void *weight = projection == 0 ? t->b : t->up;
                const size_t ldc = projection == 0 ? p->n : n;
                float *output = projection == 0 ? gate : up;
                status = dense_glu_pack(weight, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
                for (size_t r = 0; r < rows; r += DENSE_GLU_BLOCK_M) {
                    if (t->context->is_cancelled(t->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    const size_t m = rows - r < DENSE_GLU_BLOCK_M ? rows - r : DENSE_GLU_BLOCK_M;
                    dense_glu_sgemm(101, 111, 112, (int)m, (int)n, (int)k, 1.0f,
                        a + r * lda, (int)lda, b_pack, (int)k,
                        start == 0 ? 0.0f : 1.0f, output + r * ldc, (int)ldc);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            if ((r & 7) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t col = 0;
            for (; n - col >= 4; col += 4) {
                float *destination = gate + r * p->n + col;
                const float32x4_t value = activate4(vld1q_f32(destination), vld1q_f32(up + r * n + col));
                vst1q_f32(destination, value);
            }
            for (; col < n; ++col) {
                const size_t output = (row + r) * p->n + column + col;
                dense_glu_store(t, output, gate[r * p->n + col], up[r * n + col]);
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}
#endif

static TrillimStatus dense_glu_blas_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseGluBlasTask *job = opaque;
#if defined(__APPLE__)
    if (job->reuse > 1) {
        return dense_glu_blas_grouped(job, index);
    }
#endif
    const DenseGluTask *t = &job->task;
    const DenseGluParameters *p = t->p;
    size_t width = dense_glu_block_n(p);
    size_t depth = dense_glu_block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / DENSE_GLU_BLOCK_M + (p->m % DENSE_GLU_BLOCK_M != 0);
    float *a_pack = job->scratch + index * DENSE_GLU_WORKSPACE;
    float *b_pack = a_pack + DENSE_GLU_BLOCK_M * DENSE_GLU_BLOCK_K;
    float *c_pack = b_pack + DENSE_GLU_BLOCK_K * DENSE_GLU_BLOCK_N;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / row_tiles;
        size_t row = tile / columns % row_tiles * DENSE_GLU_BLOCK_M;
        size_t column = tile % columns * width;
        size_t m = p->m - row < DENSE_GLU_BLOCK_M ? p->m - row : DENSE_GLU_BLOCK_M;
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        size_t output = (batch * p->m + row) * p->n + column;
        float *c = (float *)t->output + output;
        int ldc = (int)p->n;
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
            int a_trans = 111, lda = (int)k;
            TrillimStatus status;
            if (p->a_dtype == TRILLIM_FP32 && p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                lda = (int)p->a_row_stride;
            } else if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                a_trans = 112;
                lda = (int)p->a_col_stride;
            } else {
                status = dense_glu_pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            for (size_t projection = 0; projection < 2; ++projection) {
                const void *weight = projection == 0 ? t->b : t->up;
                const float *b = b_pack;
                int b_trans = 111, ldb = (int)n;
                if (p->b_dtype == TRILLIM_FP32 && p->b_col_stride == 1 && p->b_row_stride >= p->n && p->b_row_stride <= INT32_MAX) {
                    b = (const float *)weight + b_index;
                    ldb = (int)p->b_row_stride;
                } else if (p->b_dtype == TRILLIM_FP32 && p->b_row_stride == 1 && p->b_col_stride >= p->k && p->b_col_stride <= INT32_MAX) {
                    b = (const float *)weight + b_index;
                    b_trans = 112;
                    ldb = (int)p->b_col_stride;
                } else if (p->b_row_stride == 1) {
                    status = dense_glu_pack(weight, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                    b_trans = 112;
                    ldb = (int)k;
                } else {
                    status = dense_glu_pack(weight, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                float *destination = projection == 0 ? c : c_pack;
                int destination_stride = projection == 0 ? ldc : (int)n;
                dense_glu_sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                      a, lda, b, ldb, first == 0 ? 0.0f : 1.0f, destination, destination_stride);
            }
        }
        for (size_t r = 0; r < m; ++r) {
            size_t col = 0;
            for (; n - col >= 4; col += 4) {
                float32x4_t value = activate4(vld1q_f32(c + r * (size_t)ldc + col),
                                             vld1q_f32(c_pack + r * n + col));
                vst1q_f32((float *)t->output + output + r * p->n + col, value);
            }
            for (; col < n; ++col) {
                dense_glu_store(t, output + r * p->n + col, c[r * (size_t)ldc + col], c_pack[r * n + col]);
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus dense_glu_blas_project(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    DenseGluBlasTask *job = opaque;
    if (dense_glu_get_threading != NULL && dense_glu_set_threading != NULL) {
        const unsigned previous = dense_glu_get_threading();
        if (previous != 1) {
            if (dense_glu_set_threading(1) == 0) {
                const TrillimStatus status = dense_glu_blas_project_impl(opaque, index, worker);
                (void)dense_glu_set_threading(previous);
                return status;
            }
            if (job->reuse > 1) {
                atomic_store_explicit(&job->single_threaded, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
        }
    }
#endif
    return dense_glu_blas_project_impl(opaque, index, worker);
}

#if defined(__APPLE__)
typedef struct {
    const DenseGluTask *task;
    float *packed;
} DenseGluPackTask;

enum { DENSE_GLU_PACK_ROWS = 16 };

static TrillimStatus dense_glu_pack_rows(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const DenseGluPackTask *job = opaque;
    const DenseGluTask *task = job->task;
    const DenseGluParameters *p = task->p;
    const size_t row = tile * DENSE_GLU_PACK_ROWS;
    const size_t count = p->m - row < DENSE_GLU_PACK_ROWS ? p->m - row : DENSE_GLU_PACK_ROWS;
    return dense_glu_pack(task->a, p->a_dtype, row * p->a_row_stride, count, p->k, p->a_row_stride,
        1, job->packed + row * p->k, task->context);
}
#endif

static TrillimStatus dense_glu_run(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    const DenseGluParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    DenseGluTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    if (dense_glu_use_blas(p)) {
        size_t rows = p->m / DENSE_GLU_BLOCK_M + (p->m % DENSE_GLU_BLOCK_M != 0);
        size_t columns = p->n / dense_glu_block_n(p) + (p->n % dense_glu_block_n(p) != 0);
        size_t count = rows * columns;
        size_t lanes = count < DENSE_GLU_LANES ? count : DENSE_GLU_LANES;
        DenseGluBlasTask job = {task, scratch, count, lanes, 1, DENSE_GLU_WORKSPACE, ATOMIC_VAR_INIT(true)};
#if defined(__APPLE__)
        const size_t workers = context->thread_pool->worker_count;
        if (p->b_dtype != TRILLIM_FP32 && rows > 1 && columns >= workers &&
            workers != 0 && dense_glu_get_threading != NULL && dense_glu_set_threading != NULL) {
            const size_t width = dense_glu_block_n(p);
            const size_t depth = dense_glu_block_k(p);
            size_t reuse = rows < DENSE_GLU_MAX_ROW_REUSE ? rows : DENSE_GLU_MAX_ROW_REUSE;
            for (; reuse > 1; --reuse) {
                const size_t workspace = reuse * DENSE_GLU_BLOCK_M * (depth + width) + depth * width;
                /* Reuse the space reserved for legacy BLAS tiles. */
                if (workspace <= lanes * DENSE_GLU_WORKSPACE / workers) {
                    job.reuse = reuse;
                    job.workspace = workspace;
                    job.count = (rows / reuse + (rows % reuse != 0)) * columns;
                    job.lanes = workers;
                    break;
                }
            }
        }
        DenseGluParameters prepacked;
        const size_t allocated = lanes * DENSE_GLU_WORKSPACE;
        const size_t used = job.lanes * job.workspace;
        if (job.reuse > 1 && p->a_dtype != TRILLIM_FP32 && p->k <= INT32_MAX &&
            p->m <= (allocated - used) / p->k) {
            float *activations = (float *)scratch + used;
            DenseGluPackTask packing = {&task, activations};
            const size_t tiles = p->m / DENSE_GLU_PACK_ROWS + (p->m % DENSE_GLU_PACK_ROWS != 0);
            TrillimStatus status = context->thread_pool->run(
                context->thread_pool->context, tiles, dense_glu_pack_rows, &packing
            );
            if (status != TRILLIM_OK) {
                return status;
            }
            prepacked = *p;
            prepacked.a_dtype = TRILLIM_FP32;
            prepacked.a_row_stride = p->k;
            job.task.p = &prepacked;
            job.task.a = activations;
        }
        if (job.reuse > 1) {
            const TrillimStatus status = context->thread_pool->run(
                context->thread_pool->context, job.lanes, dense_glu_blas_project, &job);
            if (status != TRILLIM_OK || atomic_load_explicit(&job.single_threaded, memory_order_relaxed)) {
                return status;
            }
            job.task = task;
            job.count = count;
            job.lanes = lanes;
            job.reuse = 1;
            job.workspace = DENSE_GLU_WORKSPACE;
        }
#endif
        return context->thread_pool->run(context->thread_pool->context, lanes, dense_glu_blas_project, &job);
    }
    size_t height = dense_glu_tile_rows(p);
    size_t rows = p->m / height + (p->m % height != 0);
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    if (p->m >= 12 && p->m <= 32 && columns / 2 >= context->thread_pool->worker_count &&
        p->a_col_stride == 1 && p->b_row_stride == 1) {
        size_t tiles = p->n / 16 + (p->n % 16 != 0);
        return context->thread_pool->run(context->thread_pool->context, tiles, dense_glu_project_columns, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, rows * columns, dense_glu_project, &task);
}


static const TrillimParameterSchema *dense_glu_schema(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(DenseGluParameters, batches), NULL},
        {"m", "size_t", offsetof(DenseGluParameters, m), NULL},
        {"n", "size_t", offsetof(DenseGluParameters, n), NULL},
        {"k", "size_t", offsetof(DenseGluParameters, k), NULL},
        {"rank", "size_t", offsetof(DenseGluParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(DenseGluParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(DenseGluParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(DenseGluParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(DenseGluParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(DenseGluParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(DenseGluParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(DenseGluParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(DenseGluParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(DenseGluParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(DenseGluParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(DenseGluParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(DenseGluParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(DenseGluParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(DenseGluParameters, b_storage_columns), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(DenseGluParameters), TRILLIM_ALIGNOF(DenseGluParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, dense_glu_requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

static const TrillimCapabilities *dense_glu_get_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}

/* Dense projection. */

#if defined(__FLT16_MANT_DIG__)
__extension__ typedef _Float16 DenseTrillimFloat16;
#endif

#define DENSE_BLOCK_M 64
#define DENSE_BLOCK_N 128
#define DENSE_BLOCK_K 1024
#define DENSE_LANES 32
#define DENSE_WORKSPACE (DENSE_BLOCK_M * DENSE_BLOCK_K + DENSE_BLOCK_K * DENSE_BLOCK_N + DENSE_BLOCK_M * DENSE_BLOCK_N)

static bool dense_initialize_blas(void);

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
} DenseParameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const DenseParameters *p;
    const TrillimKernelContext *context;
} DenseTask;

static bool dense_precise(const DenseParameters *p) {
    return p->out_dtype != TRILLIM_FP32;
}

static size_t dense_block_n(const DenseParameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t dense_block_k(const DenseParameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t block_m(const DenseParameters *p) {
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
    return direct && p->m >= 128 ? 256 : DENSE_BLOCK_M;
}

static bool dense_blas_shape(const DenseParameters *p, size_t workers) {
    if (dense_precise(p) || p->n > INT32_MAX || p->n < 32 || p->k < 32) {
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

static bool dense_reserve_blas(const DenseParameters *p) {
    if (dense_precise(p) || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    bool possibly_contiguous = p->k >= 256 && p->n > 32767 / p->k;
    return large || medium || vocabulary || possibly_contiguous;
}

static bool dense_use_blas(const DenseParameters *p, const TrillimKernelContext *context) {
    return dense_blas_shape(p, context->thread_pool->worker_count) && dense_initialize_blas();
}

static bool dense_matrix_span(size_t base, size_t rows, size_t columns,
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

static TrillimStatus dense_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const DenseParameters *p = parameters;
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
        if (!dense_matrix_span(a_span ? a_span - 1 : 0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !dense_matrix_span(b_span ? b_span - 1 : 0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = a_count * trillim_itemsize(p->a_dtype);
        inputs[1] = b_count * trillim_itemsize(p->b_dtype);
    }
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    /* Reserve optional-library workspace independently of host availability. */
    if (dense_reserve_blas(p)) {
        (void)dense_initialize_blas();
        size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t count = p->batches * rows * columns;
        *scratch = (count < DENSE_LANES ? count : DENSE_LANES) * DENSE_WORKSPACE * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline float32x4_t dense_load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void dense_store(const DenseTask *t, size_t index, float value) {
    TrillimValue result = {.f = value};
    trillim_store(t->output, index, t->p->out_dtype, result);
}

static inline bool ordered_narrow_shape(const DenseParameters *p) {
    return p->a_col_stride == 1 && p->b_row_stride == 1 &&
        p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32 &&
        p->n % 8 == 0 && p->k >= 4 && p->k % 4 == 0;
}

static size_t dense_tile_rows(const DenseParameters *p, size_t workers) {
    if (dense_precise(p)) {
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t tiles = p->batches * (p->m / 8) * columns;
        return p->m >= 8 && tiles >= workers && ordered_narrow_shape(p) ? 8 : 4;
    }
    return p->m >= 12 && p->a_col_stride == 1 && p->b_row_stride == 1 ? 6 : 4;
}

static size_t tile_columns(const DenseParameters *p, size_t workers) {
    if (p->m == 1 && dense_precise(p) && p->a_col_stride == 1 && p->b_row_stride == 1 &&
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
    const DenseTask *t, size_t a_base, size_t b_base, size_t output,
    size_t first, size_t last, uint32_t dtype, size_t rows
) {
    const DenseParameters *p = t->p;
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
                dense_store(t, index + lane, values[lane]);
            }
        }
    }
    return TRILLIM_OK;
}

static inline TrillimStatus ordered_narrow_tile(
    const DenseTask *t, size_t a_base, size_t b_base, size_t output,
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
} DenseVectorPair;

static inline DenseVectorPair dense_load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        const float *p = (const float *)data + index;
        DenseVectorPair value = {vld1q_f32(p), vld1q_f32(p + 4)};
        return value;
    }
    uint16x8_t bits = vld1q_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        DenseVectorPair value = {vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bits))),
                            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bits)))};
        return value;
    }
    DenseVectorPair value = {vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits), 16)),
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits), 16))};
    return value;
}

static inline __attribute__((always_inline)) TrillimStatus dense_decode_typed_tile(const DenseTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseParameters *p = t->p;
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
            DenseVectorPair x0 = dense_load8(activation, a_base + k, a_dtype);
            DenseVectorPair x1 = dense_load8(activation, a_base + k + 8, a_dtype);
            DenseVectorPair w0_0 = dense_load8(weights, w0 + k, b_dtype);
            s00 = vfmaq_f32(s00, x0.low, w0_0.low);
            s01 = vfmaq_f32(s01, x0.high, w0_0.high);
            DenseVectorPair w0_1 = dense_load8(weights, w0 + k + 8, b_dtype);
            s02 = vfmaq_f32(s02, x1.low, w0_1.low);
            s03 = vfmaq_f32(s03, x1.high, w0_1.high);
            DenseVectorPair w1_0 = dense_load8(weights, w1 + k, b_dtype);
            s10 = vfmaq_f32(s10, x0.low, w1_0.low);
            s11 = vfmaq_f32(s11, x0.high, w1_0.high);
            DenseVectorPair w1_1 = dense_load8(weights, w1 + k + 8, b_dtype);
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
            dense_store(t, output + column + c, sums[c]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dense_decode_tile(const DenseTask *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const DenseParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_BF16) {
        return dense_decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_BF16);
    }
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32) {
        return dense_decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_FP32);
    }
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP16) {
        return dense_decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_FP16);
    }
    return dense_decode_typed_tile(t, a_base, b_base, output, first, last, p->a_dtype, p->b_dtype);
}

static inline __attribute__((always_inline)) float32x4x4_t ordered_weights(const DenseTask *t, size_t base, size_t column, size_t k, uint32_t dtype) {
    size_t stride = t->p->b_col_stride;
    float32x4_t w0 = dense_load4(t->b, base + column * stride + k, dtype);
    float32x4_t w1 = dense_load4(t->b, base + (column + 1) * stride + k, dtype);
    float32x4_t w2 = dense_load4(t->b, base + (column + 2) * stride + k, dtype);
    float32x4_t w3 = dense_load4(t->b, base + (column + 3) * stride + k, dtype);
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

static inline __attribute__((always_inline)) TrillimStatus ordered_decode16_typed(const DenseTask *t, size_t a_base, size_t b_base, size_t output, size_t column, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseParameters *p = t->p;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t k = 0;
    for (; p->k - k >= 4; k += 4) {
        if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t x = dense_load4(t->a, a_base + k, a_dtype);
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
            dense_store(t, output + column + group * 4 + lane, values[lane]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus ordered_decode16(const DenseTask *t, size_t a_base, size_t b_base, size_t output, size_t column) {
    const DenseParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP16 && p->b_dtype == TRILLIM_FP16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_FP16, TRILLIM_FP16);
    }
    if (p->a_dtype == TRILLIM_BF16 && p->b_dtype == TRILLIM_BF16) {
        return ordered_decode16_typed(t, a_base, b_base, output, column, TRILLIM_BF16, TRILLIM_BF16);
    }
    return ordered_decode16_typed(t, a_base, b_base, output, column, p->a_dtype, p->b_dtype);
}

static TrillimStatus ordered_dot_tile(const DenseTask *t, size_t a_base, size_t b_base,
                                      size_t output, size_t rows, size_t first, size_t last) {
    const DenseParameters *p = t->p;
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
            float32x4_t w0 = dense_load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = dense_load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = dense_load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = dense_load4(t->b, b3 + k, p->b_dtype);
            float32x4x2_t even = vtrnq_f32(w0, w1);
            float32x4x2_t odd = vtrnq_f32(w2, w3);
            float32x4_t c0 = vcombine_f32(vget_low_f32(even.val[0]), vget_low_f32(odd.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(even.val[1]), vget_low_f32(odd.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(even.val[0]), vget_high_f32(odd.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(even.val[1]), vget_high_f32(odd.val[1]));
            float32x4_t x0 = dense_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0 = vfmaq_laneq_f32(s0, c0, x0, 0);
            s0 = vfmaq_laneq_f32(s0, c1, x0, 1);
            s0 = vfmaq_laneq_f32(s0, c2, x0, 2);
            s0 = vfmaq_laneq_f32(s0, c3, x0, 3);
            if (rows > 1) {
                float32x4_t x1 = dense_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1 = vfmaq_laneq_f32(s1, c0, x1, 0);
                s1 = vfmaq_laneq_f32(s1, c1, x1, 1);
                s1 = vfmaq_laneq_f32(s1, c2, x1, 2);
                s1 = vfmaq_laneq_f32(s1, c3, x1, 3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2 = vfmaq_laneq_f32(s2, c0, x2, 0);
                s2 = vfmaq_laneq_f32(s2, c1, x2, 1);
                s2 = vfmaq_laneq_f32(s2, c2, x2, 2);
                s2 = vfmaq_laneq_f32(s2, c3, x2, 3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
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
                dense_store(t, output + row * p->n + column + c, sums[row][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dot_tile(const DenseTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseParameters *p = t->p;
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
            float32x4_t w0 = dense_load4(t->b, b0 + k, p->b_dtype);
            float32x4_t w1 = dense_load4(t->b, b1 + k, p->b_dtype);
            float32x4_t w2 = dense_load4(t->b, b2 + k, p->b_dtype);
            float32x4_t w3 = dense_load4(t->b, b3 + k, p->b_dtype);
            float32x4_t x0 = dense_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s00 = vfmaq_f32(s00, x0, w0);
            s01 = vfmaq_f32(s01, x0, w1);
            s02 = vfmaq_f32(s02, x0, w2);
            s03 = vfmaq_f32(s03, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = dense_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s10 = vfmaq_f32(s10, x1, w0);
                s11 = vfmaq_f32(s11, x1, w1);
                s12 = vfmaq_f32(s12, x1, w2);
                s13 = vfmaq_f32(s13, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s20 = vfmaq_f32(s20, x2, w0);
                s21 = vfmaq_f32(s21, x2, w1);
                s22 = vfmaq_f32(s22, x2, w2);
                s23 = vfmaq_f32(s23, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
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
                dense_store(t, output + r * p->n + column + c, sums[r][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dot_tile6(const DenseTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseParameters *p = t->p;
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
            float32x4_t w0 = dense_load4(t->b, b_base + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w1 = dense_load4(t->b, b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w2 = dense_load4(t->b, b_base + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t w3 = dense_load4(t->b, b_base + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride + k, p->b_dtype);
            float32x4_t x0 = dense_load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            s0_0 = vfmaq_f32(s0_0, x0, w0);
            s0_1 = vfmaq_f32(s0_1, x0, w1);
            s0_2 = vfmaq_f32(s0_2, x0, w2);
            s0_3 = vfmaq_f32(s0_3, x0, w3);
            if (rows > 1) {
                float32x4_t x1 = dense_load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                s1_0 = vfmaq_f32(s1_0, x1, w0);
                s1_1 = vfmaq_f32(s1_1, x1, w1);
                s1_2 = vfmaq_f32(s1_2, x1, w2);
                s1_3 = vfmaq_f32(s1_3, x1, w3);
            }
            if (rows > 2) {
                float32x4_t x2 = dense_load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                s2_0 = vfmaq_f32(s2_0, x2, w0);
                s2_1 = vfmaq_f32(s2_1, x2, w1);
                s2_2 = vfmaq_f32(s2_2, x2, w2);
                s2_3 = vfmaq_f32(s2_3, x2, w3);
            }
            if (rows > 3) {
                float32x4_t x3 = dense_load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                s3_0 = vfmaq_f32(s3_0, x3, w0);
                s3_1 = vfmaq_f32(s3_1, x3, w1);
                s3_2 = vfmaq_f32(s3_2, x3, w2);
                s3_3 = vfmaq_f32(s3_3, x3, w3);
            }
            if (rows > 4) {
                float32x4_t x4 = dense_load4(t->a, a_base + 4 * p->a_row_stride + k, p->a_dtype);
                s4_0 = vfmaq_f32(s4_0, x4, w0);
                s4_1 = vfmaq_f32(s4_1, x4, w1);
                s4_2 = vfmaq_f32(s4_2, x4, w2);
                s4_3 = vfmaq_f32(s4_3, x4, w3);
            }
            if (rows > 5) {
                float32x4_t x5 = dense_load4(t->a, a_base + 5 * p->a_row_stride + k, p->a_dtype);
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
                dense_store(t, output + r * p->n + column + c, sums[r][c]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus outer_tile(const DenseTask *t, size_t a_base, size_t b_base,
                                size_t output, size_t rows, size_t first, size_t last) {
    const DenseParameters *p = t->p;
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
            float32x4_t w0 = dense_load4(t->b, b_index, p->b_dtype);
            float32x4_t w1 = dense_load4(t->b, b_index + 4, p->b_dtype);
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
                dense_store(t, output + r * p->n + column + c, sums[r][c]);
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
            dense_store(t, output + r * p->n + column, sums[r]);
        }
    }
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus row_vector_typed_tile(
    const DenseTask *t, size_t a_base, size_t b_base, size_t output, size_t rows,
    size_t first, size_t last, uint32_t a_dtype, uint32_t b_dtype) {
    const DenseParameters *p = t->p;
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
                    x0 = dense_load4(t->a, a_base + (k + 0) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 0) * p->a_col_stride + r, a_dtype).f;
                    }
                    x0 = vld1q_f32(values);
                }
                float32x4_t x1;
                if (rows == 4) {
                    x1 = dense_load4(t->a, a_base + (k + 1) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 1) * p->a_col_stride + r, a_dtype).f;
                    }
                    x1 = vld1q_f32(values);
                }
                float32x4_t x2;
                if (rows == 4) {
                    x2 = dense_load4(t->a, a_base + (k + 2) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 2) * p->a_col_stride + r, a_dtype).f;
                    }
                    x2 = vld1q_f32(values);
                }
                float32x4_t x3;
                if (rows == 4) {
                    x3 = dense_load4(t->a, a_base + (k + 3) * p->a_col_stride, a_dtype);
                } else {
                    float values[4] = {0};
                    for (size_t r = 0; r < rows; ++r) {
                        values[r] = trillim_load(t->a, a_base + (k + 3) * p->a_col_stride + r, a_dtype).f;
                    }
                    x3 = vld1q_f32(values);
                }
                {
                    float32x4_t w0 = dense_load4(t->b, b_base + k + (column + (columns > 0 ? 0 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = dense_load4(t->b, b_base + k + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = dense_load4(t->b, b_base + k + (column + (columns > 2 ? 2 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = dense_load4(t->b, b_base + k + (column + (columns > 3 ? 3 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = dense_load4(t->b, b_base + k + (column + (columns > 4 ? 4 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = dense_load4(t->b, b_base + k + (column + (columns > 5 ? 5 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = dense_load4(t->b, b_base + k + (column + (columns > 6 ? 6 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = dense_load4(t->b, b_base + k + (column + (columns > 7 ? 7 : 0)) * p->b_col_stride, b_dtype);
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
                    float32x4_t w0 = dense_load4(t->b, b_base + k + (column + (columns > 8 ? 8 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w1 = dense_load4(t->b, b_base + k + (column + (columns > 9 ? 9 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w2 = dense_load4(t->b, b_base + k + (column + (columns > 10 ? 10 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w3 = dense_load4(t->b, b_base + k + (column + (columns > 11 ? 11 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w4 = dense_load4(t->b, b_base + k + (column + (columns > 12 ? 12 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w5 = dense_load4(t->b, b_base + k + (column + (columns > 13 ? 13 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w6 = dense_load4(t->b, b_base + k + (column + (columns > 14 ? 14 : 0)) * p->b_col_stride, b_dtype);
                    float32x4_t w7 = dense_load4(t->b, b_base + k + (column + (columns > 15 ? 15 : 0)) * p->b_col_stride, b_dtype);
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
                x = dense_load4(t->a, a_base + k * p->a_col_stride, a_dtype);
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
                dense_store(t, output + r * p->n + column + c, values[c][r]);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus row_vector_tile(const DenseTask *t, size_t a_base, size_t b_base,
                              size_t output, size_t rows, size_t first, size_t last) {
    const DenseParameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32) {
        return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, TRILLIM_FP32, TRILLIM_FP32);
    }
    return row_vector_typed_tile(t, a_base, b_base, output, rows, first, last, p->a_dtype, p->b_dtype);
}

static TrillimStatus dense_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseTask *t = opaque;
    const DenseParameters *p = t->p;
    size_t width = tile_columns(p, t->context->thread_pool->worker_count);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t height = dense_tile_rows(p, t->context->thread_pool->worker_count);
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
        if (dense_precise(p)) {
            return ordered_dot_tile(t, a_base, b_base, output, rows, first, last);
        }
        if (height == 6) {
            return dot_tile6(t, a_base, b_base, output, rows, first, last);
        }
        return rows == 1 && p->k >= 64 ?
            dense_decode_tile(t, a_base, b_base, output, first, last) :
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
            dense_store(t, output + r * p->n + c, sum);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus dense_project_columns(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseTask *t = opaque;
    const DenseParameters *p = t->p;
    size_t columns = p->n / 16 + (p->n % 16 != 0);
    size_t batch = index / columns;
    size_t first = index % columns * 16;
    size_t last = p->n - first < 16 ? p->n : first + 16;
    size_t height = dense_tile_rows(p, t->context->thread_pool->worker_count);
    size_t a_batch = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides);
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    /* Reuse this weight tile in one worker before advancing to other columns. */
    for (size_t row = 0; row < p->m;) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t rows = p->m - row < height ? p->m - row : height;
        TrillimStatus status = dot_tile6(t, a_batch + row * p->a_row_stride, b_base,
                                        (batch * p->m + row) * p->n, rows, first, last);
        if (status != TRILLIM_OK) {
            return status;
        }
        row += rows;
    }
    return TRILLIM_OK;
}

typedef void (*DenseSgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static DenseSgemm dense_sgemm;
#ifdef __APPLE__
typedef unsigned (*DenseGetThreading)(void);
typedef int (*DenseSetThreading)(unsigned);
static DenseGetThreading dense_get_threading;
static DenseSetThreading dense_set_threading;
#endif
static atomic_uint dense_blas_state = 0;

static bool dense_initialize_blas(void) {
    if (atomic_load_explicit(&dense_blas_state, memory_order_acquire) == 2) {
        return dense_sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&dense_blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(dense_sgemm) == sizeof(address), "function pointer size");
                    memcpy(&dense_sgemm, &address, sizeof(dense_sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&dense_get_threading, &getter, sizeof(dense_get_threading));
                        memcpy(&dense_set_threading, &setter, sizeof(dense_set_threading));
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
                    _Static_assert(sizeof(dense_sgemm) == sizeof(address), "function pointer size");
                    memcpy(&dense_sgemm, &address, sizeof(dense_sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&dense_get_threading, &getter, sizeof(dense_get_threading));
                        memcpy(&dense_set_threading, &setter, sizeof(dense_set_threading));
                    }
#endif
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&dense_blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&dense_blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return dense_sgemm != NULL;
}

static inline float dense_load(const void *data, size_t index, uint32_t dtype) {
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
    DenseTrillimFloat16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
#else
    return trillim_fp16(bits);
#endif
}

static TrillimStatus dense_pack(const void *source, uint32_t dtype, size_t base,
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
                DenseTrillimFloat16 value;
                memcpy(&value, (const uint8_t *)source + (index + column) * sizeof(value), sizeof(value));
                target[row * columns + column] = (float)value;
            }
#endif
        } else {
            for (size_t column = 0; column < columns; ++column) {
                target[row * columns + column] = dense_load(source, index + column * col_stride, dtype);
            }
        }
    }
    return TRILLIM_OK;
}

static size_t row_reuse(const DenseParameters *p, const TrillimKernelContext *context) {
    size_t workers = context->thread_pool->worker_count;
    size_t minimum_rows = workers <= 8 || p->n >= 2048 ? 96 : 512;
    if (p->m < minimum_rows || p->out_dtype != TRILLIM_FP32 || p->b_dtype == TRILLIM_FP32 || p->k == 0) {
        return 1;
    }
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t columns = p->n / dense_block_n(p) + (p->n % dense_block_n(p) != 0);
    if (columns == 0) {
        return 1;
    }
    size_t target = workers == 0 ? 1 : (workers < DENSE_LANES / 2 ? workers * 2 : DENSE_LANES);
    size_t groups = target / columns + (target % columns != 0);
    size_t reuse = rows / groups;
    return reuse < 2 ? 1 : (reuse < 8 ? reuse : 8);
}

typedef struct {
    DenseTask task;
    float *scratch;
    size_t count;
    size_t lanes;
    size_t row_reuse;
} DenseBlasTask;

static TrillimStatus ungrouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseBlasTask *job = opaque;
    const DenseTask *t = &job->task;
    const DenseParameters *p = t->p;
    size_t width = dense_block_n(p);
    size_t depth = dense_block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / block_m(p) + (p->m % block_m(p) != 0);
    float *a_pack = job->scratch + index * DENSE_WORKSPACE;
    float *b_pack = a_pack + DENSE_BLOCK_M * DENSE_BLOCK_K;
    float *c_pack = b_pack + DENSE_BLOCK_K * DENSE_BLOCK_N;
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
                status = dense_pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
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
                status = dense_pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = dense_pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            dense_sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
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

static TrillimStatus grouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const DenseBlasTask *job = opaque;
    const DenseTask *t = &job->task;
    const DenseParameters *p = t->p;
    size_t width = dense_block_n(p), depth = dense_block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t groups = rows / job->row_reuse + (rows % job->row_reuse != 0);
    float *a_pack = job->scratch + index * DENSE_WORKSPACE;
    float *b_pack = a_pack + DENSE_BLOCK_M * DENSE_BLOCK_K;
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
                status = dense_pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = dense_pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
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
                    status = dense_pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                size_t output = (batch * p->m + row) * p->n + column;
                float *c = (float *)t->output + output;
                dense_sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                      a, lda, b_pack, ldb, first == 0 ? 0.0f : 1.0f, c, (int)p->n);
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus dense_blas_project_impl(void *opaque, size_t index, size_t worker) {
    const DenseBlasTask *job = opaque;
    return job->row_reuse > 1 ? grouped_project_impl(opaque, index, worker) :
        ungrouped_project_impl(opaque, index, worker);
}

static TrillimStatus dense_blas_project(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (dense_get_threading != NULL && dense_set_threading != NULL) {
        unsigned previous = dense_get_threading();
        if (previous != 1 && dense_set_threading(1) == 0) {
            TrillimStatus status = dense_blas_project_impl(opaque, index, worker);
            (void)dense_set_threading(previous);
            return status;
        }
    }
#endif
    return dense_blas_project_impl(opaque, index, worker);
}

static TrillimStatus dense_run(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    const DenseParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    DenseTask task = {outputs[0], inputs[0], inputs[1], p, context};
    if (dense_use_blas(p, context)) {
        size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
        size_t columns = p->n / dense_block_n(p) + (p->n % dense_block_n(p) != 0);
        size_t reuse = row_reuse(p, context);
        size_t groups = rows / reuse + (rows % reuse != 0);
        size_t count = p->batches * groups * columns;
        size_t lanes = count < DENSE_LANES ? count : DENSE_LANES;
        DenseBlasTask job = {task, scratch, count, lanes, reuse};
        return context->thread_pool->run(context->thread_pool->context, lanes, dense_blas_project, &job);
    }
    size_t height = dense_tile_rows(p, context->thread_pool->worker_count);
    size_t rows = p->m / height + (p->m % height != 0);
    size_t width = tile_columns(p, context->thread_pool->worker_count);
    size_t columns = p->n / width + (p->n % width != 0);
    if (p->m >= 12 && p->m <= 32 && columns / 2 >= context->thread_pool->worker_count &&
        height == 6 && p->a_col_stride == 1 && p->b_row_stride == 1) {
        size_t tiles = p->n / 16 + (p->n % 16 != 0);
        return context->thread_pool->run(context->thread_pool->context, p->batches * tiles, dense_project_columns, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, p->batches * rows * columns, dense_project, &task);
}


static const TrillimParameterSchema *dense_schema(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(DenseParameters, batches), NULL},
        {"m", "size_t", offsetof(DenseParameters, m), NULL},
        {"n", "size_t", offsetof(DenseParameters, n), NULL},
        {"k", "size_t", offsetof(DenseParameters, k), NULL},
        {"rank", "size_t", offsetof(DenseParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(DenseParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(DenseParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(DenseParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(DenseParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(DenseParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(DenseParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(DenseParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(DenseParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(DenseParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(DenseParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(DenseParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(DenseParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(DenseParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(DenseParameters, b_storage_columns), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(DenseParameters), TRILLIM_ALIGNOF(DenseParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, dense_requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

static const TrillimCapabilities *dense_get_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}

typedef struct {
    size_t rows;
    size_t input_width;
    size_t hidden_width;
    size_t output_width;
    size_t x_row_stride;
    size_t x_storage_rows;
    size_t x_storage_columns;
    uint32_t x_dtype;
    uint32_t gate_dtype;
    uint32_t down_dtype;
} Parameters;

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
} Values;

static const TrillimParameter value_fields[] = {
        {"batches", "size_t", offsetof(Values, batches), NULL},
        {"m", "size_t", offsetof(Values, m), NULL},
        {"n", "size_t", offsetof(Values, n), NULL},
        {"k", "size_t", offsetof(Values, k), NULL},
        {"rank", "size_t", offsetof(Values, rank), NULL},
        {"batch_shape", "size_t", offsetof(Values, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(Values, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(Values, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(Values, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(Values, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Values, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(Values, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Values, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Values, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(Values, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(Values, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Values, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Values, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Values, b_storage_columns), NULL},
    };

enum { FIELD_COUNT = sizeof(value_fields) / sizeof(value_fields[0]), MAX_PARAMETERS = 4096 };

typedef struct {
    size_t source;
    size_t destination;
    size_t bytes;
} FieldCopy;

typedef struct {
    const TrillimParameterSchema *schema;
    FieldCopy fields[FIELD_COUNT];
} Component;

typedef struct {
    size_t inputs[4];
    size_t output;
    size_t hidden_span;
    size_t scratch;
} Layout;

static Component glu_component, dense_component;
static size_t input_alignments[] = {1, 1, 1, 1};
static size_t output_alignments[] = {4};
static TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, UINT64_MAX};
static atomic_uint initialization = 0;
static bool initialized;

static bool valid_alignment(size_t value) {
    return value != 0 && value <= 64 && (value & (value - 1)) == 0;
}

static bool map_component(Component *component, const TrillimParameterSchema *schema, size_t inputs) {
    if (schema == NULL || schema->size > MAX_PARAMETERS || schema->field_count != FIELD_COUNT ||
        schema->input_count != inputs || schema->output_count != 1 || schema->requirements == NULL ||
        schema->fields == NULL || schema->input_alignments == NULL || schema->output_alignments == NULL ||
        !valid_alignment(schema->alignment) || !valid_alignment(schema->scratch_alignment) ||
        !valid_alignment(schema->output_alignments[0])) {
        return false;
    }
    for (size_t i = 0; i < inputs; ++i) {
        if (!valid_alignment(schema->input_alignments[i])) {
            return false;
        }
    }
    bool seen[FIELD_COUNT] = {false};
    for (size_t i = 0; i < FIELD_COUNT; ++i) {
        const TrillimParameter *target = &schema->fields[i];
        if (target->name == NULL || target->type == NULL) {
            return false;
        }
        size_t source = 0;
        while (source < FIELD_COUNT && strcmp(value_fields[source].name, target->name) != 0) {
            ++source;
        }
        if (source == FIELD_COUNT || seen[source] || strcmp(value_fields[source].type, target->type) != 0 ||
            (value_fields[source].length == NULL) != (target->length == NULL) ||
            (target->length != NULL && strcmp(value_fields[source].length, target->length) != 0)) {
            return false;
        }
        seen[source] = true;
        size_t bytes = target->length != NULL ? sizeof(void *) :
            (strcmp(target->type, "size_t") == 0 ? sizeof(size_t) : sizeof(uint32_t));
        if (bytes > schema->size || target->offset > schema->size - bytes) {
            return false;
        }
        component->fields[i] = (FieldCopy){value_fields[source].offset, target->offset, bytes};
    }
    component->schema = schema;
    return true;
}

static bool initialize(void) {
    if (atomic_load_explicit(&initialization, memory_order_acquire) == 2) {
        return initialized;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&initialization, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
        const TrillimCapabilities *glu = dense_glu_get_capabilities();
        const TrillimCapabilities *dense = dense_get_capabilities();
        initialized = glu != NULL && dense != NULL && glu->abi_version == TRILLIM_KERNEL_ABI_VERSION &&
            dense->abi_version == TRILLIM_KERNEL_ABI_VERSION &&
            map_component(&glu_component, dense_glu_schema(), 3) &&
            map_component(&dense_component, dense_schema(), 2);
        if (initialized) {
            for (size_t i = 0; i < 3; ++i) {
                input_alignments[i] = glu_component.schema->input_alignments[i];
            }
            input_alignments[3] = dense_component.schema->input_alignments[1];
            output_alignments[0] = dense_component.schema->output_alignments[0];
            capabilities.required = glu->required | dense->required;
        }
        atomic_store_explicit(&initialization, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&initialization, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return initialized;
}

static void pack(const Component *component, const Values *values, void *parameters) {
    memset(parameters, 0, component->schema->size);
    for (size_t i = 0; i < FIELD_COUNT; ++i) {
        const FieldCopy *field = &component->fields[i];
        memcpy((uint8_t *)parameters + field->destination,
               (const uint8_t *)values + field->source, field->bytes);
    }
}

static TrillimStatus prepare(const Parameters *p, void *glu_parameters, void *dense_parameters, Layout *layout) {
    if (p->rows == 0 || p->output_width == 0) {
        memset(layout, 0, sizeof(*layout));
        return TRILLIM_OK;
    }
    if (!initialize()) {
        return TRILLIM_ERROR;
    }
    Values glu = {
        .batches = 1, .m = p->rows, .n = p->hidden_width, .k = p->input_width,
        .a_row_stride = p->x_row_stride, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->input_width,
        .a_dtype = p->x_dtype, .b_dtype = p->gate_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->x_storage_rows, .a_storage_columns = p->x_storage_columns,
        .b_storage_rows = p->hidden_width, .b_storage_columns = p->input_width
    };
    Values dense = {
        .batches = 1, .m = p->rows, .n = p->output_width, .k = p->hidden_width,
        .a_row_stride = p->hidden_width, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->hidden_width,
        .a_dtype = TRILLIM_FP32, .b_dtype = p->down_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->rows, .a_storage_columns = p->hidden_width,
        .b_storage_rows = p->output_width, .b_storage_columns = p->hidden_width
    };
    pack(&glu_component, &glu, glu_parameters);
    pack(&dense_component, &dense, dense_parameters);
    size_t glu_inputs[3], glu_outputs[1], glu_scratch;
    size_t dense_inputs[2], dense_outputs[1], dense_scratch;
    TrillimStatus status = glu_component.schema->requirements(glu_parameters, glu_inputs, glu_outputs, &glu_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = dense_component.schema->requirements(dense_parameters, dense_inputs, dense_outputs, &dense_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (glu_outputs[0] != dense_inputs[0] || glu_outputs[0] > SIZE_MAX - 63) {
        return TRILLIM_ERROR;
    }
    layout->hidden_span = (glu_outputs[0] + 63) / 64 * 64;
    size_t workspace = glu_scratch > dense_scratch ? glu_scratch : dense_scratch;
    if (workspace > SIZE_MAX - layout->hidden_span) {
        return TRILLIM_ERROR;
    }
    for (size_t i = 0; i < 3; ++i) {
        layout->inputs[i] = glu_inputs[i];
    }
    layout->inputs[3] = dense_inputs[1];
    layout->output = dense_outputs[0];
    layout->scratch = layout->hidden_span + workspace;
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    for (size_t i = 0; i < 4; ++i) {
        inputs[i] = 0;
    }
    outputs[0] = *scratch = 0;
    _Alignas(64) uint8_t glu_parameters[MAX_PARAMETERS], dense_parameters[MAX_PARAMETERS];
    Layout layout;
    TrillimStatus status = prepare(parameters, glu_parameters, dense_parameters, &layout);
    if (status != TRILLIM_OK) {
        return status;
    }
    memcpy(inputs, layout.inputs, sizeof(layout.inputs));
    outputs[0] = layout.output;
    *scratch = layout.scratch;
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    /* The projections finish reading x before the down projection writes output. */
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_dense_ffn(void *const *outputs, const void *const *inputs, void *scratch,
                                    const void *parameters, const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->rows == 0 || p->output_width == 0) {
        return TRILLIM_OK;
    }
    _Alignas(64) uint8_t glu_parameters[MAX_PARAMETERS], dense_parameters[MAX_PARAMETERS];
    Layout layout;
    TrillimStatus status = prepare(p, glu_parameters, dense_parameters, &layout);
    if (status != TRILLIM_OK) {
        return status;
    }
    void *hidden = scratch;
    void *workspace = layout.hidden_span != 0 ? (uint8_t *)scratch + layout.hidden_span : scratch;
    void *glu_outputs[] = {hidden};
    const void *glu_inputs[] = {inputs[0], inputs[1], inputs[2]};
    status = dense_glu_run(glu_outputs, glu_inputs, workspace, glu_parameters, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const void *dense_inputs[] = {hidden, inputs[3]};
    return dense_run(outputs, dense_inputs, workspace, dense_parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_dense_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _constant_expression, _DTYPE_IDS\n"
           "from trillim.compiler.kernels.kernel_dataclasses import _KernelBindingContext\n"
           "from trillim.compiler.passes.infer import _promote\n"
           "def bind(c):\n"
           "    x, gate, up, down = (c.inputs[name] for name in ('x', 'gate', 'up', 'down'))\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or gate.dtype != up.dtype or gate.shape != up.shape or len(gate.shape) != 2 or len(down.shape) != 2 or down.shape[1] != gate.shape[0] or _promote(x.dtype, gate.dtype) != 'fp32' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('fused FFN requires compatible floating projections to FP32')\n"
           "    for weight in (gate, up, down):\n"
           "        if weight.dtype not in {'fp16', 'bf16', 'fp32'} or (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "            raise ValueError('fused FFN requires canonical floating weights')\n"
           "    base = _linear(_KernelBindingContext({'x': x, 'weight': gate}, c.outputs, c.attrs))\n"
           "    if base['a_col_stride'] != _constant_expression(1):\n"
           "        raise ValueError('fused FFN requires contiguous input channels')\n"
           "    return {'rows': base['m'], 'input_width': base['k'], 'hidden_width': base['n'], 'output_width': down.shape[0], 'x_row_stride': base['a_row_stride'], 'x_storage_rows': base['a_storage_rows'], 'x_storage_columns': base['a_storage_columns'], 'x_dtype': base['a_dtype'], 'gate_dtype': base['b_dtype'], 'down_dtype': _DTYPE_IDS[down.dtype]}\n"
           "def pattern(x: Tensor, gate: Tensor, up: Tensor, down: Tensor, *, up_bias=None, down_bias=None, gate_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if gate is None or up_bias is not None or down_bias is not None or gate_bias is not None or activation is not Silu:\n"
           "        raise ValueError('fused FFN requires an unbiased SiLU gate')\n"
           "    return (FFN(x, up=up, down=down, gate=gate, activation=activation),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_ffn_parameters(void) {
    (void)initialize();
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"input_width", "size_t", offsetof(Parameters, input_width), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"output_width", "size_t", offsetof(Parameters, output_width), NULL},
        {"x_row_stride", "size_t", offsetof(Parameters, x_row_stride), NULL},
        {"x_storage_rows", "size_t", offsetof(Parameters, x_storage_rows), NULL},
        {"x_storage_columns", "size_t", offsetof(Parameters, x_storage_columns), NULL},
        {"x_dtype", "uint32_t", offsetof(Parameters, x_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, can_alias, input_alignments, output_alignments, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_ffn_capabilities(void) {
    (void)initialize();
    return &capabilities;
}
