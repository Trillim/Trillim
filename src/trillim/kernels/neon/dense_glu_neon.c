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
    const void *up;
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

static bool reserve_blas(const Parameters *p) {
    if (precise(p) || p->n < 32 || p->k < 32) {
        return false;
    }
    bool large = p->m >= 64 && p->n > 65535 / p->k;
    bool medium = p->m >= 32 && p->n > 2097152 / p->k;
    bool vocabulary = p->m >= 16 && p->n >= 32768 && p->k >= 512;
    return large || medium || vocabulary;
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
        if (!matrix_span(a_span ? a_span - 1 : 0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !matrix_span(b_span ? b_span - 1 : 0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = a_count * trillim_itemsize(p->a_dtype);
        inputs[1] = b_count * trillim_itemsize(p->b_dtype);
    }
    inputs[2] = inputs[1];
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    /* Reserve optional-library workspace independently of host availability. */
    if (reserve_blas(p)) {
        (void)initialize_blas();
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / 64 + (p->n % 64 != 0);
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

static inline void store(const Task *t, size_t index, float gate, float up) {
    ((float *)t->output)[index] = activate(gate, up);
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
            VectorPair x0 = load8(t->a, a_base + k, a_dtype);
            VectorPair x1 = load8(t->a, a_base + k + 8, a_dtype);
            VectorPair gate = load8(t->b, weight + k, b_dtype);
            VectorPair up = load8(t->up, weight + k, b_dtype);
            g0 = vfmaq_f32(g0, x0.low, gate.low);
            g1 = vfmaq_f32(g1, x0.high, gate.high);
            u0 = vfmaq_f32(u0, x0.low, up.low);
            u1 = vfmaq_f32(u1, x0.high, up.high);
            gate = load8(t->b, weight + k + 8, b_dtype);
            up = load8(t->up, weight + k + 8, b_dtype);
            g2 = vfmaq_f32(g2, x1.low, gate.low);
            g3 = vfmaq_f32(g3, x1.high, gate.high);
            u2 = vfmaq_f32(u2, x1.low, up.low);
            u3 = vfmaq_f32(u3, x1.high, up.high);
        }
        for (; p->k - k >= 4; k += 4) {
            float32x4_t x = load4(t->a, a_base + k, a_dtype);
            g0 = vfmaq_f32(g0, x, load4(t->b, weight + k, b_dtype));
            u0 = vfmaq_f32(u0, x, load4(t->up, weight + k, b_dtype));
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
        store(t, output + last - buffered + lane, gates[lane], ups[lane]);
    }
    return TRILLIM_OK;
}

static TrillimStatus decode_tile(const Task *t, size_t a_base, size_t b_base,
                                 size_t output, size_t first, size_t last) {
    const Parameters *p = t->p;
    if (p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_BF16) {
        return decode_typed_tile(t, a_base, b_base, output, first, last, TRILLIM_FP32, TRILLIM_BF16);
    }
    return decode_typed_tile(t, a_base, b_base, output, first, last, p->a_dtype, p->b_dtype);
}

static TrillimStatus project_tile(const Task *t, size_t a_base, size_t b_base,
                                  size_t output, size_t rows, size_t first, size_t last) {
    const Parameters *p = t->p;
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
            float32x4_t gw0 = load4(t->b, index0, p->b_dtype);
            float32x4_t uw0 = load4(t->up, index0, p->b_dtype);
            size_t index1 = b_base + (column + (columns > 1 ? 1 : 0)) * p->b_col_stride + k;
            float32x4_t gw1 = load4(t->b, index1, p->b_dtype);
            float32x4_t uw1 = load4(t->up, index1, p->b_dtype);
            float32x4_t x0 = load4(t->a, a_base + 0 * p->a_row_stride + k, p->a_dtype);
            g00 = vfmaq_f32(g00, x0, gw0);
            g01 = vfmaq_f32(g01, x0, gw1);
            u00 = vfmaq_f32(u00, x0, uw0);
            u01 = vfmaq_f32(u01, x0, uw1);
            if (rows > 1) {
                float32x4_t x1 = load4(t->a, a_base + 1 * p->a_row_stride + k, p->a_dtype);
                g10 = vfmaq_f32(g10, x1, gw0);
                g11 = vfmaq_f32(g11, x1, gw1);
                u10 = vfmaq_f32(u10, x1, uw0);
                u11 = vfmaq_f32(u11, x1, uw1);
            }
            if (rows > 2) {
                float32x4_t x2 = load4(t->a, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                g20 = vfmaq_f32(g20, x2, gw0);
                g21 = vfmaq_f32(g21, x2, gw1);
                u20 = vfmaq_f32(u20, x2, uw0);
                u21 = vfmaq_f32(u21, x2, uw1);
            }
            if (rows > 3) {
                float32x4_t x3 = load4(t->a, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                g30 = vfmaq_f32(g30, x3, gw0);
                g31 = vfmaq_f32(g31, x3, gw1);
                u30 = vfmaq_f32(u30, x3, uw0);
                u31 = vfmaq_f32(u31, x3, uw1);
            }
            if (rows > 4) {
                float32x4_t x4 = load4(t->a, a_base + 4 * p->a_row_stride + k, p->a_dtype);
                g40 = vfmaq_f32(g40, x4, gw0);
                g41 = vfmaq_f32(g41, x4, gw1);
                u40 = vfmaq_f32(u40, x4, uw0);
                u41 = vfmaq_f32(u41, x4, uw1);
            }
            if (rows > 5) {
                float32x4_t x5 = load4(t->a, a_base + 5 * p->a_row_stride + k, p->a_dtype);
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
                store(t, output + row * p->n + column + c, gsums[row * 2 + c], usums[row * 2 + c]);
            }
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
    size_t row = index / columns * height;
    size_t first = index % columns * 64;
    size_t last = p->n - first < 64 ? p->n : first + 64;
    size_t rows = p->m - row < height ? p->m - row : height;
    size_t a_base = row * p->a_row_stride;
    size_t output = row * p->n;
    return rows == 1 ? decode_tile(t, a_base, 0, output, first, last) :
        project_tile(t, a_base, 0, output, rows, first, last);
}

static TrillimStatus project_columns(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    size_t first = index * 16;
    size_t last = p->n - first < 16 ? p->n : first + 16;
    size_t height = tile_rows(p);
    /* Reuse this weight tile in one worker before advancing to other columns. */
    for (size_t row = 0; row < p->m;) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t rows = p->m - row < height ? p->m - row : height;
        size_t a_base = row * p->a_row_stride;
        size_t output = row * p->n;
        TrillimStatus status = rows == 1 ? decode_tile(t, a_base, 0, output, first, last) :
            project_tile(t, a_base, 0, output, rows, first, last);
        if (status != TRILLIM_OK) {
            return status;
        }
        row += rows;
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
        static const char *const names[] = {"libopenblas.dll", "openblas.dll"};
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
        static const char *const names[] = {"libopenblas.so.0", "libopenblas.so"};
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
                status = pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
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
                    status = pack(weight, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                    b_trans = 112;
                    ldb = (int)k;
                } else {
                    status = pack(weight, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                float *destination = projection == 0 ? c : c_pack;
                int destination_stride = projection == 0 ? ldc : (int)n;
                sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
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
                store(t, output + r * p->n + col, c[r * (size_t)ldc + col], c_pack[r * n + col]);
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

TRILLIM_KERNEL TrillimStatus trillim_neon_dense_glu(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    if (use_blas(p)) {
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
        size_t count = rows * columns;
        size_t lanes = count < LANES ? count : LANES;
        BlasTask job = {task, scratch, count, lanes};
        return context->thread_pool->run(context->thread_pool->context, lanes, blas_project, &job);
    }
    size_t height = tile_rows(p);
    size_t rows = p->m / height + (p->m % height != 0);
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    if (p->m >= 12 && p->m <= 32 && columns / 2 >= context->thread_pool->worker_count &&
        p->a_col_stride == 1 && p->b_row_stride == 1) {
        size_t tiles = p->n / 16 + (p->n % 16 != 0);
        return context->thread_pool->run(context->thread_pool->context, tiles, project_columns, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, rows * columns, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_dense_glu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides\n"
           "from trillim.compiler.kernels.kernel_dataclasses import _KernelBindingContext\n"
           "from trillim.compiler.passes.infer import _promote\n"
           "def bind(c):\n"
           "    x, gate, up = c.inputs['x'], c.inputs['gate_weight'], c.inputs['up_weight']\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or gate.dtype != up.dtype or gate.shape != up.shape or len(gate.shape) != 2 or _promote(x.dtype, gate.dtype) != 'fp32' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('fused GLU requires native floating projections to FP32')\n"
           "    for weight in (gate, up):\n"
           "        if weight.dtype not in {'fp16', 'bf16', 'fp32'} or (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "            raise ValueError('fused GLU requires canonical floating weights')\n"
           "    return _linear(_KernelBindingContext({'x': x, 'weight': gate}, c.outputs, c.attrs))\n"
           "def pattern(x: Tensor, gate_weight: Tensor, up_weight: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (SiluMul(Linear(x, gate_weight), Linear(x, up_weight), dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_glu_parameters(void) {
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
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_glu_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
