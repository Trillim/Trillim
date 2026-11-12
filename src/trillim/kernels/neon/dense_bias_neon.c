#include "common.h"
#include <arm_neon.h>
#include <math.h>

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
} Task;

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
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0);
    size_t batch = index / columns / row_tiles;
    size_t row = index / columns % row_tiles * 4;
    size_t first = index % columns * 64;
    size_t last = p->n - first < 64 ? p->n : first + 64;
    size_t rows = p->m - row < 4 ? p->m - row : 4;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    size_t output = (batch * p->m + row) * p->n;
    if (p->a_col_stride == 1 && p->b_row_stride == 1) {
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

TRILLIM_KERNEL TrillimStatus trillim_neon_dense_bias(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t rows = p->m / 4 + (p->m % 4 != 0);
    size_t columns = p->n / 64 + (p->n % 64 != 0);
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
        fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_bias_capabilities(void) {
    /* Advanced SIMD is part of the AArch64 baseline. */
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
