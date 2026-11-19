#ifndef TRILLIM_NEON_ORDERED_MATMUL_H
#define TRILLIM_NEON_ORDERED_MATMUL_H

/* Include after Task and store so each kernel retains its output rounding. */
static bool ordered_narrow_shape(const Parameters *p) {
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

static TrillimStatus ordered_narrow_tile(
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

#endif
