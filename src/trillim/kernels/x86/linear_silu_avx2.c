#include "common.h"
#include <cpuid.h>
#include <immintrin.h>
#include <math.h>

#define MODE 1
#define VECTOR __attribute__((target("avx2")))

typedef struct {
    size_t batches, m, n, k, rank;
    const size_t *batch_shape, *a_batch_strides, *b_batch_strides;
    size_t a_row_stride, a_col_stride, b_row_stride, b_col_stride;
    uint32_t a_dtype, b_dtype, out_dtype;
    size_t a_storage_rows, a_storage_columns, b_storage_rows, b_storage_columns;
#if MODE != 3
    uint32_t linear_dtype, bias_dtype;
#endif
} Parameters;

typedef struct {
    void *output;
    const void *input, *weight, *third;
    const Parameters *p;
    const TrillimKernelContext *context;
    bool f16c;
    bool fma;
} Task;

static bool matrix_span(size_t base, size_t rows, size_t columns, size_t row_stride, size_t col_stride, size_t capacity) {
    if (!rows || !columns) {
        return true;
    }
    if (base >= capacity || (row_stride && rows - 1 > (capacity - 1 - base) / row_stride)) {
        return false;
    }
    base += (rows - 1) * row_stride;
    return !col_stride || columns - 1 <= (capacity - 1 - base) / col_stride;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) || !trillim_float(p->out_dtype)
#if MODE != 3
        || !trillim_float(p->bias_dtype) || !trillim_float(p->linear_dtype)
#endif
#if MODE == 1 || MODE == 2
        || p->a_dtype != TRILLIM_FP32 || p->out_dtype != TRILLIM_FP32
#endif
#if MODE == 3
        || p->out_dtype != TRILLIM_FP32 || (p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32)
        || p->rank || p->batches != 1 || p->a_col_stride != 1 || p->b_row_stride != 1 || p->b_col_stride != p->k
#endif
    ) {
        return TRILLIM_ERROR;
    }
    if (!p->batches || !p->m || !p->n) {
        return TRILLIM_OK;
    }
    size_t a_span, b_span, batches = 1;
    if (!trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span) ||
        p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / trillim_itemsize(p->out_dtype)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->batch_shape[axis] && batches > SIZE_MAX / p->batch_shape[axis]) {
            return TRILLIM_ERROR;
        }
        batches *= p->batch_shape[axis];
    }
    if (batches != p->batches) {
        return TRILLIM_ERROR;
    }
    if (p->k) {
        if ((p->a_storage_columns && p->a_storage_rows > SIZE_MAX / trillim_itemsize(p->a_dtype) / p->a_storage_columns) ||
            (p->b_storage_columns && p->b_storage_rows > SIZE_MAX / trillim_itemsize(p->b_dtype) / p->b_storage_columns)) {
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
#if MODE == 3
    inputs[2] = inputs[1];
#else
    if (p->n > SIZE_MAX / trillim_itemsize(p->bias_dtype)) {
        return TRILLIM_ERROR;
    }
    inputs[2] = p->n * trillim_itemsize(p->bias_dtype);
#endif
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    return TRILLIM_OK;
}

static unsigned optional_features(void) {
    unsigned eax, ebx, ecx, edx;
    return __get_cpuid(1, &eax, &ebx, &ecx, &edx) ? ecx : 0;
}

__attribute__((target("avx2,f16c"))) static __m256 load_half(const uint16_t *source) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)source));
}

VECTOR static inline __attribute__((always_inline)) __m256 load8(const void *data, size_t index, uint32_t dtype, bool f16c) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    const uint16_t *source = (const uint16_t *)data + index;
    if (dtype == TRILLIM_BF16) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)source)), 16));
    }
    if (f16c) {
        return load_half(source);
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = trillim_fp16(source[lane]);
    }
    return _mm256_loadu_ps(values);
}

VECTOR static float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

VECTOR static void store(const Task *task, size_t index, size_t column, float value, float up) {
    const Parameters *p = task->p;
#if MODE == 3
    (void)column;
    float decay = expf(-fabsf(value));
    float sigmoid = value >= 0 ? 1.0f / (1.0f + decay) : decay / (1.0f + decay);
    value = value * sigmoid * up;
#else
    (void)up;
    if (p->linear_dtype == TRILLIM_FP16) {
        value = trillim_fp16(trillim_to_fp16(value));
    } else if (p->linear_dtype == TRILLIM_BF16) {
        value = trillim_bf16(trillim_to_bf16(value));
    }
    value += trillim_load(task->third, column, p->bias_dtype).f;
#if MODE == 1
    float decay = expf(-fabsf(value));
    float sigmoid = value >= 0 ? 1.0f / (1.0f + decay) : decay / (1.0f + decay);
    value *= sigmoid;
#elif MODE == 2
    value = value * (0.5f * (1.0f + erff(value * 0.70710678118654752440f)));
#endif
#endif
    TrillimValue result = {.f = value};
    trillim_store(task->output, index, p->out_dtype, result);
}


#define DENSE_FAST __attribute__((target("avx2,fma,f16c")))

DENSE_FAST static inline __attribute__((always_inline)) __m256 fast_load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    if (dtype == TRILLIM_FP16) {
        return _mm256_cvtph_ps(bits);
    }
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

DENSE_FAST static TrillimStatus fast_dense_project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0);
    size_t column_tiles = p->n / 32 + (p->n % 32 != 0);
    size_t batch = tile / column_tiles / row_tiles;
    size_t row = tile / column_tiles % row_tiles * 4;
    size_t rows = p->m - row < 4 ? p->m - row : 4;
    size_t first = tile % column_tiles * 32, end = p->n - first < 32 ? p->n : first + 32;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    for (size_t column = first; column < end; column += 2) {
        size_t second = column + 1 < end ? column + 1 : column;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        size_t k = 0;
        for (; p->k - k >= 8; k += 8) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 w0 = fast_load8(task->weight, b_base + column * p->b_col_stride + k, p->b_dtype);
            __m256 w1 = fast_load8(task->weight, b_base + second * p->b_col_stride + k, p->b_dtype);
            __m256 x = fast_load8(task->input, a_base + k, p->a_dtype);
            a0 = _mm256_fmadd_ps(x, w0, a0);
            a1 = _mm256_fmadd_ps(x, w1, a1);
            if (rows > 1) {
                x = fast_load8(task->input, a_base + p->a_row_stride + k, p->a_dtype);
                b0 = _mm256_fmadd_ps(x, w0, b0);
                b1 = _mm256_fmadd_ps(x, w1, b1);
            }
            if (rows > 2) {
                x = fast_load8(task->input, a_base + 2 * p->a_row_stride + k, p->a_dtype);
                c0 = _mm256_fmadd_ps(x, w0, c0);
                c1 = _mm256_fmadd_ps(x, w1, c1);
            }
            if (rows > 3) {
                x = fast_load8(task->input, a_base + 3 * p->a_row_stride + k, p->a_dtype);
                d0 = _mm256_fmadd_ps(x, w0, d0);
                d1 = _mm256_fmadd_ps(x, w1, d1);
            }
        }
        float sums[4][2] = {{sum8(a0), sum8(a1)}, {sum8(b0), sum8(b1)}, {sum8(c0), sum8(c1)}, {sum8(d0), sum8(d1)}};
        for (; k < p->k; ++k) {
            for (size_t r = 0; r < rows; ++r) {
                float x = trillim_load(task->input, a_base + r * p->a_row_stride + k, p->a_dtype).f;
                sums[r][0] += x * trillim_load(task->weight, b_base + column * p->b_col_stride + k, p->b_dtype).f;
                sums[r][1] += x * trillim_load(task->weight, b_base + second * p->b_col_stride + k, p->b_dtype).f;
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            size_t index = (batch * p->m + row + r) * p->n;
            store(task, index + column, column, sums[r][0], 0);
            if (second != column) {
                store(task, index + second, second, sums[r][1], 0);
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#if MODE == 3
DENSE_FAST static TrillimStatus fast_glu_project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row_tiles = p->m / 2 + (p->m % 2 != 0);
    size_t column_tiles = p->n / 32 + (p->n % 32 != 0);
    size_t batch = tile / column_tiles / row_tiles;
    size_t row = tile / column_tiles % row_tiles * 2;
    size_t rows = p->m - row < 2 ? p->m - row : 2;
    size_t first = tile % column_tiles * 32, end = p->n - first < 32 ? p->n : first + 32;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    for (size_t column = first; column < end; column += 2) {
        size_t second = column + 1 < end ? column + 1 : column;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        size_t k = 0;
        for (; p->k - k >= 8; k += 8) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t first_weight = b_base + column * p->b_col_stride + k;
            size_t second_weight = b_base + second * p->b_col_stride + k;
            __m256 w0 = fast_load8(task->weight, first_weight, p->b_dtype);
            __m256 w1 = fast_load8(task->weight, second_weight, p->b_dtype);
            __m256 u0 = fast_load8(task->third, first_weight, p->b_dtype);
            __m256 u1 = fast_load8(task->third, second_weight, p->b_dtype);
            __m256 x = fast_load8(task->input, a_base + k, p->a_dtype);
            a0 = _mm256_fmadd_ps(x, w0, a0);
            a1 = _mm256_fmadd_ps(x, w1, a1);
            c0 = _mm256_fmadd_ps(x, u0, c0);
            c1 = _mm256_fmadd_ps(x, u1, c1);
            if (rows > 1) {
                x = fast_load8(task->input, a_base + p->a_row_stride + k, p->a_dtype);
                b0 = _mm256_fmadd_ps(x, w0, b0);
                b1 = _mm256_fmadd_ps(x, w1, b1);
                d0 = _mm256_fmadd_ps(x, u0, d0);
                d1 = _mm256_fmadd_ps(x, u1, d1);
            }
        }
        float gates[2][2] = {{sum8(a0), sum8(a1)}, {sum8(b0), sum8(b1)}};
        float ups[2][2] = {{sum8(c0), sum8(c1)}, {sum8(d0), sum8(d1)}};
        for (; k < p->k; ++k) {
            for (size_t r = 0; r < rows; ++r) {
                float x = trillim_load(task->input, a_base + r * p->a_row_stride + k, p->a_dtype).f;
                size_t first_weight = b_base + column * p->b_col_stride + k;
                size_t second_weight = b_base + second * p->b_col_stride + k;
                gates[r][0] += x * trillim_load(task->weight, first_weight, p->b_dtype).f;
                gates[r][1] += x * trillim_load(task->weight, second_weight, p->b_dtype).f;
                ups[r][0] += x * trillim_load(task->third, first_weight, p->b_dtype).f;
                ups[r][1] += x * trillim_load(task->third, second_weight, p->b_dtype).f;
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            size_t index = (batch * p->m + row + r) * p->n;
            store(task, index + column, column, gates[r][0], ups[r][0]);
            if (second != column) {
                store(task, index + second, second, gates[r][1], ups[r][1]);
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}
#endif

VECTOR static TrillimStatus project_columns(const Task *task, size_t a_base, size_t b_base, size_t batch, size_t first_row, size_t rows, size_t first, size_t end) {
    const Parameters *p = task->p;
    for (size_t column = first; column < end; column += 8) {
        size_t count = end - column < 8 ? end - column : 8;
        __m256 sums[4], ups[4];
        for (size_t row = 0; row < rows; ++row) {
            sums[row] = ups[row] = _mm256_setzero_ps();
        }
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t index = b_base + k * p->b_row_stride + column;
            __m256 weight, up_weight = _mm256_setzero_ps();
            if (count == 8) {
                weight = load8(task->weight, index, p->b_dtype, task->f16c);
#if MODE == 3
                up_weight = load8(task->third, index, p->b_dtype, task->f16c);
#endif
            } else {
                float lanes[8] = {0}, up_lanes[8] = {0};
                for (size_t lane = 0; lane < count; ++lane) {
                    lanes[lane] = trillim_load(task->weight, index + lane, p->b_dtype).f;
#if MODE == 3
                    up_lanes[lane] = trillim_load(task->third, index + lane, p->b_dtype).f;
#endif
                }
                weight = _mm256_loadu_ps(lanes);
                up_weight = _mm256_loadu_ps(up_lanes);
            }
            for (size_t row = 0; row < rows; ++row) {
                __m256 input = _mm256_set1_ps(trillim_load(task->input, a_base + row * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f);
                sums[row] = _mm256_add_ps(sums[row], _mm256_mul_ps(input, weight));
#if MODE == 3
                ups[row] = _mm256_add_ps(ups[row], _mm256_mul_ps(input, up_weight));
#else
                (void)up_weight;
#endif
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float lanes[8], up_lanes[8];
            _mm256_storeu_ps(lanes, sums[row]);
            _mm256_storeu_ps(up_lanes, ups[row]);
            for (size_t lane = 0; lane < count; ++lane) {
                store(task, (batch * p->m + first_row + row) * p->n + column + lane, column + lane, lanes[lane], up_lanes[lane]);
            }
        }
    }
    return TRILLIM_OK;
}

VECTOR static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->fma && task->f16c && p->a_col_stride == 1 && p->b_row_stride == 1) {
#if MODE == 3
        if (task->third != NULL) {
            return fast_glu_project(opaque, tile, worker);
        }
#endif
        return fast_dense_project(opaque, tile, worker);
    }
    const size_t tile_rows = MODE == 3 && task->third != NULL ? 2 : 4;
    size_t row_tiles = p->m / tile_rows + (p->m % tile_rows != 0);
    size_t column_tiles = p->n / 32 + (p->n % 32 != 0);
    size_t batch = tile / column_tiles / row_tiles;
    size_t first_row = tile / column_tiles % row_tiles * tile_rows;
    size_t rows = p->m - first_row < tile_rows ? p->m - first_row : tile_rows;
    size_t first = tile % column_tiles * 32;
    size_t end = p->n - first < 32 ? p->n : first + 32;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + first_row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    if (p->b_col_stride == 1) {
        return project_columns(task, a_base, b_base, batch, first_row, rows, first, end);
    }
    for (size_t column = first; column < end; column += 2) {
        size_t columns = end - column < 2 ? 1 : 2;
        __m256 sums[4][2], ups[4][2];
        for (size_t row = 0; row < rows; ++row) {
            for (size_t output = 0; output < columns; ++output) {
                sums[row][output] = _mm256_setzero_ps();
                ups[row][output] = _mm256_setzero_ps();
            }
        }
        size_t k = 0;
        if (p->a_col_stride == 1 && p->b_row_stride == 1) {
            for (; p->k - k >= 8; k += 8) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                __m256 weights[2];
#if MODE == 3
                __m256 up_weights[2];
#endif
                for (size_t output = 0; output < columns; ++output) {
                    size_t index = b_base + (column + output) * p->b_col_stride + k;
                    weights[output] = load8(task->weight, index, p->b_dtype, task->f16c);
#if MODE == 3
                    up_weights[output] = load8(task->third, index, p->b_dtype, task->f16c);
#endif
                }
                for (size_t row = 0; row < rows; ++row) {
                    __m256 input = load8(task->input, a_base + row * p->a_row_stride + k, p->a_dtype, task->f16c);
                    for (size_t output = 0; output < columns; ++output) {
                        sums[row][output] = _mm256_add_ps(sums[row][output], _mm256_mul_ps(input, weights[output]));
#if MODE == 3
                        ups[row][output] = _mm256_add_ps(ups[row][output], _mm256_mul_ps(input, up_weights[output]));
#endif
                    }
                }
            }
        }
        float values[4][2], up_values[4][2];
        for (size_t row = 0; row < rows; ++row) {
            for (size_t output = 0; output < columns; ++output) {
                values[row][output] = sum8(sums[row][output]);
                up_values[row][output] = sum8(ups[row][output]);
            }
        }
        for (; k < p->k; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            for (size_t row = 0; row < rows; ++row) {
                float input = trillim_load(task->input, a_base + row * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f;
                for (size_t output = 0; output < columns; ++output) {
                    size_t index = b_base + k * p->b_row_stride + (column + output) * p->b_col_stride;
                    values[row][output] += input * trillim_load(task->weight, index, p->b_dtype).f;
#if MODE == 3
                    up_values[row][output] += input * trillim_load(task->third, index, p->b_dtype).f;
#endif
                }
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t output = 0; output < columns; ++output) {
                size_t index = (batch * p->m + first_row + row) * p->n + column + output;
                store(task, index, column + output, values[row][output], up_values[row][output]);
            }
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_linear_silu(void *const *outputs, const void *const *inputs, void *scratch, const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_spans[3], output_span[1], scratch_span;
    TrillimStatus status = requirements(parameters, input_spans, output_span, &scratch_span);
    if (status != TRILLIM_OK || !output_span[0]) {
        return status;
    }
    const Parameters *p = parameters;
    size_t rows = MODE == 3 ? 2 : 4;
    size_t count = p->batches * (p->m / rows + (p->m % rows != 0)) * (p->n / 32 + (p->n % 32 != 0));
    unsigned features = optional_features();
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context, (features & (1u << 29)) != 0, (features & (1u << 12)) != 0};
    return context->thread_pool->run(context->thread_pool->context, count, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_linear_silu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _DTYPE_IDS\n"
           "\n"
           "def bind(c):\n"
           "    x, weight, bias = (c.inputs[name] for name in ('x', 'weight', 'bias'))\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or len(weight.shape) != 2 or x.shape[-1] != weight.shape[1]:\n"
           "        raise ValueError('Linear-SiLU requires compatible channel and matrix dimensions')\n"
           "    if x.dtype != 'fp32' or weight.dtype not in {'fp16', 'bf16', 'fp32'} or bias.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != 'fp32':\n"
           "        raise ValueError('Linear-SiLU requires FP32 activations with floating weights and bias')\n"
           "    if bias.shape != weight.shape[:1] or output.shape != (*x.shape[:-1], weight.shape[0]):\n"
           "        raise ValueError('Linear-SiLU requires a channel bias and the projected output shape')\n"
           "    for value in (bias, output):\n"
           "        if (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('Linear-SiLU requires contiguous bias and output')\n"
           "    return dict(_linear(c), linear_dtype=_DTYPE_IDS['fp32'], bias_dtype=_DTYPE_IDS[bias.dtype])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Silu(Linear(x, weight, bias=bias)),)\n";
}


TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_linear_silu_parameters(void) {
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
        {"linear_dtype", "uint32_t", offsetof(Parameters, linear_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_linear_silu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
