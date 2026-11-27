#include "common.h"
#include <immintrin.h>

#define AVX2 __attribute__((target("avx2")))

typedef struct {
    size_t tokens;
    size_t width;
    size_t outputs;
    uint32_t output_dtype;
} LinearParameters;

typedef struct {
    void *output;
    const int8_t *codes;
    const uint8_t *weight;
    const float *scales;
    const LinearParameters *params;
    float gamma;
    size_t decode_columns;
    int64_t decode_sum;
    const TrillimKernelContext *context;
} LinearTask;

static size_t packed_row_bytes(size_t width) {
    return (width / 128 + (width % 128 != 0)) * 32;
}

AVX2 static inline __m256i collect_invalid(__m256i invalid, __m256i packed) {
    return _mm256_or_si256(invalid, _mm256_and_si256(packed, _mm256_srli_epi16(packed, 1)));
}

AVX2 static inline bool malformed(__m256i invalid) {
    return !_mm256_testz_si256(invalid, _mm256_set1_epi8(0x55));
}

static bool tail_dot(const uint8_t *weight, const int8_t *input, size_t count, size_t rows,
                     size_t stride, int32_t *sums) {
    for (size_t k = 0; k < count; ++k) {
        int32_t code = (weight[k % 32] >> (6 - k / 32 * 2)) & 3;
        if (code == 3) {
            return false;
        }
        for (size_t row = 0; row < rows; ++row) {
            sums[row] += code * input[row * stride + k];
        }
    }
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const LinearParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!p->width || p->width > INT64_MAX / 256 || groups > SIZE_MAX / 32 ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        p->outputs > (SIZE_MAX - sizeof(float)) / (groups * 32) ||
        p->tokens > SIZE_MAX / bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * 32 + sizeof(float);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

AVX2 static int32_t sum_ints(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static TrillimStatus activation_sum(const LinearTask *task, size_t row, int64_t *result) {
    const LinearParameters *p = task->params;
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    int64_t sum = 0;
    size_t k = 0;
    for (; p->width - k >= 32; k += 32) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256i codes = _mm256_loadu_si256((const __m256i *)(task->codes + row * p->width + k));
        sum += sum_ints(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, codes), ones16));
    }
    for (; k < p->width; ++k) {
        sum += task->codes[row * p->width + k];
    }
    *result = sum;
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) void store(const LinearTask *task, size_t index, int64_t sum, float multiplier) {
    TrillimValue value = {.f = (float)sum * multiplier};
    if (task->params->output_dtype == TRILLIM_BF16) {
        uint16_t bits = trillim_to_bf16(value.f);
        memcpy((uint8_t *)task->output + index * 2, &bits, sizeof(bits));
    } else if (task->params->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + index * 4, &value.f, sizeof(value.f));
    } else {
        trillim_store(task->output, index, task->params->output_dtype, value);
    }
}

AVX2 static inline __m256i packed_dot(
    __m256i packed, __m256i y0, __m256i y1, __m256i y2, __m256i y3
) {
    const __m256i mask = _mm256_set1_epi8(3);
    __m256i a = _mm256_maddubs_epi16(
        _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask),
        y0);
    __m256i b = _mm256_maddubs_epi16(
        _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask),
        y1);
    __m256i c = _mm256_maddubs_epi16(
        _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask),
        y2);
    __m256i d = _mm256_maddubs_epi16(
        _mm256_and_si256(packed, mask),
        y3);
    return _mm256_add_epi16(_mm256_add_epi16(a, b), _mm256_add_epi16(c, d));
}

AVX2 static TrillimStatus decode(void *task_context, size_t task_index, size_t worker_index) {
    (void)worker_index;
    const LinearTask *task = (const LinearTask *)task_context;
    const LinearParameters *p = task->params;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    /* Short rows poll within 1M products and 256 KiB of packed weights. */
    bool poll_inner = p->width > 32768;
    size_t begin = task_index * task->decode_columns;
    size_t end = p->outputs - begin < task->decode_columns
        ? p->outputs : begin + task->decode_columns;
    size_t packed_row = p->width / 4;
    const int8_t *input = task->codes;
    float multiplier = task->gamma / task->scales[0];
    const __m256i ones16 = _mm256_set1_epi16(1);
    int64_t row_sum = task->decode_sum;
    for (size_t column = begin; column < end; column += 4) {
        if (!poll_inner && column != begin && ((column - begin) & 31) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t lanes = end - column < 4 ? end - column : 4;
        const uint8_t *w0 = task->weight + column * packed_row;
        const uint8_t *w1 = lanes > 1 ? w0 + packed_row : w0;
        const uint8_t *w2 = lanes > 2 ? w1 + packed_row : w0;
        const uint8_t *w3 = lanes > 3 ? w2 + packed_row : w0;
        int64_t sum0 = 0;
        int64_t sum1 = 0;
        int64_t sum2 = 0;
        int64_t sum3 = 0;
        for (size_t k = 0; k < p->width;) {
            if (poll_inner && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256i dp0 = _mm256_setzero_si256();
            __m256i dp1 = _mm256_setzero_si256();
            __m256i dp2 = _mm256_setzero_si256();
            __m256i dp3 = _mm256_setzero_si256();
            /* 2048 terms keep each signed 16-bit lane in [-32768, 32512]. */
            size_t limit = p->width - k < 2048 ? p->width : k + 2048;
            for (; k < limit; k += 128) {
                __m256i y0 = _mm256_loadu_si256((const __m256i *)(input + k));
                __m256i y1 = _mm256_loadu_si256((const __m256i *)(input + k + 32));
                __m256i y2 = _mm256_loadu_si256((const __m256i *)(input + k + 64));
                __m256i y3 = _mm256_loadu_si256((const __m256i *)(input + k + 96));
                dp0 = _mm256_add_epi16(dp0, packed_dot(
                    _mm256_loadu_si256((const __m256i *)(w0 + k / 4)), y0, y1, y2, y3));
                dp1 = _mm256_add_epi16(dp1, packed_dot(
                    _mm256_loadu_si256((const __m256i *)(w1 + k / 4)), y0, y1, y2, y3));
                dp2 = _mm256_add_epi16(dp2, packed_dot(
                    _mm256_loadu_si256((const __m256i *)(w2 + k / 4)), y0, y1, y2, y3));
                dp3 = _mm256_add_epi16(dp3, packed_dot(
                    _mm256_loadu_si256((const __m256i *)(w3 + k / 4)), y0, y1, y2, y3));
            }
            sum0 += sum_ints(_mm256_madd_epi16(dp0, ones16));
            sum1 += sum_ints(_mm256_madd_epi16(dp1, ones16));
            sum2 += sum_ints(_mm256_madd_epi16(dp2, ones16));
            sum3 += sum_ints(_mm256_madd_epi16(dp3, ones16));
        }
        store(task, column, sum0 - row_sum, multiplier);
        if (lanes > 1) {
            store(task, column + 1, sum1 - row_sum, multiplier);
        }
        if (lanes > 2) {
            store(task, column + 2, sum2 - row_sum, multiplier);
        }
        if (lanes > 3) {
            store(task, column + 3, sum3 - row_sum, multiplier);
        }
    }
    return TRILLIM_OK;
}

AVX2 static inline __m256i prefill_dot(
    __m256i c0, __m256i c1, __m256i c2, __m256i c3, const int8_t *input
) {
    __m256i a = _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *)input));
    __m256i b = _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *)(input + 32)));
    __m256i c = _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *)(input + 64)));
    __m256i d = _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *)(input + 96)));
    return _mm256_add_epi16(_mm256_add_epi16(a, b), _mm256_add_epi16(c, d));
}

AVX2 static TrillimStatus project(void *task_context, size_t task_index, size_t worker_index) {
    (void)worker_index;
    const LinearTask *task = (const LinearTask *)task_context;
    const LinearParameters *p = task->params;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    /* Poll within 4 rows * 8 columns * 32768 terms for short rows. */
    bool poll_inner = p->width > 32768;
    size_t columns_per_task = 16;
    size_t column_tiles = p->outputs / columns_per_task + (p->outputs % columns_per_task != 0);
    size_t first_token = (task_index / column_tiles) * 4;
    size_t first_column = (task_index % column_tiles) * columns_per_task;
    size_t rows = p->tokens - first_token < 4 ? p->tokens - first_token : 4;
    size_t columns = p->outputs - first_column < columns_per_task
        ? p->outputs - first_column : columns_per_task;
    size_t stride = p->width;
    const int8_t *x0 = task->codes + first_token * stride;
    const int8_t *x1 = rows > 1 ? x0 + stride : x0;
    const int8_t *x2 = rows > 2 ? x1 + stride : x1;
    const int8_t *x3 = rows > 3 ? x2 + stride : x2;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    int64_t row_sums[4];
    float multipliers[4];
    for (size_t row = 0; row < rows; ++row) {
        multipliers[row] = task->gamma / task->scales[first_token + row];
        TrillimStatus status = activation_sum(task, first_token + row, &row_sums[row]);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    for (size_t column = 0; column < columns; ++column) {
        if (!poll_inner && column != 0 && (column & 7) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint8_t *weight = task->weight + (first_column + column) * (p->width / 4);
        int64_t sum0 = 0;
        int64_t sum1 = 0;
        int64_t sum2 = 0;
        int64_t sum3 = 0;
        for (size_t k = 0; k < p->width;) {
            if (poll_inner && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256i dp0 = _mm256_setzero_si256();
            __m256i dp1 = _mm256_setzero_si256();
            __m256i dp2 = _mm256_setzero_si256();
            __m256i dp3 = _mm256_setzero_si256();
            /* 2048 terms keep each signed 16-bit lane in [-32768, 32512]. */
            size_t end = p->width - k < 2048 ? p->width : k + 2048;
            for (; k < end; k += 128) {
                __m256i packed = _mm256_loadu_si256((const __m256i *)(weight + k / 4));
                __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
                __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
                __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
                __m256i c3 = _mm256_and_si256(packed, mask);
                dp0 = _mm256_add_epi16(dp0, prefill_dot(c0, c1, c2, c3, x0 + k));
                if (rows > 1) {
                    dp1 = _mm256_add_epi16(dp1, prefill_dot(c0, c1, c2, c3, x1 + k));
                }
                if (rows > 2) {
                    dp2 = _mm256_add_epi16(dp2, prefill_dot(c0, c1, c2, c3, x2 + k));
                }
                if (rows > 3) {
                    dp3 = _mm256_add_epi16(dp3, prefill_dot(c0, c1, c2, c3, x3 + k));
                }
            }
            sum0 += sum_ints(_mm256_madd_epi16(dp0, ones16));
            sum1 += sum_ints(_mm256_madd_epi16(dp1, ones16));
            sum2 += sum_ints(_mm256_madd_epi16(dp2, ones16));
            sum3 += sum_ints(_mm256_madd_epi16(dp3, ones16));
        }
        const int64_t sums[4] = {sum0, sum1, sum2, sum3};
        for (size_t lane = 0; lane < rows; ++lane) {
            size_t token = first_token + lane;
            size_t output_column = first_column + column;
            store(task, token * p->outputs + output_column, sums[lane] - row_sums[lane], multipliers[lane]);
        }
    }
    return TRILLIM_OK;
}

AVX2 static TrillimStatus project_tail(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->params;
    const size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    const size_t row = tile / tiles * 4;
    const size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    const size_t first = tile % tiles * 16;
    const size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    const size_t full_width = p->width - p->width % 128;
    const int8_t *input = task->codes + row * p->width;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    int64_t row_sums[4];
    for (size_t r = 0; r < rows; ++r) {
        TrillimStatus status = activation_sum(task, row + r, row_sums + r);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    for (size_t column = first; column < end; ++column) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint8_t *weight = task->weight + column * packed_row_bytes(p->width);
        int64_t totals[4] = {0};
        __m256i invalid = _mm256_setzero_si256();
        for (size_t k = 0; k < full_width; k += 128) {
            if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256i packed = _mm256_loadu_si256((const __m256i *)(weight + k / 4));
            invalid = collect_invalid(invalid, packed);
            __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
            __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
            __m256i c3 = _mm256_and_si256(packed, mask);
            for (size_t r = 0; r < rows; ++r) {
                const int8_t *x = input + r * p->width + k;
                totals[r] += sum_ints(_mm256_madd_epi16(prefill_dot(c0, c1, c2, c3, x), ones));
            }
        }
        int32_t suffix[4] = {0};
        if (malformed(invalid) || !tail_dot(weight + full_width / 4, input + full_width,
                p->width - full_width, rows, p->width, suffix)) {
            return TRILLIM_ERROR;
        }
        for (size_t r = 0; r < rows; ++r) {
            int64_t sum = totals[r] + suffix[r] - row_sums[r];
            TrillimValue value = {.f = ((float)sum * task->gamma) / task->scales[row + r]};
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_quantized_linear_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const LinearParameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    float gamma;
    memcpy(&gamma, (const uint8_t *)inputs[1] + p->outputs * packed_row_bytes(p->width), sizeof(gamma));
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t per_worker = p->outputs / workers + (p->outputs % workers != 0);
    size_t per_task = per_worker / 4 + (per_worker % 4 != 0);
    size_t decode_columns = (per_task / 4 + (per_task % 4 != 0)) * 4;
    LinearTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, gamma, decode_columns, 0, context};
    if (p->width % 128 != 0) {
        size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
        size_t columns = p->outputs / 16 + (p->outputs % 16 != 0);
        size_t jobs = rows * columns;
        return jobs == 1 ? project_tail(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, jobs, project_tail, &task);
    }
    if (p->tokens == 1) {
        TrillimStatus status = activation_sum(&task, 0, &task.decode_sum);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (p->width <= 65536 / p->outputs) {
            task.decode_columns = p->outputs;
            return decode(&task, 0, 0);
        }
        size_t tasks = p->outputs / decode_columns + (p->outputs % decode_columns != 0);
        return context->thread_pool->run(context->thread_pool->context, tasks, decode, &task);
    }
    size_t column_tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t token_tiles = p->tokens / 4 + (p->tokens % 4 != 0);
    size_t tasks = token_tiles * column_tiles;
    return tasks == 1 ? project(&task, 0, 0)
        : context->thread_pool->run(context->thread_pool->context, tasks, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_quantized_linear_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantized_linear as _bind, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return _bind(context)\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_quantized_linear_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(LinearParameters, tokens), NULL},
        {"width", "size_t", offsetof(LinearParameters, width), NULL},
        {"outputs", "size_t", offsetof(LinearParameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(LinearParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(LinearParameters), TRILLIM_ALIGNOF(LinearParameters),
        4, fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_quantized_linear_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
