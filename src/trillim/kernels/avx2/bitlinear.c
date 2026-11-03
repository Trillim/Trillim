#include "bitlinear_avx2.h"

#include <float.h>
#include <immintrin.h>
#include <string.h>

#define AVX2 __attribute__((target("avx2")))
#define DECODE_OUTPUTS_PER_TASK 128
#define ACTIVATION_METADATA_BYTES 32



static const TrillimParameter parameter_fields[] = {
    {"tokens", "size_t", offsetof(TrillimBitLinearParameters, tokens), NULL},
    {"width", "size_t", offsetof(TrillimBitLinearParameters, width), NULL},
    {"outputs", "size_t", offsetof(TrillimBitLinearParameters, outputs), NULL}
};

static TrillimStatus requirements(
    const void *parameters, size_t *input_bytes, size_t *output_bytes, size_t *scratch_bytes
) {
    TrillimStatus status = trillim_bitlinear_requirements(parameters, input_bytes, output_bytes, scratch_bytes);
    const TrillimBitLinearParameters *p = (const TrillimBitLinearParameters *)parameters;
    if (status != TRILLIM_OK || p->tokens == 0 || p->outputs == 0) {
        return status;
    }
    if (p->width % 128 != 0 || p->width > INT64_MAX / 254 ||
        p->width > SIZE_MAX - ACTIVATION_METADATA_BYTES ||
        p->tokens > SIZE_MAX / (p->width + ACTIVATION_METADATA_BYTES)) {
        return TRILLIM_ERROR;
    }
    *scratch_bytes = p->tokens * (p->width + ACTIVATION_METADATA_BYTES);
    return TRILLIM_OK;
}

static const size_t input_alignments[] = {4, 1};
static const size_t output_alignments[] = {4};
static const TrillimParameterSchema parameter_schema = {
    sizeof(TrillimBitLinearParameters), TRILLIM_ALIGNOF(TrillimBitLinearParameters),
    sizeof(parameter_fields) / sizeof(parameter_fields[0]), parameter_fields,
    2, 1, requirements, NULL, input_alignments, output_alignments, ACTIVATION_METADATA_BYTES
};

AVX2 static int32_t sum_ints(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static TrillimStatus quantize(void *task_context, size_t token, size_t worker_index) {
    (void)worker_index;
    const TrillimAvx2BitLinearTask *task = (const TrillimAvx2BitLinearTask *)task_context;
    size_t width = task->params->width;
    const float *input = task->input + token * width;
    uint8_t *storage = task->scratch + token * (width + ACTIVATION_METADATA_BYTES);
    int8_t *quantized = (int8_t *)storage;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 finite_limit = _mm256_set1_ps(FLT_MAX);
    __m256 maxima = _mm256_set1_ps(1.0e-5f);
    for (size_t k = 0; k < width; k += 8) {
        if ((k & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 magnitude = _mm256_andnot_ps(sign, _mm256_loadu_ps(input + k));
        __m256 finite = _mm256_cmp_ps(magnitude, finite_limit, _CMP_LE_OQ);
        if (_mm256_movemask_ps(finite) != 255) {
            return TRILLIM_ERROR;
        }
        maxima = _mm256_max_ps(maxima, magnitude);
    }
    __m128 maximum4 = _mm_max_ps(_mm256_castps256_ps128(maxima),
                                _mm256_extractf128_ps(maxima, 1));
    maximum4 = _mm_max_ps(maximum4, _mm_movehl_ps(maximum4, maximum4));
    maximum4 = _mm_max_ss(maximum4,
                        _mm_shuffle_ps(maximum4, maximum4, _MM_SHUFFLE(1, 1, 1, 1)));
    float maximum = _mm_cvtss_f32(maximum4);
    float multiplier = 127.0f / maximum;
    float scale = 1.0f / multiplier;
    const __m256 factor = _mm256_set1_ps(multiplier);
    const __m256 lower = _mm256_set1_ps(-128.0f);
    const __m256 upper = _mm256_set1_ps(127.0f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    int64_t row_sum = 0;
    for (size_t k = 0; k < width; k += 32) {
        if ((k & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256i parts[4];
        for (size_t part = 0; part < 4; ++part) {
            __m256 values = _mm256_mul_ps(_mm256_loadu_ps(input + k + part * 8), factor);
            values = _mm256_min_ps(upper, _mm256_max_ps(lower, values));
            values = _mm256_round_ps(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            parts[part] = _mm256_cvttps_epi32(values);
        }
        __m256i packed = _mm256_packs_epi16(
            _mm256_packs_epi32(parts[0], parts[1]),
            _mm256_packs_epi32(parts[2], parts[3]));
        packed = _mm256_permutevar8x32_epi32(packed, order);
        _mm256_storeu_si256((__m256i *)(quantized + k), packed);
        row_sum += sum_ints(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, packed), ones16));
    }
    memcpy(storage + width, &scale, sizeof(scale));
    memcpy(storage + width + 8, &row_sum, sizeof(row_sum));
    return TRILLIM_OK;
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
    const TrillimAvx2BitLinearTask *task = (const TrillimAvx2BitLinearTask *)task_context;
    const TrillimBitLinearParameters *p = task->params;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    /* Short rows poll within 1M products and 256 KiB of packed weights. */
    bool poll_inner = p->width > 32768;
    size_t begin = task_index * task->decode_columns;
    size_t end = p->outputs - begin < task->decode_columns
        ? p->outputs : begin + task->decode_columns;
    size_t packed_row = p->width / 4;
    const int8_t *input = (const int8_t *)task->scratch;
    const __m256i ones16 = _mm256_set1_epi16(1);
    float scale;
    int64_t row_sum;
    memcpy(&scale, task->scratch + p->width, sizeof(scale));
    memcpy(&row_sum, task->scratch + p->width + 8, sizeof(row_sum));
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
        task->output[column] = (float)(sum0 - row_sum) * scale * task->gamma;
        if (lanes > 1) {
            task->output[column + 1] = (float)(sum1 - row_sum) * scale * task->gamma;
        }
        if (lanes > 2) {
            task->output[column + 2] = (float)(sum2 - row_sum) * scale * task->gamma;
        }
        if (lanes > 3) {
            task->output[column + 3] = (float)(sum3 - row_sum) * scale * task->gamma;
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
    const TrillimAvx2BitLinearTask *task = (const TrillimAvx2BitLinearTask *)task_context;
    const TrillimBitLinearParameters *p = task->params;
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
    size_t stride = p->width + ACTIVATION_METADATA_BYTES;
    const int8_t *x0 = (const int8_t *)(task->scratch + first_token * stride);
    const int8_t *x1 = rows > 1 ? x0 + stride : x0;
    const int8_t *x2 = rows > 2 ? x1 + stride : x1;
    const int8_t *x3 = rows > 3 ? x2 + stride : x2;
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
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
            const uint8_t *storage = task->scratch + token * stride;
            float scale;
            int64_t row_sum;
            memcpy(&scale, storage + p->width, sizeof(scale));
            memcpy(&row_sum, storage + p->width + 8, sizeof(row_sum));
            task->output[token * p->outputs + output_column] =
                (float)(sums[lane] - row_sum) * scale * task->gamma;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_avx2_bitlinear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimBitLinearParameters *p = (const TrillimBitLinearParameters *)parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    size_t decode_columns = DECODE_OUTPUTS_PER_TASK;
    if (p->tokens == 1) {
        size_t workers = context->thread_pool->worker_count;
        if (workers == 0) {
            return TRILLIM_ERROR;
        }
        size_t per_worker = p->outputs / workers + (p->outputs % workers != 0);
        size_t per_task = per_worker / 4 + (per_worker % 4 != 0);
        decode_columns = (per_task / 4 + (per_task % 4 != 0)) * 4;
    }
    float gamma;
    memcpy(&gamma, (const uint8_t *)inputs[1] + p->outputs * (p->width / 4), sizeof(gamma));
    TrillimAvx2BitLinearTask task = {
        (float *)outputs[0], (const float *)inputs[0], (const uint8_t *)inputs[1],
        gamma, (uint8_t *)scratch, p, context, decode_columns
    };
    TrillimStatus status = p->tokens == 1
        ? quantize(&task, 0, 0)
        : context->thread_pool->run(
            context->thread_pool->context, p->tokens, quantize, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (p->tokens == 1) {
        size_t tasks = p->outputs / decode_columns
            + (p->outputs % decode_columns != 0);
        return context->thread_pool->run(
            context->thread_pool->context, tasks, decode, &task);
    }
    size_t columns_per_task = 16;
    size_t column_tiles = p->outputs / columns_per_task + (p->outputs % columns_per_task != 0);
    size_t token_tiles = p->tokens / 4 + (p->tokens % 4 != 0);
    return context->thread_pool->run(
        context->thread_pool->context, token_tiles * column_tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_avx2_bitlinear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import avx2_bitlinear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (BitLinear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_avx2_bitlinear_parameters(void) {
    return &parameter_schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_avx2_bitlinear_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "avx2", "avx2", "m"};
    return &capabilities;
}
