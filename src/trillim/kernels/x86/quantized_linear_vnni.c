#include "common.h"
#include <immintrin.h>

#define AVX2 __attribute__((target("avx2,avxvnni")))

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
    int32_t *row_sums;
    float *multipliers;
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
    if (!p->width || p->width > INT32_MAX / 256 || groups > SIZE_MAX / 32 ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / (sizeof(int32_t) + sizeof(float)) ||
        p->outputs > (SIZE_MAX - sizeof(float)) / (groups * 32) ||
        p->tokens > SIZE_MAX / bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * 32 + sizeof(float);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    *scratch = p->tokens * (sizeof(int32_t) + sizeof(float));
    return TRILLIM_OK;
}

AVX2 static int32_t sum8(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static TrillimStatus row_sum(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->params;
    __m256i sum = _mm256_setzero_si256(), ones = _mm256_set1_epi8(1);
    size_t k = 0;
    for (; p->width - k >= 32; k += 32) {
        if ((k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        sum = _mm256_dpbusd_avx_epi32(sum, ones, _mm256_loadu_si256((const __m256i *)(task->codes + row * p->width + k)));
    }
    int32_t total = sum8(sum);
    for (; k < p->width; ++k) {
        total += task->codes[row * p->width + k];
    }
    task->row_sums[row] = total;
    task->multipliers[row] = task->gamma / task->scales[row];
    return TRILLIM_OK;
}

AVX2 static inline __attribute__((always_inline)) void store(const LinearTask *task, size_t row, size_t column, __m256i sum) {
    TrillimValue value = {.f = (float)(sum8(sum) - task->row_sums[row]) * task->multipliers[row]};
    size_t index = row * task->params->outputs + column;
    if (task->params->output_dtype == TRILLIM_BF16) {
        uint16_t bits = trillim_to_bf16(value.f);
        memcpy((uint8_t *)task->output + index * 2, &bits, sizeof(bits));
    } else if (task->params->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + index * 4, &value.f, sizeof(value.f));
    } else {
        trillim_store(task->output, index, task->params->output_dtype, value);
    }
}

AVX2 static __m256i dot(__m256i sum, __m256i w0, __m256i w1, __m256i w2, __m256i w3, const int8_t *input) {
    sum = _mm256_dpbusd_avx_epi32(sum, w0, _mm256_loadu_si256((const __m256i *)input));
    sum = _mm256_dpbusd_avx_epi32(sum, w1, _mm256_loadu_si256((const __m256i *)(input + 32)));
    sum = _mm256_dpbusd_avx_epi32(sum, w2, _mm256_loadu_si256((const __m256i *)(input + 64)));
    return _mm256_dpbusd_avx_epi32(sum, w3, _mm256_loadu_si256((const __m256i *)(input + 96)));
}

AVX2 static TrillimStatus prefill(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->params;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = (tile / tiles) * 8, column = (tile % tiles) * 16;
    size_t rows = p->tokens - row < 8 ? p->tokens - row : 8;
    size_t end = p->outputs - column < 16 ? p->outputs : column + 16;
    const int8_t *x0 = task->codes + row * p->width;
    const int8_t *x1 = rows > 1 ? x0 + p->width : x0;
    const int8_t *x2 = rows > 2 ? x1 + p->width : x0;
    const int8_t *x3 = rows > 3 ? x2 + p->width : x0;
    const int8_t *x4 = rows > 4 ? x3 + p->width : x0;
    const int8_t *x5 = rows > 5 ? x4 + p->width : x0;
    const int8_t *x6 = rows > 6 ? x5 + p->width : x0;
    const int8_t *x7 = rows > 7 ? x6 + p->width : x0;
    const __m256i mask = _mm256_set1_epi8(3);
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    for (; column < end; ++column) {
        const uint8_t *weight = task->weight + column * (p->width / 4);
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        __m256i s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (size_t k = 0; k < p->width; k += 128) {
            if (p->width > 32768 && (k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256i packed = _mm256_loadu_si256((const __m256i *)(weight + k / 4));
            __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
            __m256i w3 = _mm256_and_si256(packed, mask);
            s0 = dot(s0, w0, w1, w2, w3, x0 + k);
            if (rows > 1) s1 = dot(s1, w0, w1, w2, w3, x1 + k);
            if (rows > 2) s2 = dot(s2, w0, w1, w2, w3, x2 + k);
            if (rows > 3) s3 = dot(s3, w0, w1, w2, w3, x3 + k);
            if (rows > 4) s4 = dot(s4, w0, w1, w2, w3, x4 + k);
            if (rows > 5) s5 = dot(s5, w0, w1, w2, w3, x5 + k);
            if (rows > 6) s6 = dot(s6, w0, w1, w2, w3, x6 + k);
            if (rows > 7) s7 = dot(s7, w0, w1, w2, w3, x7 + k);
        }
        store(task, row, column, s0);
        if (rows > 1) store(task, row + 1, column, s1);
        if (rows > 2) store(task, row + 2, column, s2);
        if (rows > 3) store(task, row + 3, column, s3);
        if (rows > 4) store(task, row + 4, column, s4);
        if (rows > 5) store(task, row + 5, column, s5);
        if (rows > 6) store(task, row + 6, column, s6);
        if (rows > 7) store(task, row + 7, column, s7);
    }
    return TRILLIM_OK;
}

AVX2 static __m256i decode_dot(__m256i sum, __m256i packed, __m256i x0, __m256i x1, __m256i x2, __m256i x3) {
    const __m256i mask = _mm256_set1_epi8(3);
    sum = _mm256_dpbusd_avx_epi32(sum, _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask), x0);
    sum = _mm256_dpbusd_avx_epi32(sum, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask), x1);
    sum = _mm256_dpbusd_avx_epi32(sum, _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask), x2);
    return _mm256_dpbusd_avx_epi32(sum, _mm256_and_si256(packed, mask), x3);
}

AVX2 static TrillimStatus decode(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->params;
    size_t begin = tile * task->decode_columns;
    size_t end = p->outputs - begin < task->decode_columns ? p->outputs : begin + task->decode_columns;
    size_t stride = p->width / 4;
    for (size_t column = begin; column < end; column += 4) {
        if ((column == begin || (column & 63) == 0) && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        size_t count = end - column < 4 ? end - column : 4;
        const uint8_t *w0 = task->weight + column * stride;
        const uint8_t *w1 = count > 1 ? w0 + stride : w0;
        const uint8_t *w2 = count > 2 ? w1 + stride : w0;
        const uint8_t *w3 = count > 3 ? w2 + stride : w0;
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (size_t k = 0; k < p->width; k += 128) {
            if (p->width > 32768 && (k & 4095) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(task->codes + k));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(task->codes + k + 32));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(task->codes + k + 64));
            __m256i x3 = _mm256_loadu_si256((const __m256i *)(task->codes + k + 96));
            s0 = decode_dot(s0, _mm256_loadu_si256((const __m256i *)(w0 + k / 4)), x0, x1, x2, x3);
            s1 = decode_dot(s1, _mm256_loadu_si256((const __m256i *)(w1 + k / 4)), x0, x1, x2, x3);
            s2 = decode_dot(s2, _mm256_loadu_si256((const __m256i *)(w2 + k / 4)), x0, x1, x2, x3);
            s3 = decode_dot(s3, _mm256_loadu_si256((const __m256i *)(w3 + k / 4)), x0, x1, x2, x3);
        }
        store(task, 0, column, s0);
        if (count > 1) store(task, 0, column + 1, s1);
        if (count > 2) store(task, 0, column + 2, s2);
        if (count > 3) store(task, 0, column + 3, s3);
    }
    return TRILLIM_OK;
}

AVX2 static TrillimStatus project_tail(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const LinearTask *task = opaque;
    const LinearParameters *p = task->params;
    const size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    const size_t row = tile / tiles * 8;
    const size_t rows = p->tokens - row < 8 ? p->tokens - row : 8;
    const size_t first = tile % tiles * 16;
    const size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    const size_t full_width = p->width - p->width % 128;
    const int8_t *input = task->codes + row * p->width;
    const __m256i mask = _mm256_set1_epi8(3);
    for (size_t column = first; column < end; ++column) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint8_t *weight = task->weight + column * packed_row_bytes(p->width);
        __m256i totals[8] = {0};
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
                totals[r] = dot(totals[r], c0, c1, c2, c3, x);
            }
        }
        int32_t suffix[8] = {0};
        if (malformed(invalid) || !tail_dot(weight + full_width / 4, input + full_width,
                p->width - full_width, rows, p->width, suffix)) {
            return TRILLIM_ERROR;
        }
        for (size_t r = 0; r < rows; ++r) {
            int32_t sum = sum8(totals[r]) + suffix[r] - task->row_sums[row + r];
            TrillimValue value = {.f = ((float)sum * task->gamma) / task->scales[row + r]};
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_quantized_linear_vnni(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const LinearParameters *p = parameters;
    if (!p->tokens || !p->outputs) return TRILLIM_OK;
    float gamma;
    memcpy(&gamma, (const uint8_t *)inputs[1] + p->outputs * packed_row_bytes(p->width), sizeof(gamma));
    size_t workers = context->thread_pool->worker_count;
    if (!workers) return TRILLIM_ERROR;
    size_t per_worker = p->outputs / workers + (p->outputs % workers != 0);
    size_t columns = ((per_worker + 15) / 16) * 4;
    LinearTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, gamma, columns, scratch, (float *)scratch + p->tokens, context};
    TrillimStatus status = p->tokens == 1 ? row_sum(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, row_sum, &task);
    if (status != TRILLIM_OK) return status;
    if (p->width % 128 != 0) {
        size_t rows = p->tokens / 8 + (p->tokens % 8 != 0);
        size_t columns = p->outputs / 16 + (p->outputs % 16 != 0);
        size_t jobs = rows * columns;
        return jobs == 1 ? project_tail(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, jobs, project_tail, &task);
    }
    if (p->tokens == 1) {
        if (p->width <= 65536 / p->outputs) {
            task.decode_columns = p->outputs;
            return decode(&task, 0, 0);
        }
        return context->thread_pool->run(context->thread_pool->context,
            p->outputs / columns + (p->outputs % columns != 0), decode, &task);
    }
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t rows = p->tokens / 8 + (p->tokens % 8 != 0);
    size_t tasks = rows * tiles;
    return tasks == 1 ? prefill(&task, 0, 0)
        : context->thread_pool->run(context->thread_pool->context, tasks, prefill, &task);
}

TRILLIM_KERNEL const char *trillim_x86_quantized_linear_vnni_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_quantized_linear_vnni_parameters(void) {
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
        4, fields, 3, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_quantized_linear_vnni_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_AVX_VNNI};
    return &capabilities;
}
