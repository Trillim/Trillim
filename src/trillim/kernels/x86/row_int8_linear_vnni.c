#include "common.h"
#include <immintrin.h>

#define AVX2 __attribute__((target("avx2,avxvnni")))

typedef struct {
    size_t tokens, width, outputs;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    void *output;
    const int8_t *input, *weight;
    const float *scales;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype) || !p->width || p->width > INT32_MAX / (128 * 127) ||
        p->width > SIZE_MAX - 2 || p->outputs > SIZE_MAX / (p->width + 2) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) return TRILLIM_ERROR;
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * (p->width + 2);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

AVX2 static int32_t sum8(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 256 + (p->outputs % 256 != 0);
    size_t row = tile / tiles;
    size_t begin = (tile % tiles) * 256;
    size_t end = p->outputs - begin < 256 ? p->outputs : begin + 256;
    const int8_t *input = task->input + row * p->width;
    float multiplier = 1.0f / task->scales[row];
    for (size_t column = begin; column < end; column += 4) {
        __m256i sum0 = _mm256_setzero_si256(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        size_t count = end - column < 4 ? end - column : 4;
        const int8_t *w0 = task->weight + column * p->width;
        const int8_t *w1 = count > 1 ? w0 + p->width : w0;
        const int8_t *w2 = count > 2 ? w1 + p->width : w0;
        const int8_t *w3 = count > 3 ? w2 + p->width : w0;
        size_t k = 0;
        for (; p->width - k >= 32; k += 32) {
            if (p->width > 32768 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256i a = _mm256_loadu_si256((const __m256i *)(input + k));
            __m256i absolute = _mm256_abs_epi8(a);
            __m256i b0 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w0 + k)), a);
            __m256i b1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w1 + k)), a);
            __m256i b2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w2 + k)), a);
            __m256i b3 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w3 + k)), a);
            sum0 = _mm256_dpbusd_avx_epi32(sum0, absolute, b0);
            sum1 = _mm256_dpbusd_avx_epi32(sum1, absolute, b1);
            sum2 = _mm256_dpbusd_avx_epi32(sum2, absolute, b2);
            sum3 = _mm256_dpbusd_avx_epi32(sum3, absolute, b3);
        }
        int32_t sums[4] = {sum8(sum0), sum8(sum1), sum8(sum2), sum8(sum3)};
        for (size_t i = 0; i < count; ++i) {
            int32_t sum = sums[i];
            for (size_t j = k; j < p->width; ++j) sum += (int32_t)input[j] * task->weight[(column + i) * p->width + j];
            uint16_t bits;
            memcpy(&bits, (const uint8_t *)task->weight + p->outputs * p->width + (column + i) * 2, sizeof(bits));
            TrillimValue value = {.f = (float)sum * (trillim_bf16(bits) * multiplier)};
            size_t index = row * p->outputs + column + i;
            if (p->output_dtype == TRILLIM_FP32) {
                memcpy((uint8_t *)task->output + index * 4, &value.f, sizeof(value.f));
            } else {
                trillim_store(task->output, index, p->output_dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_row_int8_linear_vnni(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t tiles = p->outputs / 256 + (p->outputs % 256 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->tokens * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_row_int8_linear_vnni_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _row_int8_linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_row_int8_linear_vnni_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_row_int8_linear_vnni_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_AVX_VNNI};
    return &capabilities;
}
