#include "common.h"
#include <immintrin.h>

#define SIMD __attribute__((target("avx2")))

typedef struct { size_t tokens, width, outputs; uint32_t output_dtype; } Parameters;
typedef struct {
    void *output;
    const int8_t *input, *weight;
    const float *scales;
    const Parameters *p;
    size_t tiles_per_task;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype) || !p->width || p->width % 32 || p->outputs % 32 ||
        p->width > INT32_MAX / (128 * 127) || p->width > SIZE_MAX - 2 ||
        p->outputs > SIZE_MAX / (p->width + 2) || p->tokens > SIZE_MAX / p->width ||
        p->tokens > SIZE_MAX / sizeof(float) || (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) return TRILLIM_ERROR;
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * (p->width + 2);
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

SIMD static int32_t sum8(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

SIMD static TrillimStatus project(void *opaque, size_t task_index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 32;
    size_t chunks = tiles / task->tiles_per_task + (tiles % task->tiles_per_task != 0);
    size_t row = task_index / chunks;
    size_t first = (task_index % chunks) * task->tiles_per_task;
    size_t last = tiles - first < task->tiles_per_task ? tiles : first + task->tiles_per_task;
    const int8_t *input = task->input + row * p->width;
    float multiplier = 1.0f / task->scales[row];
    for (size_t tile = first; tile < last; ++tile) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256i sums[32];
        for (size_t i = 0; i < 32; ++i) sums[i] = _mm256_setzero_si256();
        const int8_t *weight = task->weight + tile * 32 * p->width;
        for (size_t k = 0; k < p->width; k += 32) {
            if (p->width > 32768 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256i lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(input + k)));
            __m256i hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(input + k + 16)));
            const int8_t *block = weight + k * 32;
            for (size_t i = 0; i < 32; ++i) {
                __m256i wlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(block + i * 32)));
                __m256i whi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(block + i * 32 + 16)));
                sums[i] = _mm256_add_epi32(sums[i], _mm256_add_epi32(_mm256_madd_epi16(lo, wlo), _mm256_madd_epi16(hi, whi)));
            }
        }
        for (size_t i = 0; i < 32; ++i) {
            size_t column = tile * 32 + i;
            uint16_t bits;
            memcpy(&bits, (const uint8_t *)task->weight + p->outputs * p->width + column * 2, sizeof(bits));
            TrillimValue value = {.f = (float)sum8(sums[i]) * (trillim_bf16(bits) * multiplier)};
            size_t index = row * p->outputs + column;
            if (p->output_dtype == TRILLIM_FP32) memcpy((uint8_t *)task->output + index * 4, &value.f, sizeof(value.f));
            else trillim_store(task->output, index, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_row_int8_blocked_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) return TRILLIM_OK;
    size_t workers = context->thread_pool->worker_count;
    if (!workers) return TRILLIM_ERROR;
    size_t tiles = p->outputs / 32;
    size_t per_worker = tiles / workers + (tiles % workers != 0);
    size_t per_task = per_worker / 4 + (per_worker % 4 != 0);
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, per_task, context};
    size_t chunks = tiles / per_task + (tiles % per_task != 0);
    return context->thread_pool->run(context->thread_pool->context, p->tokens * chunks, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_row_int8_blocked_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _row_int8_blocked as bind\n"
           "packings = {'weight': 'row_int8_block32'}\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_row_int8_blocked_avx2_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_row_int8_blocked_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
