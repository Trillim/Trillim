#include "common.h"
#include <immintrin.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

typedef struct { size_t count; uint32_t dtype; } Parameters;
typedef struct {
    void *output;
    const void *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

TRILLIM_AVX2 static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits,
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)))), 16);
    __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
        _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->dtype);
    if ((p->dtype != TRILLIM_BF16 && p->dtype != TRILLIM_FP32) || p->count > SIZE_MAX / bytes) return TRILLIM_ERROR;
    inputs[0] = inputs[1] = outputs[0] = p->count * bytes;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus add(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = task->p;
    size_t begin = tile * 4096;
    size_t end = p->count - begin < 4096 ? p->count : begin + 4096;
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        store8(task->output, i, p->dtype, _mm256_add_ps(load8(task->a, i, p->dtype), load8(task->b, i, p->dtype)));
    }
    for (; i < end; ++i) {
        TrillimValue value = {.f = trillim_load(task->a, i, p->dtype).f + trillim_load(task->b, i, p->dtype).f};
        trillim_store(task->output, i, p->dtype, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_add_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    return p->count <= 4096 ? add(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->count / 4096 + (p->count % 4096 != 0), add, &task);
}

TRILLIM_KERNEL const char *trillim_x86_add_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_binary as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Add(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_add_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_add_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
