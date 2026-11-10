#include "common.h"
#include <immintrin.h>
#define SIMD __attribute__((target("avx2")))
typedef struct {
    size_t rows;
    size_t width;
    uint32_t input_dtype;
    uint32_t bias_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
SIMD static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

SIMD static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i rounded =
        _mm256_srli_epi32(_mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                                                  _mm256_and_si256(upper, _mm256_set1_epi32(1)))),
                          16);
    __m256i nan =
        _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
                     _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_FP32 && p->input_dtype != TRILLIM_BF16) ||
        (p->output_dtype != TRILLIM_FP32 && p->output_dtype != TRILLIM_BF16))
        return TRILLIM_ERROR;
    inputs[1] = 0;
    if ((p->bias_dtype != TRILLIM_FP32 && p->bias_dtype != TRILLIM_BF16) || !p->width || p->width > SIZE_MAX / 4 ||
        p->rows > SIZE_MAX / 4 / p->width)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
SIMD static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t begin = tile * p->width, i = 0;
    for (; p->width - i >= 8; i += 8) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        store8(t->output, begin + i, p->output_dtype,
               _mm256_add_ps(load8(t->input, begin + i, p->input_dtype), load8(t->bias, i, p->bias_dtype)));
    }
    for (; i < p->width; ++i) {
        TrillimValue value = {.f = trillim_load(t->input, begin + i, p->input_dtype).f +
                                   trillim_load(t->bias, i, p->bias_dtype).f};
        trillim_store(t->output, begin + i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_bias_add_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                       const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->rows;
    return tiles == 1 ? apply(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_x86_bias_add_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bias_add as bind\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Add(a, b) if dtype is None else Cast(Add(a, b), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_bias_add_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_bias_add_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
