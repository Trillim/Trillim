#include "common.h"
#include <immintrin.h>
#include <math.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

typedef struct {
    size_t count;
    uint32_t input_dtype, output_dtype;
} TrillimActivationParameters;

typedef struct {
    void *output;
    const void *gate;
    const void *up;
    const TrillimActivationParameters *parameters;
    const TrillimKernelContext *context;
} TrillimActivationTask;

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

static inline TrillimStatus trillim_activation_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                            size_t *scratch) {
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->input_dtype), output_bytes = trillim_itemsize(p->output_dtype);
    if ((p->input_dtype != TRILLIM_FP32 && p->input_dtype != TRILLIM_BF16) ||
        (p->output_dtype != TRILLIM_FP32 && p->output_dtype != TRILLIM_BF16) || p->count > SIZE_MAX / bytes ||
        p->count > SIZE_MAX / output_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * bytes;
    inputs[1] = inputs[0];
    outputs[0] = p->count * output_bytes;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_activation_task_avx2(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimActivationTask *task = (const TrillimActivationTask *)opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = task->parameters->count - begin < 1024 ? task->parameters->count : begin + 1024;
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256 gate = load8(task->gate, i, task->parameters->input_dtype);
        float probabilities[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            float scalar = trillim_load(task->gate, i + lane, task->parameters->input_dtype).f;
            float e = expf(-fabsf(scalar));
            probabilities[lane] = scalar >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
        }
        __m256 value = _mm256_mul_ps(gate, _mm256_loadu_ps(probabilities));
        store8(task->output, i, task->parameters->output_dtype,
               _mm256_mul_ps(value, load8(task->up, i, task->parameters->input_dtype)));
    }
    for (; i < end; ++i) {
        float gate = trillim_load(task->gate, i, task->parameters->input_dtype).f;
        float e = expf(-fabsf(gate));
        float value = gate * (gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
        TrillimValue output = {.f = value * trillim_load(task->up, i, task->parameters->input_dtype).f};
        trillim_store(task->output, i, task->parameters->output_dtype, output);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const TrillimActivationParameters *p = parameters;
    return output == 0 && input < 2 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_silu_mul_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                       const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    TrillimActivationTask task = {outputs[0], inputs[0], inputs[1], p, context};
    if (p->count <= 1024) return trillim_activation_task_avx2(&task, 0, 0);
    return context->thread_pool->run(context->thread_pool->context, p->count / 1024 + (p->count % 1024 != 0),
                                     trillim_activation_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_silu_mul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _silu_mul as bind\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (SiluMul(a, b, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_silu_mul_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimActivationParameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimActivationParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimActivationParameters, output_dtype), NULL}};
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(TrillimActivationParameters),
                                                  TRILLIM_ALIGNOF(TrillimActivationParameters),
                                                  3,
                                                  fields,
                                                  2,
                                                  1,
                                                  trillim_activation_requirements,
                                                  can_alias,
                                                  inputs,
                                                  outputs,
                                                  1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_silu_mul_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
