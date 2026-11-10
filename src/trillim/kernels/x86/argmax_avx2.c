#include "common.h"
#include <immintrin.h>

#define SIMD __attribute__((target("avx2")))

typedef struct { size_t count; uint32_t output_dtype; } Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->count > INT32_MAX || p->count > SIZE_MAX / sizeof(float) ||
        (p->output_dtype != TRILLIM_INT32 && p->output_dtype != TRILLIM_INT64)) return TRILLIM_ERROR;
    inputs[0] = p->count * sizeof(float);
    outputs[0] = trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

SIMD TRILLIM_KERNEL TrillimStatus trillim_x86_argmax_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    const float *input = inputs[0];
    int64_t selected = INT64_MAX;
    float best = 0.0f;
    size_t index = 0;
    bool nan = false;
    if (p->count >= 8) {
        __m256 maximum = _mm256_loadu_ps(input);
        __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        nan = _mm256_movemask_ps(_mm256_cmp_ps(maximum, maximum, _CMP_UNORD_Q)) != 0;
        for (index = 8; p->count - index >= 8; index += 8) {
            if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256 value = _mm256_loadu_ps(input + index);
            nan |= _mm256_movemask_ps(_mm256_cmp_ps(value, value, _CMP_UNORD_Q)) != 0;
            __m256 choose = _mm256_cmp_ps(value, maximum, _CMP_GT_OQ);
            maximum = _mm256_blendv_ps(maximum, value, choose);
            __m256i positions = _mm256_add_epi32(_mm256_set1_epi32((int32_t)index), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            indices = _mm256_blendv_epi8(indices, positions, _mm256_castps_si256(choose));
        }
        float values[8];
        int32_t positions[8];
        _mm256_storeu_ps(values, maximum);
        _mm256_storeu_si256((__m256i *)positions, indices);
        best = values[0];
        selected = positions[0];
        for (size_t lane = 1; lane < 8; ++lane) {
            if (values[lane] > best || (values[lane] == best && positions[lane] < selected)) {
                best = values[lane];
                selected = positions[lane];
            }
        }
    }
    /* The scalar comparison also defines the ordering when a NaN occurs. */
    if (nan) { index = 0; selected = INT64_MAX; }
    for (; index < p->count; ++index) {
        if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
        if (selected == INT64_MAX || !(best >= input[index])) {
            best = input[index];
            selected = (int64_t)index;
        }
    }
    TrillimValue result = {.i = selected};
    trillim_store(outputs[0], 0, p->output_dtype, result);
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_argmax_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _vector_argmax as bind\n"
           "def pattern(x: Tensor, *, axis: int = -1, rank: int) -> tuple[Tensor, ...]:\n"
           "    return (Argmax(x, axis=axis, rank=rank),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_argmax_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_argmax_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
