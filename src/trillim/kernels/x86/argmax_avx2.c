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
        __m256 maxima[4];
        __m256i positions[4];
        const __m256i lanes = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
        for (size_t v = 0; v < 4; ++v) {
            maxima[v] = _mm256_loadu_ps(input);
            positions[v] = lanes;
        }
        for (; p->count - index >= 32; index += 32) {
            if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            for (size_t v = 0; v < 4; ++v) {
                __m256 x = _mm256_loadu_ps(input + index + v * 8);
                nan |= _mm256_movemask_ps(_mm256_cmp_ps(x,x,_CMP_UNORD_Q)) != 0;
                __m256 choose = _mm256_cmp_ps(x,maxima[v],_CMP_GT_OQ);
                maxima[v] = _mm256_blendv_ps(maxima[v],x,choose);
                __m256i ids = _mm256_add_epi32(lanes,_mm256_set1_epi32((int32_t)(index + v * 8)));
                positions[v] = _mm256_blendv_epi8(positions[v],ids,_mm256_castps_si256(choose));
            }
        }
        for (; p->count - index >= 8; index += 8) {
            if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256 x = _mm256_loadu_ps(input + index);
            nan |= _mm256_movemask_ps(_mm256_cmp_ps(x,x,_CMP_UNORD_Q)) != 0;
            __m256 choose = _mm256_cmp_ps(x,maxima[0],_CMP_GT_OQ);
            maxima[0] = _mm256_blendv_ps(maxima[0],x,choose);
            positions[0] = _mm256_blendv_epi8(positions[0],_mm256_add_epi32(lanes,_mm256_set1_epi32((int32_t)index)),_mm256_castps_si256(choose));
        }
        for (size_t v = 0; v < 4; ++v) {
            float values[8]; int32_t ids[8];
            _mm256_storeu_ps(values,maxima[v]);
            _mm256_storeu_si256((__m256i *)ids,positions[v]);
            for (size_t lane = 0; lane < 8; ++lane) {
                if (selected == INT64_MAX || values[lane] > best || (values[lane] == best && ids[lane] < selected)) {
                    best = values[lane]; selected = ids[lane];
                }
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
