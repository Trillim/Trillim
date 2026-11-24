#include "common.h"
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))
TRILLIM_AVX2 static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    if (dtype == TRILLIM_FP16) {
        __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        __m256i h = _mm256_cvtepu16_epi32(raw);
        __m256i fraction = _mm256_and_si256(h, _mm256_set1_epi32(1023));
        __m256i exponent = _mm256_and_si256(h, _mm256_set1_epi32(0x7c00));
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
        __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x7fff)), 13),
                                          _mm256_set1_epi32(112 << 23));
        __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
        normal = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                    _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
        __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000), _mm256_slli_epi32(fraction, 13));
        normal = _mm256_blendv_epi8(normal, special, _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00)));
        return _mm256_castsi256_ps(_mm256_or_si256(normal, sign));
    }
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

TRILLIM_AVX2 static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        float lanes[8];
        _mm256_storeu_ps(lanes, value);
        for (size_t lane = 0; lane < 8; ++lane) {
            TrillimValue item = {.f = lanes[lane]};
            trillim_store(data, index + lane, dtype, item);
        }
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

typedef struct { size_t count; } Parameters;
typedef struct { uint16_t *output; const uint16_t *a, *b; const Parameters *p; const TrillimKernelContext *context; } Task;
static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->count > SIZE_MAX / 2) return TRILLIM_ERROR;
    inputs[0] = inputs[1] = outputs[0] = p->count * 2;
    return TRILLIM_OK;
}
static bool can_alias(const void *opaque, size_t output, size_t input) { (void)opaque; return output == 0 && input < 2; }
TRILLIM_AVX2 static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t i = tile * 4096, end = t->p->count - i < 4096 ? t->p->count : i + 4096;
    for (; end - i >= 8; i += 8) {
        store8(t->output,i,TRILLIM_BF16,_mm256_add_ps(load8(t->a,i,TRILLIM_BF16),load8(t->b,i,TRILLIM_BF16)));
    }
    for (; i < end; ++i) t->output[i] = trillim_to_bf16(trillim_bf16(t->a[i]) + trillim_bf16(t->b[i]));
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_bf16_add(void *const *outputs, const void *const *inputs,
        void *scratch, const void *opaque, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = opaque;
    Task task = {outputs[0],inputs[0],inputs[1],p,context};
    size_t count = p->count / 4096 + (p->count % 4096 != 0);
    return count == 1 ? apply(&task,0,0) : context->thread_pool->run(context->thread_pool->context,count,apply,&task);
}
TRILLIM_KERNEL const char *trillim_x86_bf16_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_binary, _contiguous_strides\n"
           "def bind(c):\n"
           "    output = c.outputs[0]\n"
           "    stride = _contiguous_strides(output.shape)\n"
           "    if output.dtype != 'bf16' or (output.strides or stride) != stride:\n"
           "        raise ValueError('requires contiguous BF16 output')\n"
           "    return dict(count=_contiguous_binary(c)['count'])\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Add(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_bf16_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {2, 2}, outputs[] = {2};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_bf16_add_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
