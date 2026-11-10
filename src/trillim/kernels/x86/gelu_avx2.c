#include "common.h"
#include <math.h>
#include <immintrin.h>
#define SIMD __attribute__((target("avx2,fma")))
typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input;
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

SIMD static __m256 exp_negative(__m256 x) {
    __m256 original = x;
    x = _mm256_max_ps(_mm256_set1_ps(-104.0f), _mm256_min_ps(x, _mm256_setzero_ps()));
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(0.00021219444005469058f), r);
    // Range reduction bounds |r| by ln(2)/2 for this Taylor polynomial.
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 720.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 120.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 24.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 6.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(0.5f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_cvttps_epi32(n);
    __m256i low = _mm256_min_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(126)), _mm256_setzero_si256());
    __m256i high = _mm256_sub_epi32(exponent, low);
    __m256 main_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(high, _mm256_set1_epi32(127)), 23));
    __m256 tail_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(low, _mm256_set1_epi32(127)), 23));
    value = _mm256_mul_ps(_mm256_mul_ps(value, main_scale), tail_scale);
    return _mm256_blendv_ps(value, original, _mm256_cmp_ps(original, original, _CMP_UNORD_Q));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_FP32 && p->input_dtype != TRILLIM_BF16) ||
        (p->output_dtype != TRILLIM_FP32 && p->output_dtype != TRILLIM_BF16))
        return TRILLIM_ERROR;
    if (p->count > SIZE_MAX / 4) return TRILLIM_ERROR;
    inputs[0] = p->count * trillim_itemsize(p->input_dtype);
    outputs[0] = p->count * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
SIMD static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t begin = tile * 1024, end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256 x = load8(t->input, i, p->input_dtype);
        __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));
        __m256 z = _mm256_mul_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), x), _mm256_set1_ps(0.7071067811865476f));
        __m256 u =
            _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_fmadd_ps(z, _mm256_set1_ps(0.3275911f), _mm256_set1_ps(1.0f)));
        // Approximate erf with a polynomial in 1/(1 + 0.3275911*abs(x)).
        __m256 value = _mm256_set1_ps(1.061405429f);
        value = _mm256_fmadd_ps(value, u, _mm256_set1_ps(-1.453152027f));
        value = _mm256_fmadd_ps(value, u, _mm256_set1_ps(1.421413741f));
        value = _mm256_fmadd_ps(value, u, _mm256_set1_ps(-0.284496736f));
        value = _mm256_fmadd_ps(value, u, _mm256_set1_ps(0.254829592f));
        __m256 exponent = exp_negative(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(z, z)));
        value = _mm256_fnmadd_ps(_mm256_mul_ps(value, u), exponent, _mm256_set1_ps(1.0f));
        value = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_xor_ps(value, sign));
        store8(t->output, i, p->output_dtype, _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)), value));
    }
    for (; i < end; ++i) {

        float x = trillim_load(t->input, i, p->input_dtype).f;
        TrillimValue value = {.f = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f))};
        trillim_store(t->output, i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_gelu_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                   const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    return tiles == 1 ? apply(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}
TRILLIM_KERNEL const char *trillim_x86_gelu_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _gelu as bind\n"
           "def pattern(x: Tensor, *, approximate: str = 'none', dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Gelu(x, approximate=approximate) if dtype is None else Cast(Gelu(x, approximate=approximate), "
           "dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_gelu_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_gelu_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION,
                                             TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &caps;
}
