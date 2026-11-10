#include "common.h"
#include <immintrin.h>
#include <math.h>
#define SIMD __attribute__((target("avx2,fma")))
typedef struct {
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
    float eps;
} Parameters;
typedef struct {
    void *output;
    const void *input, *weight, *bias;
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
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    uint32_t types[] = {p->input_dtype, p->weight_dtype, p->bias_dtype, p->output_dtype};
    for (size_t i = 0; i < 4; ++i)
        if (types[i] != TRILLIM_BF16 && types[i] != TRILLIM_FP32) return TRILLIM_ERROR;
    if (!p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width || !isfinite(p->eps) || p->eps < 0)
        return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    inputs[2] = p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
SIMD static float sum8(__m256 v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}
SIMD static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    __m256 total = _mm256_setzero_ps();
    size_t i = 0, offset = row * p->width;
    for (; p->width - i >= 8; i += 8) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        total = _mm256_add_ps(total, load8(t->input, offset + i, p->input_dtype));
    }
    float mean = sum8(total);
    for (; i < p->width; ++i)
        mean += trillim_load(t->input, offset + i, p->input_dtype).f;
    mean /= (float)p->width;
    __m256 center = _mm256_set1_ps(mean);
    total = _mm256_setzero_ps();
    i = 0;
    for (; p->width - i >= 8; i += 8) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 v = _mm256_sub_ps(load8(t->input, offset + i, p->input_dtype), center);
        total = _mm256_fmadd_ps(v, v, total);
    }
    float variance = sum8(total);
    for (; i < p->width; ++i) {
        float v = trillim_load(t->input, offset + i, p->input_dtype).f - mean;
        variance += v * v;
    }
    float inv = 1.0f / sqrtf(variance / (float)p->width + p->eps);
    __m256 inverse = _mm256_set1_ps(inv);
    i = 0;
    for (; p->width - i >= 8; i += 8) {
        if ((i & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(load8(t->input, offset + i, p->input_dtype), center), inverse);
        v = _mm256_add_ps(_mm256_mul_ps(v, load8(t->weight, i, p->weight_dtype)), load8(t->bias, i, p->bias_dtype));
        store8(t->output, offset + i, p->output_dtype, v);
    }
    for (; i < p->width; ++i) {
        TrillimValue v = {.f = (trillim_load(t->input, offset + i, p->input_dtype).f - mean) * inv *
                                   trillim_load(t->weight, i, p->weight_dtype).f +
                               trillim_load(t->bias, i, p->bias_dtype).f};
        trillim_store(t->output, offset + i, p->output_dtype, v);
    }
    return TRILLIM_OK;
}
static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (!parameters) return false;
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_layernorm_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                        const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0)
                        : context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}
TRILLIM_KERNEL const char *trillim_x86_layernorm_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _layernorm as bind\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, axes: tuple[int, ...] = (-1,), eps: float = 1e-5, "
           "dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (LayerNorm(x, weight=weight, bias=bias, axes=axes, eps=eps) if dtype is None else "
           "Cast(LayerNorm(x, weight=weight, bias=bias, axes=axes, eps=eps), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_layernorm_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 3, 1, requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_layernorm_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &capabilities;
}
