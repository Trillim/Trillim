#include "common.h"
#include <immintrin.h>

#define SIMD __attribute__((target("avx2,fma")))

typedef struct {
    size_t channels, spatial;
    float eps;
} Parameters;

typedef struct {
    float *output;
    const float *input, *mean, *variance, *weight, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

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

SIMD static __m256 gelu8(__m256 x) {
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
    return _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)), value);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    for (size_t i = 1; i < 5; ++i) {
        inputs[i] = 0;
    }
    if (p->channels > SIZE_MAX / sizeof(float) ||
        (p->channels && p->spatial > SIZE_MAX / sizeof(float) / p->channels)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->channels * p->spatial * sizeof(float);
    for (size_t i = 1; i < 5; ++i) {
        inputs[i] = p->channels * sizeof(float);
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus normalize(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t blocks = p->spatial / 4096 + (p->spatial % 4096 != 0);
    size_t channel = tile / blocks;
    size_t first = tile % blocks * 4096;
    size_t end = p->spatial - first < 4096 ? p->spatial : first + 4096;
    const float *input = task->input + channel * p->spatial;
    float *output = task->output + channel * p->spatial;
    float scale = task->weight[channel] / sqrtf(task->variance[channel] + p->eps);
    float mean = task->mean[channel], bias = task->bias[channel];
    __m256 a = _mm256_set1_ps(scale), b = _mm256_set1_ps(bias), center = _mm256_set1_ps(mean);
    size_t i = first;
    for (; end - i >= 8; i += 8) {
        __m256 centered = _mm256_sub_ps(_mm256_loadu_ps(input + i), center);
        __m256 x = _mm256_fmadd_ps(centered, a, b);
        _mm256_storeu_ps(output + i, gelu8(x));
    }
    for (; i < end; ++i) {
        float x = (input[i] - mean) * scale + bias;
        output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_batchnorm_gelu(void *const *outputs, const void *const *inputs,
        void *scratch, const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], p, context};
    size_t blocks = p->spatial / 4096 + (p->spatial % 4096 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->channels * blocks, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_batchnorm_gelu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def pattern(x: Tensor, mean: Tensor, variance: Tensor, weight: Tensor, bias: Tensor, *, eps: float = 1e-5, rank: int = 3) -> tuple[Tensor, ...]:\n"
           "    return (Gelu(BatchNorm(x, running_mean=mean, running_var=variance, weight=weight, bias=bias, eps=eps, rank=rank)),)\n"
           "def bind(c):\n"
           "    x,o=c.inputs['x'],c.outputs[0]\n"
           "    if len(x.shape)<1 or c.attrs.get('rank',3)!=len(x.shape) or x.dtype!='fp32' or o.dtype!='fp32' or x.shape!=o.shape:\n"
           "        raise ValueError('batch normalization requires matching channel-first fp32 tensors')\n"
           "    if (x.strides or _contiguous_strides(x.shape))!=_contiguous_strides(x.shape) or (o.strides or _contiguous_strides(o.shape))!=_contiguous_strides(o.shape):\n"
           "        raise ValueError('batch normalization requires contiguous activations')\n"
           "    for name in ('mean','variance','weight','bias'):\n"
           "        v=c.inputs[name]\n"
           "        if v.dtype!='fp32' or v.shape!=x.shape[:1] or (v.strides or _contiguous_strides(v.shape))!=_contiguous_strides(v.shape):\n"
           "            raise ValueError('batch normalization requires contiguous fp32 channel vectors')\n"
           "    return dict(channels=x.shape[0],spatial=_product(x.shape[1:]),eps=c.attrs.get('eps',1e-5))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_batchnorm_gelu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"channels", "size_t", offsetof(Parameters, channels), NULL},
        {"spatial", "size_t", offsetof(Parameters, spatial), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
    };
    static const size_t inputs[] = {1,1,1,1,1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields,
        5, 1, requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_batchnorm_gelu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION,
        TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &caps;
}
