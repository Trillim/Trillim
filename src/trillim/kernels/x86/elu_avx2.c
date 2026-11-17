#include "common.h"
#include <math.h>
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

TRILLIM_AVX2 static inline __m256 madd8(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(a, _mm256_mul_ps(b, c));
}

TRILLIM_AVX2 static inline __m256 maddn8(__m256 a, __m256 b, float c) {
    return _mm256_add_ps(a, _mm256_mul_ps(b, _mm256_set1_ps(c)));
}

TRILLIM_AVX2 static inline __m256 muln8(__m256 a, float b) {
    return _mm256_mul_ps(a, _mm256_set1_ps(b));
}

TRILLIM_AVX2 static inline __m256 select8(__m256i m, __m256 a, __m256 b) {
    return _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m));
}

TRILLIM_AVX2 static inline __m256i ge8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_GE_OQ));
}

TRILLIM_AVX2 static inline __m256i le8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_LE_OQ));
}

TRILLIM_AVX2 static inline __m256 min8(__m256 a, __m256 b) {
    return select8(le8(a,b),a,b);
}

TRILLIM_AVX2 static inline uint32_t all8(__m256i a) {
    return _mm256_testc_si256(a,_mm256_set1_epi32(-1)) ? UINT32_MAX : 0;
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    float alpha;
} Parameters;

typedef struct {
    float *output;
    const float *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t count = 1, span;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] && count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    if (count != p->count || !isfinite(p->alpha) ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count ? span * sizeof(float) : 0;
    outputs[0] = count * sizeof(float);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static float activate(float value, float alpha) {
    return value >= 0.0f ? value : (expf(value) - 1.0f) * alpha;
}

TRILLIM_AVX2 static __m256 exp8(__m256 x) {
    if (all8(_mm256_and_si256(ge8(x, _mm256_set1_ps(-87.0f)),
                              le8(x, _mm256_set1_ps(0.0f)))) != UINT32_MAX) {
        float lanes[8];
        _mm256_storeu_ps(lanes, x);
        for (size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] = expf(lanes[lane]);
        }
        return _mm256_loadu_ps(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    __m256i power = _mm256_cvtps_epi32(muln8(x, 1.4426950408889634f));
    __m256 integer = _mm256_cvtepi32_ps(power);
    __m256 r = maddn8(x, integer, -0.693145751953125f);
    r = maddn8(r, integer, -1.428606765330187e-6f);
    __m256 p = _mm256_set1_ps(1.0f / 5040.0f);
    p = madd8(_mm256_set1_ps(1.0f / 720.0f), p, r);
    p = madd8(_mm256_set1_ps(1.0f / 120.0f), p, r);
    p = madd8(_mm256_set1_ps(1.0f / 24.0f), p, r);
    p = madd8(_mm256_set1_ps(1.0f / 6.0f), p, r);
    p = madd8(_mm256_set1_ps(0.5f), p, r);
    p = madd8(_mm256_set1_ps(1.0f), p, r);
    p = madd8(_mm256_set1_ps(1.0f), p, r);
    __m256i exponent = _mm256_slli_epi32(_mm256_add_epi32(power, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exponent));
}

TRILLIM_AVX2 static __m256 activate8(__m256 x, float alpha) {
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 negative = muln8(_mm256_sub_ps(exp8(min8(x, zero)), _mm256_set1_ps(1.0f)), alpha);
    __m256 result = select8(ge8(x, zero), x, negative);
    return _mm256_blendv_ps(x, result, _mm256_cmp_ps(x, x, _CMP_ORD_Q));
}

TRILLIM_AVX2 static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t first = tile * 1024, end = p->count - first < 1024 ? p->count : first + 1024;
    size_t i = first;
    bool contiguous = trillim_contiguous(p->rank, p->shape, p->strides);
    for (; end - i >= 8; i += 8) {
        __m256 x;
        if (contiguous) {
            x = _mm256_loadu_ps(task->input + i);
        } else {
            float values[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                values[lane] = task->input[trillim_offset(i + lane, p->rank, p->shape, p->strides)];
            }
            x = _mm256_loadu_ps(values);
        }
        _mm256_storeu_ps(task->output + i, activate8(x, p->alpha));
    }
    for (; i < end; ++i) {
        size_t source = contiguous ? i : trillim_offset(i, p->rank, p->shape, p->strides);
        task->output[i] = activate(task->input[source], p->alpha);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_elu_avx2(void *const *outputs, const void *const *inputs,
                                void *scratch, const void *parameters,
                                const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    size_t count = p->count / 1024 + (p->count % 1024 != 0);
    return count == 1 ? apply(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, count, apply, &task);
}

TRILLIM_KERNEL const char *trillim_x86_elu_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "import math\n"
           "\n"
           "def pattern(x: Tensor, *, alpha: float=1.0) -> tuple[Tensor, ...]:\n"
           "    return (Elu(x,alpha=alpha),)\n"
           "\n"
           "def bind(c):\n"
           "    x,o=c.inputs['x'],c.outputs[0]\n"
           "    if x.dtype!='fp32' or o.dtype!='fp32' or x.shape!=o.shape or (o.strides or _contiguous_strides(o.shape))!=_contiguous_strides(o.shape):\n"
           "        raise ValueError('ELU requires FP32 input and contiguous output')\n"
           "    alpha=c.attrs.get('alpha',1.0)\n"
           "    if not isinstance(alpha,(int,float)) or not math.isfinite(alpha):\n"
           "        raise ValueError('ELU requires a finite alpha')\n"
           "    if alpha!=1.0:\n"
           "        raise ValueError('vector ELU requires alpha one')\n"
           "    return dict(count=_product(x.shape),rank=len(x.shape),shape=x.shape,strides=x.strides or _contiguous_strides(x.shape),alpha=float(alpha))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_elu_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"alpha", "float", offsetof(Parameters, alpha), NULL},
    };
    static const size_t alignments[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, alignments, alignments, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_elu_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
