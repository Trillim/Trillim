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

TRILLIM_AVX2 static inline __m256 abs8(__m256 a) {
    return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

TRILLIM_AVX2 static inline __m256 neg8(__m256 a) {
    return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
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

TRILLIM_AVX2 static inline uint32_t all8(__m256i a) {
    return _mm256_testc_si256(a,_mm256_set1_epi32(-1)) ? UINT32_MAX : 0;
}

TRILLIM_AVX2 static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

TRILLIM_AVX2 static inline __m256 round_bf16_8(__m256 value) {
    __m256i bits = _mm256_castps_si256(value);
    __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
        _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
    return _mm256_castsi256_ps(_mm256_and_si256(_mm256_add_epi32(bits, bias), _mm256_set1_epi32(0xffff0000)));
}

typedef struct {
    size_t count;
    uint32_t dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->dtype != TRILLIM_BF16 && p->dtype != TRILLIM_FP32) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->dtype);
    outputs[0] = p->count * trillim_itemsize(TRILLIM_FP32);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

TRILLIM_AVX2 static float sigmoid(float value, uint32_t dtype) {
    float e = round_dtype(expf(-fabsf(value)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    return value >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
}

TRILLIM_AVX2 static __m256 exp_reduced8(__m256 x) {
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
    return exp_reduced8(x);
}

TRILLIM_AVX2 static __m256 sigmoid8(__m256 value, uint32_t dtype) {
    __m256 x = neg8(abs8(value));
    if (dtype != TRILLIM_FP32 && (_mm_getcsr() & 0xffc0) != 0x1f80) {
        float lanes[8];
        _mm256_storeu_ps(lanes, value);
        for (size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] = sigmoid(lanes[lane], dtype);
        }
        return _mm256_loadu_ps(lanes);
    }

    __m256 exponential = exp8(x);
    unsigned refine = 0;
    if (dtype == TRILLIM_BF16) {
        /* Resolve expf rounding near a BF16 midpoint. */
        __m256i bits = _mm256_castps_si256(exponential);
        __m256i low = _mm256_and_si256(bits, _mm256_set1_epi32(0xffff));
        __m256i distance = _mm256_abs_epi32(_mm256_sub_epi32(low, _mm256_set1_epi32(0x8000)));
        __m256 near = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(65), distance));
        near = _mm256_and_ps(near, _mm256_cmp_ps(x, _mm256_set1_ps(-87.0f), _CMP_GE_OQ));
        refine = (unsigned)_mm256_movemask_ps(_mm256_or_ps(near, _mm256_cmp_ps(value, value, _CMP_UNORD_Q)));
        exponential = round_bf16_8(exponential);
    }
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denominator = _mm256_add_ps(exponential, one);
    if (dtype == TRILLIM_BF16) {
        denominator = round_bf16_8(denominator);
    }
    __m256i positive = ge8(value, _mm256_setzero_ps());
    __m256 quotient = _mm256_div_ps(select8(positive, one, exponential), denominator);
    __m256 result = dtype == TRILLIM_BF16 ? select8(positive, quotient, round_bf16_8(quotient)) : quotient;
    if (refine) {
        float input[8], output[8];
        _mm256_storeu_ps(input, value);
        _mm256_storeu_ps(output, result);
        for (size_t lane = 0; lane < 8; ++lane) {
            if (refine & (1u << lane)) {
                output[lane] = sigmoid(input[lane], dtype);
            }
        }
        result = _mm256_loadu_ps(output);
    }
    return result;
}

TRILLIM_AVX2 static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256 input = load8(task->input, i, p->dtype);
        __m256 result = _mm256_mul_ps(input, sigmoid8(input, p->dtype));
        _mm256_storeu_ps((float *)task->output + i, result);
    }
    for (; i < end; ++i) {
        float value = trillim_load(task->input, i, p->dtype).f;
        ((float *)task->output)[i] = value * sigmoid(value, p->dtype);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_silu_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_x86_silu_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    if x.dtype not in {'bf16', 'fp32'} or output.dtype != 'fp32' or output.shape != x.shape:\n"
           "        raise ValueError('SiLU requires native floating tensors')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('SiLU requires contiguous input')\n"
           "    return {'count': _product(x.shape), 'dtype': _native_dtype(x.dtype)}\n"
           "\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Silu(x),)\n";
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_silu_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_silu_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
