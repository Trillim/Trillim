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

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

TRILLIM_AVX2 static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    __m256i bits = _mm256_cvtepu16_epi32(raw);
    if (dtype == TRILLIM_BF16) return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
    __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
    __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7c00));
    __m256 normal = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(magnitude, 13), _mm256_set1_epi32(0x38000000)));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude), _mm256_set1_ps(0x1p-24f));
    __m256 special = _mm256_castsi256_ps(_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x3ff)), 13), _mm256_set1_epi32(0x7f800000)));
    __m256 value = _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256())));
    value = _mm256_blendv_ps(value, special, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00))));
    return _mm256_xor_ps(value, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)), 16)));
}

TRILLIM_AVX2 static inline void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i absolute = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    __m256i nan = _mm256_cmpgt_epi32(absolute, _mm256_set1_epi32(0x7f800000));
    __m256i rounded;
    if (dtype == TRILLIM_BF16) {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)));
        rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    } else {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0xfff), _mm256_and_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(1)));
        rounded = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(absolute, bias), 13), _mm256_set1_epi32(112 << 10));
        rounded = _mm256_min_epi32(rounded, _mm256_set1_epi32(0x7c00));
        __m256 small = _mm256_mul_ps(_mm256_castsi256_ps(absolute), _mm256_set1_ps(0x1p24f));
        __m256i subnormal = _mm256_cvttps_epi32(_mm256_round_ps(small, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        rounded = _mm256_blendv_epi8(rounded, subnormal, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), absolute));
        __m256i payload = _mm256_or_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(0x200));
        payload = _mm256_and_si256(payload, _mm256_set1_epi32(0x3ff));
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(payload, _mm256_set1_epi32(0x7c00)), nan);
        rounded = _mm256_or_si256(rounded, _mm256_and_si256(upper, _mm256_set1_epi32(0x8000)));
    }
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index), _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

TRILLIM_AVX2 static __m256 round8(__m256 value, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return value;
    uint16_t bits[16];
    store8(bits, 0, dtype, value);
    return load8(bits, 0, dtype);
}

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const float *up;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

TRILLIM_AVX2 static float gelu_tanh(float x, uint32_t dtype) {
    float square = round_dtype(x * x, dtype);
    float cube = round_dtype(square * x, dtype);
    float correction = round_dtype(cube * round_dtype(0.044715f, dtype), dtype);
    float sum = round_dtype(x + correction, dtype);
    float argument = round_dtype(sum * round_dtype(0.7978845608028654f, dtype), dtype);
    float scaled = round_dtype(argument * 2.0f, dtype);
    float e = round_dtype(expf(-fabsf(scaled)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    float probability = scaled >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
    float cdf = probability * 2.0f - 1.0f;
    float half = round_dtype(x * 0.5f, dtype);
    return half * (cdf + 1.0f);
}

TRILLIM_AVX2 static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(TRILLIM_FP32) || !trillim_float(TRILLIM_FP32) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(TRILLIM_FP32);
    outputs[0] = inputs[1] = p->count * sizeof(float);
    return TRILLIM_OK;
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

TRILLIM_AVX2 static __m256 gelu_tanh8(__m256 x, uint32_t dtype) {
    __m256 square = round8(_mm256_mul_ps(x, x), dtype);
    __m256 cube = round8(_mm256_mul_ps(square, x), dtype);
    __m256 correction = round8(muln8(cube, round_dtype(0.044715f, dtype)), dtype);
    __m256 sum = round8(_mm256_add_ps(x, correction), dtype);
    __m256 argument = round8(muln8(sum, round_dtype(0.7978845608028654f, dtype)), dtype);
    __m256 scaled = round8(muln8(argument, 2.0f), dtype);
    __m256 e = round8(exp8(neg8(abs8(scaled))), dtype);
    __m256 denominator = round8(_mm256_add_ps(e, _mm256_set1_ps(1.0f)), dtype);
    __m256 positive = _mm256_div_ps(_mm256_set1_ps(1.0f), denominator);
    __m256 negative = round8(_mm256_div_ps(e, denominator), dtype);
    __m256 probability = select8(ge8(scaled, _mm256_set1_ps(0.0f)), positive, negative);
    __m256 cdf = _mm256_sub_ps(muln8(probability, 2.0f), _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(round8(muln8(x, 0.5f), dtype), _mm256_add_ps(cdf, _mm256_set1_ps(1.0f)));
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
        __m256 input = load8(task->input, i, TRILLIM_FP32);
        store8(task->output, i, TRILLIM_FP32, _mm256_mul_ps(gelu_tanh8(input, TRILLIM_FP32), _mm256_loadu_ps(task->up + i)));
    }
    for (; i < end; ++i) {
        float input = trillim_load(task->input, i, TRILLIM_FP32).f;
        TrillimValue value = {.f = gelu_tanh(input, TRILLIM_FP32) * task->up[i]};
        trillim_store(task->output, i, TRILLIM_FP32, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_gelu_tanh_mul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[1], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_x86_gelu_tanh_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    a, b, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    for value in (a, b, output):\n"
           "        strides = _contiguous_strides(value.shape)\n"
           "        if value.dtype != 'fp32' or value.shape != output.shape or (value.strides or strides) != strides:\n"
           "            raise ValueError('GELU gate requires matching contiguous FP32 tensors')\n"
           "    return {'count': _product(output.shape)}\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(a, Gelu(b, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_gelu_tanh_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_gelu_tanh_mul_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
