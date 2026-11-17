#include "common.h"
#include <math.h>
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

TRILLIM_AVX2 static inline __m256 madd8(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(a, _mm256_mul_ps(b, c));
}

TRILLIM_AVX2 static inline __m256 msub8(__m256 a, __m256 b, __m256 c) {
    return _mm256_sub_ps(a, _mm256_mul_ps(b, c));
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

TRILLIM_AVX2 static inline __m256i eq8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_EQ_OQ));
}

TRILLIM_AVX2 static inline __m256 max8(__m256 a, __m256 b) {
    return select8(ge8(a,b),a,b);
}

TRILLIM_AVX2 static inline __m256 min8(__m256 a, __m256 b) {
    return select8(le8(a,b),a,b);
}

TRILLIM_AVX2 static inline float max_lane8(__m256 a) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_max_ps(s,_mm_movehl_ps(s,s));
    s = _mm_max_ss(s,_mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}

TRILLIM_AVX2 static inline float min_lane8(__m256 a) {
    return -max_lane8(neg8(a));
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

TRILLIM_AVX2 static __m256 load_strided8(const float *data, size_t offset, size_t step) {
    if (step == 0) return _mm256_set1_ps(data[offset]);
    if (step == 1) return _mm256_loadu_ps(data + offset);
    if (step <= INT32_MAX / 7) {
        __m256i offsets = _mm256_mullo_epi32(_mm256_setr_epi32(0,1,2,3,4,5,6,7), _mm256_set1_epi32((int)step));
        return _mm256_i32gather_ps(data + offset, offsets, 4);
    }
    float values[8];
    for (size_t lane=0; lane<8; ++lane) values[lane]=data[offset+lane*step];
    return _mm256_loadu_ps(values);
}

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
} Parameters;

typedef struct {
    float *output;
    const float *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rank && (!p->shape || !p->a_strides || !p->b_strides)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (!p->shape[axis]) {
            return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
        }
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    size_t a, b;
    if (count != p->count || count > SIZE_MAX / sizeof(float) ||
        !trillim_span(p->rank, p->shape, p->a_strides, &a) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b) ||
        a > SIZE_MAX / sizeof(float) || b > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a * sizeof(float);
    inputs[1] = b * sizeof(float);
    outputs[0] = count * sizeof(float);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static float activate(float gate, float up) {
    /* Compare/Select clamps send NaN gates to +7 and NaN up values to -7. */
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float activated = gate * probability;
    float shifted = up + 1.0f;
    return activated * shifted;
}

TRILLIM_AVX2 static inline __m256 exp_negative(__m256 x) {
    __m256 original = x;
    x = max8(_mm256_set1_ps(-104.0f), min8(x, _mm256_set1_ps(0)));
    __m256 n = _mm256_floor_ps(madd8(_mm256_set1_ps(0.5f), x, _mm256_set1_ps(1.4426950408889634f)));
    __m256 r = msub8(x, n, _mm256_set1_ps(0.693359375f));
    r = madd8(r, n, _mm256_set1_ps(0.00021219444005469058f));
    /* Range reduction bounds |r| by ln(2)/2 for the degree-seven polynomial. */
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = madd8(_mm256_set1_ps(1.0f / 720.0f), value, r);
    value = madd8(_mm256_set1_ps(1.0f / 120.0f), value, r);
    value = madd8(_mm256_set1_ps(1.0f / 24.0f), value, r);
    value = madd8(_mm256_set1_ps(1.0f / 6.0f), value, r);
    value = madd8(_mm256_set1_ps(0.5f), value, r);
    value = madd8(_mm256_set1_ps(1.0f), value, r);
    value = madd8(_mm256_set1_ps(1.0f), value, r);
    __m256i exponent = _mm256_cvttps_epi32(n);
    __m256i low = _mm256_min_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(126)), _mm256_set1_epi32(0));
    __m256i high = _mm256_sub_epi32(exponent, low);
    __m256 main_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(high, _mm256_set1_epi32(127)), 23));
    __m256 tail_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(low, _mm256_set1_epi32(127)), 23));
    value = _mm256_mul_ps(_mm256_mul_ps(value, main_scale), tail_scale);
    return select8(eq8(original, original), value, original);
}

TRILLIM_AVX2 static __m256 activate8(__m256 gate, __m256 up) {
    __m256 low = _mm256_set1_ps(-7.0f), high = _mm256_set1_ps(7.0f);
    gate = select8(le8(gate, high), gate, high);
    up = select8(ge8(up, low), up, low);
    up = select8(le8(up, high), up, high);
    __m256 scaled = muln8(gate, 1.702f);
    __m256 e = exp_negative(neg8(abs8(scaled)));
    __m256 denominator = _mm256_add_ps(e, _mm256_set1_ps(1.0f));
    __m256 probability = select8(ge8(scaled, _mm256_set1_ps(0)),
        _mm256_div_ps(_mm256_set1_ps(1.0f), denominator), _mm256_div_ps(e, denominator));
    __m256 activated = _mm256_mul_ps(gate, probability);
    __m256 shifted = _mm256_add_ps(up, _mm256_set1_ps(1.0f));
    __m256 result = _mm256_mul_ps(activated, shifted);
    if (min_lane8(scaled) < -80.0f) {
        /* libm preserves subnormal rounding and the -Inf * 0 invalid result. */
        float gates[8], ups[8], values[8], arguments[8];
        _mm256_storeu_ps(gates, gate);
        _mm256_storeu_ps(ups, up);
        _mm256_storeu_ps(values, result);
        _mm256_storeu_ps(arguments, scaled);
        for (size_t lane = 0; lane < 8; ++lane) {
            if (arguments[lane] < -80.0f) {
                values[lane] = activate(gates[lane], ups[lane]);
            }
        }
        result = _mm256_loadu_ps(values);
    }
    return result;
}

TRILLIM_AVX2 static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t index = tile * 2048;
    size_t end = p->count - index < 2048 ? p->count : index + 2048;
    while (index < end) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t a = trillim_offset(index, p->rank, p->shape, p->a_strides);
        size_t b = trillim_offset(index, p->rank, p->shape, p->b_strides);
        size_t a_step = p->rank ? p->a_strides[p->rank - 1] : 0;
        size_t b_step = p->rank ? p->b_strides[p->rank - 1] : 0;
        size_t width = p->rank ? p->shape[p->rank - 1] : 1;
        size_t remaining = width - index % width;
        size_t limit = end - index < remaining ? end : index + remaining;
        for (; limit - index >= 8; index += 8, a += a_step * 8, b += b_step * 8) {
            if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 gate, up;
            gate = load_strided8(task->a, a, a_step);
            up = load_strided8(task->b, b, b_step);
            _mm256_storeu_ps(task->output + index, activate8(gate, up));
        }
        for (; index < limit; ++index, a += a_step, b += b_step) {
            if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            task->output[index] = activate(task->a[a], task->b[b]);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (output != 0 || input >= 2) {
        return false;
    }
    if (!parameters) {
        return true;
    }
    const Parameters *p = parameters;
    return trillim_contiguous(p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_x86_clipped_swiglu_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->count) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->count / 2048 + (p->count % 2048 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_x86_clipped_swiglu_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary\n"
           "def bind(context):\n"
           "    a, b, output = context.inputs['a'], context.inputs['b'], context.outputs[0]\n"
           "    if any(value.dtype != 'fp32' for value in (a, b, output)):\n"
           "        raise ValueError('clipped SwiGLU requires FP32 tensors')\n"
           "    parameters = _binary(context)\n"
           "    del parameters['dtype']\n"
           "    return parameters\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    gate = Minimum(a, 7.0)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(b, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_clipped_swiglu_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_clipped_swiglu_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
