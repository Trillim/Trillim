#include "common.h"
#include <math.h>
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

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

TRILLIM_AVX2 static inline __m256i eq8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_EQ_OQ));
}

TRILLIM_AVX2 static inline __m256 maxnum8(__m256 a, __m256 b) {
    return select8(eq8(b,b),_mm256_max_ps(a,b),a);
}

TRILLIM_AVX2 static inline float sum8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

TRILLIM_AVX2 static inline float max_lane8(__m256 a) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_max_ps(s,_mm_movehl_ps(s,s));
    s = _mm_max_ss(s,_mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}

TRILLIM_AVX2 static inline uint32_t all8(__m256i a) {
    return _mm256_testc_si256(a,_mm256_set1_epi32(-1)) ? UINT32_MAX : 0;
}

TRILLIM_AVX2 static __m256 round8(__m256 value, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return value;
    uint16_t bits[16];
    store8(bits, 0, dtype, value);
    return load8(bits, 0, dtype);
}

typedef struct {
    size_t rows, width, rank;
    const size_t *shape, *sink_strides;
    uint32_t dtype, sink_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input, *sinks;
    float *scratch;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    size_t span;
    if (!trillim_float(p->dtype) || !trillim_float(p->sink_dtype) ||
        !trillim_float(p->output_dtype) || !p->width ||
        p->rows > SIZE_MAX / sizeof(float) / p->width ||
        !trillim_span(p->rank, p->shape, p->sink_strides, &span) ||
        span > SIZE_MAX / trillim_itemsize(p->sink_dtype)) {
        return TRILLIM_ERROR;
    }
    size_t rows = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] && rows > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        rows *= p->shape[axis];
    }
    if (rows != p->rows) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    inputs[1] = span * trillim_itemsize(p->sink_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    *scratch = p->rows * p->width * sizeof(float);
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

TRILLIM_AVX2 static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

TRILLIM_AVX2 static TrillimStatus softmax_precise(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float maximum = -INFINITY;
    size_t column = 0;
    __m256 maxima = _mm256_set1_ps(-INFINITY);
    bool ordered = true;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 value = load8(task->input, start + column, p->dtype);
        ordered = ordered && all8(_mm256_castps_si256(_mm256_cmp_ps(value, value, _CMP_EQ_OQ))) == UINT32_MAX;
        maxima = _mm256_max_ps(maxima, value);
    }
    if (ordered) maximum = max_lane8(maxima);
    else column = 0;
    for (; column < p->width; ++column) {

        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->dtype).f;
        maximum = maximum >= value ? maximum : value;
    }
    float sink = trillim_load(task->sinks,
                              trillim_offset(row, p->rank, p->shape, p->sink_strides),
                              p->sink_dtype).f;
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 difference = round8(_mm256_sub_ps(load8(task->input, start + column, p->dtype), _mm256_set1_ps(maximum)), p->output_dtype);
        __m256 numerator = round8(exp8(difference), p->output_dtype);
        _mm256_storeu_ps(task->scratch + start + column, numerator);
        for (size_t lane = 0; lane < 8; ++lane) sum += task->scratch[start + column + lane];
    }
    for (; column < p->width; ++column) {

        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float difference = round_dtype(trillim_load(task->input, start + column, p->dtype).f - maximum, p->output_dtype);
        float numerator = round_dtype(expf(difference), p->output_dtype);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    sum = round_dtype(sum, p->output_dtype);
    float sink_difference = round_dtype(sink - maximum, p->output_dtype);
    float sink_numerator = round_dtype(expf(sink_difference), p->output_dtype);
    sum = round_dtype(sum + sink_numerator, p->output_dtype);
    float denominator = sum > 0.0f ? sum : 1.0f;
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 value = _mm256_div_ps(_mm256_loadu_ps(task->scratch + start + column), _mm256_set1_ps(denominator));
        store8(task->output, start + column, p->output_dtype, value);
    }
    for (; column < p->width; ++column) {

        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus softmax(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve low-precision rounding at scalar-reference boundaries. */
        return softmax_precise(opaque, row, worker);
    }
    size_t start = row * p->width, column = 0;
    __m256 maximum8 = _mm256_set1_ps(-INFINITY);
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 value = load8(task->input, start + column, p->dtype);
        if (all8(eq8(value, value)) != UINT32_MAX) {
            return softmax_precise(opaque, row, worker);
        }
        maximum8 = maxnum8(maximum8, value);
    }
    float maximum = max_lane8(maximum8);
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->dtype).f;
        if (isnan(value)) {
            return softmax_precise(opaque, row, worker);
        }
        maximum = maximum >= value ? maximum : value;
    }
    float sink = trillim_load(task->sinks,
                              trillim_offset(row, p->rank, p->shape, p->sink_strides),
                              p->sink_dtype).f;
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    __m256 sum_vector = _mm256_set1_ps(0.0f), maximum_vector = _mm256_set1_ps(maximum);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 numerator = exp8(_mm256_sub_ps(load8(task->input, start + column, p->dtype),
                                              maximum_vector));
        _mm256_storeu_ps(task->scratch + start + column, numerator);
        sum_vector = _mm256_add_ps(sum_vector, numerator);
    }
    float sum = sum8(sum_vector) + expf(sink - maximum);
    for (; column < p->width; ++column) {
        float numerator = expf(trillim_load(task->input, start + column, p->dtype).f - maximum);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    float denominator = sum > 0.0f ? sum : 1.0f;
    __m256 denominator_vector = _mm256_set1_ps(denominator);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, start + column, p->output_dtype,
               _mm256_div_ps(_mm256_loadu_ps(task->scratch + start + column), denominator_vector));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_normalize_softmax_sinks_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], inputs[1], scratch, p, context};
    if (p->rows == 1) {
        return softmax(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     softmax, &task);
}

TRILLIM_KERNEL const char *trillim_x86_normalize_softmax_sinks_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _native_dtype, _product\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "\n"
           "def bind(context):\n"
           "    x, sinks, output = context.inputs['x'], context.inputs['sinks'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('Softmax requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or sinks.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'} or output.shape != x.shape:\n"
           "        raise ValueError('Softmax requires floating tensors')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Softmax requires contiguous input')\n"
           "    shape = x.shape[:-1] + (_constant_expression(1),)\n"
           "    strides = _broadcast_strides(sinks.shape, shape, sinks.strides)\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'rank': len(shape) - 1, 'shape': shape[:-1], 'sink_strides': strides[:-1], 'dtype': _native_dtype(x.dtype), 'sink_dtype': _native_dtype(sinks.dtype), 'output_dtype': _native_dtype(output.dtype)}\n"
           "\n"
           "def pattern(x: Tensor, sinks: Tensor, *, axis: int = -1) -> tuple[Tensor, ...]:\n"
           "    return (Softmax(x, axis=axis, sinks=sinks),)\n";
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_normalize_softmax_sinks_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"sink_strides", "size_t", offsetof(Parameters, sink_strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"sink_dtype", "uint32_t", offsetof(Parameters, sink_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 8, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_normalize_softmax_sinks_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
