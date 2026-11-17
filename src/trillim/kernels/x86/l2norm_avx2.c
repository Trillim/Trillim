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

TRILLIM_AVX2 static inline float sum8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}

TRILLIM_AVX2 static inline float lane8(__m256 a, size_t lane) {
    float values[8];
    _mm256_storeu_ps(values,a);
    return values[lane];
}

TRILLIM_AVX2 static __m256 round8(__m256 value, uint32_t dtype) { uint16_t bits[8]; if (dtype == TRILLIM_FP32) return value; store8(bits,0,dtype,value); return load8(bits,0,dtype); }

typedef struct {
    size_t rows, width;
    uint32_t dtype;
    float eps;
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
    if (!trillim_float(p->dtype) || !p->width ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        p->rows > SIZE_MAX / trillim_itemsize(p->dtype) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    return TRILLIM_OK;
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

TRILLIM_AVX2 static TrillimStatus normalize_precise(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width, column = 0;
    float sum = 0.0f;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 x = load8(task->input, start + column, p->dtype);
        __m256 squared = round8(_mm256_mul_ps(x, x), p->dtype);
        sum += lane8(squared, 0);
        sum += lane8(squared, 1);
        sum += lane8(squared, 2);
        sum += lane8(squared, 3);
        sum += lane8(squared, 4);
        sum += lane8(squared, 5);
        sum += lane8(squared, 6);
        sum += lane8(squared, 7);
    }
    for (; column < p->width; ++column) {
        float x = trillim_load(task->input, start + column, p->dtype).f;
        sum += round_dtype(x * x, p->dtype);
    }
    sum = round_dtype(sum, p->dtype);
    float added = round_dtype(sum + round_dtype(p->eps, p->dtype), p->dtype);
    float denominator = round_dtype(sqrtf(added), p->dtype);
    __m256 denominator8 = _mm256_set1_ps(denominator);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, start + column, p->dtype,
               _mm256_div_ps(load8(task->input, start + column, p->dtype), denominator8));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {
            .f = trillim_load(task->input, start + column, p->dtype).f / denominator
        };
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve each primitive dtype boundary and its reduction order. */
        return normalize_precise(opaque, row, worker);
    }
    size_t start = row * p->width, column = 0;
    __m256 sums0 = _mm256_set1_ps(0.0f), sums1 = sums0;
    for (; p->width - column >= 16; column += 16) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 a = load8(task->input, start + column, p->dtype);
        __m256 b = load8(task->input, start + column + 8, p->dtype);
        sums0 = madd8(sums0, a, a);
        sums1 = madd8(sums1, b, b);
    }
    float sum = sum8(_mm256_add_ps(sums0, sums1));
    for (; column < p->width; ++column) {
        float x = trillim_load(task->input, start + column, p->dtype).f;
        sum += x * x;
    }
    float denominator = sqrtf(sum + p->eps);
    __m256 denominator8 = _mm256_set1_ps(denominator);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, start + column, p->dtype,
               _mm256_div_ps(load8(task->input, start + column, p->dtype), denominator8));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {
            .f = trillim_load(task->input, start + column, p->dtype).f / denominator
        };
        trillim_store(task->output, start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_l2norm_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows == 1) {
        return normalize(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, p->rows,
                                     normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_l2norm_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('L2Norm requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('L2Norm requires matching floating input and output')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('L2Norm requires contiguous input')\n"
           "    eps = context.attrs.get('eps', 1e-6)\n"
           "    if not isinstance(eps, (int, float)):\n"
           "        raise ValueError('invalid L2Norm epsilon')\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'dtype': _native_dtype(x.dtype), 'eps': float(eps)}\n"
           "def pattern(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (L2Norm(x, axis=axis, eps=eps),)\n";
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_l2norm_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_l2norm_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
