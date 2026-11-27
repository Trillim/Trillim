#include "common.h"
#include <math.h>
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#define ROWS_PER_TASK 8

typedef struct {
    size_t rank, rows, width;
    const size_t *shape, *strides;
    uint32_t dtype;
    float eps;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t span;
    if (!trillim_float(p->dtype) || !p->rank || !p->width ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        p->width != p->shape[p->rank - 1]) {
        return TRILLIM_ERROR;
    }
    size_t itemsize = trillim_itemsize(p->dtype);
    if (span > SIZE_MAX / itemsize || p->rows > SIZE_MAX / itemsize / p->width) {
        return TRILLIM_ERROR;
    }
    size_t rows = 1;
    for (size_t axis = 0; axis < p->rank - 1; ++axis) {
        size_t dim = p->shape[axis];
        if (dim && rows > SIZE_MAX / dim) {
            return TRILLIM_ERROR;
        }
        rows *= dim;
    }
    if (p->rows != rows) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * itemsize;
    outputs[0] = p->rows * p->width * itemsize;
    return TRILLIM_OK;
}

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

TRILLIM_AVX2 static __m256 round8(__m256 value, uint32_t dtype) { uint16_t bits[8]; if (dtype == TRILLIM_FP32) return value; store8(bits,0,dtype,value); return load8(bits,0,dtype); }

TRILLIM_AVX2 static __m256 load_row8(const void *data, size_t index, size_t stride, uint32_t dtype) {
    if (stride == 1) return load8(data,index,dtype);
    if (stride == 0) return _mm256_set1_ps(trillim_load(data,index,dtype).f);
    float lanes[8];
    for (size_t i=0;i<8;++i) lanes[i]=trillim_load(data,index+i*stride,dtype).f;
    return _mm256_loadu_ps(lanes);
}

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
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

TRILLIM_AVX2 static TrillimStatus normalize_fast(const Task *task, size_t row) {
    const Parameters *p = task->p;
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

TRILLIM_AVX2 static TrillimStatus normalize_row(const Task *task, size_t row) {
    const Parameters *p = task->p;
    if (p->dtype == TRILLIM_FP32 && trillim_contiguous(p->rank,p->shape,p->strides)) return normalize_fast(task,row);
    const TrillimKernelContext *context = task->context;
    size_t start = trillim_offset(row, p->rank - 1, p->shape, p->strides);
    size_t output_start = row * p->width;
    size_t stride = p->strides[p->rank - 1];
    size_t column = 0;
    float sum = 0.0f;
    /* Match the primitive Mul and the left-to-right Sum. */
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 x = load_row8(task->input, start + column * stride, stride, p->dtype);
        __m256 squared = round8(_mm256_mul_ps(x, x), p->dtype);
        float lanes[8]; _mm256_storeu_ps(lanes,squared);
        for (size_t lane=0;lane<8;++lane) sum+=lanes[lane];
    }
    for (; column < p->width; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(task->input, start + column * stride, p->dtype).f;
        sum += round_dtype(x * x, p->dtype);
    }
    sum = round_dtype(sum, p->dtype);
    float added = round_dtype(sum + round_dtype(p->eps, p->dtype), p->dtype);
    float denominator = round_dtype(sqrtf(added), p->dtype);
    __m256 denominator4 = _mm256_set1_ps(denominator);
    column = 0;
    for (; stride == 1 && p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task->output, output_start + column, p->dtype,
               _mm256_div_ps(load8(task->input, start + column, p->dtype), denominator4));
    }
    for (; column < p->width; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {
            .f = trillim_load(task->input, start + column * stride, p->dtype).f / denominator
        };
        trillim_store(task->output, output_start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_rows(const Task *task, size_t start, size_t count) {
    for (size_t offset = 0; offset < count; ++offset) {
        TrillimStatus status = normalize_row(task, start + offset);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t start = index * ROWS_PER_TASK;
    size_t count = task->p->rows - start;
    return normalize_rows(task, start, count < ROWS_PER_TASK ? count : ROWS_PER_TASK);
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
    if (p->rows <= 1 || p->rows <= 4096 / p->width) {
        return normalize_rows(&task, 0, p->rows);
    }
    size_t tasks = (p->rows - 1) / ROWS_PER_TASK + 1;
    return context->thread_pool->run(context->thread_pool->context, tasks,
                                     normalize_task, &task);
}

TRILLIM_KERNEL const char *trillim_x86_l2norm_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from struct import pack, unpack\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('L2Norm requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('L2Norm requires matching floating input and output')\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    eps = context.attrs.get('eps', 1e-6)\n"
           "    if not isinstance(eps, (int, float)):\n"
           "        raise ValueError('invalid L2Norm epsilon')\n"
           "    # Round the literal directly to FP16 before packing the float field.\n"
           "    if x.dtype == 'fp16':\n"
           "        try:\n"
           "            eps = unpack('e', pack('e', eps))[0]\n"
           "        except OverflowError:\n"
           "            pass\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'rank': len(x.shape), 'shape': x.shape, 'strides': strides, 'dtype': _native_dtype(x.dtype), 'eps': float(eps)}\n"
           "def pattern(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (L2Norm(x, axis=axis, eps=eps),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const Parameters *p = parameters;
    return output == 0 && input == 0 &&
        (!p || trillim_contiguous(p->rank, p->shape, p->strides));
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_l2norm_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_l2norm_avx2_capabilities(void) {
    static const TrillimCapabilities caps={TRILLIM_KERNEL_ABI_VERSION,TRILLIM_CAP_X86_AVX2|TRILLIM_CAP_LIBM};
    return &caps;
}
