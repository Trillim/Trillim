#include "common.h"
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

/* Mul rounds before Sum; contraction would change the primitive result. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

TRILLIM_AVX2 static __m256 round8(__m256 value, uint32_t dtype) { uint16_t bits[8]; if (dtype == TRILLIM_FP32) return value; store8(bits,0,dtype,value); return load8(bits,0,dtype); }

typedef struct {
    size_t heads, reduce, columns, rank;
    const size_t *a_strides, *b_strides;
    size_t a_rows, a_columns, b_rows, b_columns;
    uint32_t a_dtype, b_dtype, dtype;
} Parameters;

typedef struct {
    void *output;
    const void *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    uint32_t promoted = p->a_dtype == p->b_dtype ? p->a_dtype : TRILLIM_FP32;
    if (p->rank != 3 || !p->a_strides || !p->b_strides ||
        !trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        !trillim_float(p->dtype) || p->dtype != promoted) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->columns) {
        return TRILLIM_OK;
    }
    if (p->heads > SIZE_MAX / trillim_itemsize(p->dtype) / p->columns) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->heads * p->columns * trillim_itemsize(p->dtype);
    if (!p->reduce) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->reduce, p->columns};
    const size_t *strides[] = {p->a_strides, p->b_strides};
    size_t rows[] = {p->a_rows, p->b_rows};
    size_t columns[] = {p->a_columns, p->b_columns};
    uint32_t dtypes[] = {p->a_dtype, p->b_dtype};
    for (size_t i = 0; i < 2; ++i) {
        size_t span;
        if ((columns[i] && rows[i] > SIZE_MAX / columns[i]) ||
            !trillim_span(3, shape, strides[i], &span) ||
            span > rows[i] * columns[i] || span > SIZE_MAX / trillim_itemsize(dtypes[i])) {
            return TRILLIM_ERROR;
        }
        inputs[i] = span * trillim_itemsize(dtypes[i]);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static float native_float(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(float), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)data + index * sizeof(bits), sizeof(bits));
    return dtype == TRILLIM_FP16 ? trillim_fp16(bits) : trillim_bf16(bits);
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

TRILLIM_AVX2 static __m256 load_strided8(const void *data, size_t base, size_t stride, uint32_t dtype) {
    if (stride == 1) {
        return load8(data, base, dtype);
    }
    if (stride == 0) {
        return _mm256_set1_ps(native_float(data, base, dtype));
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = native_float(data, base + lane * stride, dtype);
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static TrillimStatus reduce_columns(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->columns / 64 + (p->columns % 64 != 0);
    size_t head = tile / tiles, first = tile % tiles * 64;
    size_t count = p->columns - first < 64 ? p->columns - first : 64;
    size_t abase = head * p->a_strides[0] + first * p->a_strides[2];
    size_t bbase = head * p->b_strides[0] + first * p->b_strides[2];
    size_t vectors = count / 8;
    __m256 sums[16];
    float tail[7] = {0};
    for (size_t lane = 0; lane < vectors; ++lane) {
        sums[lane] = _mm256_set1_ps(0.0f);
    }
    for (size_t k = 0; k < p->reduce; ++k) {
        if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t arow = abase + k * p->a_strides[1];
        size_t brow = bbase + k * p->b_strides[1];
        if (p->a_strides[2] == 1 && p->b_strides[2] == 0) {
            __m256 b = _mm256_set1_ps(native_float(task->b, brow, p->b_dtype));
            for (size_t lane = 0; lane < vectors; ++lane) {
                __m256 product = round8(_mm256_mul_ps(load8(task->a, arow + lane * 8, p->a_dtype), b), p->dtype);
                sums[lane] = _mm256_add_ps(sums[lane], product);
            }
        } else {
            for (size_t lane = 0; lane < vectors; ++lane) {
                __m256 a = load_strided8(task->a, arow + lane * 8 * p->a_strides[2], p->a_strides[2], p->a_dtype);
                __m256 b = load_strided8(task->b, brow + lane * 8 * p->b_strides[2], p->b_strides[2], p->b_dtype);
                __m256 product = round8(_mm256_mul_ps(a, b), p->dtype);
                sums[lane] = _mm256_add_ps(sums[lane], product);
            }
        }
        for (size_t lane = vectors * 8; lane < count; ++lane) {
            float a = native_float(task->a, arow + lane * p->a_strides[2], p->a_dtype);
            float b = native_float(task->b, brow + lane * p->b_strides[2], p->b_dtype);
            float product = round_dtype(a * b, p->dtype);
            tail[lane - vectors * 8] += product;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t output = head * p->columns + first;
    for (size_t lane = 0; lane < vectors; ++lane) {
        store8(task->output, output + lane * 8, p->dtype, sums[lane]);
    }
    for (size_t lane = vectors * 8; lane < count; ++lane) {
        TrillimValue value = {.f = tail[lane - vectors * 8]};
        trillim_store(task->output, output + lane, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_reduce_sum_mul_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->heads || !p->columns) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->columns / 64 + (p->columns % 64 != 0);
    return p->heads * tiles == 1 ? reduce_columns(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->heads * tiles, reduce_columns, &task);
}

TRILLIM_KERNEL const char *trillim_x86_reduce_sum_mul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _native_dtype, _storage_shape\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression, _promote\n"
           "\n"
           "def bind(context):\n"
           "    a, b, output = context.inputs['a'], context.inputs['b'], context.outputs[0]\n"
           "    rank = max(len(a.shape), len(b.shape))\n"
           "    if rank not in {2, 3} or a.dtype not in {'fp16', 'bf16', 'fp32'} or b.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('fused multiply-sum requires rank-two or rank-three native floating tensors')\n"
           "    if output.dtype != _promote(a.dtype, b.dtype):\n"
           "        raise ValueError('fused multiply-sum requires the promoted product dtype')\n"
           "    one = _constant_expression(1)\n"
           "    aa = (one,) * (rank - len(a.shape)) + a.shape\n"
           "    bb = (one,) * (rank - len(b.shape)) + b.shape\n"
           "    shape = tuple(_expression('broadcast', x, y) for x, y in zip(aa, bb, strict=True))\n"
           "    if output.shape != shape[:-2] + (shape[-1],):\n"
           "        raise ValueError('fused multiply-sum requires reduction of the penultimate axis')\n"
           "    physical_shape = (one,) * (3 - rank) + shape\n"
           "    ar, ac = _storage_shape(a)\n"
           "    br, bc = _storage_shape(b)\n"
           "    return {'heads': physical_shape[0], 'reduce': physical_shape[1], 'columns': physical_shape[2], 'rank': 3, 'a_strides': _broadcast_strides(a.shape, physical_shape, a.strides), 'b_strides': _broadcast_strides(b.shape, physical_shape, b.strides), 'a_rows': ar, 'a_columns': ac, 'b_rows': br, 'b_columns': bc, 'a_dtype': _native_dtype(a.dtype), 'b_dtype': _native_dtype(b.dtype), 'dtype': _native_dtype(output.dtype)}\n"
           "\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Sum(Mul(a, b), axes=(-2,)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_reduce_sum_mul_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"reduce", "size_t", offsetof(Parameters, reduce), NULL},
        {"columns", "size_t", offsetof(Parameters, columns), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"},
        {"a_rows", "size_t", offsetof(Parameters, a_rows), NULL},
        {"a_columns", "size_t", offsetof(Parameters, a_columns), NULL},
        {"b_rows", "size_t", offsetof(Parameters, b_rows), NULL},
        {"b_columns", "size_t", offsetof(Parameters, b_columns), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, b_dtype), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 13, fields, 2, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_reduce_sum_mul_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
