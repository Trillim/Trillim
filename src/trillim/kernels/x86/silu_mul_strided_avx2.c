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

TRILLIM_AVX2 static inline __m256i gt8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_GT_OQ));
}

TRILLIM_AVX2 static inline __m256i le8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_LE_OQ));
}

TRILLIM_AVX2 static inline __m256i lt8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_LT_OQ));
}

TRILLIM_AVX2 static inline __m256i eq8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_EQ_OQ));
}

TRILLIM_AVX2 static inline uint32_t all8(__m256i a) {
    return _mm256_testc_si256(a,_mm256_set1_epi32(-1)) ? UINT32_MAX : 0;
}

TRILLIM_AVX2 static inline uint32_t any8(__m256i a) {
    return _mm256_testz_si256(a,a) ? 0 : UINT32_MAX;
}

TRILLIM_AVX2 static inline void storei8(void *data, __m256i a) {
    _mm256_storeu_si256((__m256i *)data,a);
}

TRILLIM_AVX2 static inline __m256i not8(__m256i a) {
    return _mm256_xor_si256(a,_mm256_set1_epi32(-1));
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t rows;
    size_t width;
    size_t a_row_stride;
    size_t a_column_stride;
    size_t b_row_stride;
    size_t b_column_stride;
} Parameters;

typedef struct {
    float *output;
    const float *a;
    const float *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static float activate(float a, float b) {
    float e = expf(-fabsf(a));
    float denominator = 1.0f + e;
    float probability = a >= 0.0f ? 1.0f / denominator : e / denominator;
    return (a * probability) * b;
}

TRILLIM_AVX2 static bool input_span(
    size_t rows, size_t width, size_t row_stride, size_t column_stride, size_t *bytes
) {
    bool row_major = column_stride == 1 && row_stride >= width;
    bool column_major = row_stride == 1 && column_stride >= rows;
    if (!row_major && !column_major) {
        return false;
    }
    size_t count = 1;
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - count) / row_stride) {
        return false;
    }
    count += (rows - 1) * row_stride;
    if (column_stride != 0 && width - 1 > (SIZE_MAX - count) / column_stride) {
        return false;
    }
    count += (width - 1) * column_stride;
    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *bytes = count * sizeof(float);
    return true;
}

TRILLIM_AVX2 static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width / sizeof(float) ||
        !input_span(p->rows, p->width, p->a_row_stride, p->a_column_stride, &inputs[0]) ||
        !input_span(p->rows, p->width, p->b_row_stride, p->b_column_stride, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
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

TRILLIM_AVX2 static __m256 activate8(__m256 a, __m256 b) {
    __m256 e = exp8(neg8(abs8(a)));
    __m256 denominator = _mm256_add_ps(_mm256_set1_ps(1.0f), e);
    __m256 positive = _mm256_div_ps(_mm256_set1_ps(1.0f), denominator);
    __m256 negative = _mm256_div_ps(e, denominator);
    __m256 probability = select8(ge8(a, _mm256_set1_ps(0.0f)), positive, negative);
    __m256 value = _mm256_mul_ps(_mm256_mul_ps(a, probability), b);
    __m256 magnitude = abs8(value);
    __m256i sensitive = _mm256_or_si256(
        lt8(magnitude, _mm256_set1_ps(0x1p-125f)),
        gt8(magnitude, _mm256_set1_ps(0x1p127f))
    );
    sensitive = _mm256_or_si256(sensitive, not8(eq8(value, value)));
    if (any8(sensitive) != 0) {
        /* Resolve IEEE boundary cases with libm before the staged products. */
        float values[8], gates[8], ups[8];
        uint32_t mask[8];
        _mm256_storeu_ps(values, value);
        _mm256_storeu_ps(gates, a);
        _mm256_storeu_ps(ups, b);
        storei8(mask, sensitive);
        for (size_t lane = 0; lane < 8; ++lane) {
            if (mask[lane] != 0) {
                values[lane] = activate(gates[lane], ups[lane]);
            }
        }
        value = _mm256_loadu_ps(values);
    }
    return value;
}

TRILLIM_AVX2 static __m256 load_channels(const float *input, size_t index, size_t stride) {
    if (stride == 1) {
        return _mm256_loadu_ps(input + index);
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = input[(index + lane) * stride];
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t count = p->rows * p->width;
    size_t end = count - begin < 1024 ? count : begin + 1024;
    while (begin < end) {
        size_t row = begin / p->width;
        size_t column = begin % p->width;
        size_t length = p->width - column;
        if (length > end - begin) {
            length = end - begin;
        }
        const float *a = task->a + row * p->a_row_stride + column * p->a_column_stride;
        const float *b = task->b + row * p->b_row_stride + column * p->b_column_stride;
        float *output = task->output + begin;
        size_t i = 0;
        for (; length - i >= 8; i += 8) {
            _mm256_storeu_ps(output + i, activate8(load_channels(a, i, p->a_column_stride), load_channels(b, i, p->b_column_stride)));
        }
        for (; i < length; ++i) {
            output[i] = activate(a[i * p->a_column_stride], b[i * p->b_column_stride]);
        }
        begin += length;
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus apply_transposed(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t row_tiles = p->rows / 8;
    size_t row = (tile % row_tiles) * 8;
    size_t column = (tile / row_tiles) * 256;
    size_t end = p->width - column < 256 ? p->width : column + 256;
    for (; end - column >= 8; column += 8) {
        __m256 v[8], t[8], u[8];
        for (size_t lane=0;lane<8;++lane) {
            __m256 a = _mm256_loadu_ps(task->a + row + (column+lane)*p->a_column_stride);
            __m256 b = _mm256_loadu_ps(task->b + row + (column+lane)*p->b_column_stride);
            v[lane] = activate8(a,b);
        }
        for(size_t i=0;i<4;++i) {
            t[2*i] = _mm256_unpacklo_ps(v[2*i],v[2*i+1]);
            t[2*i+1] = _mm256_unpackhi_ps(v[2*i],v[2*i+1]);
        }
        for(size_t i=0;i<2;++i) {
            u[4*i] = _mm256_shuffle_ps(t[4*i],t[4*i+2],0x44);
            u[4*i+1] = _mm256_shuffle_ps(t[4*i],t[4*i+2],0xee);
            u[4*i+2] = _mm256_shuffle_ps(t[4*i+1],t[4*i+3],0x44);
            u[4*i+3] = _mm256_shuffle_ps(t[4*i+1],t[4*i+3],0xee);
        }
        for(size_t i=0;i<4;++i) {
            _mm256_storeu_ps(task->output+(row+i)*p->width+column,_mm256_permute2f128_ps(u[i],u[i+4],0x20));
            _mm256_storeu_ps(task->output+(row+i+4)*p->width+column,_mm256_permute2f128_ps(u[i],u[i+4],0x31));
        }
    }
    for(;column<end;++column) {
        __m256 a = _mm256_loadu_ps(task->a+row+column*p->a_column_stride);
        __m256 b = _mm256_loadu_ps(task->b+row+column*p->b_column_stride);
        float values[8];
        _mm256_storeu_ps(values,activate8(a,b));
        for(size_t lane=0;lane<8;++lane) task->output[(row+lane)*p->width+column]=values[lane];
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL || output != 0 || input >= 2) {
        return false;
    }
    const Parameters *p = parameters;
    size_t row_stride = input == 0 ? p->a_row_stride : p->b_row_stride;
    size_t column_stride = input == 0 ? p->a_column_stride : p->b_column_stride;
    return column_stride == 1 && (p->rows <= 1 || row_stride == p->width);
}

TRILLIM_KERNEL TrillimStatus trillim_x86_silu_mul_strided_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t count = p->rows * p->width;
    bool transposed = p->rows % 8 == 0 && p->a_row_stride == 1 && p->b_row_stride == 1;
    size_t tiles = transposed
        ? (p->rows / 8) * (p->width / 256 + (p->width % 256 != 0))
        : count / 1024 + (count % 1024 != 0);
    TrillimTask worker = transposed ? apply_transposed : apply;
    if (count <= 4096) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = worker(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, worker, &task);
}

TRILLIM_KERNEL const char *trillim_x86_silu_mul_strided_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "def bind(c):\n"
           "    a, b, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    if len(a.shape) != 2 or a.shape != b.shape or a.shape != output.shape:\n"
           "        raise ValueError('strided SiLU-Mul requires matching matrices')\n"
           "    if any(value.dtype != 'fp32' for value in (a, b, output)):\n"
           "        raise ValueError('strided SiLU-Mul requires FP32 arithmetic')\n"
           "    if (output.strides or _contiguous_strides(output.shape)) != _contiguous_strides(output.shape):\n"
           "        raise ValueError('strided SiLU-Mul requires contiguous output')\n"
           "    rows, width = a.shape\n"
           "    if width.op != 'const' or width.value is None:\n"
           "        raise ValueError('strided SiLU-Mul requires fixed channel width')\n"
           "    one = _constant_expression(1)\n"
           "    strides = [value.strides or _contiguous_strides(value.shape) for value in (a, b)]\n"
           "    for row, column in strides:\n"
           "        row_major = column == one and row.op == 'const' and row.value is not None and row.value >= width.value\n"
           "        column_major = row == one and column == rows\n"
           "        if not row_major and not column_major:\n"
           "            raise ValueError('strided SiLU-Mul requires row-major or transposed matrices')\n"
           "    if all(row == width and column == one for row, column in strides):\n"
           "        raise ValueError('canonical SiLU-Mul uses the existing kernels')\n"
           "    return dict(rows=rows, width=width, a_row_stride=strides[0][0], a_column_stride=strides[0][1], b_row_stride=strides[1][0], b_column_stride=strides[1][1])\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (SiluMul(a, b, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_silu_mul_strided_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"a_row_stride", "size_t", offsetof(Parameters, a_row_stride), NULL},
        {"a_column_stride", "size_t", offsetof(Parameters, a_column_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Parameters, b_row_stride), NULL},
        {"b_column_stride", "size_t", offsetof(Parameters, b_column_stride), NULL}
    };
    static const size_t inputs[] = {4, 4};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_silu_mul_strided_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
