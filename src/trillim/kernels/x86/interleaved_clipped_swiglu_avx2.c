#include "common.h"
#include <immintrin.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t rows, width;
} Parameters;

typedef struct {
    float *output;
    const float *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->width % 4) {
        return TRILLIM_ERROR;
    }
    if (!p->width || !p->rows) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / (2 * sizeof(float)) / p->width) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
    inputs[0] = outputs[0] * 2;
    return TRILLIM_OK;
}

/* Match the staged FP32 arithmetic in the decomposed operations. */
static float activate(float gate, float up) {
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    return (gate * probability) * (up + 1.0f);
}

#define SIMD __attribute__((target("avx2,fma")))

SIMD static __m256 exp8(__m256 x) {
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(0.00021219444005469058f), r);
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 720.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 120.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 24.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 6.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(0.5f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_mul_ps(value, _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23)));
}

SIMD static __m256 activate8(__m256 gate, __m256 up) {
    gate = _mm256_min_ps(gate, _mm256_set1_ps(7.0f));
    up = _mm256_min_ps(_mm256_max_ps(up, _mm256_set1_ps(-7.0f)), _mm256_set1_ps(7.0f));
    __m256 scaled = _mm256_mul_ps(gate, _mm256_set1_ps(1.702f));
    if (_mm256_movemask_ps(_mm256_cmp_ps(scaled, _mm256_set1_ps(-80.0f), _CMP_LT_OQ))) {
        float gates[8], ups[8];
        _mm256_storeu_ps(gates, gate);
        _mm256_storeu_ps(ups, up);
        for (size_t lane = 0; lane < 8; ++lane) {
            gates[lane] = activate(gates[lane], ups[lane]);
        }
        return _mm256_loadu_ps(gates);
    }
    __m256 e = exp8(_mm256_or_ps(scaled, _mm256_set1_ps(-0.0f)));
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denominator = _mm256_add_ps(e, one);
    __m256 numerator = _mm256_blendv_ps(e, one, _mm256_cmp_ps(scaled, _mm256_setzero_ps(), _CMP_GE_OQ));
    __m256 probability = _mm256_div_ps(numerator, denominator);
    return _mm256_mul_ps(_mm256_mul_ps(gate, probability), _mm256_add_ps(up, one));
}

SIMD static __m256 activate_interleaved(__m256 first, __m256 second) {
    const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    __m256 gate = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(first, second, 0x88), order);
    __m256 up = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(first, second, 0xdd), order);
    return activate8(gate, up);
}

SIMD static TrillimStatus apply(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t width = task->p->width;
    const float *source = task->input + row * width * 2;
    float *output = task->output + row * width;
    size_t i = 0;
    for (; width - i >= 8; i += 8) {
        if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 first = _mm256_loadu_ps(source + 2 * i);
        __m256 second = _mm256_loadu_ps(source + 2 * i + 8);
        _mm256_storeu_ps(output + i, activate_interleaved(first, second));
    }
    for (; i < width; ++i) {
        output[i] = activate(source[2 * i], source[2 * i + 1]);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_interleaved_clipped_swiglu_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows > 4 && p->rows > 16384 / p->width) {
        return context->thread_pool->run(context->thread_pool->context, p->rows, apply, &task);
    }
    for (size_t row = 0; row < p->rows; ++row) {
        TrillimStatus status = apply(&task, row, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_interleaved_clipped_swiglu_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if len(x.shape) != 3 or len(output.shape) != 3 or x.dtype != 'fp32' or output.dtype != 'fp32':\n"
           "        raise ValueError('interleaved clipped SwiGLU needs rank-three FP32 tensors')\n"
           "    width = output.shape[-1]\n"
           "    if width.value is None or width.value % 4 or x.shape[-1].value != width.value * 2 or x.shape[:-1] != output.shape[:-1]:\n"
           "        raise ValueError('interleaved clipped SwiGLU needs a gate and value per output channel')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('interleaved clipped SwiGLU needs contiguous projected rows')\n"
           "    return dict(rows=_product(output.shape[:-1]), width=width)\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    if not isinstance(x, Tensor) or x.dtype != 'fp32' or len(x.shape.dimensions) != 3:\n"
           "        raise ValueError('expected a rank-three FP32 tensor')\n"
           "    width = x.shape.dimensions[-1]\n"
           "    if width is None or width <= 0 or width % 4:\n"
           "        raise ValueError('expected a fixed channel width divisible by four')\n"
           "    gate_ids = Iota(width * 2, step=2)\n"
           "    up_ids = Iota(width * 2, start=1, step=2)\n"
           "    gate = Minimum(Take(x, gate_ids, axis=2, rank=3), 7.0)\n"
           "    up = Take(x, up_ids, axis=2, rank=3)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(up, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_interleaved_clipped_swiglu_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_interleaved_clipped_swiglu_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &capabilities;
}
