#include "common.h"
#include <math.h>
#include <float.h>
#include <immintrin.h>

#define AVX2 __attribute__((target("avx2")))

typedef struct {
    size_t tokens;
    size_t width;
    uint32_t input_dtype;
} QuantizeParameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input;
    const QuantizeParameters *parameters;
    const TrillimKernelContext *context;
} QuantizeTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const QuantizeParameters *p = parameters;
    inputs[0] = outputs[0] = outputs[1] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->input_dtype);
    if (!trillim_float(p->input_dtype) || (p->tokens && !p->width) ||
        p->tokens > SIZE_MAX / sizeof(float) ||
        (p->width && p->tokens > SIZE_MAX / bytes / p->width)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * bytes;
    outputs[0] = p->tokens * p->width;
    outputs[1] = p->tokens * sizeof(float);
    return TRILLIM_OK;
}

AVX2 static __m256 load8(const QuantizeTask *task, size_t index) {
    uint32_t dtype = task->parameters->input_dtype;
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)task->input + index);
    }
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)task->input + index));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    float values[8];
    for (size_t i = 0; i < 8; ++i) {
        values[i] = trillim_load(task->input, index + i, dtype).f;
    }
    return _mm256_loadu_ps(values);
}

AVX2 static __m256i round8(__m256 values, __m256 scale) {
    __m256 scaled = _mm256_mul_ps(values, scale);
    scaled = _mm256_min_ps(_mm256_max_ps(scaled, _mm256_set1_ps(-127.0f)), _mm256_set1_ps(127.0f));
    return _mm256_cvttps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

AVX2 static TrillimStatus quantize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const QuantizeTask *task = opaque;
    const QuantizeParameters *p = task->parameters;
    size_t start = row * p->width;
    size_t column = 0;
    __m256 maximum8 = _mm256_set1_ps(1e-5f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 magnitude = _mm256_andnot_ps(sign, load8(task, start + column));
        if (_mm256_movemask_ps(_mm256_cmp_ps(magnitude, _mm256_set1_ps(FLT_MAX), _CMP_LE_OQ)) != 255) {
            return TRILLIM_ERROR;
        }
        maximum8 = _mm256_max_ps(maximum8, magnitude);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, maximum8);
    float maximum = 1e-5f;
    for (size_t i = 0; i < 8; ++i) {
        maximum = fmaxf(maximum, lanes[i]);
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        maximum = fmaxf(maximum, fabsf(value));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    __m256 multiplier = _mm256_set1_ps(scale);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    column = 0;
    for (; p->width - column >= 32; column += 32) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256i a = round8(load8(task, start + column), multiplier);
        __m256i b = round8(load8(task, start + column + 8), multiplier);
        __m256i c = round8(load8(task, start + column + 16), multiplier);
        __m256i d = round8(load8(task, start + column + 24), multiplier);
        __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
        packed = _mm256_permutevar8x32_epi32(packed, order);
        _mm256_storeu_si256((__m256i *)(task->codes + start + column), packed);
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f * scale;
        float magnitude = fminf(fabsf(value), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[start + column] = (int8_t)(value < 0 ? -code : code);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_quantize_int8_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const QuantizeParameters *p = parameters;
    QuantizeTask task = {outputs[0], outputs[1], inputs[0], p, context};
    return p->tokens == 1 ? quantize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, quantize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_quantize_int8_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _quantize_int8 as bind\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(x)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_quantize_int8_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(QuantizeParameters, tokens), NULL},
        {"width", "size_t", offsetof(QuantizeParameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(QuantizeParameters, input_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(QuantizeParameters), TRILLIM_ALIGNOF(QuantizeParameters),
        3, fields, 1, 2, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_quantize_int8_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
