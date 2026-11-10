#include "common.h"
#include <immintrin.h>
#include <float.h>
#include <math.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

typedef struct {
    size_t rows, width;
    uint32_t input_dtype, weight_dtype, output_dtype;
    float eps, weight_offset;
} Parameters;

typedef struct {
    int8_t *codes;
    float *scales;
    const void *input, *weight;
    void *normalized;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

TRILLIM_AVX2 static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits,
        _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)))), 16);
    __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
        _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = outputs[1] = *scratch = 0;
    if ((p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP32) ||
        (p->weight_dtype != TRILLIM_BF16 && p->weight_dtype != TRILLIM_FP32) ||
        (p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32) ||
        !p->width || p->width > SIZE_MAX / 4 || p->rows > SIZE_MAX / 4 / p->width) return TRILLIM_ERROR;
    inputs[0] = p->rows * p->width * trillim_itemsize(p->input_dtype);
    inputs[1] = p->width * trillim_itemsize(p->weight_dtype);
    outputs[0] = p->rows * p->width;
    outputs[1] = p->rows * sizeof(float);
    *scratch = p->rows * p->width * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t start = row * p->width;
    __m256 sum = _mm256_setzero_ps();
    size_t column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 value = load8(task->input, start + column, p->input_dtype);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(value, value));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float total = 0;
    for (size_t i = 0; i < 8; ++i) total += lanes[i];
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->input_dtype).f;
        total += value * value;
    }
    float inverse = 1.0f / sqrtf(total / (float)p->width + p->eps);
    __m256 multiplier = _mm256_set1_ps(inverse), offset = _mm256_set1_ps(p->weight_offset);
    __m256 maximum = _mm256_set1_ps(1e-5f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 value = _mm256_mul_ps(_mm256_mul_ps(load8(task->input, start + column, p->input_dtype), multiplier),
            _mm256_add_ps(load8(task->weight, column, p->weight_dtype), offset));
        store8(task->normalized, start + column, p->output_dtype, value);
        value = load8(task->normalized, start + column, p->output_dtype);
        __m256 magnitude = _mm256_andnot_ps(sign, value);
        if (_mm256_movemask_ps(_mm256_cmp_ps(magnitude, _mm256_set1_ps(FLT_MAX), _CMP_LE_OQ)) != 255) return TRILLIM_ERROR;
        maximum = _mm256_max_ps(maximum, magnitude);
    }
    _mm256_storeu_ps(lanes, maximum);
    float absmax = 1e-5f;
    for (size_t i = 0; i < 8; ++i) absmax = fmaxf(absmax, lanes[i]);
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = trillim_load(task->input, start + column, p->input_dtype).f * inverse *
            (trillim_load(task->weight, column, p->weight_dtype).f + p->weight_offset)};
        trillim_store(task->normalized, start + column, p->output_dtype, value);
        float rounded = trillim_load(task->normalized, start + column, p->output_dtype).f;
        if (!isfinite(rounded)) return TRILLIM_ERROR;
        absmax = fmaxf(absmax, fabsf(rounded));
    }
    float scale = 127.0f / absmax;
    task->scales[row] = scale;
    multiplier = _mm256_set1_ps(scale);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    column = 0;
    for (; p->width - column >= 32; column += 32) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256i codes[4];
        for (size_t i = 0; i < 4; ++i) {
            __m256 scaled = _mm256_mul_ps(load8(task->normalized, start + column + i * 8, p->output_dtype), multiplier);
            scaled = _mm256_min_ps(_mm256_max_ps(scaled, _mm256_set1_ps(-127)), _mm256_set1_ps(127));
            codes[i] = _mm256_cvttps_epi32(_mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
        __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(codes[0], codes[1]), _mm256_packs_epi32(codes[2], codes[3]));
        _mm256_storeu_si256((__m256i *)(task->codes + start + column), _mm256_permutevar8x32_epi32(packed, order));
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->normalized, start + column, p->output_dtype).f * scale;
        task->codes[start + column] = (int8_t)nearbyintf(fminf(fmaxf(value, -127), 127));
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_rmsnorm_quantize_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0], outputs[1], inputs[0], inputs[1], scratch, p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rmsnorm_quantize_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm_quantize as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return QuantizeInt8(RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_rmsnorm_quantize_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1, 4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 2, 2,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rmsnorm_quantize_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
