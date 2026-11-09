#include "common.h"
#include <immintrin.h>
#include <math.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

static inline float trillim_read_float(const void *data, size_t index, uint32_t dtype) {
    TrillimValue value = trillim_load(data, index, dtype);
    return trillim_float(dtype) ? value.f :
        (trillim_unsigned(dtype) ? (float)value.u : (float)value.i);
}

typedef struct {
    size_t heads;
    size_t tokens;
    size_t dim;
    size_t rotary_dim;
    size_t rank;
    const size_t *strides;
    size_t position_stride;
    uint32_t position_dtype;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t interleaved;
    float frequency_step;
    float amplitude;
} TrillimRotaryParameters;

typedef struct {
    void *output;
    const void *input;
    const void *positions;
    const TrillimRotaryParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRotaryTask;

static inline TrillimStatus trillim_rotary_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t position_bytes = trillim_itemsize(p->position_dtype);
    if (p->rank != 3 || p->rotary_dim == 0 || p->rotary_dim % 2 != 0 ||
        p->rotary_dim > p->dim || position_bytes == 0 ||
        !trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) || p->interleaved > 1) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->tokens, p->dim};
    size_t span;
    if (!trillim_span(3, shape, p->strides, &span) || span > SIZE_MAX / trillim_itemsize(p->input_dtype) ||
        p->heads > SIZE_MAX / p->tokens || p->heads * p->tokens > SIZE_MAX / trillim_itemsize(p->output_dtype) / p->dim ||
        (p->position_stride != 0 && p->tokens - 1 > (SIZE_MAX - 1) / p->position_stride)) {
        return TRILLIM_ERROR;
    }
    size_t positions = (p->tokens - 1) * p->position_stride + 1;
    if (positions > SIZE_MAX / position_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * trillim_itemsize(p->input_dtype);
    inputs[1] = positions * position_bytes;
    outputs[0] = p->heads * p->tokens * p->dim * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

static void store(const TrillimRotaryTask *task, size_t index, float result) {
    TrillimValue value;
    value.f = result;
    trillim_store(task->output, index, task->parameters->output_dtype, value);
}

static inline void trillim_rotate_pair(const TrillimRotaryTask *task, size_t base, size_t output, float position, size_t pair) {
    const TrillimRotaryParameters *p = task->parameters;
    size_t a = p->interleaved ? pair * 2 : pair;
    size_t b = p->interleaved ? a + 1 : pair + p->rotary_dim / 2;
    float frequency = expf((float)pair * p->frequency_step);
    float angle = position * frequency;
    float cosine = cosf(angle) * p->amplitude;
    float sine = sinf(angle) * p->amplitude;
    float first = trillim_read_float(task->input, base + a * p->strides[2], p->input_dtype);
    float second = trillim_read_float(task->input, base + b * p->strides[2], p->input_dtype);
    store(task, output + a, first * cosine - second * sine);
    store(task, output + b, second * cosine + first * sine);
}

TRILLIM_AVX2 static __m256 load8(const TrillimRotaryTask *task, size_t index) {
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
        values[i] = trillim_read_float(task->input, index + i, dtype);
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static void store8(const TrillimRotaryTask *task, size_t index, __m256 value) {
    if (task->parameters->output_dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)task->output + index, value);
    } else {
        float values[8];
        _mm256_storeu_ps(values, value);
        for (size_t i = 0; i < 8; ++i) {
            store(task, index + i, values[i]);
        }
    }
}

TRILLIM_AVX2 static TrillimStatus trillim_rotary_task_avx2(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index / p->tokens;
    size_t token = index % p->tokens;
    size_t base = head * p->strides[0] + token * p->strides[1];
    size_t output = (token * p->heads + head) * p->dim;
    size_t column = p->rotary_dim;
    for (; p->dim - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store8(task, output + column, load8(task, base + column));
    }
    for (; column < p->dim; ++column) {
        store(task, output + column, trillim_read_float(task->input, base + column, p->input_dtype));
    }
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype);
    size_t pairs = p->rotary_dim / 2;
    size_t pair = 0;
    for (; pairs - pair >= 8; pair += 8) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float cosine[8], sine[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            float angle = position * expf((float)(pair + lane) * p->frequency_step);
            cosine[lane] = cosf(angle) * p->amplitude;
            sine[lane] = sinf(angle) * p->amplitude;
        }
        __m256 a, b;
        if (p->interleaved) {
            __m256 low = load8(task, base + pair * 2);
            __m256 high = load8(task, base + pair * 2 + 8);
            __m256i even = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
            __m256i odd = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
            a = _mm256_blend_ps(_mm256_permutevar8x32_ps(low, even), _mm256_permutevar8x32_ps(high, even), 0xf0);
            b = _mm256_blend_ps(_mm256_permutevar8x32_ps(low, odd), _mm256_permutevar8x32_ps(high, odd), 0xf0);
        } else {
            a = load8(task, base + pair);
            b = load8(task, base + pair + pairs);
        }
        __m256 c = _mm256_loadu_ps(cosine);
        __m256 s = _mm256_loadu_ps(sine);
        __m256 first = _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s));
        __m256 second = _mm256_add_ps(_mm256_mul_ps(b, c), _mm256_mul_ps(a, s));
        if (p->interleaved) {
            __m256 low = _mm256_unpacklo_ps(first, second);
            __m256 high = _mm256_unpackhi_ps(first, second);
            store8(task, output + pair * 2, _mm256_permute2f128_ps(low, high, 0x20));
            store8(task, output + pair * 2 + 8, _mm256_permute2f128_ps(low, high, 0x31));
        } else {
            store8(task, output + pair, first);
            store8(task, output + pair + pairs, second);
        }
    }
    for (; pair < pairs; ++pair) {
        trillim_rotate_pair(task, base, output, position, pair);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const TrillimRotaryParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype &&
        p->strides[2] == 1 && (p->heads <= 1 || p->strides[0] == p->dim) &&
        (p->tokens <= 1 || p->strides[1] == p->heads * p->dim);
}

TRILLIM_KERNEL TrillimStatus trillim_x86_rotary_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    TrillimRotaryTask task = {outputs[0], inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context, p->heads * p->tokens, trillim_rotary_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rotary_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary as bind\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_rotary_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(TrillimRotaryParameters, heads), NULL},
        {"tokens", "size_t", offsetof(TrillimRotaryParameters, tokens), NULL},
        {"dim", "size_t", offsetof(TrillimRotaryParameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(TrillimRotaryParameters, rotary_dim), NULL},
        {"rank", "size_t", offsetof(TrillimRotaryParameters, rank), NULL},
        {"strides", "size_t", offsetof(TrillimRotaryParameters, strides), "rank"},
        {"position_stride", "size_t", offsetof(TrillimRotaryParameters, position_stride), NULL},
        {"position_dtype", "uint32_t", offsetof(TrillimRotaryParameters, position_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRotaryParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRotaryParameters, output_dtype), NULL},
        {"interleaved", "uint32_t", offsetof(TrillimRotaryParameters, interleaved), NULL},
        {"frequency_step", "float", offsetof(TrillimRotaryParameters, frequency_step), NULL},
        {"amplitude", "float", offsetof(TrillimRotaryParameters, amplitude), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRotaryParameters), TRILLIM_ALIGNOF(TrillimRotaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        trillim_rotary_requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rotary_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
