#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

static inline float trillim_read_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (trillim_itemsize(dtype) != 0) {
        TrillimValue value = trillim_load(data, index, dtype);
        if (trillim_float(dtype)) {
            return value.f;
        }
        return trillim_unsigned(dtype) ? (float)value.u : (float)value.i;
    }
    const uint8_t *values = (const uint8_t *)data;
    size_t row = index / columns;
    size_t column = index % columns;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        int8_t code;
        memcpy(&bits, values + rows * columns + row * 2, 2);
        memcpy(&code, values + index, 1);
        return (float)code * trillim_bf16(bits);
    }
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_BINARY ||
          dtype == TRILLIM_GRP_TERNARY) ? 128 : 32);
    size_t groups = columns / group + (columns % group != 0);
    size_t total = rows * groups;
    size_t block = row * groups + column / group;
    size_t offset = column % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        size_t code = (values[block * 32 + offset % 32] >> (6 - (offset / 32) * 2)) & 3;
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
        return code == 3 ? NAN : (float)((int32_t)code - 1) * scale;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        uint8_t code = (values[block * 16 + offset / 2] >> ((offset % 2) * 4)) & 15;
        uint8_t exponent = values[total * 16 + block];
        float value = magnitudes[code & 7];
        if ((code & 8) != 0) {
            value = -value;
        }
        return exponent == 255 ? NAN : ldexpf(value, (int)exponent - 127);
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 :
        (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    uint16_t bits;
    memcpy(&bits, values + scale_start + block * 2, 2);
    float scale = trillim_bf16(bits);
    if (dtype == TRILLIM_GRP_INT8) {
        int8_t code;
        memcpy(&code, values + block * 32 + offset, 1);
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        int32_t code = ((values[block * 16 + offset / 8] >> (7 - offset % 8)) & 1) ? 1 : -1;
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t code = (values[block * 16 + offset % 16] >> ((offset / 16) * 4)) & 15;
        if (dtype == TRILLIM_GRP_INT5) {
            code |= ((values[scale_start + total * 2 + block * 4 + offset / 8] >> (offset % 8)) & 1) << 4;
        }
        return (float)(code - (dtype == TRILLIM_GRP_INT4 ? 8 : 16)) * scale;
    }
    size_t half = offset / 128;
    size_t lane = (offset % 128) / 32;
    size_t low = block * 128 + half * 64 + (lane / 2) * 32 + offset % 32;
    size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
    int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
    code |= ((values[high] >> (lane * 2)) & 3) << 4;
    int8_t subscale;
    memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
    return (float)((code - 32) * subscale) * scale;
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
    uint32_t interleaved;
    float frequency_step;
    float amplitude;
} TrillimRotaryParameters;

typedef struct {
    float *output;
    const float *input;
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
        p->rotary_dim > p->dim || position_bytes == 0 || p->interleaved > 1) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->tokens, p->dim};
    size_t span;
    if (!trillim_span(3, shape, p->strides, &span) || span > SIZE_MAX / sizeof(float) ||
        p->heads > SIZE_MAX / p->tokens || p->heads * p->tokens > SIZE_MAX / sizeof(float) / p->dim ||
        (p->position_stride != 0 && p->tokens - 1 > (SIZE_MAX - 1) / p->position_stride)) {
        return TRILLIM_ERROR;
    }
    size_t positions = (p->tokens - 1) * p->position_stride + 1;
    if (positions > SIZE_MAX / position_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * sizeof(float);
    inputs[1] = positions * position_bytes;
    outputs[0] = p->heads * p->tokens * p->dim * sizeof(float);
    return TRILLIM_OK;
}

static inline void trillim_rotate_pair(const TrillimRotaryTask *task, size_t base, size_t output, float position, size_t pair) {
    const TrillimRotaryParameters *p = task->parameters;
    size_t a = p->interleaved ? pair * 2 : pair;
    size_t b = p->interleaved ? a + 1 : pair + p->rotary_dim / 2;
    float frequency = expf((float)pair * p->frequency_step);
    float angle = position * frequency;
    float cosine = cosf(angle) * p->amplitude;
    float sine = sinf(angle) * p->amplitude;
    float first = task->input[base + a * p->strides[2]];
    float second = task->input[base + b * p->strides[2]];
    task->output[output + a] = first * cosine - second * sine;
    task->output[output + b] = second * cosine + first * sine;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_rotary_task_avx2(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = (const TrillimRotaryTask *)opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index / p->tokens;
    size_t token = index % p->tokens;
    size_t base = head * p->strides[0] + token * p->strides[1];
    size_t output = index * p->dim;
    size_t column = 0;
    for (; p->dim - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        _mm256_storeu_ps(task->output + output + column, _mm256_loadu_ps(task->input + base + column));
    }
    for (; column < p->dim; ++column) {
        task->output[output + column] = task->input[base + column];
    }
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype, 1, p->tokens);
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
            __m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
            a = _mm256_i32gather_ps(task->input + base + pair * 2, indices, 4);
            b = _mm256_i32gather_ps(task->input + base + pair * 2 + 1, indices, 4);
        } else {
            a = _mm256_loadu_ps(task->input + base + pair);
            b = _mm256_loadu_ps(task->input + base + pair + pairs);
        }
        __m256 c = _mm256_loadu_ps(cosine);
        __m256 s = _mm256_loadu_ps(sine);
        __m256 first = _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s));
        __m256 second = _mm256_add_ps(_mm256_mul_ps(b, c), _mm256_mul_ps(a, s));
        if (p->interleaved) {
            __m256 low = _mm256_unpacklo_ps(first, second);
            __m256 high = _mm256_unpackhi_ps(first, second);
            _mm256_storeu_ps(task->output + output + pair * 2, _mm256_permute2f128_ps(low, high, 0x20));
            _mm256_storeu_ps(task->output + output + pair * 2 + 8, _mm256_permute2f128_ps(low, high, 0x31));
        } else {
            _mm256_storeu_ps(task->output + output + pair, first);
            _mm256_storeu_ps(task->output + output + pair + pairs, second);
        }
    }
    for (; pair < pairs; ++pair) {
        trillim_rotate_pair(task, base, output, position, pair);
    }
    return TRILLIM_OK;
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
    TrillimRotaryTask task = {(float *)outputs[0], (const float *)inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context, p->heads * p->tokens, trillim_rotary_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rotary_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary as bind\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, amplitude: float = 1.0) -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude),)\n";
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
        {"interleaved", "uint32_t", offsetof(TrillimRotaryParameters, interleaved), NULL},
        {"frequency_step", "float", offsetof(TrillimRotaryParameters, frequency_step), NULL},
        {"amplitude", "float", offsetof(TrillimRotaryParameters, amplitude), NULL}
    };
    static const size_t inputs[] = {4, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRotaryParameters), TRILLIM_ALIGNOF(TrillimRotaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        trillim_rotary_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rotary_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
