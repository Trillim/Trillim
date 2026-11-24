#include "common.h"
#include <immintrin.h>
#include <math.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

static inline float trillim_read_float(const void *data, size_t index, uint32_t dtype) {
    TrillimValue value = trillim_load(data, index, dtype);
    return trillim_float(dtype) ? value.f : (trillim_unsigned(dtype) ? (float)value.u : (float)value.i);
}

typedef struct {
    size_t tokens, heads, dim, rotary_dim;
    float amplitude;
} TrillimRotaryParameters;

typedef struct {
    void *output;
    const void *input;
    const void *positions;
    const float *frequencies;
    float *angles;
    const TrillimRotaryParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRotaryTask;

static inline TrillimStatus trillim_rotary_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                        size_t *scratch) {
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    inputs[2] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t position_bytes = trillim_itemsize(TRILLIM_INT64);
    if (3 != 3 || p->rotary_dim == 0 || p->rotary_dim % 2 != 0 || p->rotary_dim > p->dim || position_bytes == 0 ||
        !trillim_float(TRILLIM_BF16) || !trillim_float(TRILLIM_BF16) || 0 > 1 || 1 != 1) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->tokens, p->dim};
    if (!p->dim || p->heads > SIZE_MAX / p->dim) return TRILLIM_ERROR;
    size_t strides[] = {p->dim, p->heads * p->dim, 1};
    size_t span;
    if (!trillim_span(3, shape, strides, &span) || span > SIZE_MAX / trillim_itemsize(TRILLIM_BF16) ||
        p->heads > SIZE_MAX / p->tokens ||
        p->heads * p->tokens > SIZE_MAX / trillim_itemsize(TRILLIM_BF16) / p->dim ||
        (1 != 0 && p->tokens - 1 > (SIZE_MAX - 1) / 1)) {
        return TRILLIM_ERROR;
    }
    size_t positions = (p->tokens - 1) * 1 + 1;
    if (positions > SIZE_MAX / position_bytes) {
        return TRILLIM_ERROR;
    }
    if (p->rotary_dim / 2 > SIZE_MAX / sizeof(float)) return TRILLIM_ERROR;
    inputs[2] = p->rotary_dim / 2 * sizeof(float);
    inputs[0] = span * trillim_itemsize(TRILLIM_BF16);
    inputs[1] = positions * position_bytes;
    outputs[0] = p->heads * p->tokens * p->dim * trillim_itemsize(TRILLIM_BF16);
    if (p->tokens > SIZE_MAX / sizeof(float) / p->rotary_dim) return TRILLIM_ERROR;
    *scratch = p->tokens * p->rotary_dim * sizeof(float);
    return TRILLIM_OK;
}

static void store(const TrillimRotaryTask *task, size_t index, float result) {
    TrillimValue value;
    value.f = result;
    trillim_store(task->output, index, TRILLIM_BF16, value);
}

static inline void trillim_rotate_pair(const TrillimRotaryTask *task, size_t base, size_t output, const float *angles,
                                       size_t pair) {
    const TrillimRotaryParameters *p = task->parameters;
    size_t a = 0 ? pair * 2 : pair;
    size_t b = 0 ? a + 1 : pair + p->rotary_dim / 2;
    float cosine = angles[pair], sine = angles[pair + p->rotary_dim / 2];
    float first = trillim_read_float(task->input, base + a * 1, TRILLIM_BF16);
    float second = trillim_read_float(task->input, base + b * 1, TRILLIM_BF16);
    store(task, output + a, first * cosine - second * sine);
    store(task, output + b, second * cosine + first * sine);
}

TRILLIM_AVX2 static __m256 load8(const TrillimRotaryTask *task, size_t index) {
    uint32_t dtype = TRILLIM_BF16;
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)task->input + index);
    }
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)task->input + index));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    if (dtype == TRILLIM_FP16) {
        __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)task->input + index));
        __m256i h = _mm256_cvtepu16_epi32(raw);
        __m256i fraction = _mm256_and_si256(h, _mm256_set1_epi32(1023));
        __m256i exponent = _mm256_and_si256(h, _mm256_set1_epi32(0x7c00));
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
        __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x7fff)), 13),
                                          _mm256_set1_epi32(112 << 23));
        __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
        normal = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                    _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
        __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000), _mm256_slli_epi32(fraction, 13));
        normal = _mm256_blendv_epi8(normal, special, _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00)));
        return _mm256_castsi256_ps(_mm256_or_si256(normal, sign));
    }
    float values[8];
    for (size_t i = 0; i < 8; ++i) {
        values[i] = trillim_read_float(task->input, index + i, dtype);
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static void store8(const TrillimRotaryTask *task, size_t index, __m256 value) {
    if (TRILLIM_BF16 == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)task->output + index, value);
    } else {
        float values[8];
        _mm256_storeu_ps(values, value);
        for (size_t i = 0; i < 8; ++i) {
            store(task, index + i, values[i]);
        }
    }
}

TRILLIM_AVX2 static TrillimStatus trillim_rotary_task_avx2(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t rows = p->heads * p->tokens;
    size_t block = rows >= 128 ? 8 : 1;
    size_t first = tile * block;
    size_t last = rows - first < block ? rows : first + block;
    for (size_t index = first; index < last; ++index) {
        size_t head = index % p->heads;
        size_t token = index / p->heads;
        size_t base = head * p->dim + token * (p->heads * p->dim);
        size_t output = (token * p->heads + head) * p->dim;
        size_t column = p->rotary_dim;
        for (; p->dim - column >= 8; column += 8) {
            if (((column - p->rotary_dim) & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            store8(task, output + column, load8(task, base + column));
        }
        for (; column < p->dim; ++column) {
            store(task, output + column, trillim_read_float(task->input, base + column, TRILLIM_BF16));
        }
        const float *angles = task->angles + token * p->rotary_dim;
        size_t pairs = p->rotary_dim / 2;
        size_t pair = 0;
        for (; pairs - pair >= 8; pair += 8) {
            if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 a, b;
            if (0) {
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
            __m256 c = _mm256_loadu_ps(angles + pair);
            __m256 s = _mm256_loadu_ps(angles + pairs + pair);
            __m256 first = _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s));
            __m256 second = _mm256_add_ps(_mm256_mul_ps(b, c), _mm256_mul_ps(a, s));
            if (0) {
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
            trillim_rotate_pair(task, base, output, angles, pair);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus prepare_angles(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    float position = trillim_read_float(task->positions, token * 1, TRILLIM_INT64);
    float *angles = task->angles + token * p->rotary_dim;
    size_t pairs = p->rotary_dim / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context))
            return TRILLIM_CANCELLED;
        float angle = position * task->frequencies[pair];
        angles[pair] = cosf(angle) * p->amplitude;
        angles[pairs + pair] = sinf(angle) * p->amplitude;
    }
    return TRILLIM_OK;
}

static TrillimStatus rotate_token(void *opaque, size_t token, size_t worker) {
    TrillimRotaryTask *task = opaque;
    TrillimStatus status = prepare_angles(task, token, worker);
    if (status != TRILLIM_OK) return status;
    TrillimRotaryParameters p = *task->parameters;
    p.tokens = 1;
    TrillimRotaryTask local = *task;
    local.parameters = &p;
    local.input = (const uint16_t *)task->input + token * p.heads * p.dim;
    local.output = (uint16_t *)task->output + token * p.heads * p.dim;
    local.angles = task->angles + token * p.rotary_dim;
    size_t block = p.heads >= 128 ? 8 : 1;
    for (size_t tile = 0; tile < p.heads / block + (p.heads % block != 0); ++tile) {
        status = trillim_rotary_task_avx2(&local, tile, worker);
        if (status != TRILLIM_OK) return status;
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_bf16_rotary_frequencies(void *const *outputs, const void *const *inputs,
                                                                 void *scratch, const void *parameters,
                                                                 const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    if (!p->heads || !p->tokens) return TRILLIM_OK;
    TrillimRotaryTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context};
    if (p->tokens >= 16 && p->tokens <= 128 && p->heads <= 32 && p->dim <= 256 &&
        p->tokens >= context->thread_pool->worker_count) {
        return context->thread_pool->run(context->thread_pool->context, p->tokens, rotate_token, &task);
    }
    TrillimStatus status =
        p->tokens == 1 ? prepare_angles(&task, 0, 0)
                       : context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_angles, &task);
    if (status != TRILLIM_OK) return status;
    size_t rows = p->heads * p->tokens, block = rows >= 128 ? 8 : 1;
    return context->thread_pool->run(context->thread_pool->context, rows / block + (rows % block != 0),
                                     trillim_rotary_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_bf16_rotary_frequencies_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary, _contiguous_strides\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "physical_inputs = ('x', 'positions', 'frequencies')\n"
           "def bind(c):\n"
           "    x, positions, f = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or positions.dtype != 'int64':\n"
           "        raise ValueError('requires BF16 activations and INT64 positions')\n"
           "    if output.shape != x.shape or positions.shape != x.shape[:1]:\n"
           "        raise ValueError('requires one position per token and unchanged activation shape')\n"
           "    for tensor in (x, positions, f, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    p = _avx2_rotary(c)\n"
           "    if p['interleaved']:\n"
           "        raise ValueError('requires split-half rotary')\n"
           "    if f.dtype != 'fp32' or f.shape != (_constant_expression(c.attrs['dim'] // 2),):\n"
           "        raise ValueError('requires one FP32 frequency per pair')\n"
           "    return {name: p[name] for name in ('tokens', 'heads', 'dim', 'rotary_dim', 'amplitude')}\n"
           "def pattern(x: Tensor, positions: Tensor, frequencies: Tensor, *, "
           "dim: int, theta: float = 10000.0, interleaved: bool = False, "
           "amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, frequencies=frequencies, "
           "interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_bf16_rotary_frequencies_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(TrillimRotaryParameters, tokens), NULL},
        {"heads", "size_t", offsetof(TrillimRotaryParameters, heads), NULL},
        {"dim", "size_t", offsetof(TrillimRotaryParameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(TrillimRotaryParameters, rotary_dim), NULL},
        {"amplitude", "float", offsetof(TrillimRotaryParameters, amplitude), NULL},
    };
    static const size_t inputs[] = {2, 1, 4}, outputs[] = {2};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRotaryParameters), TRILLIM_ALIGNOF(TrillimRotaryParameters), 5, fields, 3, 1,
        trillim_rotary_requirements, can_alias, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_bf16_rotary_frequencies_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
