#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t tokens, heads, dim, rotary_dim;
    float frequency_step;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const void *positions;
    float *angles;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    if (p->dim == 0 || p->heads > SIZE_MAX / p->dim) {
        return TRILLIM_ERROR;
    }
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rotary_dim == 0 || p->rotary_dim % 2 != 0 || p->rotary_dim > p->dim) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->tokens == 0) {
        return TRILLIM_OK;
    }
    if (p->heads > SIZE_MAX / p->tokens ||
        p->heads * p->tokens > SIZE_MAX / sizeof(uint16_t) / p->dim ||
        p->tokens > SIZE_MAX / sizeof(int64_t)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->heads * p->tokens * p->dim * sizeof(uint16_t);
    inputs[1] = p->tokens * sizeof(int64_t);
    if (p->tokens > SIZE_MAX / sizeof(float) / p->rotary_dim) {
        return TRILLIM_ERROR;
    }
    *scratch = p->tokens * p->rotary_dim * sizeof(float);
    return TRILLIM_OK;
}

static inline float32x4_t load4(const void *data, size_t index) {
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, float32x4_t value) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static inline __attribute__((always_inline)) TrillimStatus rotate_row(
    const Task *task, size_t head, size_t token
) {
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t base = (token * p->heads + head) * p->dim;
    size_t column = p->rotary_dim;
    for (; p->dim - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, base + column, load4(task->input, base + column));
    }
    for (; column < p->dim; ++column) {
        TrillimValue value = trillim_load(task->input, base + column, TRILLIM_BF16);
        trillim_store(task->output, base + column, TRILLIM_BF16, value);
    }
    const float *angles = task->angles + token * p->rotary_dim;
    size_t pairs = p->rotary_dim / 2;
    size_t pair = 0;
    for (; pairs - pair >= 4; pair += 4) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t a = load4(task->input, base + pair);
        float32x4_t b = load4(task->input, base + pair + pairs);
        float32x4_t c = vld1q_f32(angles + pair);
        float32x4_t s = vld1q_f32(angles + pairs + pair);
        float32x4_t first = vfmaq_f32(vnegq_f32(vmulq_f32(b, s)), a, c);
        float32x4_t second = vfmaq_f32(vmulq_f32(a, s), b, c);
        store4(task->output, base + pair, first);
        store4(task->output, base + pair + pairs, second);
    }
    for (; pair < pairs; ++pair) {
        float cosine = angles[pair], sine = angles[pair + pairs];
        size_t a = base + pair, b = a + pairs;
        float first = trillim_load(task->input, a, TRILLIM_BF16).f;
        float second = trillim_load(task->input, b, TRILLIM_BF16).f;
        trillim_store(task->output, a, TRILLIM_BF16, (TrillimValue){.f = first * cosine - second * sine});
        trillim_store(task->output, b, TRILLIM_BF16, (TrillimValue){.f = second * cosine + first * sine});
    }
    return TRILLIM_OK;
}

static TrillimStatus rotate_rows(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    size_t rows = p->heads * p->tokens;
    size_t block = rows >= 128 ? 8 : 1;
    size_t first = tile * block;
    size_t last = rows - first < block ? rows : first + block;
    for (size_t index = first; index < last; ++index) {
        const TrillimStatus status = rotate_row(task, index % p->heads, index / p->heads);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus prepare_angles(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    float position = (float)trillim_load(task->positions, token, TRILLIM_INT64).i;
    float *angles = task->angles + token * p->rotary_dim;
    size_t pairs = p->rotary_dim / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float angle = position * expf((float)pair * p->frequency_step);
        angles[pair] = cosf(angle);
        angles[pairs + pair] = sinf(angle);
    }
    return TRILLIM_OK;
}

static TrillimStatus rotate_token(void *opaque, size_t token, size_t worker) {
    const Task *task = opaque;
    const TrillimStatus status = prepare_angles(opaque, token, worker);
    if (status != TRILLIM_OK) {
        return status;
    }
    const size_t heads = task->parameters->heads;
    for (size_t head = 0; head < heads; ++head) {
        const TrillimStatus applied = rotate_row(task, head, token);
        if (applied != TRILLIM_OK) {
            return applied;
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rotary(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->heads || !p->tokens) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], scratch, p, context};
    if (p->tokens >= 16 && p->tokens <= 128 && p->heads <= 32 && p->dim <= 256 &&
        p->tokens >= context->thread_pool->worker_count) {
        return context->thread_pool->run(context->thread_pool->context, p->tokens, rotate_token, &task);
    }
    TrillimStatus status =
        p->tokens == 1 ? prepare_angles(&task, 0, 0)
                       : context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_angles, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t rows = p->heads * p->tokens, block = rows >= 128 ? 8 : 1;
    size_t tiles = rows / block + (rows % block != 0);
    if (p->tokens == 1) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus result = rotate_rows(&task, tile, 0);
            if (result != TRILLIM_OK) {
                return result;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, rotate_rows, &task);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary, _contiguous_strides\n"
           "def bind(c):\n"
           "    x, positions, output = c.inputs['x'], c.inputs['positions'], c.outputs[0]\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or positions.dtype != 'int64':\n"
           "        raise ValueError('requires BF16 activations and INT64 positions')\n"
           "    if output.shape != x.shape or positions.shape != x.shape[:1]:\n"
           "        raise ValueError('requires one position per token and unchanged activation shape')\n"
           "    for tensor in (x, positions, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    p = _avx2_rotary(c)\n"
           "    if p['interleaved'] or p['amplitude'] != 1.0:\n"
           "        raise ValueError('requires split-half rotary with unit amplitude')\n"
           "    return {name: p[name] for name in ('tokens', 'heads', 'dim', 'rotary_dim', 'frequency_step')}\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, "
           "amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rotary_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"dim", "size_t", offsetof(Parameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(Parameters, rotary_dim), NULL},
        {"frequency_step", "float", offsetof(Parameters, frequency_step), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
