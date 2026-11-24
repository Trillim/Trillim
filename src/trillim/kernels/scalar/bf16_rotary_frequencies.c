#include "common.h"
#include <math.h>


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

static TrillimStatus trillim_rotary_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index % p->heads;
    size_t token = index / p->heads;
    size_t base = head * p->dim + token * (p->heads * p->dim);
    size_t output = (token * p->heads + head) * p->dim;
    for (size_t column = p->rotary_dim; column < p->dim; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store(task, output + column, trillim_read_float(task->input, base + column, TRILLIM_BF16));
    }
    const float *angles = task->angles + token * p->rotary_dim;
    for (size_t pair = 0; pair < p->rotary_dim / 2; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_rotate_pair(task, base, output, angles, pair);
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

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_bf16_rotary_frequencies(void *const *outputs, const void *const *inputs,
                                                                 void *scratch, const void *parameters,
                                                                 const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    if (!p->heads || !p->tokens) return TRILLIM_OK;
    TrillimRotaryTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context};
    TrillimStatus status =
        p->tokens == 1 ? prepare_angles(&task, 0, 0)
                       : context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_angles, &task);
    if (status != TRILLIM_OK) return status;
    return context->thread_pool->run(context->thread_pool->context, p->heads * p->tokens,
                                     trillim_rotary_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_bf16_rotary_frequencies_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bf16_rotary_frequencies_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bf16_rotary_frequencies_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_LIBM};
    return &capabilities;
}
