#include "common.h"
#include <math.h>

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

static inline TrillimStatus trillim_rotary_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = (const TrillimRotaryTask *)opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index / p->tokens;
    size_t token = index % p->tokens;
    size_t base = head * p->strides[0] + token * p->strides[1];
    size_t output = (token * p->heads + head) * p->dim;
    for (size_t column = p->rotary_dim; column < p->dim; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store(task, output + column, trillim_read_float(task->input, base + column * p->strides[2], p->input_dtype));
    }
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype);
    for (size_t pair = 0; pair < p->rotary_dim / 2; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_rotate_pair(task, base, output, position, pair);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRotaryParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype &&
        p->strides[2] == 1 && (p->heads <= 1 || p->strides[0] == p->dim) &&
        (p->tokens <= 1 || p->strides[1] == p->heads * p->dim);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_rotary(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    TrillimRotaryTask task = {outputs[0], inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context, p->heads * p->tokens, trillim_rotary_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rotary as bind\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_rotary_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
