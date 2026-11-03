#ifndef TRILLIM_KERNEL_ROTARY_H
#define TRILLIM_KERNEL_ROTARY_H

#include "codec.h"

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

static inline TrillimStatus trillim_rotary_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = (const TrillimRotaryTask *)opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index / p->tokens;
    size_t token = index % p->tokens;
    size_t base = head * p->strides[0] + token * p->strides[1];
    size_t output = index * p->dim;
    for (size_t column = 0; column < p->dim; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        task->output[output + column] = task->input[base + column * p->strides[2]];
    }
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype, 1, p->tokens);
    for (size_t pair = 0; pair < p->rotary_dim / 2; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_rotate_pair(task, base, output, position, pair);
    }
    return TRILLIM_OK;
}

static inline const TrillimParameterSchema *trillim_rotary_schema(void) {
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

#endif
