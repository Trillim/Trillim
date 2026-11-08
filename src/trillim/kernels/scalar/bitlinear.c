#include "common.h"
#include <math.h>
#include <string.h>

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t itemsize = trillim_itemsize(dtype);
    if (columns != 0 && rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (itemsize != 0) {
        if (count > SIZE_MAX / itemsize) {
            return false;
        }
        *bytes = count * itemsize;
        return true;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        if (rows > (SIZE_MAX - count) / 2) {
            return false;
        }
        *bytes = count + rows * 2;
        return true;
    }
    size_t group;
    size_t record;
    switch (dtype) {
    case TRILLIM_MAT_TERNARY: group = 128; record = 32; break;
    case TRILLIM_GRP_INT8: group = 32; record = 34; break;
    case TRILLIM_GRP_INT4: group = 32; record = 18; break;
    case TRILLIM_GRP_INT5: group = 32; record = 22; break;
    case TRILLIM_DUBGRP_INT6: group = 256; record = 210; break;
    case TRILLIM_GRP_BINARY: group = 128; record = 18; break;
    case TRILLIM_GRP_TERNARY: group = 128; record = 34; break;
    case TRILLIM_MXFP4: group = 32; record = 17; break;
    default: return false;
    }
    size_t groups = columns / group + (columns % group != 0);
    if (rows != 0 && groups > SIZE_MAX / record / rows) {
        return false;
    }
    *bytes = rows * groups * record;
    if (dtype == TRILLIM_MAT_TERNARY) {
        if (*bytes > SIZE_MAX - sizeof(float)) {
            return false;
        }
        *bytes += sizeof(float);
    }
    return true;
}

static inline float trillim_round_even(float value) {
    float lower = floorf(value);
    float fraction = value - lower;
    if (fraction < 0.5f || (fraction == 0.5f && fmodf(lower, 2.0f) == 0.0f)) {
        return lower;
    }
    return lower + 1.0f;
}

typedef struct {
    size_t tokens;
    size_t width;
    size_t outputs;
} TrillimBitLinearParameters;

typedef struct {
    float *output;
    const float *input;
    const uint8_t *weight;
    uint8_t *scratch;
    const TrillimBitLinearParameters *parameters;
    const TrillimKernelContext *context;
} TrillimScalarBitLinearTask;

static inline TrillimStatus trillim_bitlinear_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimBitLinearParameters *p = (const TrillimBitLinearParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->width > INT64_MAX / 127 || p->width > SIZE_MAX - 8 ||
        p->tokens > SIZE_MAX / sizeof(float) / p->width || p->tokens > SIZE_MAX / sizeof(float) / p->outputs ||
        p->tokens > SIZE_MAX / (p->width + 8) ||
        !trillim_payload_size(p->outputs, p->width, TRILLIM_MAT_TERNARY, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * sizeof(float);
    outputs[0] = p->tokens * p->outputs * sizeof(float);
    *scratch = p->tokens * (p->width + 8);
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_bitlinear_quantize_scalar(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimScalarBitLinearTask *task = (const TrillimScalarBitLinearTask *)opaque;
    const TrillimBitLinearParameters *p = task->parameters;
    const float *source = task->input + row * p->width;
    uint8_t *storage = task->scratch + row * (p->width + 8);
    float maximum = 1e-5f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!isfinite(source[column])) {
            return TRILLIM_ERROR;
        }
        maximum = fmaxf(maximum, fabsf(source[column]));
    }
    float multiplier = 127.0f / maximum;
    float scale = 1.0f / multiplier;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = fminf(127.0f, fmaxf(-128.0f, source[column] * multiplier));
        int8_t code = (int8_t)trillim_round_even(value);
        memcpy(storage + column, &code, 1);
    }
    memcpy(storage + p->width, &scale, sizeof(scale));
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_bitlinear_project_scalar(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimScalarBitLinearTask *task = (const TrillimScalarBitLinearTask *)opaque;
    const TrillimBitLinearParameters *p = task->parameters;
    size_t tiles = p->outputs / 32 + (p->outputs % 32 != 0);
    size_t row = index / tiles;
    size_t begin = (index % tiles) * 32;
    size_t end = p->outputs - begin < 32 ? p->outputs : begin + 32;
    size_t packed_row = (p->width / 128 + (p->width % 128 != 0)) * 32;
    const uint8_t *storage = task->scratch + row * (p->width + 8);
    const int8_t *input = (const int8_t *)storage;
    float scale, gamma;
    memcpy(&scale, storage + p->width, sizeof(scale));
    memcpy(&gamma, task->weight + p->outputs * packed_row, sizeof(gamma));
    for (size_t column = begin; column < end; ++column) {
        int64_t sum = 0;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t packed = task->weight[column * packed_row + (k / 128) * 32 + k % 32];
            int32_t code = (packed >> (6 - ((k % 128) / 32) * 2)) & 3;
            if (code == 3) {
                return TRILLIM_ERROR;
            }
            sum += (int64_t)input[k] * (code - 1);
        }
        task->output[row * p->outputs + column] = (float)sum * scale * gamma;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_bitlinear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimBitLinearParameters *p = (const TrillimBitLinearParameters *)parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    TrillimScalarBitLinearTask task = {(float *)outputs[0], (const float *)inputs[0],
        (const uint8_t *)inputs[1], (uint8_t *)scratch, p, context};
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context,
        p->tokens, trillim_bitlinear_quantize_scalar, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t columns = p->outputs / 32 + (p->outputs % 32 != 0);
    return context->thread_pool->run(context->thread_pool->context,
        p->tokens * columns, trillim_bitlinear_project_scalar, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_bitlinear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bitlinear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (BitLinear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bitlinear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(TrillimBitLinearParameters, tokens), NULL},
        {"width", "size_t", offsetof(TrillimBitLinearParameters, width), NULL},
        {"outputs", "size_t", offsetof(TrillimBitLinearParameters, outputs), NULL}
    };
    static const size_t inputs[] = {4, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimBitLinearParameters), TRILLIM_ALIGNOF(TrillimBitLinearParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        trillim_bitlinear_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bitlinear_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
