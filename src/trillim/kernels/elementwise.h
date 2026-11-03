#ifndef TRILLIM_KERNEL_ELEMENTWISE_H
#define TRILLIM_KERNEL_ELEMENTWISE_H

#include "tensor.h"

typedef enum {
    TRILLIM_ADD, TRILLIM_SUB, TRILLIM_MUL, TRILLIM_DIV,
    TRILLIM_FLOOR_DIV, TRILLIM_MOD, TRILLIM_EXP, TRILLIM_LOG,
    TRILLIM_SQRT, TRILLIM_SIN, TRILLIM_COS, TRILLIM_ERF
} TrillimElementwiseOperation;

typedef struct {
    TrillimBinaryTask binary;
    TrillimElementwiseOperation operation;
} TrillimElementwiseTask;

typedef struct {
    void *output;
    const void *input;
    const TrillimUnaryParameters *parameters;
    const TrillimKernelContext *context;
    TrillimElementwiseOperation operation;
} TrillimUnaryTask;

static inline TrillimStatus trillim_binary_value(
    TrillimElementwiseOperation operation, TrillimValue a, TrillimValue b,
    uint32_t dtype, TrillimValue *result
) {
    if (trillim_float(dtype)) {
        switch (operation) {
        case TRILLIM_ADD: result->f = a.f + b.f; break;
        case TRILLIM_SUB: result->f = a.f - b.f; break;
        case TRILLIM_MUL: result->f = a.f * b.f; break;
        case TRILLIM_DIV: result->f = a.f / b.f; break;
        default: return TRILLIM_ERROR;
        }
    } else {
        switch (operation) {
        case TRILLIM_ADD: result->u = a.u + b.u; break;
        case TRILLIM_SUB: result->u = a.u - b.u; break;
        case TRILLIM_MUL: result->u = a.u * b.u; break;
        case TRILLIM_FLOOR_DIV: case TRILLIM_MOD:
            if (b.u == 0) {
                return TRILLIM_ERROR;
            }
            if (trillim_unsigned(dtype)) {
                result->u = operation == TRILLIM_MOD ? a.u % b.u : a.u / b.u;
            } else if (a.i == INT64_MIN && b.i == -1) {
                result->i = operation == TRILLIM_MOD ? 0 : INT64_MIN;
            } else {
                int64_t quotient = a.i / b.i;
                int64_t remainder = a.i % b.i;
                if (remainder != 0 && (remainder < 0) != (b.i < 0)) {
                    --quotient;
                    remainder += b.i;
                }
                result->i = operation == TRILLIM_MOD ? remainder : quotient;
            }
            break;
        default: return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_binary_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimElementwiseTask *task = (const TrillimElementwiseTask *)opaque;
    const TrillimBinaryParameters *p = task->binary.p;
    if (task->binary.context->is_cancelled(task->binary.context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        TrillimValue a = trillim_load(task->binary.a, trillim_offset(index, p->rank, p->shape, p->a_strides), p->dtype);
        TrillimValue b = trillim_load(task->binary.b, trillim_offset(index, p->rank, p->shape, p->b_strides), p->dtype);
        TrillimValue result;
        TrillimStatus status = trillim_binary_value(task->operation, a, b, p->dtype, &result);
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->binary.output, index, p->dtype, result);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_binary_compute(
    TrillimElementwiseOperation operation, void *const *outputs,
    const void *const *inputs, const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimBinaryParameters *p = (const TrillimBinaryParameters *)parameters;
    TrillimElementwiseTask task = {{outputs[0], inputs[0], inputs[1], p, context}, operation};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_binary_task, &task);
}

static inline const TrillimParameterSchema *trillim_binary_schema(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimBinaryParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimBinaryParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimBinaryParameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(TrillimBinaryParameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(TrillimBinaryParameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimBinaryParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimBinaryParameters), TRILLIM_ALIGNOF(TrillimBinaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        trillim_binary_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

static inline TrillimStatus trillim_unary_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimUnaryParameters *p = (const TrillimUnaryParameters *)parameters;
    size_t bytes = trillim_itemsize(p->dtype);
    size_t span;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (!trillim_float(p->dtype) || !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > SIZE_MAX / bytes || p->count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->count != 0) {
        inputs[0] = span * bytes;
        outputs[0] = p->count * bytes;
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_unary_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimUnaryTask *task = (const TrillimUnaryTask *)opaque;
    const TrillimUnaryParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        TrillimValue value = trillim_load(task->input, trillim_offset(index, p->rank, p->shape, p->strides), p->dtype);
        switch (task->operation) {
        case TRILLIM_EXP: value.f = expf(value.f); break;
        case TRILLIM_LOG: value.f = logf(value.f); break;
        case TRILLIM_SQRT: value.f = sqrtf(value.f); break;
        case TRILLIM_SIN: value.f = sinf(value.f); break;
        case TRILLIM_COS: value.f = cosf(value.f); break;
        case TRILLIM_ERF: value.f = erff(value.f); break;
        default: return TRILLIM_ERROR;
        }
        trillim_store(task->output, index, p->dtype, value);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_unary_compute(
    TrillimElementwiseOperation operation, void *const *outputs,
    const void *const *inputs, const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimUnaryParameters *p = (const TrillimUnaryParameters *)parameters;
    TrillimUnaryTask task = {outputs[0], inputs[0], p, context, operation};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_unary_task, &task);
}

static inline const TrillimParameterSchema *trillim_unary_schema(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimUnaryParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimUnaryParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimUnaryParameters, shape), "rank"},
        {"strides", "size_t", offsetof(TrillimUnaryParameters, strides), "rank"},
        {"dtype", "uint32_t", offsetof(TrillimUnaryParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimUnaryParameters), TRILLIM_ALIGNOF(TrillimUnaryParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_unary_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

#endif
