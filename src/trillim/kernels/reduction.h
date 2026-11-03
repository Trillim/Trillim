#ifndef TRILLIM_KERNEL_REDUCTION_H
#define TRILLIM_KERNEL_REDUCTION_H

#include "cast.h"

typedef struct {
    size_t outputs;
    size_t output_rank;
    size_t reduce_rank;
    size_t reduce_count;
    const size_t *output_shape;
    const size_t *output_strides;
    const size_t *reduce_shape;
    const size_t *reduce_strides;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimReductionParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimReductionParameters *parameters;
    const TrillimKernelContext *context;
    uint32_t operation;
} TrillimReductionTask;

enum { TRILLIM_SUM, TRILLIM_MAX, TRILLIM_ARGMAX };

static inline TrillimStatus trillim_reduction_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimReductionParameters *p = (const TrillimReductionParameters *)parameters;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t outer_span, inner_span;
    if (bytes == 0 || !trillim_span(p->output_rank, p->output_shape, p->output_strides, &outer_span) ||
        !trillim_span(p->reduce_rank, p->reduce_shape, p->reduce_strides, &inner_span) ||
        p->outputs > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->outputs != 0) {
        outputs[0] = p->outputs * bytes;
        if (p->reduce_count != 0 && !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
            return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

static inline bool trillim_greater_equal(TrillimValue a, TrillimValue b, uint32_t dtype) {
    if (trillim_float(dtype)) {
        return a.f >= b.f;
    }
    return trillim_unsigned(dtype) ? a.u >= b.u : a.i >= b.i;
}

static inline TrillimStatus trillim_reduction_task(void *opaque, size_t output_index, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = (const TrillimReductionTask *)opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t base = trillim_offset(output_index, p->output_rank, p->output_shape, p->output_strides);
    TrillimValue accumulator;
    accumulator.u = 0;
    if (trillim_float(p->output_dtype)) {
        accumulator.f = 0.0f;
    }
    int64_t selected = INT64_MAX;
    bool valid = false;
    uint32_t compare_dtype = trillim_itemsize(p->input_dtype) == 0 ? TRILLIM_FP32 : p->input_dtype;
    for (size_t start = 0; start < p->reduce_count; start += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->reduce_count - start < 1024 ? p->reduce_count : start + 1024;
        for (size_t index = start; index < end; ++index) {
            size_t source = base + trillim_offset(index, p->reduce_rank, p->reduce_shape, p->reduce_strides);
            TrillimValue value;
            if (trillim_itemsize(p->input_dtype) == 0) {
                value.f = trillim_read_float(task->input, source, p->input_dtype, p->storage_rows, p->storage_columns);
            } else {
                value = trillim_load(task->input, source, p->input_dtype);
            }
            if (task->operation == TRILLIM_SUM) {
                TrillimValue converted;
                TrillimStatus status = trillim_convert(value, compare_dtype, p->output_dtype, &converted);
                if (status != TRILLIM_OK) {
                    return status;
                }
                if (trillim_float(p->output_dtype)) {
                    accumulator.f += converted.f;
                } else {
                    accumulator.u += converted.u;
                }
            } else if (!valid || !trillim_greater_equal(accumulator, value, compare_dtype)) {
                accumulator = value;
                selected = (int64_t)index;
            }
            valid = true;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (task->operation == TRILLIM_ARGMAX) {
        accumulator.i = selected;
    } else if (task->operation == TRILLIM_MAX && valid) {
        TrillimValue converted;
        TrillimStatus status = trillim_convert(accumulator, compare_dtype, p->output_dtype, &converted);
        if (status != TRILLIM_OK) {
            return status;
        }
        accumulator = converted;
    }
    trillim_store(task->output, output_index, p->output_dtype, accumulator);
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_reduction_compute(
    uint32_t operation, void *const *outputs, const void *const *inputs,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimReductionParameters *p = (const TrillimReductionParameters *)parameters;
    TrillimReductionTask task = {outputs[0], inputs[0], p, context, operation};
    return context->thread_pool->run(context->thread_pool->context, p->outputs, trillim_reduction_task, &task);
}

static inline const TrillimParameterSchema *trillim_reduction_schema(void) {
    static const TrillimParameter fields[] = {
        {"outputs", "size_t", offsetof(TrillimReductionParameters, outputs), NULL},
        {"output_rank", "size_t", offsetof(TrillimReductionParameters, output_rank), NULL},
        {"reduce_rank", "size_t", offsetof(TrillimReductionParameters, reduce_rank), NULL},
        {"reduce_count", "size_t", offsetof(TrillimReductionParameters, reduce_count), NULL},
        {"output_shape", "size_t", offsetof(TrillimReductionParameters, output_shape), "output_rank"},
        {"output_strides", "size_t", offsetof(TrillimReductionParameters, output_strides), "output_rank"},
        {"reduce_shape", "size_t", offsetof(TrillimReductionParameters, reduce_shape), "reduce_rank"},
        {"reduce_strides", "size_t", offsetof(TrillimReductionParameters, reduce_strides), "reduce_rank"},
        {"storage_rows", "size_t", offsetof(TrillimReductionParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimReductionParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimReductionParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimReductionParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimReductionParameters), TRILLIM_ALIGNOF(TrillimReductionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_reduction_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

#endif
