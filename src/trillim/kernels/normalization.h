#ifndef TRILLIM_KERNEL_NORMALIZATION_H
#define TRILLIM_KERNEL_NORMALIZATION_H

#include "codec.h"

typedef struct {
    size_t rows;
    size_t width;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    float eps;
    float weight_offset;
} TrillimRMSNormParameters;

typedef struct {
    void *output;
    const void *input;
    const void *weight;
    const TrillimRMSNormParameters *parameters;
    const TrillimKernelContext *context;
    bool weighted;
} TrillimRMSNormTask;

static inline TrillimStatus trillim_rms_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch, bool weighted
) {
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    inputs[0] = 0;
    if (weighted) {
        inputs[1] = 0;
    }
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (!trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width || p->rows * p->width > SIZE_MAX / bytes ||
        p->storage_columns == 0 || p->storage_rows > SIZE_MAX / p->storage_columns ||
        p->rows * p->width > p->storage_rows * p->storage_columns ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (weighted && !trillim_payload_size(1, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * bytes;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_rms_weighted_requirements(const void *p, size_t *inputs, size_t *outputs, size_t *scratch) {
    return trillim_rms_requirements(p, inputs, outputs, scratch, true);
}

static inline TrillimStatus trillim_rms_plain_requirements(const void *p, size_t *inputs, size_t *outputs, size_t *scratch) {
    return trillim_rms_requirements(p, inputs, outputs, scratch, false);
}

static inline TrillimStatus trillim_rms_task(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = (const TrillimRMSNormTask *)opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    float sum = 0.0f;
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t column = begin; column < end; ++column) {
            float value = trillim_read_float(task->input, row * p->width + column,
                p->input_dtype, p->storage_rows, p->storage_columns);
            sum += value * value;
        }
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    for (size_t begin = 0; begin < p->width; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - begin < 1024 ? p->width : begin + 1024;
        for (size_t column = begin; column < end; ++column) {
            TrillimValue value;
            value.f = trillim_read_float(task->input, row * p->width + column,
                p->input_dtype, p->storage_rows, p->storage_columns) / denominator;
            if (task->weighted) {
                float weight = trillim_read_float(task->weight, column, p->weight_dtype, 1, p->width);
                value.f *= weight + p->weight_offset;
            }
            trillim_store(task->output, row * p->width + column, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_rms_compute(
    bool weighted, void *const *outputs, const void *const *inputs,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], weighted ? inputs[1] : NULL, p, context, weighted};
    return context->thread_pool->run(context->thread_pool->context, p->rows, trillim_rms_task, &task);
}

static inline const TrillimParameterSchema *trillim_rms_schema(bool weighted) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(TrillimRMSNormParameters, rows), NULL},
        {"width", "size_t", offsetof(TrillimRMSNormParameters, width), NULL},
        {"storage_rows", "size_t", offsetof(TrillimRMSNormParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimRMSNormParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, output_dtype), NULL},
        {"eps", "float", offsetof(TrillimRMSNormParameters, eps), NULL},
        {"weight_offset", "float", offsetof(TrillimRMSNormParameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema weighted_schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, trillim_rms_weighted_requirements,
        NULL, inputs, outputs, 1
    };
    static const TrillimParameterSchema plain_schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1, trillim_rms_plain_requirements,
        NULL, inputs, outputs, 1
    };
    return weighted ? &weighted_schema : &plain_schema;
}

#endif
