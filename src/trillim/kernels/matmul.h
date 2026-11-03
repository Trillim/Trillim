#ifndef TRILLIM_KERNEL_MATMUL_H
#define TRILLIM_KERNEL_MATMUL_H

#include "codec.h"

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const TrillimMatmulParameters *p;
    const TrillimKernelContext *context;
} TrillimMatmulTask;

static inline bool trillim_matrix_span(size_t base, size_t rows, size_t columns, size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - base) / row_stride) {
        return false;
    }
    base += (rows - 1) * row_stride;
    if (col_stride != 0 && columns - 1 > (SIZE_MAX - base) / col_stride) {
        return false;
    }
    return base + (columns - 1) * col_stride <= capacity;
}

static inline TrillimStatus trillim_matmul_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    size_t bytes = trillim_itemsize(p->out_dtype);
    size_t a_span, b_span;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (bytes == 0 || !trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span)) {
        return TRILLIM_ERROR;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if (p->a_storage_columns == 0 || p->b_storage_columns == 0 ||
            p->a_storage_rows > SIZE_MAX / p->a_storage_columns ||
            p->b_storage_rows > SIZE_MAX / p->b_storage_columns ||
            !trillim_matrix_span(a_span, p->m, p->k, p->a_row_stride, p->a_col_stride, p->a_storage_rows * p->a_storage_columns) ||
            !trillim_matrix_span(b_span, p->k, p->n, p->b_row_stride, p->b_col_stride, p->b_storage_rows * p->b_storage_columns) ||
            !trillim_payload_size(p->a_storage_rows, p->a_storage_columns, p->a_dtype, &inputs[0]) ||
            !trillim_payload_size(p->b_storage_rows, p->b_storage_columns, p->b_dtype, &inputs[1])) {
            return TRILLIM_ERROR;
        }
    }
    outputs[0] = p->batches * p->m * p->n * bytes;
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_matmul_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimMatmulTask *task = (const TrillimMatmulTask *)opaque;
    const TrillimMatmulParameters *p = task->p;
    size_t batch = index / p->m;
    size_t row = index % p->m;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    for (size_t column = 0; column < p->n; ++column) {
        TrillimValue sum;
        if (trillim_float(p->out_dtype)) {
            sum.f = 0.0f;
        } else {
            sum.u = 0;
        }
        for (size_t start = 0; start < p->k; start += 1024) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t end = p->k - start < 1024 ? p->k : start + 1024;
            for (size_t k = start; k < end; ++k) {
                size_t a_index = a_base + k * p->a_col_stride;
                size_t b_index = b_base + k * p->b_row_stride + column * p->b_col_stride;
                if (trillim_float(p->out_dtype)) {
                    float a = trillim_read_float(task->a, a_index, p->a_dtype, p->a_storage_rows, p->a_storage_columns);
                    float b = trillim_read_float(task->b, b_index, p->b_dtype, p->b_storage_rows, p->b_storage_columns);
                    sum.f += a * b;
                } else {
                    TrillimValue a = trillim_load(task->a, a_index, p->a_dtype);
                    TrillimValue b = trillim_load(task->b, b_index, p->b_dtype);
                    sum.u += a.u * b.u;
                }
            }
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_store(task->output, (batch * p->m + row) * p->n + column, p->out_dtype, sum);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_matmul_compute(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    TrillimMatmulTask task = {outputs[0], inputs[0], inputs[1], p, context};
    return context->thread_pool->run(context->thread_pool->context, p->batches * p->m, trillim_matmul_task, &task);
}


static inline const TrillimParameterSchema *trillim_matmul_schema(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(TrillimMatmulParameters, batches), NULL},
        {"m", "size_t", offsetof(TrillimMatmulParameters, m), NULL},
        {"n", "size_t", offsetof(TrillimMatmulParameters, n), NULL},
        {"k", "size_t", offsetof(TrillimMatmulParameters, k), NULL},
        {"rank", "size_t", offsetof(TrillimMatmulParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(TrillimMatmulParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(TrillimMatmulParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(TrillimMatmulParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(TrillimMatmulParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(TrillimMatmulParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(TrillimMatmulParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(TrillimMatmulParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(TrillimMatmulParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(TrillimMatmulParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(TrillimMatmulParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(TrillimMatmulParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(TrillimMatmulParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(TrillimMatmulParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(TrillimMatmulParameters, b_storage_columns), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimMatmulParameters), TRILLIM_ALIGNOF(TrillimMatmulParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, trillim_matmul_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}


#endif
