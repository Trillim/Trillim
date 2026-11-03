#ifndef TRILLIM_KERNEL_INDEXING_H
#define TRILLIM_KERNEL_INDEXING_H

#include "cast.h"
#include "elementwise.h"

typedef struct {
    size_t count;
    size_t items;
    size_t coordinates;
    size_t index_rank;
    size_t slice_rank;
    size_t slice_size;
    const size_t *index_shape;
    const size_t *index_strides;
    const size_t *mask_strides;
    size_t coordinate_stride;
    const size_t *indexed_shape;
    const size_t *indexed_strides;
    const size_t *output_indexed_strides;
    const size_t *slice_shape;
    const size_t *slice_strides;
    size_t values_rank;
    const size_t *values_shape;
    const size_t *values_strides;
    size_t storage_rows;
    size_t storage_columns;
    size_t output_rows;
    size_t output_columns;
    uint32_t dtype;
    uint32_t masked;
    uint32_t mode;
    uint64_t fill_bits;
    float fill_float;
} TrillimIndexParameters;

typedef struct {
    void *output;
    const void *const *inputs;
    float *scratch;
    const TrillimIndexParameters *parameters;
    const TrillimKernelContext *context;
    bool scatter;
} TrillimIndexTask;

static inline bool trillim_index_table_size(size_t items, size_t *capacity) {
    size_t result = 1;
    if (items > SIZE_MAX / 2) {
        return false;
    }
    while (result < items * 2) {
        if (result > SIZE_MAX / 2) {
            return false;
        }
        result *= 2;
    }
    *capacity = result;
    return result <= SIZE_MAX / sizeof(size_t);
}

static inline TrillimStatus trillim_index_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch,
    bool scatter, bool masked
) {
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    size_t count = scatter ? (masked ? 4 : 3) : (masked ? 3 : 2);
    for (size_t i = 0; i < count; ++i) {
        inputs[i] = 0;
    }
    outputs[0] = 0;
    *scratch = 0;
    if (p->masked != (uint32_t)masked || p->mode > 1 ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->dtype, &inputs[0]) ||
        !trillim_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->items != 0) {
        size_t span;
        if (!trillim_span(p->index_rank, p->index_shape, p->index_strides, &span)) {
            return TRILLIM_ERROR;
        }
        if (p->coordinates != 0) {
            if (p->coordinate_stride != 0 && p->coordinates - 1 > (SIZE_MAX - span) / p->coordinate_stride) {
                return TRILLIM_ERROR;
            }
            span += (p->coordinates - 1) * p->coordinate_stride;
            if (span > SIZE_MAX / sizeof(int64_t)) {
                return TRILLIM_ERROR;
            }
            inputs[1] = span * sizeof(int64_t);
        }
        if (masked) {
            if (!trillim_span(p->index_rank, p->index_shape, p->mask_strides, &span)) {
                return TRILLIM_ERROR;
            }
            inputs[scatter ? 3 : 2] = span;
        }
        if (scatter && p->slice_size != 0) {
            size_t bytes = trillim_itemsize(p->dtype);
            if (bytes == 0 || !trillim_span(p->values_rank, p->values_shape, p->values_strides, &span) || span > SIZE_MAX / bytes) {
                return TRILLIM_ERROR;
            }
            inputs[2] = span * bytes;
        }
    }
    if (scatter && p->mode == 0 && p->items != 0) {
        size_t capacity;
        if (!trillim_index_table_size(p->items, &capacity)) {
            return TRILLIM_ERROR;
        }
        *scratch = capacity * sizeof(size_t);
    }
    if (trillim_itemsize(p->dtype) == 0) {
        if (p->count > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->count * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_coordinates(
    const TrillimIndexTask *task, size_t item, size_t *source, size_t *destination, size_t *key, bool *active
) {
    const TrillimIndexParameters *p = task->parameters;
    *source = 0;
    *destination = 0;
    *key = 0;
    *active = true;
    if (p->masked != 0) {
        size_t index = trillim_offset(item, p->index_rank, p->index_shape, p->mask_strides);
        if (((const uint8_t *)task->inputs[task->scatter ? 3 : 2])[index] == 0) {
            *active = false;
            return TRILLIM_OK;
        }
    }
    size_t base = trillim_offset(item, p->index_rank, p->index_shape, p->index_strides);
    for (size_t axis = 0; axis < p->coordinates; ++axis) {
        if ((axis & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t coordinate;
        memcpy(&coordinate, (const uint8_t *)task->inputs[1] + (base + axis * p->coordinate_stride) * sizeof(int64_t), sizeof(coordinate));
        if (coordinate < 0 || (uint64_t)coordinate >= p->indexed_shape[axis]) {
            return TRILLIM_ERROR;
        }
        if (p->indexed_shape[axis] != 0 && *key > (SIZE_MAX - (size_t)coordinate) / p->indexed_shape[axis]) {
            return TRILLIM_ERROR;
        }
        *key = *key * p->indexed_shape[axis] + (size_t)coordinate;
        *source += (size_t)coordinate * p->indexed_strides[axis];
        *destination += (size_t)coordinate * p->output_indexed_strides[axis];
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_gather_task(void *opaque, size_t item, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = (const TrillimIndexTask *)opaque;
    const TrillimIndexParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t source, destination, key;
    bool active;
    TrillimStatus status = trillim_coordinates(task, item, &source, &destination, &key, &active);
    if (status != TRILLIM_OK) {
        return status;
    }
    for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->slice_size - begin < 1024 ? p->slice_size : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            size_t output_index = item * p->slice_size + i;
            size_t input_index = source + trillim_offset(i, p->slice_rank, p->slice_shape, p->slice_strides);
            if (trillim_itemsize(p->dtype) == 0) {
                task->scratch[output_index] = active ?
                    trillim_read_float(task->inputs[0], input_index, p->dtype, p->storage_rows, p->storage_columns) : p->fill_float;
            } else {
                TrillimValue value;
                if (active) {
                    value = trillim_load(task->inputs[0], input_index, p->dtype);
                } else if (trillim_float(p->dtype)) {
                    value.f = p->fill_float;
                } else {
                    value.u = p->fill_bits;
                }
                trillim_store(task->output, output_index, p->dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_index_compute(
    bool scatter, void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    TrillimIndexTask task = {outputs[0], inputs, (float *)scratch, p, context, scatter};
    TrillimStatus status;
    if (!scatter) {
        status = context->thread_pool->run(context->thread_pool->context, p->items, trillim_gather_task, &task);
    } else {
        if (p->slice_size != 0) {
            for (size_t begin = 0; begin < p->count; begin += 1024) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
                for (size_t i = begin; i < end; ++i) {
                    size_t source = trillim_offset(i / p->slice_size, p->coordinates, p->indexed_shape, p->indexed_strides) +
                        trillim_offset(i % p->slice_size, p->slice_rank, p->slice_shape, p->slice_strides);
                    TrillimValue value = trillim_load(inputs[0], source, p->dtype);
                    trillim_store(outputs[0], i, p->dtype, value);
                }
            }
        }
        size_t table_capacity = 0;
        size_t *table = (size_t *)scratch;
        if (p->mode == 0 && p->items != 0) {
            if (!trillim_index_table_size(p->items, &table_capacity)) {
                return TRILLIM_ERROR;
            }
            for (size_t begin = 0; begin < table_capacity; begin += 128) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t count = table_capacity - begin < 128 ? table_capacity - begin : 128;
                memset(table + begin, 255, count * sizeof(size_t));
            }
        }
        for (size_t item = 0; item < p->items; ++item) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t source, destination, key;
            bool active;
            status = trillim_coordinates(&task, item, &source, &destination, &key, &active);
            if (status != TRILLIM_OK) {
                return status;
            }
            if (!active) {
                continue;
            }
            if (p->mode == 0) {
                if (key == SIZE_MAX) {
                    return TRILLIM_ERROR;
                }
                size_t slot = (key * (size_t)UINT64_C(0x9e3779b97f4a7c15)) & (table_capacity - 1);
                size_t probes = 0;
                while (table[slot] != SIZE_MAX) {
                    if ((probes++ & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    if (table[slot] == key) {
                        return TRILLIM_ERROR;
                    }
                    slot = (slot + 1) & (table_capacity - 1);
                }
                table[slot] = key;
            }
            for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = p->slice_size - begin < 1024 ? p->slice_size : begin + 1024;
                for (size_t i = begin; i < end; ++i) {
                    size_t index = trillim_offset(item * p->slice_size + i, p->values_rank, p->values_shape, p->values_strides);
                    TrillimValue value = trillim_load(inputs[2], index, p->dtype);
                    if (p->mode != 0) {
                        TrillimValue previous = trillim_load(outputs[0], destination + i, p->dtype);
                        status = trillim_binary_value(TRILLIM_ADD, previous, value, p->dtype, &value);
                        if (status != TRILLIM_OK) {
                            return status;
                        }
                    }
                    trillim_store(outputs[0], destination + i, p->dtype, value);
                }
            }
        }
        status = TRILLIM_OK;
    }
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->dtype) != 0) {
        return status;
    }
    return trillim_encode_quant((const float *)scratch, p->output_rows, p->output_columns, p->dtype, outputs[0], context);
}

#endif
