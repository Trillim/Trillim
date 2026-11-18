#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"
#include <math.h>
#include <string.h>

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
        !trillim_packed_payload_size(p->storage_rows, p->storage_columns, p->dtype, &inputs[0]) ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
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
        if (((const uint8_t *)task->inputs[2])[index] == 0) {
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
    if (active && p->dtype > TRILLIM_BOOL && p->dtype != TRILLIM_FP16 &&
        p->dtype != TRILLIM_BF16 && trillim_itemsize(p->dtype) &&
        trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides)) {
        size_t bytes = trillim_itemsize(p->dtype);
        for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->slice_size - begin < 1024 ? p->slice_size - begin : 1024;
            memcpy((uint8_t *)task->output + (item * p->slice_size + begin) * bytes,
                   (const uint8_t *)task->inputs[0] + (source + begin) * bytes, count * bytes);
        }
        return TRILLIM_OK;
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
                float value = active ?
                    trillim_packed_read_float(task->inputs[0], input_index, p->dtype, p->storage_rows, p->storage_columns) : p->fill_float;
                if (!isfinite(value)) {
                    return TRILLIM_ERROR;
                }
                task->scratch[output_index] = value;
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

static size_t gather_batch_size(const TrillimIndexParameters *p) {
    return p->slice_size && p->slice_size < 1024 ? 1024 / p->slice_size : 1;
}

static TrillimStatus gather_batch(void *opaque, size_t batch, size_t worker) {
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    size_t size = gather_batch_size(p), first = batch * size;
    size_t end = p->items - first < size ? p->items : first + size;
    for (size_t item = first; item < end; ++item) {
        TrillimStatus status = trillim_gather_task(opaque, item, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static bool trillim_mxfp4_copyable(const TrillimIndexParameters *p) {
    if (p->dtype != TRILLIM_MXFP4 || p->masked || p->coordinates != 1 ||
        !p->storage_columns || p->storage_columns % 32 || p->output_columns % 32 ||
        p->slice_size != p->storage_columns || p->indexed_shape[0] != p->storage_rows ||
        p->indexed_strides[0] != p->storage_columns ||
        !trillim_contiguous(p->index_rank, p->index_shape, p->index_strides) ||
        !trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides) ||
        p->items > SIZE_MAX / p->slice_size || p->items * p->slice_size != p->count ||
        (p->output_columns && p->output_rows > SIZE_MAX / p->output_columns) ||
        p->output_rows * p->output_columns != p->count) {
        return false;
    }
    size_t items = 1;
    for (size_t axis = 0; axis < p->index_rank; ++axis) {
        if (p->index_shape[axis] && items > SIZE_MAX / p->index_shape[axis]) {
            return false;
        }
        items *= p->index_shape[axis];
    }
    return items == p->items;
}

static TrillimStatus trillim_mxfp4_copy_slice(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    const uint8_t *input = task->inputs[0];
    const uint8_t *indices = task->inputs[1];
    uint8_t *output = task->output;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t groups = p->storage_columns / 32;
    size_t tiles = groups / 32768 + (groups % 32768 != 0);
    size_t item = tile / tiles;
    size_t first = tile % tiles * 32768;
    size_t count = groups - first < 32768 ? groups - first : 32768;
    int64_t selected;
    memcpy(&selected, indices + item * sizeof(selected), sizeof(selected));
    size_t source = (size_t)selected * groups + first;
    size_t destination = item * groups + first;
    const uint8_t *scales = input + p->storage_rows * groups * 16 + source;
    unsigned exceptional = 0;
    for (size_t i = 0; i < count; ++i) {
        exceptional |= scales[i] >= 253;
    }
    if (exceptional) {
        for (size_t i = 0; i < count; ++i) {
            if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t exponent = scales[i];
            if (exponent == 255) {
                return TRILLIM_ERROR;
            }
            if (exponent < 253) {
                continue;
            }
            uint8_t limit = exponent == 253 ? 6 : 4;
            const uint8_t *codes = input + (source + i) * 16;
            for (size_t j = 0; j < 16; ++j) {
                if ((codes[j] & 7) >= limit || ((codes[j] >> 4) & 7) >= limit) {
                    return TRILLIM_ERROR;
                }
            }
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    memcpy(output + destination * 16, input + source * 16, count * 16);
    memcpy(output + p->items * groups * 16 + destination, scales, count);
    return TRILLIM_OK;
}

static TrillimStatus trillim_mxfp4_gather(TrillimIndexTask *task) {
    const TrillimIndexParameters *p = task->parameters;
    for (size_t item = 0; item < p->items; ++item) {
        if ((item & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t selected;
        memcpy(&selected, (const uint8_t *)task->inputs[1] + item * sizeof(selected), sizeof(selected));
        if (selected < 0 || (uint64_t)selected >= p->storage_rows) {
            return TRILLIM_ERROR;
        }
    }
    size_t groups = p->storage_columns / 32;
    size_t tiles = groups / 32768 + (groups % 32768 != 0);
    size_t tasks = p->items * tiles;
    return tasks == 1 ? trillim_mxfp4_copy_slice(task, 0, 0)
                      : task->context->thread_pool->run(task->context->thread_pool->context,
                                                       tasks, trillim_mxfp4_copy_slice, task);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    TrillimStatus status = trillim_index_requirements(parameters, inputs, outputs, scratch, false, false);
    if (status == TRILLIM_OK && trillim_mxfp4_copyable(parameters)) {
        *scratch = 0;
    }
    return status;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_gather(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    TrillimIndexTask task = {outputs[0], inputs, (float *)scratch, p, context};
    if (trillim_mxfp4_copyable(p)) {
        return trillim_mxfp4_gather(&task);
    }
    size_t size = gather_batch_size(p);
    size_t batches = p->items / size + (p->items % size != 0);
    TrillimStatus status = batches == 1 ? gather_batch(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, batches, gather_batch, &task);
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->dtype) != 0) {
        return status;
    }
    return trillim_packed_encode_quant((const float *)scratch, p->output_rows, p->output_columns, p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_gather_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _gather as bind\n"
           "physical_inputs = ('x', 'indices')\n"
           "def pattern(x: Tensor, indices: Tensor, *, fill = 0) -> tuple[Tensor, ...]:\n"
           "    return (Gather(x, indices=indices, fill=fill),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gather_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimIndexParameters, count), NULL},
        {"items", "size_t", offsetof(TrillimIndexParameters, items), NULL},
        {"coordinates", "size_t", offsetof(TrillimIndexParameters, coordinates), NULL},
        {"index_rank", "size_t", offsetof(TrillimIndexParameters, index_rank), NULL},
        {"slice_rank", "size_t", offsetof(TrillimIndexParameters, slice_rank), NULL},
        {"slice_size", "size_t", offsetof(TrillimIndexParameters, slice_size), NULL},
        {"index_shape", "size_t", offsetof(TrillimIndexParameters, index_shape), "index_rank"},
        {"index_strides", "size_t", offsetof(TrillimIndexParameters, index_strides), "index_rank"},
        {"mask_strides", "size_t", offsetof(TrillimIndexParameters, mask_strides), "index_rank"},
        {"coordinate_stride", "size_t", offsetof(TrillimIndexParameters, coordinate_stride), NULL},
        {"indexed_shape", "size_t", offsetof(TrillimIndexParameters, indexed_shape), "coordinates"},
        {"indexed_strides", "size_t", offsetof(TrillimIndexParameters, indexed_strides), "coordinates"},
        {"output_indexed_strides", "size_t", offsetof(TrillimIndexParameters, output_indexed_strides), "coordinates"},
        {"slice_shape", "size_t", offsetof(TrillimIndexParameters, slice_shape), "slice_rank"},
        {"slice_strides", "size_t", offsetof(TrillimIndexParameters, slice_strides), "slice_rank"},
        {"values_rank", "size_t", offsetof(TrillimIndexParameters, values_rank), NULL},
        {"values_shape", "size_t", offsetof(TrillimIndexParameters, values_shape), "values_rank"},
        {"values_strides", "size_t", offsetof(TrillimIndexParameters, values_strides), "values_rank"},
        {"storage_rows", "size_t", offsetof(TrillimIndexParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimIndexParameters, storage_columns), NULL},
        {"output_rows", "size_t", offsetof(TrillimIndexParameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(TrillimIndexParameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(TrillimIndexParameters, dtype), NULL},
        {"masked", "uint32_t", offsetof(TrillimIndexParameters, masked), NULL},
        {"mode", "uint32_t", offsetof(TrillimIndexParameters, mode), NULL},
        {"fill_bits", "uint64_t", offsetof(TrillimIndexParameters, fill_bits), NULL},
        {"fill_float", "float", offsetof(TrillimIndexParameters, fill_float), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimIndexParameters), TRILLIM_ALIGNOF(TrillimIndexParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(size_t)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gather_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
