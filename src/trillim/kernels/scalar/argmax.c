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

static inline float trillim_read_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (trillim_itemsize(dtype) != 0) {
        TrillimValue value = trillim_load(data, index, dtype);
        if (trillim_float(dtype)) {
            return value.f;
        }
        return trillim_unsigned(dtype) ? (float)value.u : (float)value.i;
    }
    const uint8_t *values = (const uint8_t *)data;
    size_t row = index / columns;
    size_t column = index % columns;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        int8_t code;
        memcpy(&bits, values + rows * columns + row * 2, 2);
        memcpy(&code, values + index, 1);
        return (float)code * trillim_bf16(bits);
    }
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_BINARY ||
          dtype == TRILLIM_GRP_TERNARY) ? 128 : 32);
    size_t groups = columns / group + (columns % group != 0);
    size_t total = rows * groups;
    size_t block = row * groups + column / group;
    size_t offset = column % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        size_t code = (values[block * 32 + offset % 32] >> (6 - (offset / 32) * 2)) & 3;
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
        return code == 3 ? NAN : (float)((int32_t)code - 1) * scale;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        uint8_t code = (values[block * 16 + offset / 2] >> ((offset % 2) * 4)) & 15;
        uint8_t exponent = values[total * 16 + block];
        float value = magnitudes[code & 7];
        if ((code & 8) != 0) {
            value = -value;
        }
        return exponent == 255 ? NAN : ldexpf(value, (int)exponent - 127);
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 :
        (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    uint16_t bits;
    memcpy(&bits, values + scale_start + block * 2, 2);
    float scale = trillim_bf16(bits);
    if (dtype == TRILLIM_GRP_INT8) {
        int8_t code;
        memcpy(&code, values + block * 32 + offset, 1);
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        int32_t code = ((values[block * 16 + offset / 8] >> (7 - offset % 8)) & 1) ? 1 : -1;
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t code = (values[block * 16 + offset % 16] >> ((offset / 16) * 4)) & 15;
        if (dtype == TRILLIM_GRP_INT5) {
            code |= ((values[scale_start + total * 2 + block * 4 + offset / 8] >> (offset % 8)) & 1) << 4;
        }
        return (float)(code - (dtype == TRILLIM_GRP_INT4 ? 8 : 16)) * scale;
    }
    size_t half = offset / 128;
    size_t lane = (offset % 128) / 32;
    size_t low = block * 128 + half * 64 + (lane / 2) * 32 + offset % 32;
    size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
    int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
    code |= ((values[high] >> (lane * 2)) & 3) << 4;
    int8_t subscale;
    memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
    return (float)((code - 32) * subscale) * scale;
}

static inline TrillimStatus trillim_convert(TrillimValue input, uint32_t from, uint32_t to, TrillimValue *output) {
    if (trillim_float(to)) {
        output->f = trillim_float(from) ? input.f :
            (trillim_unsigned(from) ? (float)input.u : (float)input.i);
    } else if (to == TRILLIM_BOOL) {
        output->u = trillim_float(from) ? input.f != 0.0f : input.u != 0;
    } else if (trillim_float(from)) {
        if (!isfinite(input.f)) {
            return TRILLIM_ERROR;
        }
        size_t bits = trillim_itemsize(to) * 8;
        long double modulus = ldexpl(1.0L, (int)bits);
        long double value = fmodl(truncl((long double)input.f), modulus);
        if (value < 0.0L) {
            value += modulus;
        }
        output->u = (uint64_t)value;
    } else {
        output->u = input.u;
    }
    return TRILLIM_OK;
}

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

TRILLIM_KERNEL TrillimStatus trillim_scalar_argmax(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimReductionParameters *p = (const TrillimReductionParameters *)parameters;
    TrillimReductionTask task = {outputs[0], inputs[0], p, context, TRILLIM_ARGMAX};
    return context->thread_pool->run(context->thread_pool->context, p->outputs, trillim_reduction_task, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_argmax_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _argmax as bind\n"
           "def pattern(x: Tensor, *, axis: int = -1, rank: int) -> tuple[Tensor, ...]:\n"
           "    return (Argmax(x, axis=axis, rank=rank),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_argmax_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_argmax_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
