#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"
#include <math.h>
#include <string.h>

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *strides;
    size_t source_offset;
    size_t source_rows;
    size_t source_columns;
    size_t output_rows;
    size_t output_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimCastParameters;

typedef struct {
    void *output;
    const void *input;
    float *scratch;
    const TrillimCastParameters *parameters;
    const TrillimKernelContext *context;
} TrillimCastTask;

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

static inline TrillimStatus trillim_cast_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimCastParameters *p = (const TrillimCastParameters *)parameters;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    size_t span;
    if (p->source_columns == 0 || p->output_columns == 0 ||
        p->source_rows > SIZE_MAX / p->source_columns ||
        p->output_rows > SIZE_MAX / p->output_columns ||
        p->count != p->output_rows * p->output_columns ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > p->source_rows * p->source_columns ||
        p->source_offset > p->source_rows * p->source_columns - span ||
        !trillim_packed_payload_size(p->source_rows, p->source_columns, p->input_dtype, &inputs[0]) ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->output_dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (trillim_itemsize(p->output_dtype) == 0) {
        if (p->count > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->count * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_cast_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimCastTask *task = (const TrillimCastTask *)opaque;
    const TrillimCastParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        size_t source_index = p->source_offset + trillim_offset(index, p->rank, p->shape, p->strides);
        if (trillim_itemsize(p->output_dtype) == 0) {
            float value = trillim_packed_read_float(task->input, source_index, p->input_dtype,
                                                    p->source_rows, p->source_columns);
            if (!isfinite(value)) {
                return TRILLIM_ERROR;
            }
            task->scratch[index] = value;
        } else {
            TrillimValue input;
            uint32_t input_dtype = p->input_dtype;
            if (trillim_itemsize(input_dtype) == 0) {
                input.f = trillim_packed_read_float(task->input, source_index, input_dtype, p->source_rows, p->source_columns);
                input_dtype = TRILLIM_FP32;
            } else {
                input = trillim_load(task->input, source_index, input_dtype);
            }
            TrillimValue output;
            TrillimStatus status = trillim_convert(input, input_dtype, p->output_dtype, &output);
            if (status != TRILLIM_OK) {
                return status;
            }
            trillim_store(task->output, index, p->output_dtype, output);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_cast(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimCastParameters *p = (const TrillimCastParameters *)parameters;
    TrillimCastTask task = {outputs[0], inputs[0], (float *)scratch, p, context};
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_cast_task, &task);
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->output_dtype) != 0) {
        return status;
    }
    return trillim_packed_encode_quant((const float *)scratch, p->output_rows, p->output_columns,
        p->output_dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_cast_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _cast as bind\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_cast_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimCastParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimCastParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimCastParameters, shape), "rank"},
        {"strides", "size_t", offsetof(TrillimCastParameters, strides), "rank"},
        {"source_offset", "size_t", offsetof(TrillimCastParameters, source_offset), NULL},
        {"source_rows", "size_t", offsetof(TrillimCastParameters, source_rows), NULL},
        {"source_columns", "size_t", offsetof(TrillimCastParameters, source_columns), NULL},
        {"output_rows", "size_t", offsetof(TrillimCastParameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(TrillimCastParameters, output_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimCastParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimCastParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimCastParameters), TRILLIM_ALIGNOF(TrillimCastParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1, trillim_cast_requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_cast_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
