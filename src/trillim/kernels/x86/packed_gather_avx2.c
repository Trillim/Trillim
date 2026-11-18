#define TRILLIM_ENABLE_PACKED_AVX2
#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"

typedef struct {
    size_t rows;
    size_t width;
    size_t tokens;
    size_t output_rows;
    size_t output_columns;
    uint32_t dtype;
    uint32_t index_dtype;
} Parameters;

typedef struct {
    float *decoded;
    const uint8_t *weight;
    const void *indices;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static bool index_dtype(uint32_t dtype) {
    return dtype == TRILLIM_INT32 || dtype == TRILLIM_UINT32 ||
           dtype == TRILLIM_INT64 || dtype == TRILLIM_UINT64;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0 ||
        p->dtype < TRILLIM_ROW_INT8 || p->dtype > TRILLIM_MXFP4 ||
        !index_dtype(p->index_dtype) ||
        !trillim_packed_payload_size(p->rows, p->width, p->dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t id_size = trillim_itemsize(p->index_dtype);
    if (p->tokens > SIZE_MAX / id_size || p->tokens > SIZE_MAX / sizeof(float) / p->width ||
        !p->output_columns || p->output_rows > SIZE_MAX / p->output_columns ||
        p->output_rows * p->output_columns != p->tokens * p->width ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->tokens * id_size;
    *scratch = p->tokens * p->width * sizeof(float);
    return TRILLIM_OK;
}

static size_t tiles_per_row(size_t width) {
    return width / 1024 + (width % 1024 != 0);
}

__attribute__((target("avx2"))) static TrillimStatus gather(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = tiles_per_row(p->width);
    size_t token = tile / tiles;
    size_t column = (tile % tiles) * 1024;
    size_t end = p->width - column < 1024 ? p->width : column + 1024;
    TrillimValue coordinate = trillim_load(task->indices, token, p->index_dtype);
    uint64_t row = trillim_unsigned(p->index_dtype) ? coordinate.u : (uint64_t)coordinate.i;
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t source = (size_t)row * p->width;
    size_t destination = token * p->width;
    for (; end - column >= 8; column += 8) {
        __m256 values = trillim_packed_decode8(task->weight, p->rows, p->width, p->dtype,
                                               (size_t)row, column);
        if (!trillim_packed_finite8(values)) {
            return TRILLIM_ERROR;
        }
        _mm256_storeu_ps(task->decoded + destination + column, values);
    }
    for (; column < end; ++column) {
        float value = trillim_packed_read_float(task->weight, source + column, p->dtype, p->rows, p->width);
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        task->decoded[destination + column] = value;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_gather_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    Task task = {scratch, inputs[0], inputs[1], p, context};
    size_t tiles = p->tokens * tiles_per_row(p->width);
    TrillimStatus status = tiles == 1 ? gather(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, gather, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    return trillim_packed_encode_quant(task.decoded, p->output_rows, p->output_columns,
                                      p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_x86_packed_gather_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'indices')\n"
           "def bind(context):\n"
           "    weight, indices = context.inputs['x'], context.inputs['indices']\n"
           "    output = context.outputs[0]\n"
           "    if _DTYPE_IDS[weight.dtype] < _DTYPE_IDS['row_int8'] or output.dtype != weight.dtype:\n"
           "        raise ValueError('packed gather requires the same packed input and output dtype')\n"
           "    if not weight.shape:\n"
           "        raise ValueError('packed gather requires a source row axis')\n"
           "    rows, width = _storage_shape(weight)\n"
           "    if (rows, width) != (weight.shape[0], _product(weight.shape[1:])):\n"
           "        raise ValueError('packed gather requires a complete canonical source')\n"
           "    if not indices.shape or indices.shape[-1].value != 1 or indices.dtype not in ('int32', 'uint32', 'int64', 'uint64'):\n"
           "        raise ValueError('packed gather requires single integer coordinates')\n"
           "    for tensor in (weight, indices):\n"
           "        if tensor.strides and tensor.strides != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('packed gather requires contiguous inputs')\n"
           "    output_rows, output_columns = _storage_shape(output)\n"
           "    return dict(rows=rows, width=width, tokens=_product(indices.shape[:-1]), output_rows=output_rows, output_columns=output_columns, dtype=_DTYPE_IDS[weight.dtype], index_dtype=_DTYPE_IDS[indices.dtype])\n"
           "def pattern(x: Tensor, indices: Tensor, *, fill=0) -> tuple[Tensor, ...]:\n"
           "    return (Gather(x, indices=indices, fill=fill),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_gather_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"output_rows", "size_t", offsetof(Parameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(Parameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"index_dtype", "uint32_t", offsetof(Parameters, index_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_gather_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
