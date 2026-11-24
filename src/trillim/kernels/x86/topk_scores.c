#include "common.h"

typedef struct {
    size_t source_count, index_rows, rows, count, width, previous_row_stride;
    uint32_t dtype;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = inputs[3] = outputs[0] = *scratch = 0;
    if (!p->width || !trillim_float(p->dtype)) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->dtype);
    size_t previous_shape[] = {p->rows, p->count};
    size_t previous_strides[] = {p->previous_row_stride, 1};
    size_t previous_span;
    if (p->index_rows > SIZE_MAX / sizeof(int64_t) || p->source_count > SIZE_MAX / bytes ||
        !trillim_span(2, previous_shape, previous_strides, &previous_span) ||
        previous_span > SIZE_MAX / bytes ||
        (p->count && p->rows > SIZE_MAX / bytes / p->count)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->source_count * bytes;
    inputs[1] = p->index_rows * sizeof(int64_t);
    inputs[2] = p->count;
    inputs[3] = previous_span * bytes;
    outputs[0] = p->rows * p->count * bytes;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_topk_scores(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    const int64_t *indices = inputs[1];
    const uint8_t *replace = inputs[2];
    if (p->index_rows != p->rows && p->index_rows != 1) {
        return TRILLIM_ERROR;
    }
    for (size_t row = 0; row < p->index_rows; ++row) {
        if ((row & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        /* Gather validates the wrapped flat offset even when no column is replaced. */
        uint64_t coordinate = (uint64_t)row * p->width + (uint64_t)indices[row];
        if (coordinate > INT64_MAX || coordinate >= p->source_count) {
            return TRILLIM_ERROR;
        }
    }
    if (!p->count) {
        return TRILLIM_OK;
    }
    for (size_t row = 0; row < p->rows; ++row) {
        if ((row & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t score_row = p->index_rows == 1 ? 0 : row;
        uint64_t coordinate = (uint64_t)score_row * p->width + (uint64_t)indices[score_row];
        TrillimValue score = trillim_load(inputs[0], coordinate, p->dtype);
        size_t base = row * p->count;
        size_t previous = row * p->previous_row_stride;
        for (size_t column = 0; column < p->count; ++column) {
            if ((column & 1023) == 1023 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            TrillimValue value = replace[column] ? score : trillim_load(inputs[3], previous + column, p->dtype);
            trillim_store(outputs[0], base + column, p->dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_topk_scores_tdl(void) {
    return "import numpy as np\n"
           "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _native_dtype\n"
           "physical_inputs = ('x', 'indices', 'condition', 'when_false')\n"
           "def bind(c):\n"
           "    x, indices, condition, previous = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    if len(indices.shape) not in (1, 2) or (len(indices.shape) == 2 and indices.shape[1].value != 1):\n"
           "        raise ValueError('score update needs one index per row')\n"
           "    if (len(x.shape), len(condition.shape), len(previous.shape), len(out.shape)) != (1, 1, 2, 2):\n"
           "        raise ValueError('score update needs flat logits, a column mask and rank-two scores')\n"
           "    if indices.dtype != 'int64' or condition.dtype != 'bool' or not (x.dtype == previous.dtype == out.dtype and x.dtype in {'fp16', 'bf16', 'fp32'}):\n"
           "        raise ValueError('score update needs int64 indices, a bool mask and matching floating dtypes')\n"
           "    if previous.shape[1:] != out.shape[1:] or condition.shape != out.shape[1:]:\n"
           "        raise ValueError('score update needs one previous score per output and a mask for each column')\n"
           "    if any(v.strides and v.strides != _contiguous_strides(v.shape) for v in (x, indices, condition, previous)):\n"
           "        raise ValueError('score update needs contiguous inputs')\n"
           "    values = c.inputs['value_shape'].values\n"
           "    if values is None or len(values) != 2 or values[0] != indices.shape[0] or values[1].value != 1:\n"
           "        raise ValueError('score update needs a single gathered score per row')\n"
           "    previous_row_stride, _ = _broadcast_strides(previous.shape, out.shape, previous.strides)\n"
           "    return dict(source_count=x.shape[0], index_rows=indices.shape[0], rows=out.shape[0], count=out.shape[1],\n"
           "                width=c.attrs['width'], previous_row_stride=previous_row_stride, dtype=_native_dtype(x.dtype))\n"
           "def pattern(value_shape: Tensor, x: Tensor, indices: Tensor, condition: Tensor, when_false: Tensor,\n"
           "            *, width: int = 32) -> tuple[Tensor, ...]:\n"
           "    if any(not isinstance(v, Tensor) for v in (value_shape, x, indices, condition, when_false)):\n"
           "        raise ValueError('score update requires tensor inputs')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or indices.dtype != 'int64' or condition.dtype != 'bool' or when_false.dtype != x.dtype:\n"
           "        raise ValueError('expected logits, indices, a bool mask and previous scores')\n"
           "    flat = Reshape(indices, (-1,))\n"
           "    extent = Constant(np.asarray(width, dtype=np.int64), dtype='int64')\n"
           "    coordinates = Add(Mul(Iota(Dim(flat, 0)), extent), flat)\n"
           "    scores = Gather(x, indices=Reshape(coordinates, (-1, 1)))\n"
           "    values = Reshape(scores, value_shape)\n"
           "    return (Select(condition, values, when_false),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_topk_scores_parameters(void) {
    static const TrillimParameter fields[] = {
        {"source_count", "size_t", offsetof(Parameters, source_count), NULL},
        {"index_rows", "size_t", offsetof(Parameters, index_rows), NULL},
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"previous_row_stride", "size_t", offsetof(Parameters, previous_row_stride), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
    };
    static const size_t inputs[] = {1, 8, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_topk_scores_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
