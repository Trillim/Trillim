#include "common.h"

typedef struct {
    size_t rows, width, index_stride, mask_row_stride, mask_column_stride;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    inputs[3] = sizeof(uint8_t);
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->rows, p->width};
    size_t mask_strides[] = {p->mask_row_stride, p->mask_column_stride};
    size_t index_strides[] = {p->index_stride, 0};
    size_t index_span;
    if (p->rows > SIZE_MAX / p->width || p->width > SIZE_MAX / sizeof(int64_t) ||
        !trillim_span(2, shape, index_strides, &index_span) ||
        index_span > SIZE_MAX / sizeof(int64_t) ||
        !trillim_span(2, shape, mask_strides, &inputs[2])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->width * sizeof(int64_t);
    inputs[1] = index_span * sizeof(int64_t);
    outputs[0] = p->rows * p->width;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_topk_mask(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->width) {
        return TRILLIM_OK;
    }
    const int64_t *columns = inputs[0], *selected = inputs[1];
    const uint8_t *mask = inputs[2];
    bool replacement = *(const uint8_t *)inputs[3] != 0;
    uint8_t *remaining = outputs[0];
    for (size_t row = 0; row < p->rows; ++row) {
        int64_t index = selected[row * p->index_stride];
        const uint8_t *mask_row = mask + row * p->mask_row_stride;
        uint8_t *output_row = remaining + row * p->width;
        for (size_t column = 0; column < p->width; ++column) {
            if ((column & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            output_row[column] = columns[column] == index ? replacement : mask_row[column * p->mask_column_stride] != 0;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_topk_mask_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides\n"
           "physical_inputs = ('columns', 'selected', 'when_false', 'when_true')\n"
           "def bind(c):\n"
           "    columns, selected, mask, replacement = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    if (len(columns.shape), len(selected.shape), len(mask.shape), len(out.shape)) != (1, 2, 2, 2) or replacement.shape:\n"
           "        raise ValueError('mask update needs column IDs, one selected index per row and a rank-two mask')\n"
           "    if (columns.dtype, selected.dtype, mask.dtype, replacement.dtype, out.dtype) != ('int64', 'int64', 'bool', 'bool', 'bool'):\n"
           "        raise ValueError('mask update needs int64 indices and bool masks')\n"
           "    if selected.shape[1].value != 1 or columns.shape != out.shape[-1:] or (columns.strides and columns.strides != _contiguous_strides(columns.shape)):\n"
           "        raise ValueError('mask update needs contiguous columns and a single selection per row')\n"
           "    index_stride, _ = _broadcast_strides(selected.shape, out.shape, selected.strides)\n"
           "    mask_row_stride, mask_column_stride = _broadcast_strides(mask.shape, out.shape, mask.strides)\n"
           "    return dict(rows=out.shape[0], width=out.shape[1], index_stride=index_stride,\n"
           "                mask_row_stride=mask_row_stride, mask_column_stride=mask_column_stride)\n"
           "def pattern(columns: Tensor, selected: Tensor, when_false: Tensor, when_true: Tensor) -> tuple[Tensor, ...]:\n"
           "    if any(not isinstance(x, Tensor) for x in (columns, selected, when_false, when_true)):\n"
           "        raise ValueError('mask update requires tensor inputs')\n"
           "    if (columns.dtype, selected.dtype, when_false.dtype, when_true.dtype) != ('int64', 'int64', 'bool', 'bool'):\n"
           "        raise ValueError('expected two indices and a mask')\n"
           "    if tuple(len(x.shape.dimensions) for x in (columns, selected, when_false, when_true)) != (1, 2, 2, 0):\n"
           "        raise ValueError('expected columns, per-row indices, a matrix mask and a scalar replacement')\n"
           "    return (Select(Compare(columns, selected, predicate='eq'), when_true, when_false),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_topk_mask_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"index_stride", "size_t", offsetof(Parameters, index_stride), NULL},
        {"mask_row_stride", "size_t", offsetof(Parameters, mask_row_stride), NULL},
        {"mask_column_stride", "size_t", offsetof(Parameters, mask_column_stride), NULL},
    };
    static const size_t inputs[] = {8, 8, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_topk_mask_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
