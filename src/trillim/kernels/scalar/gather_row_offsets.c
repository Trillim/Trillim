#include "common.h"

typedef struct {
    size_t source_count;
    size_t rows;
    uint64_t width;
    uint32_t dtype;
} TrillimGatherRowOffsetsParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimGatherRowOffsetsParameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || p->width == 0 || p->rows > SIZE_MAX / sizeof(int64_t)) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->dtype);
    if (p->source_count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->source_count * bytes;
    inputs[1] = p->rows * sizeof(int64_t);
    outputs[0] = p->rows * bytes;
    *scratch = 0;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_gather_row_offsets(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimGatherRowOffsetsParameters *p = parameters;
    const uint8_t *source = inputs[0];
    const uint8_t *indices = inputs[1];
    uint8_t *output = outputs[0];
    size_t bytes = trillim_itemsize(p->dtype);
    uint16_t infinity = p->dtype == TRILLIM_FP16 ? 0x7c00 : 0x7f80;
    uint16_t quiet = p->dtype == TRILLIM_FP16 ? 0x0200 : 0x0040;
    for (size_t row = 0; row < p->rows; ++row) {
        if ((row & 1023) == 1023 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint64_t index;
        memcpy(&index, indices + row * sizeof(index), sizeof(index));
        /* The original integer arithmetic wraps before Gather checks the flat coordinate. */
        uint64_t coordinate = (uint64_t)row * p->width + index;
        if (coordinate > INT64_MAX || coordinate >= p->source_count) {
            return TRILLIM_ERROR;
        }
        if (bytes == sizeof(uint16_t)) {
            uint16_t bits;
            memcpy(&bits, source + coordinate * bytes, sizeof(bits));
            bits |= (bits & 0x7fff) > infinity ? quiet : 0;
            memcpy(output + row * bytes, &bits, sizeof(bits));
        } else {
            memcpy(output + row * bytes, source + coordinate * bytes, bytes);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_gather_row_offsets_tdl(void) {
    return "import numpy as np\n"
           "from trillim.tdl import Add, Constant, Dim, Gather, Iota, Mul, Reshape, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype\n"
           "physical_inputs = ('x', 'indices')\n"
           "def pattern(x: Tensor, indices: Tensor, *, width: int = 32) -> tuple[Tensor, ...]:\n"
           "    flat = Reshape(indices, (-1,))\n"
           "    extent = Constant(np.asarray(width, dtype=np.int64), dtype='int64')\n"
           "    coordinates = Add(Mul(Iota(Dim(flat, 0)), extent), flat)\n"
           "    return (Gather(x, indices=Reshape(coordinates, (-1, 1))),)\n"
           "def bind(context):\n"
           "    source = context.inputs['x']\n"
           "    indices = context.inputs['indices']\n"
           "    output = context.outputs[0]\n"
           "    if (len(source.shape) != 1 or source.dtype not in {'fp16', 'bf16', 'fp32'}\n"
           "        or len(indices.shape) != 2 or indices.shape[1].value != 1\n"
           "        or indices.dtype != 'int64' or output.shape != indices.shape[:1]\n"
           "        or output.dtype != source.dtype\n"
           "        or source.strides != _contiguous_strides(source.shape)\n"
           "        or indices.strides[:1] != _contiguous_strides(indices.shape)[:1]):\n"
           "        raise ValueError('row-offset gather requires contiguous floating values and int64 row indices')\n"
           "    return dict(source_count=source.shape[0], rows=indices.shape[0],\n"
           "                width=context.attrs['width'], dtype=_native_dtype(source.dtype))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gather_row_offsets_parameters(void) {
    static const TrillimParameter fields[] = {
        {"source_count", "size_t", offsetof(TrillimGatherRowOffsetsParameters, source_count), NULL},
        {"rows", "size_t", offsetof(TrillimGatherRowOffsetsParameters, rows), NULL},
        {"width", "uint64_t", offsetof(TrillimGatherRowOffsetsParameters, width), NULL},
        {"dtype", "uint32_t", offsetof(TrillimGatherRowOffsetsParameters, dtype), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimGatherRowOffsetsParameters), TRILLIM_ALIGNOF(TrillimGatherRowOffsetsParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gather_row_offsets_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
