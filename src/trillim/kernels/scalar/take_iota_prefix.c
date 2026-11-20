#include "common.h"

typedef struct {
    size_t source_rows;
    size_t rows;
    size_t width;
    uint32_t dtype;
} Parameters;

typedef struct {
    uint8_t *output;
    const uint8_t *input;
    size_t count;
    uint32_t dtype;
    const TrillimKernelContext *context;
} Task;

enum { COPY_TILE = 16384, CANCEL_GRAIN = 1024 };

static bool prefix_count(const Parameters *p, size_t *count) {
    size_t bytes = trillim_itemsize(p->dtype);
    if (bytes == 0 || p->rows > p->source_rows || p->rows > SIZE_MAX / sizeof(int64_t) ||
        (p->rows != 0 && p->width > SIZE_MAX / bytes / p->rows)) {
        return false;
    }
    *count = p->rows * p->width;
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    size_t count;
    if (!prefix_count(p, &count)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = count * trillim_itemsize(p->dtype);
    *scratch = 0;
    return TRILLIM_OK;
}

static TrillimStatus copy_prefix(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const TrillimKernelContext *context = task->context;
    size_t begin = tile * COPY_TILE;
    size_t remaining = task->count - begin;
    size_t end = remaining < COPY_TILE ? task->count : begin + COPY_TILE;
    size_t bytes = trillim_itemsize(task->dtype);
    bool narrow = task->dtype == TRILLIM_FP16 || task->dtype == TRILLIM_BF16;
    uint16_t infinity = task->dtype == TRILLIM_FP16 ? 0x7c00 : 0x7f80;
    uint16_t quiet = task->dtype == TRILLIM_FP16 ? 0x0200 : 0x0040;
    while (begin < end) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - begin < CANCEL_GRAIN ? end - begin : CANCEL_GRAIN;
        const uint8_t *input = task->input + begin * bytes;
        uint8_t *output = task->output + begin * bytes;
        if (narrow) {
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits;
                memcpy(&bits, input + i * sizeof(bits), sizeof(bits));
                /* Match Gather's signaling NaN handling and preserve payloads. */
                bits |= (bits & 0x7fff) > infinity ? quiet : 0;
                memcpy(output + i * sizeof(bits), &bits, sizeof(bits));
            }
        } else if (task->dtype == TRILLIM_BOOL) {
            for (size_t i = 0; i < count; ++i) {
                output[i] = input[i] != 0;
            }
        } else {
            memcpy(output, input, count * bytes);
        }
        begin += count;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_take_iota_prefix(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t count;
    if (!prefix_count(p, &count)) {
        return TRILLIM_ERROR;
    }
    if (count == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], count, p->dtype, context};
    size_t tiles = count / COPY_TILE + (count % COPY_TILE != 0);
    return tiles == 1 ? copy_prefix(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tiles, copy_prefix, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_take_iota_prefix_tdl(void) {
    return "from trillim.tdl import Iota, Take, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "physical_inputs = ('x',)\n"
           "def pattern(x: Tensor, stop: Tensor, start: Tensor, *, axis: int = 0, rank: int = 1, index_rank: int = 1) -> tuple[Tensor, ...]:\n"
           "    if axis != 0 or index_rank != 1:\n"
           "        raise ValueError('expected axis-zero vector indices')\n"
           "    return (Take(x, Iota(stop, start=start), axis=axis, rank=rank, index_rank=index_rank),)\n"
           "def bind(c):\n"
           "    start, stop = c.inputs['start'], c.inputs['stop']\n"
           "    if start.dtype != 'int64' or stop.dtype != 'int64' or start.shape != () or stop.shape != ():\n"
           "        raise ValueError('expected scalar int64 Iota bounds')\n"
           "    if start.values is None or len(start.values) != 1 or start.values[0].op != 'const' or start.values[0].value != 0:\n"
           "        raise ValueError('expected zero Iota start')\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if not x.shape or x.dtype != output.dtype or x.shape[1:] != output.shape[1:]:\n"
           "        raise ValueError('expected matching prefix rows')\n"
           "    for value in (x, output):\n"
           "        if value.strides and value.strides != _contiguous_strides(value.shape):\n"
           "            raise ValueError('expected contiguous prefix rows')\n"
           "    return {'source_rows': x.shape[0], 'rows': output.shape[0], 'width': _product(x.shape[1:]), 'dtype': _native_dtype(x.dtype)}\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_take_iota_prefix_parameters(void) {
    static const TrillimParameter fields[] = {
        {"source_rows", "size_t", offsetof(Parameters, source_rows), NULL},
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
    };
    static const size_t alignments[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, alignments, alignments, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_take_iota_prefix_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
