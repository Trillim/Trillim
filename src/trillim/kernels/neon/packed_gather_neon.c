#define TRILLIM_ENABLE_PACKED_NEON
#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"
#include <arm_neon.h>
#include <float.h>

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
    void *destination;
    const uint8_t *weight;
    const void *indices;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static bool index_dtype(uint32_t dtype) {
    return dtype == TRILLIM_INT32 || dtype == TRILLIM_UINT32 ||
           dtype == TRILLIM_INT64 || dtype == TRILLIM_UINT64;
}

static bool copyable_mxfp4(const Parameters *p) {
    return p->dtype == TRILLIM_MXFP4 && p->width % 32 == 0 && p->output_columns % 32 == 0;
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
    if (p->tokens > SIZE_MAX / id_size || p->tokens > SIZE_MAX / p->width ||
        !p->output_columns || p->output_rows > SIZE_MAX / p->output_columns ||
        p->output_rows * p->output_columns != p->tokens * p->width ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->tokens * id_size;
    if (!copyable_mxfp4(p)) {
        if (p->tokens * p->width > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->tokens * p->width * sizeof(float);
    }
    return TRILLIM_OK;
}

static size_t tiles_per_row(size_t width) {
    return width / 1024 + (width % 1024 != 0);
}

static uint64_t row_index(const Task *task, size_t token) {
    uint32_t dtype = task->parameters->index_dtype;
    TrillimValue coordinate = trillim_load(task->indices, token, dtype);
    return trillim_unsigned(dtype) ? coordinate.u : (uint64_t)coordinate.i;
}

enum { MXFP4_GROUPS_PER_TILE = 32768 };

static size_t mxfp4_tiles_per_row(size_t groups) {
    return groups / MXFP4_GROUPS_PER_TILE + (groups % MXFP4_GROUPS_PER_TILE != 0);
}

static TrillimStatus copy_mxfp4(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t groups = p->width / 32;
    size_t tiles = mxfp4_tiles_per_row(groups);
    size_t token = tile / tiles;
    uint64_t row = row_index(task, token);
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t first = tile % tiles * MXFP4_GROUPS_PER_TILE;
    size_t count = groups - first < MXFP4_GROUPS_PER_TILE ? groups - first : MXFP4_GROUPS_PER_TILE;
    size_t source = (size_t)row * groups + first;
    size_t destination = token * groups + first;
    const uint8_t *codes = task->weight + source * 16;
    const uint8_t *scales = task->weight + p->rows * groups * 16 + source;

    uint8x16_t vector_maximum = vdupq_n_u8(0);
    size_t i = 0;
    for (; count - i >= 16; i += 16) {
        vector_maximum = vmaxq_u8(vector_maximum, vld1q_u8(scales + i));
    }
    uint8_t maximum = vmaxvq_u8(vector_maximum);
    for (; i < count; ++i) {
        if (scales[i] > maximum) {
            maximum = scales[i];
        }
    }
    if (maximum >= 253) {
        for (i = 0; i < count; ++i) {
            if ((i & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t exponent = scales[i];
            if (exponent == 255) {
                return TRILLIM_ERROR;
            }
            if (exponent < 253) {
                continue;
            }
            /* The largest magnitudes overflow FP32 at these two exponents. */
            uint8x16_t packed = vld1q_u8(codes + i * 16);
            uint8x16_t mask = vdupq_n_u8(7);
            uint8x16_t low = vandq_u8(packed, mask);
            uint8x16_t high = vandq_u8(vshrq_n_u8(packed, 4), mask);
            uint8_t limit = exponent == 253 ? 6 : 4;
            if (vmaxvq_u8(vmaxq_u8(low, high)) >= limit) {
                return TRILLIM_ERROR;
            }
        }
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    uint8_t *output = task->destination;
    memcpy(output + destination * 16, codes, count * 16);
    memcpy(output + p->tokens * groups * 16 + destination, scales, count);
    return TRILLIM_OK;
}

static TrillimStatus gather(void *opaque, size_t tile, size_t worker) {
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
    uint64_t row = row_index(task, token);
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t source = (size_t)row * p->width;
    float *decoded = task->destination;
    size_t destination = token * p->width;
    if (end - column >= 16) {
        TrillimPackedNeonRow packed_row = trillim_packed_neon_row(
            task->weight, p->dtype, (size_t)row, p->width, p->rows
        );
        float32x4_t bound = vdupq_n_f32(FLT_MAX);
        for (; end - column >= 16; column += 16) {
            float32x4_t values[4];
            trillim_packed_neon_decode16(&packed_row, column, values);
            uint32x4_t finite = vcleq_f32(vabsq_f32(values[0]), bound);
            for (size_t lane = 1; lane < 4; ++lane) {
                finite = vandq_u32(finite, vcleq_f32(vabsq_f32(values[lane]), bound));
            }
            if (vminvq_u32(finite) == 0) {
                return TRILLIM_ERROR;
            }
            for (size_t lane = 0; lane < 4; ++lane) {
                vst1q_f32(decoded + destination + column + lane * 4, values[lane]);
            }
        }
    }
    for (; column < end; ++column) {
        float value = trillim_packed_read_float(task->weight, source + column, p->dtype, p->rows, p->width);
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        decoded[destination + column] = value;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_gather_neon(
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
    bool copy = copyable_mxfp4(p);
    Task task = {copy ? outputs[0] : scratch, inputs[0], inputs[1], p, context};
    size_t tiles = p->tokens * (copy ? mxfp4_tiles_per_row(p->width / 32) : tiles_per_row(p->width));
    TrillimTask operation = copy ? copy_mxfp4 : gather;
    TrillimStatus status = tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
    if (status != TRILLIM_OK || copy) {
        return status;
    }
    return trillim_packed_encode_quant(scratch, p->output_rows, p->output_columns,
                                      p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_gather_neon_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_gather_neon_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_gather_neon_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
    return &capabilities;
}
