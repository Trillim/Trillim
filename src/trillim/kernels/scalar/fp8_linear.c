#include "common.h"

enum { BLOCK_SIZE = 128, OUTPUT_TILE = 16 };
_Static_assert(BLOCK_SIZE % OUTPUT_TILE == 0, "an output tile must fit within one scale block");

typedef struct {
    size_t tokens, width, outputs;
    size_t input_row_stride, input_column_stride;
    uint32_t input_dtype, output_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    void *output;
    const float *decoded;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t tiles, blocks;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    size_t output_itemsize = trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / output_itemsize / p->outputs) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0) {
        size_t shape[] = {p->tokens, p->width};
        size_t strides[] = {p->input_row_stride, p->input_column_stride};
        size_t elements;
        size_t itemsize = trillim_itemsize(p->input_dtype);
        if (!trillim_span(2, shape, strides, &elements) || elements > SIZE_MAX / itemsize ||
            !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_FP8_E4M3, &inputs[1])) {
            return TRILLIM_ERROR;
        }
        inputs[0] = elements * itemsize;
    }
    outputs[0] = p->tokens * p->outputs * output_itemsize;
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = index / task->tiles;
    size_t first = index % task->tiles * OUTPUT_TILE;
    size_t columns = p->outputs - first < OUTPUT_TILE ? p->outputs - first : OUTPUT_TILE;
    float sums[OUTPUT_TILE] = {0};
    for (size_t start = 0; start < p->width;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->width - start < BLOCK_SIZE ? p->width - start : BLOCK_SIZE;
        size_t input = row * p->input_row_stride + start * p->input_column_stride;
        float activations[BLOCK_SIZE];
        for (size_t k = 0; k < count; ++k) {
            activations[k] = trillim_load(task->input, input + k * p->input_column_stride, p->input_dtype).f;
        }
        size_t block = (first / BLOCK_SIZE) * task->blocks + start / BLOCK_SIZE;
        float scale;
        memcpy(&scale, task->weight + p->outputs * p->width + block * sizeof(scale), sizeof(scale));
        for (size_t column = 0; column < columns; ++column) {
            const uint8_t *weight = task->weight + (first + column) * p->width + start;
            float sum = sums[column];
            for (size_t k = 0; k < count; ++k) {
                float value = task->decoded[weight[k]] * scale;
                sum = fmaf(activations[k], value, sum);
            }
            sums[column] = sum;
        }
        start += count;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t output = row * p->outputs + first;
    if (p->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + output * sizeof(float), sums, columns * sizeof(float));
    } else {
        for (size_t column = 0; column < columns; ++column) {
            TrillimValue value = {.f = sums[column]};
            trillim_store(task->output, output + column, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_fp8_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    float decoded[256];
    for (size_t code = 0; code < 256; ++code) {
        decoded[code] = trillim_fp8_e4m3((uint8_t)code);
    }
    size_t tiles = p->outputs / OUTPUT_TILE + (p->outputs % OUTPUT_TILE != 0);
    size_t blocks = p->width / BLOCK_SIZE + (p->width % BLOCK_SIZE != 0);
    Task task = {inputs[0], inputs[1], outputs[0], decoded, p, context, tiles, blocks};
    size_t jobs = p->tokens * tiles;
    return jobs == 1 ? project(&task, 0, 0)
        : context->thread_pool->run(context->thread_pool->context, jobs, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_fp8_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or weight.dtype != 'fp8_e4m3' or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('FP8 linear requires floating activations and output')\n"
           "    if len(weight.shape) != 2 or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('FP8 linear requires a complete weight matrix')\n"
           "    if (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "        raise ValueError('FP8 linear requires contiguous weights')\n"
           "    values = _linear(context)\n"
           "    outputs, width = weight.shape\n"
           "    row_stride, column_stride = values['a_row_stride'], values['a_col_stride']\n"
           "    if row_stride == values['k'] and column_stride.value == 1:\n"
           "        row_stride = width\n"
           "    return dict(tokens=values['m'], width=width, outputs=outputs, input_row_stride=row_stride, input_column_stride=column_stride, input_dtype=values['a_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased FP8 linear projection')\n"
           "    output = Linear(x, weight)\n"
           "    return (output if dtype is None else Cast(output, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_fp8_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_row_stride", "size_t", offsetof(Parameters, input_row_stride), NULL},
        {"input_column_stride", "size_t", offsetof(Parameters, input_column_stride), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_fp8_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
