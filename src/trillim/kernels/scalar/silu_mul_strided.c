#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t rows;
    size_t width;
    size_t a_row_stride;
    size_t a_column_stride;
    size_t b_row_stride;
    size_t b_column_stride;
} Parameters;

typedef struct {
    float *output;
    const float *a;
    const float *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static float activate(float a, float b) {
    float e = expf(-fabsf(a));
    float denominator = 1.0f + e;
    float probability = a >= 0.0f ? 1.0f / denominator : e / denominator;
    return (a * probability) * b;
}

static bool input_span(
    size_t rows, size_t width, size_t row_stride, size_t column_stride, size_t *bytes
) {
    bool row_major = column_stride == 1 && row_stride >= width;
    bool column_major = row_stride == 1 && column_stride >= rows;
    if (!row_major && !column_major) {
        return false;
    }
    size_t count = 1;
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - count) / row_stride) {
        return false;
    }
    count += (rows - 1) * row_stride;
    if (column_stride != 0 && width - 1 > (SIZE_MAX - count) / column_stride) {
        return false;
    }
    count += (width - 1) * column_stride;
    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *bytes = count * sizeof(float);
    return true;
}

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width / sizeof(float) ||
        !input_span(p->rows, p->width, p->a_row_stride, p->a_column_stride, &inputs[0]) ||
        !input_span(p->rows, p->width, p->b_row_stride, p->b_column_stride, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t count = p->rows * p->width;
    size_t end = count - begin < 1024 ? count : begin + 1024;
    while (begin < end) {
        size_t row = begin / p->width;
        size_t column = begin % p->width;
        size_t length = p->width - column;
        if (length > end - begin) {
            length = end - begin;
        }
        const float *a = task->a + row * p->a_row_stride + column * p->a_column_stride;
        const float *b = task->b + row * p->b_row_stride + column * p->b_column_stride;
        float *output = task->output + begin;
        size_t i = 0;

        for (; i < length; ++i) {
            output[i] = activate(a[i * p->a_column_stride], b[i * p->b_column_stride]);
        }
        begin += length;
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL || output != 0 || input >= 2) {
        return false;
    }
    const Parameters *p = parameters;
    size_t row_stride = input == 0 ? p->a_row_stride : p->b_row_stride;
    size_t column_stride = input == 0 ? p->a_column_stride : p->b_column_stride;
    return column_stride == 1 && (p->rows <= 1 || row_stride == p->width);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_silu_mul_strided(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t count = p->rows * p->width;
    size_t tiles = count / 1024 + (count % 1024 != 0);
    if (count <= 1024) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_silu_mul_strided_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "def bind(c):\n"
           "    a, b, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    if len(a.shape) != 2 or a.shape != b.shape or a.shape != output.shape:\n"
           "        raise ValueError('strided SiLU-Mul requires matching matrices')\n"
           "    if any(value.dtype != 'fp32' for value in (a, b, output)):\n"
           "        raise ValueError('strided SiLU-Mul requires FP32 arithmetic')\n"
           "    if (output.strides or _contiguous_strides(output.shape)) != _contiguous_strides(output.shape):\n"
           "        raise ValueError('strided SiLU-Mul requires contiguous output')\n"
           "    rows, width = a.shape\n"
           "    if width.op != 'const' or width.value is None:\n"
           "        raise ValueError('strided SiLU-Mul requires fixed channel width')\n"
           "    one = _constant_expression(1)\n"
           "    strides = [value.strides or _contiguous_strides(value.shape) for value in (a, b)]\n"
           "    for row, column in strides:\n"
           "        row_major = column == one and row.op == 'const' and row.value is not None and row.value >= width.value\n"
           "        column_major = row == one and column == rows\n"
           "        if not row_major and not column_major:\n"
           "            raise ValueError('strided SiLU-Mul requires row-major or transposed matrices')\n"
           "    if all(row == width and column == one for row, column in strides):\n"
           "        raise ValueError('canonical SiLU-Mul uses the existing kernels')\n"
           "    return dict(rows=rows, width=width, a_row_stride=strides[0][0], a_column_stride=strides[0][1], b_row_stride=strides[1][0], b_column_stride=strides[1][1])\n"
           "def pattern(a: Tensor, b: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (SiluMul(a, b, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_silu_mul_strided_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"a_row_stride", "size_t", offsetof(Parameters, a_row_stride), NULL},
        {"a_column_stride", "size_t", offsetof(Parameters, a_column_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Parameters, b_row_stride), NULL},
        {"b_column_stride", "size_t", offsetof(Parameters, b_column_stride), NULL}
    };
    static const size_t inputs[] = {4, 4};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_silu_mul_strided_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
