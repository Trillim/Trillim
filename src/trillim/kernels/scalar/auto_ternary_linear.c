#include "common.h"

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype, input_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    void *output;
    int8_t *codes;
    float *scales;
    bool ordinary;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->weight_dtype != TRILLIM_GRP_TERNARY || !trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t in_bytes = trillim_itemsize(p->input_dtype), out_bytes = trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / p->width ||
        (p->outputs && p->tokens > SIZE_MAX / out_bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    size_t count = p->tokens * p->width;
    if (count > SIZE_MAX / in_bytes || count > SIZE_MAX - 3 ||
        p->tokens > (SIZE_MAX - ((count + 3) & ~(size_t)3)) / sizeof(float) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count * in_bytes;
    outputs[0] = p->tokens * p->outputs * out_bytes;
    *scratch = ((count + 3) & ~(size_t)3) + p->tokens * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus prepare(Task *task) {
    const Parameters *p = task->p;
    for (size_t row = 0; row < p->tokens; ++row) {
        float maximum = 1e-5f;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f;
            if (!isfinite(x)) {
                task->ordinary = false;
                return TRILLIM_OK;
            }
            maximum = fmaxf(maximum, fabsf(x));
        }
        float scale = 127.0f / maximum;
        task->scales[row] = scale;
        for (size_t k = 0; k < p->width; ++k) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f * scale;
            float magnitude = fminf(fabsf(x), 127.0f);
            int32_t code = (int32_t)magnitude;
            float fraction = magnitude - (float)code;
            if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
                ++code;
            }
            task->codes[row * p->width + k] = (int8_t)(x < 0.0f ? -code : code);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = index / tiles, first = index % tiles * 16;
    size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    for (size_t column = first; column < end; ++column) {
        float total = 0.0f;
        if (task->ordinary) {
            for (size_t g = 0; g < groups; ++g) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t block = column * groups + g;
                const uint8_t *packed = task->weight + block * 32;
                size_t count = p->width - g * 128 < 128 ? p->width - g * 128 : 128;
                int32_t sum = 0;
                for (size_t k = 0; k < count; ++k) {
                    int32_t code = (packed[k % 32] >> (6 - 2 * (k / 32))) & 3;
                    if (code == 3) {
                        total = NAN;
                        break;
                    }
                    sum += (code - 1) * task->codes[row * p->width + g * 128 + k];
                }
                uint16_t bits;
                memcpy(&bits, task->weight + p->outputs * groups * 32 + block * 2, sizeof(bits));
                total = fmaf((float)sum, trillim_bf16(bits), total);
            }
            total /= task->scales[row];
        } else {
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f;
                float w = trillim_packed_read_float(task->weight, column * p->width + k,
                    p->weight_dtype, p->outputs, p->width);
                total = fmaf(x, w, total);
            }
        }
        TrillimValue value = {.f = total};
        trillim_store(task->output, row * p->outputs + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_auto_ternary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t count = p->tokens * p->width;
    Task task = {inputs[0], inputs[1], outputs[0], scratch,
        (float *)((uint8_t *)scratch + ((count + 3) & ~(size_t)3)), true, p, context};
    status = prepare(&task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->tokens * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_auto_ternary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if weight.dtype != 'grp_ternary' or x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires grouped ternary weights and floating activations')\n"
           "    for tensor in (x, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete packed matrix')\n"
           "    p = _linear(context)\n"
           "    return dict(tokens=p['m'], width=p['k'], outputs=p['n'], weight_dtype=p['b_dtype'], output_dtype=p['out_dtype'], input_dtype=p['a_dtype'])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased floating linear projection')\n"
           "    result = Linear(x, weight)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_auto_ternary_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_auto_ternary_linear_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
