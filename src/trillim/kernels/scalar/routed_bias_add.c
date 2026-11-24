#include "common.h"

typedef struct {
    size_t tokens, routes, experts, width;
    size_t index_tokens, index_routes;
    size_t a_token_stride, a_route_stride, index_token_stride, index_route_stride;
    uint32_t bias_dtype;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->bias_dtype) || !p->experts) {
        return TRILLIM_ERROR;
    }
    size_t index_shape[] = {p->index_tokens, p->index_routes};
    size_t index_strides[] = {p->index_token_stride, p->index_route_stride};
    size_t index_span;
    if (!trillim_span(2, index_shape, index_strides, &index_span) ||
        index_span > SIZE_MAX / sizeof(int64_t)) {
        return TRILLIM_ERROR;
    }
    inputs[2] = index_span * sizeof(int64_t);
    if (!p->tokens || !p->routes || !p->width) {
        return TRILLIM_OK;
    }
    size_t a_shape[] = {p->tokens, p->routes, p->width};
    size_t a_strides[] = {p->a_token_stride, p->a_route_stride, 1};
    size_t input_span;
    if (!trillim_span(3, a_shape, a_strides, &input_span) || input_span > SIZE_MAX / sizeof(float) ||
        (p->index_tokens != p->tokens && p->index_tokens != 1) ||
        (p->index_routes != p->routes && p->index_routes != 1) ||
        p->tokens > SIZE_MAX / sizeof(float) / p->routes / p->width ||
        p->experts > SIZE_MAX / trillim_itemsize(p->bias_dtype) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = input_span * sizeof(float);
    inputs[1] = p->experts * p->width * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->tokens * p->routes * p->width * sizeof(float);
    return TRILLIM_OK;
}

typedef struct {
    float *output;
    const float *a;
    const void *bias;
    const int64_t *indices;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static float load_bias(const void *bias, size_t offset, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return ((const float *)bias)[offset];
    }
    uint16_t bits = ((const uint16_t *)bias)[offset];
    if (dtype == TRILLIM_FP16) {
        bits |= (bits & 0x7fff) > 0x7c00 ? 0x0200 : 0;
        return trillim_fp16(bits);
    }
    bits |= (bits & 0x7fff) > 0x7f80 ? 0x0040 : 0;
    return trillim_bf16(bits);
}

static TrillimStatus add_bias(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t token = slot / p->routes, route = slot % p->routes;
    uint64_t expert = (uint64_t)task->indices[token * p->index_token_stride + route * p->index_route_stride];
    if (expert >= p->experts) {
        return TRILLIM_ERROR;
    }
    const float *a = task->a + token * p->a_token_stride + route * p->a_route_stride;
    float *out = task->output + slot * p->width;
    size_t bias_start = expert * p->width;
    size_t i = 0;
    for (; i < p->width; ++i) {
        if (i && (i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        out[i] = a[i] + load_bias(task->bias, bias_start + i, p->bias_dtype);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_routed_bias_add(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->tokens || !p->routes || !p->width) {
        if (!p->index_tokens || !p->index_routes) {
            return TRILLIM_OK;
        }
        const int64_t *indices = inputs[2];
        for (size_t token = 0; token < p->index_tokens; ++token) {
            for (size_t route = 0; route < p->index_routes; ++route) {
                if ((route & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                if ((uint64_t)indices[token * p->index_token_stride + route * p->index_route_stride] >= p->experts) {
                    return TRILLIM_ERROR;
                }
            }
        }
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t slots = p->tokens * p->routes;
    if (slots > 16 && slots > 32768 / p->width) {
        return context->thread_pool->run(context->thread_pool->context, slots, add_bias, &task);
    }
    for (size_t slot = 0; slot < slots; ++slot) {
        TrillimStatus status = add_bias(&task, slot, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_routed_bias_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _DTYPE_IDS\n"
           "physical_inputs = ('a', 'bias', 'indices')\n"
           "def bind(c):\n"
           "    a, bias, indices = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    if (len(a.shape), len(bias.shape), len(indices.shape), len(out.shape)) != (3, 2, 3, 3):\n"
           "        raise ValueError('routed bias addition needs rank-three activations and rank-two bias')\n"
           "    if a.dtype != 'fp32' or bias.dtype not in {'fp16', 'bf16', 'fp32'} or indices.dtype != 'int64' or out.dtype != 'fp32':\n"
           "        raise ValueError('routed bias addition needs FP32 activations and floating bias')\n"
           "    if a.shape[-1] != out.shape[-1] or bias.shape[-1] != out.shape[-1] or indices.shape[-1].value != 1:\n"
           "        raise ValueError('bias and activation widths must match, with one expert ID per route')\n"
           "    if bias.strides and bias.strides != _contiguous_strides(bias.shape):\n"
           "        raise ValueError('routed bias addition needs contiguous expert bias vectors')\n"
           "    a_strides = a.strides or _contiguous_strides(a.shape)\n"
           "    if a_strides[-1].value != 1:\n"
           "        raise ValueError('routed bias addition needs contiguous activation rows')\n"
           "    index_strides = indices.strides or _contiguous_strides(indices.shape)\n"
           "    a_token_stride, a_route_stride = _broadcast_strides(a.shape[:2], out.shape[:2], a_strides[:2])\n"
           "    index_token_stride, index_route_stride = _broadcast_strides(indices.shape[:2], out.shape[:2], index_strides[:2])\n"
           "    return dict(tokens=out.shape[0], routes=out.shape[1], experts=bias.shape[0], width=out.shape[2],\n"
           "                index_tokens=indices.shape[0], index_routes=indices.shape[1],\n"
           "                a_token_stride=a_token_stride, a_route_stride=a_route_stride,\n"
           "                index_token_stride=index_token_stride, index_route_stride=index_route_stride, bias_dtype=_DTYPE_IDS[bias.dtype])\n"
           "def pattern(a: Tensor, bias: Tensor, indices: Tensor) -> tuple[Tensor, ...]:\n"
           "    if any(not isinstance(x, Tensor) for x in (a, bias, indices)):\n"
           "        raise ValueError('routed bias addition requires tensor inputs')\n"
           "    if tuple(len(x.shape.dimensions) for x in (a, bias, indices)) != (3, 2, 3) or indices.dtype != 'int64' or a.dtype != 'fp32':\n"
           "        raise ValueError('expected projected rows, expert biases and routed indices')\n"
           "    return (Add(a, Cast(Gather(bias, indices=indices), 'fp32')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_routed_bias_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"routes", "size_t", offsetof(Parameters, routes), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"index_tokens", "size_t", offsetof(Parameters, index_tokens), NULL},
        {"index_routes", "size_t", offsetof(Parameters, index_routes), NULL},
        {"a_token_stride", "size_t", offsetof(Parameters, a_token_stride), NULL},
        {"a_route_stride", "size_t", offsetof(Parameters, a_route_stride), NULL},
        {"index_token_stride", "size_t", offsetof(Parameters, index_token_stride), NULL},
        {"index_route_stride", "size_t", offsetof(Parameters, index_route_stride), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
    };
    static const size_t inputs[] = {4, 2, 8}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_routed_bias_add_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
