#include "common.h"
#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { PROJECTION_TILE = 16, ACTIVATION_TILE = 1024 };

typedef struct {
    size_t tokens, width, hidden, outputs;
    uint32_t up_dtype, down_dtype;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

typedef struct {
    const LinearParameters *p;
    const void *input;
    const void *weight;
    void *output;
    const TrillimKernelContext *context;
} Projection;

typedef struct {
    float *up;
    const float *gate;
    size_t count;
    const TrillimKernelContext *context;
} Gate;

static bool grouped(uint32_t dtype) {
    return dtype >= TRILLIM_GRP_INT8 && dtype <= TRILLIM_DUBGRP_INT6;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->hidden == 0 || !grouped(p->up_dtype) || !grouped(p->down_dtype) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, p->up_dtype, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, p->down_dtype, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch) ||
        *scratch > SIZE_MAX / 2) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch *= 2;
    return TRILLIM_OK;
}

/* Selective kernel builds may omit BlockLinear. */
#if !defined(__APPLE__) && !defined(_WIN32)
__attribute__((weak)) extern TrillimStatus trillim_scalar_block_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
#endif

static TrillimKernel linked_projection(void) {
#if defined(__APPLE__)
#define TRILLIM_KERNEL_NAME(symbol) #symbol
    void *symbol = dlsym(RTLD_SELF, TRILLIM_KERNEL_NAME(trillim_scalar_block_linear));
#undef TRILLIM_KERNEL_NAME
    TrillimKernel kernel;
    memcpy(&kernel, &symbol, sizeof(kernel));
    return kernel;
#elif defined(_WIN32)
    return NULL;
#else
    return trillim_scalar_block_linear;
#endif
}

static float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Projection *task = opaque;
    const LinearParameters *p = task->p;
    const TrillimKernelContext *context = task->context;
    size_t count = p->tokens * p->outputs;
    size_t first = tile * PROJECTION_TILE;
    size_t end = count - first < PROJECTION_TILE ? count : first + PROJECTION_TILE;
    for (size_t index = first; index < end; ++index) {
        size_t row = index / p->outputs;
        size_t column = index % p->outputs;
        float sum = 0.0f;
        for (size_t k = 0; k < p->width; ++k) {
            if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f;
            float w = trillim_packed_read_float(task->weight, column * p->width + k, p->weight_dtype,
                p->outputs, p->width);
            sum = multiply_add(x, w, sum);
        }
        trillim_store(task->output, index, TRILLIM_FP32, (TrillimValue){.f = sum});
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus run_projection(
    TrillimKernel kernel, const LinearParameters *parameters, const void *input, const void *weight,
    void *output, const TrillimKernelContext *context
) {
    if (kernel != NULL) {
        const void *inputs[] = {input, weight};
        void *outputs[] = {output};
        return kernel(outputs, inputs, NULL, parameters, context);
    }
    Projection task = {parameters, input, weight, output, context};
    size_t count = parameters->tokens * parameters->outputs;
    size_t tiles = count / PROJECTION_TILE + (count % PROJECTION_TILE != 0);
    const TrillimThreadPool *pool = context->thread_pool;
    return tiles == 1 ? project(&task, 0, 0) : pool->run(pool->context, tiles, project, &task);
}

static float gelu_tanh(float x) {
    float cube = (x * x) * x;
    float correction = cube * 0.044715f;
    float argument = (x + correction) * 0.7978845608028654f;
    float scaled = argument * 2.0f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float cdf = probability * 2.0f - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}

static TrillimStatus apply_gate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Gate *task = opaque;
    const TrillimKernelContext *context = task->context;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t first = tile * ACTIVATION_TILE;
    size_t end = task->count - first < ACTIVATION_TILE ? task->count : first + ACTIVATION_TILE;
    for (size_t i = first; i < end; ++i) {
        task->up[i] *= gelu_tanh(task->gate[i]);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_quantized_gelu_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[4], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    if (scratch == NULL || pool->worker_count == 0) {
        return TRILLIM_ERROR;
    }
    const Parameters *p = parameters;
    size_t count = p->tokens * p->hidden;
    float *up = scratch, *gate = up + count;
    TrillimKernel projection = linked_projection();
    LinearParameters expand = {p->tokens, p->width, p->hidden, TRILLIM_BF16, p->up_dtype, TRILLIM_FP32};
    status = run_projection(projection, &expand, inputs[0], inputs[1], up, context);
    if (status == TRILLIM_OK) {
        status = run_projection(projection, &expand, inputs[0], inputs[2], gate, context);
    }
    if (status == TRILLIM_OK) {
        Gate task = {up, gate, count, context};
        size_t tiles = count / ACTIVATION_TILE + (count % ACTIVATION_TILE != 0);
        status = tiles == 1 ? apply_gate(&task, 0, 0) : pool->run(pool->context, tiles, apply_gate, &task);
    }
    if (status != TRILLIM_OK) {
        return status;
    }
    LinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, p->down_dtype, TRILLIM_FP32};
    return run_projection(projection, &down, up, inputs[3], outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_packed_quantized_gelu_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != 'bf16' or output.dtype != 'fp32':\n"
           "        raise ValueError('packed GELU FFN requires BF16 input and FP32 output')\n"
           "    for tensor in (x, up, gate, down, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('packed GELU FFN requires contiguous operands')\n"
           "    grouped = {'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_int8'}\n"
           "    if gate.dtype != up.dtype or up.dtype not in grouped or down.dtype not in grouped:\n"
           "        raise ValueError('packed GELU FFN requires matching grouped up and gate weights and grouped down weights')\n"
           "    if any(len(t.shape) != 2 or _storage_shape(t) != t.shape for t in (up, gate, down)):\n"
           "        raise ValueError('packed GELU FFN requires full canonical weight matrices')\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible packed GELU FFN inputs')\n"
           "    if output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed GELU FFN output')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden, outputs=down.shape[0],\n"
           "                up_dtype=_DTYPE_IDS[up.dtype], down_dtype=_DTYPE_IDS[down.dtype])\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is Silu:\n"
           "        raise ValueError('packed GELU FFN requires an unbiased GELU gate')\n"
           "    return (FFN(x, up=up, gate=gate, down=down, activation=lambda value: Gelu(value, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_quantized_gelu_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_quantized_gelu_ffn_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
