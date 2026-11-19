#include "common.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t tokens;
    size_t routes;
    size_t experts;
    size_t n;
    size_t k;
    size_t a_token_stride;
    size_t a_route_stride;
    size_t index_token_stride;
    size_t index_route_stride;
    uint32_t a_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *activation;
    const uint8_t *weight;
    const uint8_t *scales;
    const void *indices;
    const Parameters *parameters;
    const TrillimKernelContext *context;
    size_t columns;
    size_t tiles;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !p->experts || !p->n || !p->k || p->k % 32) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    size_t a_shape[] = {p->tokens, p->routes, p->k};
    size_t a_strides[] = {p->a_token_stride, p->a_route_stride, 1};
    size_t index_strides[] = {p->index_token_stride, p->index_route_stride};
    size_t a_span, index_span;
    size_t a_bytes = trillim_itemsize(p->a_dtype);
    if (!trillim_span(3, a_shape, a_strides, &a_span) || a_span > SIZE_MAX / a_bytes ||
        !trillim_span(2, a_shape, index_strides, &index_span) || index_span > SIZE_MAX / sizeof(int64_t) ||
        p->n > SIZE_MAX / p->k ||
        !trillim_packed_payload_size(p->experts, p->n * p->k, TRILLIM_MXFP4, &inputs[1]) ||
        p->tokens > SIZE_MAX / p->routes ||
        p->tokens * p->routes > SIZE_MAX / p->n / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a_span * a_bytes;
    inputs[2] = index_span * sizeof(int64_t);
    outputs[0] = p->tokens * p->routes * p->n * sizeof(float);
    return TRILLIM_OK;
}

static inline float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static inline bool mxfp4_palette(const uint8_t *codes, uint8_t exponent, float palette[16]) {
    static const float magnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    /* The packed gather rejects NaN scales and codes that decode to infinity. */
    if (exponent >= 253) {
        if (exponent == 255) {
            return false;
        }
        uint8_t limit = exponent == 253 ? 6 : 4;
        for (size_t i = 0; i < 16; ++i) {
            if ((codes[i] & 7) >= limit || ((codes[i] >> 4) & 7) >= limit) {
                return false;
            }
        }
    }
    float scale = trillim_mxfp4_scale(exponent);
    for (size_t i = 0; i < 8; ++i) {
        palette[i] = magnitudes[i] * scale;
        palette[i + 8] = -palette[i];
    }
    return true;
}

static inline bool load_activation(const void *data, size_t offset, uint32_t dtype, float row[32]) {
    switch (dtype) {
    case TRILLIM_FP32:
        memcpy(row, (const uint8_t *)data + offset * sizeof(float), 32 * sizeof(float));
        return true;
    case TRILLIM_FP16:
        for (size_t i = 0; i < 32; ++i) {
            row[i] = trillim_load(data, offset + i, TRILLIM_FP16).f;
        }
        return true;
    case TRILLIM_BF16:
        for (size_t i = 0; i < 32; ++i) {
            row[i] = trillim_load(data, offset + i, TRILLIM_BF16).f;
        }
        return true;
    default:
        return false;
    }
}

static inline int64_t expert_id(const void *indices, size_t token, size_t route, const Parameters *p) {
    size_t offset = token * p->index_token_stride + route * p->index_route_stride;
    return trillim_load(indices, offset, TRILLIM_INT64).i;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t item = tile / task->tiles;
    size_t token = item / p->routes;
    size_t route = item % p->routes;
    size_t expert = (size_t)expert_id(task->indices, token, route, p);
    size_t a_offset = token * p->a_token_stride + route * p->a_route_stride;
    size_t groups = p->k / 32;
    size_t first = tile % task->tiles * task->columns;
    size_t end = p->n - first < task->columns ? p->n : first + task->columns;
    for (size_t column = first; column < end; column += 4) {
        size_t count = end - column < 4 ? end - column : 4;
        size_t block = (expert * p->n + column) * groups;
        float sums[4] = {0};
        for (size_t group = 0; group < groups; ++group) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float palette[4][16];
            const uint8_t *codes[4];
            for (size_t j = 0; j < count; ++j) {
                size_t offset = block + j * groups + group;
                codes[j] = task->weight + offset * 16;
                if (!mxfp4_palette(codes[j], task->scales[offset], palette[j])) {
                    return TRILLIM_ERROR;
                }
            }
            float a[32];
            if (!load_activation(task->activation, a_offset + group * 32, p->a_dtype, a)) {
                return TRILLIM_ERROR;
            }
            for (size_t i = 0; i < 16; ++i) {
                for (size_t j = 0; j < count; ++j) {
                    uint8_t code = codes[j][i];
                    sums[j] = multiply_add(a[i * 2], palette[j][code & 15], sums[j]);
                    sums[j] = multiply_add(a[i * 2 + 1], palette[j][code >> 4], sums[j]);
                }
            }
        }
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        memcpy((uint8_t *)task->output + (item * p->n + column) * sizeof(float), sums, count * sizeof(float));
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_routed_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    if (!pool->worker_count) {
        return TRILLIM_ERROR;
    }
    size_t items = p->tokens * p->routes;
    for (size_t item = 0; item < items; ++item) {
        if ((item & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t expert = expert_id(inputs[2], item / p->routes, item % p->routes, p);
        if (expert < 0 || (uint64_t)expert >= p->experts) {
            return TRILLIM_ERROR;
        }
    }
    size_t columns = p->n / pool->worker_count / 4 / 16 * 16;
    if (columns < 16) {
        columns = 16;
    }
    size_t tiles = p->n / columns + (p->n % columns != 0);
    const uint8_t *weight = inputs[1];
    const uint8_t *scales = weight + p->experts * p->n * (p->k / 32) * 16;
    Task task = {outputs[0], inputs[0], weight, scales, inputs[2], p, context, columns, tiles};
    size_t tasks = items * tiles;
    return tasks == 1 ? project(&task, 0, 0) : pool->run(pool->context, tasks, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_routed_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _native_dtype, _product, _storage_shape\n"
           "physical_inputs = ('a', 'weight', 'indices')\n"
           "def bind(c):\n"
           "    a, weight, indices = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    if a.dtype not in floating or out.dtype != 'fp32' or weight.dtype != 'mxfp4' or indices.dtype != 'int64':\n"
           "        raise ValueError('routed matmul requires floating activations, MXFP4 weights, int64 indices and an FP32 output')\n"
           "    if len(a.shape) != 4 or len(weight.shape) != 3 or len(indices.shape) != 3 or len(out.shape) != 4:\n"
           "        raise ValueError('routed matmul requires a rank-three expert bank and rank-four matvecs')\n"
           "    experts, n, k = weight.shape\n"
           "    if any(d.value is None or d.value <= 0 for d in weight.shape) or k.value % 32:\n"
           "        raise ValueError('routed matmul requires fixed expert shapes and group-aligned weight rows')\n"
           "    if a.shape[2].value != 1 or indices.shape[2].value != 1 or a.shape[3] != k or out.shape[2].value != 1 or out.shape[3] != n:\n"
           "        raise ValueError('routed matmul requires one activation row and one expert index per route')\n"
           "    ws = _contiguous_strides(weight.shape)\n"
           "    if (weight.strides or ws) != ws or _storage_shape(weight) != (experts, _product(weight.shape[1:])):\n"
           "        raise ValueError('routed matmul requires contiguous expert-major weights')\n"
           "    ast = a.strides or _contiguous_strides(a.shape)\n"
           "    ist = _contiguous_strides(indices.shape)\n"
           "    if ast[3].value != 1 or (indices.strides or ist) != ist:\n"
           "        raise ValueError('routed matmul requires contiguous activation rows and indices')\n"
           "    at, ar = _broadcast_strides(a.shape[:2], out.shape[:2], ast[:2])\n"
           "    it, ir = _broadcast_strides(indices.shape[:2], out.shape[:2], ist[:2])\n"
           "    return dict(tokens=out.shape[0], routes=out.shape[1], experts=experts, n=n, k=k,\n"
           "                a_token_stride=at, a_route_stride=ar, index_token_stride=it, index_route_stride=ir,\n"
           "                a_dtype=_native_dtype(a.dtype))\n"
           "def pattern(a: Tensor, weight: Tensor, indices: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, Permute(Gather(weight, indices=indices), (0, 1, 3, 2))),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_routed_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"routes", "size_t", offsetof(Parameters, routes), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
        {"n", "size_t", offsetof(Parameters, n), NULL},
        {"k", "size_t", offsetof(Parameters, k), NULL},
        {"a_token_stride", "size_t", offsetof(Parameters, a_token_stride), NULL},
        {"a_route_stride", "size_t", offsetof(Parameters, a_route_stride), NULL},
        {"index_token_stride", "size_t", offsetof(Parameters, index_token_stride), NULL},
        {"index_route_stride", "size_t", offsetof(Parameters, index_route_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_routed_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
