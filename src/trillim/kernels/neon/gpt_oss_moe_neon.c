#include "common.h"
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { ROUTES = 4, PANEL_ROWS = 128 };

typedef struct {
    size_t rows, width, hidden, experts;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 7 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->rows > 65536 || p->experts != 32 || p->width == 0 || p->hidden == 0 ||
        p->width > 8192 || p->hidden > 8192 || p->width % 32 || p->hidden % 32) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->experts * p->width * sizeof(uint16_t);
    inputs[2] = p->experts * sizeof(uint16_t);
    inputs[3] = p->experts * p->hidden * 2 * (p->width / 32) * 17;
    inputs[4] = p->experts * p->width * (p->hidden / 32) * 17;
    inputs[5] = p->experts * p->hidden * 2 * sizeof(uint16_t);
    inputs[6] = p->experts * p->width * sizeof(uint16_t);
    outputs[0] = p->rows * p->width * sizeof(float);
    if (p->rows != 0) {
        size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
        size_t slots = rows * ROUTES;
        size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
        *scratch = slots * (projection_width + p->hidden + 1) * sizeof(float) +
            slots * sizeof(int64_t) + (2 * (p->experts + 1) + slots) * sizeof(size_t) +
            (rows * p->experts + slots) * sizeof(uint16_t);
    }
    return TRILLIM_OK;
}

typedef struct {
    size_t batches, m, n, k, rank;
    const size_t *batch_shape, *a_batch_strides, *b_batch_strides;
    size_t a_row_stride, a_col_stride, b_row_stride, b_col_stride;
    uint32_t a_dtype, b_dtype, out_dtype, linear_dtype, bias_dtype;
    size_t a_storage_rows, a_storage_columns, b_storage_rows, b_storage_columns;
} RouterParameters;

typedef struct {
    size_t tokens, routes, experts, n, k;
    size_t a_token_stride, a_route_stride, index_token_stride, index_route_stride;
    uint32_t a_dtype;
} ProjectionParameters;

typedef struct {
    size_t rows;
} TopKParameters;

typedef struct {
    size_t rows, width;
    uint32_t dtype;
} SoftmaxParameters;

typedef struct {
    size_t rows, width;
} SwiGLUParameters;

extern TrillimStatus trillim_neon_dense_bias(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern TrillimStatus trillim_neon_scalar_topk4(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern TrillimStatus trillim_neon_normalize_softmax(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern TrillimStatus trillim_neon_routed_matmul(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern TrillimStatus trillim_neon_interleaved_clipped_swiglu(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);

extern const TrillimParameterSchema *trillim_neon_dense_bias_parameters(void);
extern const TrillimParameterSchema *trillim_neon_scalar_topk4_parameters(void);
extern const TrillimParameterSchema *trillim_neon_normalize_softmax_parameters(void);
extern const TrillimParameterSchema *trillim_neon_routed_matmul_parameters(void);
extern const TrillimParameterSchema *trillim_neon_interleaved_clipped_swiglu_parameters(void);

typedef struct {
    float *projected, *hidden;
    int64_t *indices;
    size_t *routing;
    float *normalization;
    uint16_t *logits, *scores;
} Workspace;

typedef struct {
    float *projected;
    const uint16_t *bias;
    const int64_t *indices;
    size_t width;
    const TrillimKernelContext *context;
} BiasTask;

typedef struct {
    float *output;
    const float *projected;
    const uint16_t *bias, *weights;
    const int64_t *indices;
    size_t width;
    const TrillimKernelContext *context;
} OutputTask;

static inline float32x4_t bias_values(const uint16_t *bias) {
    uint16x4_t bits = vld1_u16(bias);
    uint16x4_t magnitude = vand_u16(bits, vdup_n_u16(0x7fff));
    uint16x4_t nan = vcgt_u16(magnitude, vdup_n_u16(0x7f80));
    /* Gather writes a quiet BF16 NaN before the bias is widened. */
    bits = vorr_u16(bits, vand_u16(nan, vdup_n_u16(0x0040)));
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static TrillimStatus add_up_bias(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const BiasTask *task = opaque;
    float *row = task->projected + slot * task->width;
    const uint16_t *bias = task->bias + (size_t)task->indices[slot] * task->width;
    for (size_t i = 0; i < task->width; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        vst1q_f32(row + i, vaddq_f32(vld1q_f32(row + i), bias_values(bias + i)));
    }
    return TRILLIM_OK;
}

static TrillimStatus finish_output(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const OutputTask *task = opaque;
    size_t width = task->width, slot = token * ROUTES;
    float *destination = task->output + token * width;
    const float *projected[ROUTES];
    const uint16_t *bias[ROUTES];
    float weights[ROUTES];
    for (size_t route = 0; route < ROUTES; ++route) {
        projected[route] = task->projected + (slot + route) * width;
        bias[route] = task->bias + (size_t)task->indices[slot + route] * width;
        weights[route] = trillim_bf16(task->weights[slot + route]);
    }
    for (size_t i = 0; i < width; i += 4) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t sum = vdupq_n_f32(0);
        for (size_t route = 0; route < ROUTES; ++route) {
            float32x4_t projected_with_bias = vaddq_f32(vld1q_f32(projected[route] + i), bias_values(bias[route] + i));
            sum = vaddq_f32(sum, vmulq_n_f32(projected_with_bias, weights[route]));
        }
        vst1q_f32(destination + i, sum);
    }
    return TRILLIM_OK;
}

static TrillimStatus panel(
    float *output, const uint16_t *activation, const void *const *inputs,
    const Parameters *p, size_t rows, const Workspace *workspace, const TrillimKernelContext *context
) {
    size_t width = p->width, hidden = p->hidden, experts = p->experts;
    RouterParameters router = {
        .batches = 1, .m = rows, .n = experts, .k = width,
        .a_row_stride = width, .a_col_stride = 1, .b_row_stride = 1, .b_col_stride = width,
        .a_dtype = TRILLIM_BF16, .b_dtype = TRILLIM_BF16, .out_dtype = TRILLIM_BF16,
        .linear_dtype = TRILLIM_BF16, .bias_dtype = TRILLIM_BF16,
        .a_storage_rows = rows, .a_storage_columns = width,
        .b_storage_rows = experts, .b_storage_columns = width,
    };
    TrillimStatus status = trillim_neon_dense_bias(
        (void *[]){workspace->logits}, (const void *[]){activation, inputs[1], inputs[2]},
        NULL, &router, context
    );
    if (status != TRILLIM_OK) return status;
    TopKParameters topk = {rows};
    status = trillim_neon_scalar_topk4(
        (void *[]){workspace->scores, workspace->indices}, (const void *[]){workspace->logits},
        NULL, &topk, context
    );
    if (status != TRILLIM_OK) return status;
    SoftmaxParameters softmax = {rows, ROUTES, TRILLIM_BF16};
    status = trillim_neon_normalize_softmax(
        (void *[]){workspace->scores}, (const void *[]){workspace->scores},
        workspace->normalization, &softmax, context
    );
    if (status != TRILLIM_OK) return status;
    ProjectionParameters up = {
        .tokens = rows, .routes = ROUTES, .experts = experts, .n = hidden * 2, .k = width,
        .a_token_stride = width, .a_route_stride = 0,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_BF16,
    };
    status = trillim_neon_routed_matmul(
        (void *[]){workspace->projected}, (const void *[]){activation, inputs[3], workspace->indices},
        workspace->routing, &up, context
    );
    if (status != TRILLIM_OK) return status;
    size_t slots = rows * ROUTES;
    const TrillimThreadPool *pool = context->thread_pool;
    BiasTask bias = {workspace->projected, inputs[5], workspace->indices, hidden * 2, context};
    if (slots > 16 && slots > 32768 / bias.width) {
        status = pool->run(pool->context, slots, add_up_bias, &bias);
    } else {
        for (size_t slot = 0; slot < slots && status == TRILLIM_OK; ++slot) {
            status = add_up_bias(&bias, slot, 0);
        }
    }
    if (status != TRILLIM_OK) return status;
    SwiGLUParameters swiglu = {slots, hidden};
    status = trillim_neon_interleaved_clipped_swiglu(
        (void *[]){workspace->hidden}, (const void *[]){workspace->projected},
        NULL, &swiglu, context
    );
    if (status != TRILLIM_OK) return status;
    ProjectionParameters down = {
        .tokens = rows, .routes = ROUTES, .experts = experts, .n = width, .k = hidden,
        .a_token_stride = ROUTES * hidden, .a_route_stride = hidden,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_FP32,
    };
    status = trillim_neon_routed_matmul(
        (void *[]){workspace->projected}, (const void *[]){workspace->hidden, inputs[4], workspace->indices},
        workspace->routing, &down, context
    );
    if (status != TRILLIM_OK) return status;
    OutputTask result = {output, workspace->projected, inputs[6], workspace->scores, workspace->indices, width, context};
    return rows == 1 ? finish_output(&result, 0, 0) : pool->run(pool->context, rows, finish_output, &result);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gpt_oss_moe(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->rows == 0) {
        return TRILLIM_OK;
    }
    size_t spans[7], output_span, required;
    if (!scratch || requirements(p, spans, &output_span, &required) != TRILLIM_OK ||
        trillim_neon_dense_bias_parameters()->size != sizeof(RouterParameters) ||
        trillim_neon_scalar_topk4_parameters()->size != sizeof(TopKParameters) ||
        trillim_neon_normalize_softmax_parameters()->size != sizeof(SoftmaxParameters) ||
        trillim_neon_routed_matmul_parameters()->size != sizeof(ProjectionParameters) ||
        trillim_neon_interleaved_clipped_swiglu_parameters()->size != sizeof(SwiGLUParameters)) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
    size_t slots = rows * ROUTES;
    size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
    Workspace workspace;
    workspace.projected = scratch;
    workspace.hidden = workspace.projected + slots * projection_width;
    workspace.indices = (int64_t *)(workspace.hidden + slots * p->hidden);
    workspace.routing = (size_t *)(workspace.indices + slots);
    workspace.normalization = (float *)(workspace.routing + 2 * (p->experts + 1) + slots);
    workspace.logits = (uint16_t *)(workspace.normalization + slots);
    workspace.scores = workspace.logits + rows * p->experts;

    const uint16_t *activation = inputs[0];
    float *output = outputs[0];
    for (size_t first = 0; first < p->rows; first += rows) {
        size_t length = p->rows - first < rows ? p->rows - first : rows;
        TrillimStatus status = panel(output + first * p->width, activation + first * p->width,
                                     inputs, p, length, &workspace, context);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_neon_gpt_oss_moe_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'router', 'router_bias', 'gate_up', 'down', 'gate_up_bias', 'down_bias')\n"
           "def pattern(x: Tensor, *, router: Tensor, router_bias: Tensor, gate_up: Tensor, down: Tensor,\n"
           "            gate_up_bias: Tensor, down_bias: Tensor, top_k: int = 4, limit: float = 7.0) -> tuple[Tensor, ...]:\n"
           "    if top_k != 4 or limit != 7.0 or x.dtype != 'bf16' or router.dtype != 'bf16' or gate_up.dtype != 'mxfp4' or down.dtype != 'mxfp4':\n"
           "        raise ValueError('expected four routes, BF16 logits, and MXFP4 experts')\n"
           "    return (GptOssMoE(x, router=router, router_bias=router_bias, gate_up=gate_up, down=down,\n"
           "                     gate_up_bias=gate_up_bias, down_bias=down_bias, top_k=top_k, limit=limit),)\n"
           "def bind(c):\n"
           "    x, router, rb, gate_up, down, gb, db = (c.inputs[k] for k in physical_inputs)\n"
           "    if tuple(len(v.shape) for v in (x, router, rb, gate_up, down, gb, db)) != (2, 2, 1, 3, 3, 2, 2):\n"
           "        raise ValueError('expected contiguous activation and expert banks')\n"
           "    if any(v.strides and v.strides != _contiguous_strides(v.shape) for v in (x, router, rb, gate_up, down, gb, db)):\n"
           "        raise ValueError('noncontiguous MoE input')\n"
           "    if rb.dtype != 'bf16' or gb.dtype != 'bf16' or db.dtype != 'bf16' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('expected BF16 biases and FP32 output')\n"
           "    experts, doubled, width = gate_up.shape\n"
           "    if any(_storage_shape(w) != (experts, _product(w.shape[1:])) for w in (gate_up, down)):\n"
           "        raise ValueError('expected canonical MXFP4 expert banks')\n"
           "    if router.shape != (experts, width) or rb.shape != (experts,) or gb.shape != (experts, doubled) or down.shape[:2] != (experts, width) or db.shape != (experts, width):\n"
           "        raise ValueError('incompatible expert bank dimensions')\n"
           "    hidden = down.shape[-1]\n"
           "    if hidden.value is None or doubled.value != 2 * hidden.value or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible expert projection widths')\n"
           "    out = c.outputs[0]\n"
           "    if len(out.shape) != 2 or out.shape[-1] != width or (out.strides and out.strides != _contiguous_strides(out.shape)):\n"
           "        raise ValueError('expected contiguous FP32 output rows')\n"
           "    return dict(rows=x.shape[0], width=width, hidden=hidden, experts=experts)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gpt_oss_moe_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
    };
    static const size_t inputs[] = {2, 2, 2, 1, 1, 2, 2}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields,
        7, 1, requirements, NULL, inputs, outputs, 64,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gpt_oss_moe_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
