#include "common.h"
#include <stdlib.h>

extern TrillimStatus trillim_neon_packed_ffn(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_packed_ffn_parameters(void);
extern const TrillimCapabilities *trillim_neon_packed_ffn_capabilities(void);
extern TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

#if defined(__APPLE__) || defined(__linux__)
extern TrillimStatus trillim_neon_try_packed_ffn_bf16(
    void *, const void *const *, void *, const void *, const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_bf16")
#endif
;
#endif

typedef struct {
    size_t tokens, width, hidden;
    uint32_t weight_dtype;
} Parameters;

typedef struct {
    size_t tokens, width, outputs, hidden_width;
    uint32_t input_dtype, output_dtype, gate_dtype, up_dtype, down_dtype;
} FfnParameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static FfnParameters projection_parameters(const Parameters *p) {
    return (FfnParameters){
        p->tokens, p->width, p->width, p->hidden, TRILLIM_BF16, TRILLIM_FP32,
        p->weight_dtype, p->weight_dtype, p->weight_dtype
    };
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (p->width == 0 || p->hidden == 0 ||
        (p->weight_dtype != TRILLIM_GRP_INT4 && p->weight_dtype != TRILLIM_GRP_INT8)) {
        return TRILLIM_ERROR;
    }
    const FfnParameters projection = projection_parameters(p);
    size_t projected, child_scratch, hidden;
    const TrillimStatus status = trillim_neon_packed_ffn_parameters()->requirements(
        &projection, inputs, &projected, &child_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (!trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, &hidden) ||
        child_scratch != hidden) {
        return TRILLIM_ERROR;
    }
    outputs[0] = projected / 2;
    *scratch = hidden;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_grouped_ffn_bf16(
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
    const FfnParameters projection = projection_parameters(p);
#if defined(__APPLE__) || defined(__linux__)
    bool handled;
    const TrillimStatus direct = trillim_neon_try_packed_ffn_bf16(
        outputs[0], inputs, scratch, &projection, context, &handled);
    if (direct != TRILLIM_OK || handled) {
        return direct;
    }
#endif

    /* Replay the original FP32 projection and Cast whenever AQ declines. */
    size_t projected_bytes;
    if (!trillim_packed_payload_size(p->tokens, p->width, TRILLIM_FP32, &projected_bytes)) {
        return TRILLIM_ERROR;
    }
    void *intermediate = malloc(projected_bytes);
    if (intermediate == NULL) {
        return TRILLIM_ERROR;
    }
    void *projected[] = {intermediate};
    TrillimStatus status = trillim_neon_packed_ffn(projected, inputs, scratch, &projection, context);
    if (status == TRILLIM_OK) {
        const size_t count = p->tokens * p->width, stride = 1;
        const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
        const void *cast_inputs[] = {intermediate};
        status = trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
    }
    free(intermediate);
    return status;
}

TRILLIM_KERNEL const char *trillim_neon_packed_grouped_ffn_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(c):\n"
           "    x, up, gate, down = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or x.dtype != 'bf16' or output.dtype != 'bf16' or output.shape != x.shape:\n"
           "        raise ValueError('requires matching BF16 input and output')\n"
           "    for tensor in (x, up, gate, down, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    if up.dtype not in {'grp_int4', 'grp_int8'} or gate.dtype != up.dtype or down.dtype != up.dtype:\n"
           "        raise ValueError('requires Q4 or Q8 weights of a shared format')\n"
           "    if any(len(v.shape) != 2 or _storage_shape(v) != v.shape for v in (up, gate, down)):\n"
           "        raise ValueError('requires complete packed matrices')\n"
           "    hidden, width = up.shape\n"
           "    if up.shape != gate.shape or down.shape != (width, hidden) or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible FFN dimensions')\n"
           "    if width.value is None or hidden.value is None or not width.value or not hidden.value:\n"
           "        raise ValueError('requires fixed nonempty feature widths')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden, weight_dtype=_DTYPE_IDS[up.dtype])\n"
           "def pattern(x: Tensor, up: Tensor, down: Tensor, gate: Tensor, *, activation, "
           "up_bias=None, gate_bias=None, down_bias=None, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if activation is not Silu or dtype != 'bf16' or any(b is not None for b in (up_bias, gate_bias, down_bias)):\n"
           "        raise ValueError('requires an unbiased BF16 SwiGLU cast')\n"
           "    return (Cast(FFN(x, up=up, gate=gate, down=down), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_grouped_ffn_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 4, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_grouped_ffn_bf16_capabilities(void) {
    return trillim_neon_packed_ffn_capabilities();
}
