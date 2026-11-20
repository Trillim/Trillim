#include "common.h"

extern TrillimStatus trillim_neon_packed_gelu_ffn(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_packed_gelu_ffn_parameters(void);
extern TrillimStatus trillim_neon_packed_quantized_gelu_ffn(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_packed_quantized_gelu_ffn_parameters(void);
extern TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, hidden, outputs;
    uint32_t up_dtype, down_dtype;
} Parameters;

typedef struct {
    size_t tokens, width, hidden, outputs;
} Q4Parameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    size_t projected;
    TrillimStatus status;
    if (p->up_dtype == TRILLIM_GRP_INT4) {
        if (p->down_dtype != TRILLIM_GRP_INT4) {
            return TRILLIM_ERROR;
        }
        const Q4Parameters projection = {p->tokens, p->width, p->hidden, p->outputs};
        status = trillim_neon_packed_gelu_ffn_parameters()->requirements(&projection, inputs, &projected, scratch);
    } else {
        status = trillim_neon_packed_quantized_gelu_ffn_parameters()->requirements(p, inputs, &projected, scratch);
    }
    if (status != TRILLIM_OK) {
        return status;
    }
    if (*scratch > SIZE_MAX - projected) {
        return TRILLIM_ERROR;
    }
    outputs[0] = projected / 2;
    *scratch += projected;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_gelu_ffn_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = opaque;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    const size_t hidden_bytes = p->tokens * p->hidden * 2 * sizeof(float);
    void *temporary = (uint8_t *)scratch + hidden_bytes;
    void *projected[] = {temporary};
    TrillimStatus status;
    if (p->up_dtype == TRILLIM_GRP_INT4) {
        const Q4Parameters projection = {p->tokens, p->width, p->hidden, p->outputs};
        status = trillim_neon_packed_gelu_ffn(projected, inputs, scratch, &projection, context);
    } else {
        status = trillim_neon_packed_quantized_gelu_ffn(projected, inputs, scratch, p, context);
    }
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->outputs, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {temporary};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_gelu_ffn_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 4, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_gelu_ffn_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}


TRILLIM_KERNEL const char *trillim_neon_packed_gelu_ffn_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "\n"
           "physical_inputs = FFN_INPUTS = (\"x\", \"up\", \"gate\", \"down\")\n"
           "GELU_FORMATS = {\"grp_int5\", \"dubgrp_int6\", \"grp_int8\"}\n"
           "\n"
           "def contiguous(*bindings) -> None:\n"
           "    for binding in bindings:\n"
           "        stride = _contiguous_strides(binding.shape)\n"
           "        if (binding.strides or stride) != stride:\n"
           "            raise ValueError(\"requires contiguous operands\")\n"
           "\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in FFN_INPUTS)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != \"bf16\" or output.dtype != \"bf16\":\n"
           "        raise ValueError(\"requires BF16 input and output\")\n"
           "    contiguous(x, up, gate, down, output)\n"
           "    allowed_down = {\"grp_int4\"} if up.dtype == \"grp_int4\" else GELU_FORMATS | {\"grp_int4\"}\n"
           "    if up.dtype not in GELU_FORMATS | {\"grp_int4\"} or gate.dtype != up.dtype or down.dtype not in allowed_down:\n"
           "        raise ValueError(\"unsupported GELU FFN formats\")\n"
           "    if any(len(weight.shape) != 2 or _storage_shape(weight) != weight.shape\n"
           "           for weight in (up, gate, down)):\n"
           "        raise ValueError(\"requires complete packed matrices\")\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width or output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError(\"incompatible GELU FFN dimensions\")\n"
           "    up_dtype = _DTYPE_IDS[up.dtype]\n"
           "    group = 256 if up.dtype == \"dubgrp_int6\" else 32\n"
           "    if (width.value is None or not 1024 <= width.value <= 65536\n"
           "        or width.value % group or hidden.value is None or hidden.value < 16\n"
           "        or up.dtype == \"grp_int4\" and hidden.value % 4):\n"
           "        raise ValueError(\"requires aligned fixed-width GELU projections\")\n"
           "    return {\"tokens\": _product(x.shape[:-1]), \"width\": width, \"hidden\": hidden,\n"
           "            \"outputs\": down.shape[0], \"up_dtype\": up_dtype, \"down_dtype\": _DTYPE_IDS[down.dtype]}\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, activation,\n"
           "                 up_bias=None, gate_bias=None, down_bias=None, dtype: DType = \"bf16\") -> tuple[Tensor, ...]:\n"
           "    if activation is Silu or up_bias is not None or gate_bias is not None or down_bias is not None or dtype != \"bf16\":\n"
           "        raise ValueError(\"requires unbiased BF16 GELU FFN\")\n"
           "    return (Cast(FFN(x, up=up, gate=gate, down=down,\n"
           "                     activation=lambda value: Gelu(value, approximate=\"tanh\")), \"bf16\"),)\n";
}
