#include "common.h"

extern TrillimStatus trillim_neon_rotary_frequencies(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_rotary_frequencies_parameters(void);
extern const TrillimCapabilities *trillim_neon_rotary_frequencies_capabilities(void);

typedef struct {
    size_t tokens, heads, dim, rotary_dim;
    float amplitude;
} Parameters;

typedef struct {
    size_t heads, tokens, dim, rotary_dim, rank;
    const size_t *strides;
    size_t position_stride;
    uint32_t position_dtype, input_dtype, output_dtype, interleaved;
    float amplitude;
} RotaryParameters;

static RotaryParameters expand(const Parameters *p, size_t *strides) {
    strides[0] = p->dim;
    strides[1] = p->heads * p->dim;
    strides[2] = 1;
    return (RotaryParameters){
        .heads = p->heads, .tokens = p->tokens, .dim = p->dim, .rotary_dim = p->rotary_dim,
        .rank = 3, .strides = strides, .position_stride = 1, .position_dtype = TRILLIM_INT64,
        .input_dtype = TRILLIM_BF16, .output_dtype = TRILLIM_BF16,
        .interleaved = 0, .amplitude = p->amplitude,
    };
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (p->dim == 0 || p->heads > SIZE_MAX / p->dim) {
        return TRILLIM_ERROR;
    }
    size_t strides[3];
    const RotaryParameters rotary = expand(p, strides);
    return trillim_neon_rotary_frequencies_parameters()->requirements(&rotary, inputs, outputs, scratch);
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rotary_frequencies(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    size_t strides[3];
    const RotaryParameters rotary = expand(parameters, strides);
    return trillim_neon_rotary_frequencies(outputs, inputs, scratch, &rotary, context);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rotary_frequencies_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary, _contiguous_strides\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "physical_inputs = ('x', 'positions', 'frequencies')\n"
           "def bind(c):\n"
           "    x, positions, f = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or positions.dtype != 'int64':\n"
           "        raise ValueError('requires BF16 activations and INT64 positions')\n"
           "    if output.shape != x.shape or positions.shape != x.shape[:1]:\n"
           "        raise ValueError('requires one position per token and unchanged activation shape')\n"
           "    for tensor in (x, positions, f, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    p = _avx2_rotary(c)\n"
           "    if p['interleaved']:\n"
           "        raise ValueError('requires split-half rotary')\n"
           "    if f.dtype != 'fp32' or f.shape != (_constant_expression(c.attrs['dim'] // 2),):\n"
           "        raise ValueError('requires one FP32 frequency per pair')\n"
           "    return {name: p[name] for name in ('tokens', 'heads', 'dim', 'rotary_dim', 'amplitude')}\n"
           "def pattern(x: Tensor, positions: Tensor, frequencies: Tensor, *, "
           "dim: int, theta: float = 10000.0, interleaved: bool = False, "
           "amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, frequencies=frequencies, "
           "interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rotary_frequencies_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"dim", "size_t", offsetof(Parameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(Parameters, rotary_dim), NULL},
        {"amplitude", "float", offsetof(Parameters, amplitude), NULL},
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 3, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rotary_frequencies_capabilities(void) {
    return trillim_neon_rotary_frequencies_capabilities();
}
