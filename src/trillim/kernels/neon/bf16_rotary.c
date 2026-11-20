#include "common.h"

#if defined(__APPLE__) && defined(__aarch64__)
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

extern TrillimStatus trillim_neon_rotary(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern const TrillimParameterSchema *trillim_neon_rotary_parameters(void);

typedef struct {
    size_t tokens, heads, dim, rotary_dim;
    float frequency_step;
} Parameters;

typedef struct {
    size_t heads, tokens, dim, rotary_dim, rank;
    const size_t *strides;
    size_t position_stride;
    uint32_t position_dtype, input_dtype, output_dtype, interleaved;
    float frequency_step, amplitude;
} RotaryParameters;

static RotaryParameters expand(const Parameters *p, size_t *strides) {
    strides[0] = p->dim;
    strides[1] = p->heads * p->dim;
    strides[2] = 1;
    return (RotaryParameters){
        .heads = p->heads, .tokens = p->tokens, .dim = p->dim, .rotary_dim = p->rotary_dim,
        .rank = 3, .strides = strides, .position_stride = 1, .position_dtype = TRILLIM_INT64,
        .input_dtype = TRILLIM_BF16, .output_dtype = TRILLIM_BF16,
        .interleaved = 0, .frequency_step = p->frequency_step, .amplitude = 1.0f,
    };
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (p->dim == 0 || p->heads > SIZE_MAX / p->dim) {
        return TRILLIM_ERROR;
    }
    size_t strides[3];
    const RotaryParameters rotary = expand(p, strides);
    return trillim_neon_rotary_parameters()->requirements(&rotary, inputs, outputs, scratch);
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rotary(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    size_t strides[3];
    const RotaryParameters rotary = expand(parameters, strides);
    return trillim_neon_rotary(outputs, inputs, scratch, &rotary, context);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary, _contiguous_strides\n"
           "def bind(c):\n"
           "    x, positions, output = c.inputs['x'], c.inputs['positions'], c.outputs[0]\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or positions.dtype != 'int64':\n"
           "        raise ValueError('requires BF16 activations and INT64 positions')\n"
           "    if output.shape != x.shape or positions.shape != x.shape[:1]:\n"
           "        raise ValueError('requires one position per token and unchanged activation shape')\n"
           "    for tensor in (x, positions, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    p = _avx2_rotary(c)\n"
           "    if p['interleaved'] or p['amplitude'] != 1.0:\n"
           "        raise ValueError('requires split-half rotary with unit amplitude')\n"
           "    return {name: p[name] for name in ('tokens', 'heads', 'dim', 'rotary_dim', 'frequency_step')}\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, "
           "amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rotary_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"dim", "size_t", offsetof(Parameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(Parameters, rotary_dim), NULL},
        {"frequency_step", "float", offsetof(Parameters, frequency_step), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
