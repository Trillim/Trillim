#include "common.h"

#if defined(__APPLE__) && defined(__aarch64__)
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

extern TrillimStatus trillim_neon_add(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
    uint32_t dtype;
} AddParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->count > SIZE_MAX / sizeof(uint16_t)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = inputs[1] = outputs[0] = p->count * sizeof(uint16_t);
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_aarch64_bf16_add(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    const size_t stride = 1;
    const AddParameters add = {p->count, 1, &p->count, &stride, &stride, TRILLIM_BF16};
    return trillim_neon_add(outputs, inputs, scratch, &add, context);
}

TRILLIM_KERNEL const char *trillim_neon_aarch64_bf16_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_binary, _contiguous_strides\n"
           "def bind(c):\n"
           "    output = c.outputs[0]\n"
           "    stride = _contiguous_strides(output.shape)\n"
           "    if output.dtype != 'bf16' or (output.strides or stride) != stride:\n"
           "        raise ValueError('requires contiguous BF16 output')\n"
           "    return dict(count=_contiguous_binary(c)['count'])\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Add(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_aarch64_bf16_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_aarch64_bf16_add_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED};
    return &capabilities;
}
