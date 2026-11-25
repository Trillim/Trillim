#include "common.h"

enum { CHUNK = 256 };

typedef struct {
    size_t width;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (p->width > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->width * sizeof(float);
    *scratch = 0;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_cumsum(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    const float *input = inputs[0];
    float *output = outputs[0];
    float value = 0.0f;
    for (size_t first = 0; first < p->width;) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->width - first < CHUNK ? p->width : first + CHUNK;
        for (size_t i = first; i < end; ++i) {
            value += input[i];
            output[i] = value;
        }
        first = end;
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_cumsum_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "physical_inputs = ('x',)\n"
           "\n"
           "def pattern(x: Tensor, *, rank: int) -> tuple[Tensor, ...]:\n"
           "    if rank != 1 or x.dtype != 'fp32' or x.shape is None or len(x.shape.dimensions) != 1:\n"
           "        raise ValueError('CumSum requires a rank-one FP32 vector')\n"
           "    if x.shape.dimensions[0] is None:\n"
           "        # The matcher creates a fresh placeholder without the target's maximum shape.\n"
           "        # Bound its loop counter here. Binding uses the target's actual width.\n"
           "        x._producer.args = x._producer.args.model_copy(update={'maximum_shape': (2**63 - 1,)})\n"
           "    return (CumSum(x, rank=rank),)\n"
           "\n"
           "def bind(context):\n"
           "    x = context.inputs['x']\n"
           "    output, = context.outputs\n"
           "    if (x.dtype != 'fp32' or len(x.shape) != 1 or x.strides != _contiguous_strides(x.shape) or\n"
           "            output.dtype != x.dtype or len(output.shape) != 1):\n"
           "        raise ValueError('CumSum requires matching contiguous FP32 vectors')\n"
           "    return {'width': x.shape[0]}\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_cumsum_parameters(void) {
    static const TrillimParameter fields[] = {
        {"width", "size_t", offsetof(Parameters, width), NULL},
    };
    static const size_t alignments[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 1, 1,
        requirements, NULL, alignments, alignments, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_cumsum_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
