#include "common.h"

enum { EXPERTS = 32, ROUTES = 4 };

typedef struct {
    size_t rows;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = outputs[1] = *scratch = 0;
    if (p->rows > SIZE_MAX / (EXPERTS * sizeof(uint16_t))) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * EXPERTS * sizeof(uint16_t);
    outputs[0] = p->rows * ROUTES * sizeof(uint16_t);
    outputs[1] = p->rows * ROUTES * sizeof(int64_t);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_topk4(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t row = 0; row < p->rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t source = row * EXPERTS;
        size_t target = row * ROUTES;
        uint32_t available = UINT32_MAX;
        for (size_t route = 0; route < ROUTES; ++route) {
            size_t selected = EXPERTS;
            float best = 0;
            for (size_t expert = 0; expert < EXPERTS; ++expert) {
                if (!(available & (UINT32_C(1) << expert))) {
                    continue;
                }
                float value = trillim_load(inputs[0], source + expert, TRILLIM_BF16).f;
                /* Follow the existing masked Argmax's ordered ties and NaN behavior. */
                if (selected == EXPERTS || !(best >= value)) {
                    best = value;
                    selected = expert;
                }
            }
            available &= ~(UINT32_C(1) << selected);
            trillim_store(outputs[0], target + route, TRILLIM_BF16, (TrillimValue){.f = best});
            trillim_store(outputs[1], target + route, TRILLIM_INT64, (TrillimValue){.i = (int64_t)selected});
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_topk4_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "physical_inputs = ('x',)\n"
           "def pattern(x: Tensor, *, k: int = 4, rank: int = 2) -> tuple[Tensor, ...]:\n"
           "    if k != 4 or rank != 2:\n"
           "        raise ValueError('expected four routes and rank two')\n"
           "    return TopK(x, k=k, rank=rank)\n"
           "def bind(c):\n"
           "    x = c.inputs['x']\n"
           "    scores, indices = c.outputs\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or x.shape[1].value != 32:\n"
           "        raise ValueError('expected contiguous BF16 logits for 32 experts')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('expected contiguous logits')\n"
           "    if scores.dtype != 'bf16' or indices.dtype != 'int64':\n"
           "        raise ValueError('expected BF16 scores and int64 indices')\n"
           "    if any(len(v.shape) != 2 or v.shape[1].value != 4 for v in (scores, indices)):\n"
           "        raise ValueError('expected four outputs per logits row')\n"
           "    if any(v.strides and v.strides != _contiguous_strides(v.shape) for v in (scores, indices)):\n"
           "        raise ValueError('expected contiguous score and index outputs')\n"
           "    return dict(rows=x.shape[0])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_topk4_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1, 1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields,
        1, 2, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_topk4_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
