#include "common.h"

typedef struct {
    size_t rows, width;
    float eps, weight_offset;
    uint32_t weight_dtype;
} Parameters;
typedef struct {
    const uint16_t *input;
    const void *weight;
    uint16_t *output;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if ((p->weight_dtype != TRILLIM_FP16 && p->weight_dtype != TRILLIM_BF16) ||
        p->width > SIZE_MAX / 2 || (p->width && p->rows > SIZE_MAX / 2 / p->width)) return TRILLIM_ERROR;
    inputs[0] = outputs[0] = p->rows * p->width * 2;
    inputs[1] = p->width * 2;
    return TRILLIM_OK;
}
static bool can_alias(const void *opaque, size_t output, size_t input) {
    (void)opaque;
    return output == 0 && input == 0;
}
static TrillimStatus normalize(void *opaque, size_t pair, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    size_t first = pair * 2, rows = p->rows - first < 2 ? 1 : 2;
    float denominator[2];
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *x = t->input + (first + row) * p->width;
        float sum = 0;
        size_t col = 0;
        for (; col < p->width; ++col) {
            if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            float x0 = trillim_bf16(x[col]);
            sum += x0 * x0;
        }
        denominator[row] = sqrtf(sum / (float)p->width + p->eps);
    }
    size_t col = 0;
    for (; col < p->width; ++col) {
        if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        float w = trillim_load(t->weight, col, p->weight_dtype).f + p->weight_offset;
        for (size_t row = 0; row < rows; ++row) {
            size_t i = (first + row) * p->width + col;
            t->output[i] = trillim_to_bf16((trillim_bf16(t->input[i]) / denominator[row]) * w);
        }
    }
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_bf16_rmsnorm(void *const *outputs, const void *const *inputs,
        void *scratch, const void *opaque, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = opaque;
    if (!p->width || !p->rows) return TRILLIM_OK;
    Task task = {inputs[0], inputs[1], outputs[0], p, context};
    size_t count = p->rows / 2 + (p->rows % 2 != 0);
    return count == 1 ? normalize(&task, 0, 0) : context->thread_pool->run(context->thread_pool->context, count, normalize, &task);
}
TRILLIM_KERNEL const char *trillim_scalar_bf16_rmsnorm_tdl(void) {
    return "from trillim.tdl import DType, RMSNorm, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _DTYPE_IDS\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or output.shape != x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('BF16 RMSNorm requires a matching channel vector')\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or weight.dtype not in ('bf16', 'fp16'):\n"
           "        raise ValueError('requires BF16 activations and BF16 or FP16 weights')\n"
           "    for value in (x, weight, output):\n"
           "        if value.strides and value.strides != _contiguous_strides(value.shape):\n"
           "            raise ValueError('BF16 RMSNorm requires contiguous BF16 tensors')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    return dict(rows=_product(x.shape[:-1]), width=x.shape[-1], eps=float(eps), weight_offset=float(offset), weight_dtype=_DTYPE_IDS[weight.dtype])\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bf16_rmsnorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
    };
    static const size_t inputs[] = {2, 1}, outputs[] = {2};
    static const TrillimParameterSchema schema = {sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bf16_rmsnorm_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
