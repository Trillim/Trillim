#include "common.h"
typedef struct { size_t count; } Parameters;
typedef struct { uint16_t *output; const uint16_t *a, *b; const Parameters *p; const TrillimKernelContext *context; } Task;
static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->count > SIZE_MAX / 2) return TRILLIM_ERROR;
    inputs[0] = inputs[1] = outputs[0] = p->count * 2;
    return TRILLIM_OK;
}
static bool can_alias(const void *opaque, size_t output, size_t input) { (void)opaque; return output == 0 && input < 2; }
static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t i = tile * 4096, end = t->p->count - i < 4096 ? t->p->count : i + 4096;
    for (; i < end; ++i) t->output[i] = trillim_to_bf16(trillim_bf16(t->a[i]) + trillim_bf16(t->b[i]));
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_scalar_bf16_add(void *const *outputs, const void *const *inputs,
        void *scratch, const void *opaque, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = opaque;
    Task task = {outputs[0],inputs[0],inputs[1],p,context};
    size_t count = p->count / 4096 + (p->count % 4096 != 0);
    return count == 1 ? apply(&task,0,0) : context->thread_pool->run(context->thread_pool->context,count,apply,&task);
}
TRILLIM_KERNEL const char *trillim_scalar_bf16_add_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_bf16_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {2, 2}, outputs[] = {2};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_bf16_add_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &caps;
}
