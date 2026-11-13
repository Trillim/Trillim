#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    float alpha;
} Parameters;

typedef struct {
    float *output;
    const float *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t count = 1, span;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] && count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    if (count != p->count || !isfinite(p->alpha) ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count ? span * sizeof(float) : 0;
    outputs[0] = count * sizeof(float);
    return TRILLIM_OK;
}

static float activate(float value, float alpha) {
    return value >= 0.0f ? value : (expf(value) - 1.0f) * alpha;
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t first = tile * 1024, end = p->count - first < 1024 ? p->count : first + 1024;
    size_t i = first;
    bool contiguous = trillim_contiguous(p->rank, p->shape, p->strides);
    for (; i < end; ++i) {
        size_t source = contiguous ? i : trillim_offset(i, p->rank, p->shape, p->strides);
        task->output[i] = activate(task->input[source], p->alpha);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_elu(void *const *outputs, const void *const *inputs,
                                void *scratch, const void *parameters,
                                const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    size_t count = p->count / 1024 + (p->count % 1024 != 0);
    return count == 1 ? apply(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, count, apply, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_elu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "import math\n"
           "\n"
           "def pattern(x: Tensor, *, alpha: float=1.0) -> tuple[Tensor, ...]:\n"
           "    return (Elu(x,alpha=alpha),)\n"
           "\n"
           "def bind(c):\n"
           "    x,o=c.inputs['x'],c.outputs[0]\n"
           "    if x.dtype!='fp32' or o.dtype!='fp32' or x.shape!=o.shape or (o.strides or _contiguous_strides(o.shape))!=_contiguous_strides(o.shape):\n"
           "        raise ValueError('ELU requires FP32 input and contiguous output')\n"
           "    alpha=c.attrs.get('alpha',1.0)\n"
           "    if not isinstance(alpha,(int,float)) or not math.isfinite(alpha):\n"
           "        raise ValueError('ELU requires a finite alpha')\n"
           "    return dict(count=_product(x.shape),rank=len(x.shape),shape=x.shape,strides=x.strides or _contiguous_strides(x.shape),alpha=float(alpha))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_elu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"alpha", "float", offsetof(Parameters, alpha), NULL},
    };
    static const size_t alignments[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, alignments, alignments, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_elu_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
