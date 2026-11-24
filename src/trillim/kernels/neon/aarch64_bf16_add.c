#include "common.h"
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    uint16_t *output;
    const uint16_t *a, *b;
    size_t count;
    const TrillimKernelContext *context;
} Task;

static inline float32x4_t load4(const uint16_t *data) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(data), 16));
}

static inline void store4(uint16_t *data, float32x4_t value) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16(data, vmovn_u32(rounded));
}

static TrillimStatus add(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t i = tile * 4096;
    size_t end = task->count - i < 4096 ? task->count : i + 4096;
    for (; end - i >= 4; i += 4) {
        store4(task->output + i, vaddq_f32(load4(task->a + i), load4(task->b + i)));
    }
    for (; i < end; ++i) {
        if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float a = trillim_load(task->a, i, TRILLIM_BF16).f;
        float b = trillim_load(task->b, i, TRILLIM_BF16).f;
        trillim_store(task->output, i, TRILLIM_BF16, (TrillimValue){.f = a + b});
    }
    return TRILLIM_OK;
}

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
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p->count, context};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = add(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, add, &task);
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
