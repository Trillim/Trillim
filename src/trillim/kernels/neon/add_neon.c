#include "common.h"
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
    uint32_t dtype;
} Parameters;
typedef struct {
    void *output;
    const void *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
    bool flat;
    size_t a_step, b_step;
} Task;

static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index, vreinterpret_u16_f16(vcvt_f16_f32(value)));
        return;
    }
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || (p->rank && (!p->shape || !p->a_strides || !p->b_strides))) return TRILLIM_ERROR;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (!p->shape[axis]) return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) return TRILLIM_ERROR;
        count *= p->shape[axis];
    }
    size_t a_span, b_span;
    size_t bytes = trillim_itemsize(p->dtype);
    if (count != p->count || p->count > SIZE_MAX / bytes ||
        !trillim_span(p->rank, p->shape, p->a_strides, &a_span) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b_span) ||
        a_span > SIZE_MAX / bytes || b_span > SIZE_MAX / bytes) return TRILLIM_ERROR;
    inputs[0] = a_span * bytes;
    inputs[1] = b_span * bytes;
    outputs[0] = p->count * bytes;
    return TRILLIM_OK;
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t i = tile * 4096;
    size_t end = p->count - i < 4096 ? p->count : i + 4096;
    while (i < end) {
        size_t limit = end;
        size_t a = i * task->a_step, b = i * task->b_step;
        size_t a_step = task->a_step, b_step = task->b_step;
        if (!task->flat) {
            size_t width = p->shape[p->rank - 1];
            size_t remaining = width - i % width;
            if (remaining < end - i) limit = i + remaining;
            a = trillim_offset(i, p->rank, p->shape, p->a_strides);
            b = trillim_offset(i, p->rank, p->shape, p->b_strides);
            a_step = p->a_strides[p->rank - 1];
            b_step = p->b_strides[p->rank - 1];
        }
        if (a_step <= 1 && b_step <= 1) {
            for (; limit - i >= 4; i += 4, a += a_step * 4, b += b_step * 4) {
                float32x4_t x = a_step ? load4(task->a, a, p->dtype) : vdupq_n_f32(trillim_load(task->a, a, p->dtype).f);
                float32x4_t y = b_step ? load4(task->b, b, p->dtype) : vdupq_n_f32(trillim_load(task->b, b, p->dtype).f);
                store4(task->output, i, p->dtype, vaddq_f32(x, y));
            }
        }
        for (; i < limit; ++i, a += a_step, b += b_step) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            TrillimValue value = {.f = trillim_load(task->a, a, p->dtype).f + trillim_load(task->b, b, p->dtype).f};
            trillim_store(task->output, i, p->dtype, value);
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (output != 0 || input >= 2) return false;
    if (!parameters) return true;
    const Parameters *p = parameters;
    return trillim_contiguous(p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_add(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (!p->count) return TRILLIM_OK;
    size_t a_span, b_span;
    if (!trillim_span(p->rank, p->shape, p->a_strides, &a_span) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b_span)) return TRILLIM_ERROR;
    bool a_contiguous = trillim_contiguous(p->rank, p->shape, p->a_strides);
    bool b_contiguous = trillim_contiguous(p->rank, p->shape, p->b_strides);
    bool flat = (a_contiguous || a_span == 1) && (b_contiguous || b_span == 1);
    Task task = {outputs[0], inputs[0], inputs[1], p, context, flat,
                 a_contiguous ? 1 : 0, b_contiguous ? 1 : 0};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_neon_add_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary\n"
           "def bind(c):\n"
           "    p = _binary(c)\n"
           "    if p['dtype'] not in (9,10,11):\n"
           "        raise ValueError('NEON binary kernel requires floating operands')\n"
           "    return p\n"
           "def pattern(a: Tensor,b: Tensor) -> tuple[Tensor,...]:\n"
           "    return (Add(a,b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_add_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_add_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED};
    return &capabilities;
}
