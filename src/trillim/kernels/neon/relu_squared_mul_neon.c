#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct { size_t count; uint32_t dtype; } TrillimActivationParameters;

typedef struct {
    void *output;
    const void *gate;
    const void *up;
    const TrillimActivationParameters *parameters;
    const TrillimKernelContext *context;
    bool silu;
} TrillimActivationTask;


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

static inline float32x4_t round4(float32x4_t value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) return vcvt_f32_f16(vcvt_f16_f32(value));
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    return vreinterpretq_f32_u32(vshlq_n_u32(rounded, 16));
}

static inline float round_scalar(float value, uint32_t dtype) {
    return dtype == TRILLIM_FP16 ? trillim_fp16(trillim_to_fp16(value)) : trillim_bf16(trillim_to_bf16(value));
}

static inline TrillimStatus trillim_activation_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->dtype);
    if (!trillim_float(p->dtype) || p->count > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * bytes;
    inputs[1] = inputs[0];
    outputs[0] = inputs[0];
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_activation_task_neon(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimActivationTask *task = (const TrillimActivationTask *)opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 4096;
    size_t end = task->parameters->count - begin < 4096 ? task->parameters->count : begin + 4096;
    size_t i = begin;
    for (; end - i >= 4; i += 4) {
        float32x4_t gate = load4(task->gate, i, task->parameters->dtype);
        float32x4_t value = vbslq_f32(vcgeq_f32(gate, vdupq_n_f32(0)), gate, vdupq_n_f32(0));
        value = vmulq_f32(value, value);
        if (task->parameters->dtype != TRILLIM_FP32) value = round4(value, task->parameters->dtype);
        store4(task->output, i, task->parameters->dtype, vmulq_f32(value, load4(task->up, i, task->parameters->dtype)));
    }
    for (; i < end; ++i) {
        float gate = trillim_load(task->gate, i, task->parameters->dtype).f;
        float value;
        if (task->silu) {
            float e = expf(-fabsf(gate));
            value = gate * (gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
        } else {
            value = gate >= 0.0f ? gate : 0.0f;
            value *= value;
            if (task->parameters->dtype != TRILLIM_FP32) value = round_scalar(value, task->parameters->dtype);
        }
        TrillimValue output = {.f = value * trillim_load(task->up, i, task->parameters->dtype).f};
        trillim_store(task->output, i, task->parameters->dtype, output);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_relu_squared_mul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    TrillimActivationTask task = {outputs[0], inputs[0], inputs[1], p, context, false};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = trillim_activation_task_neon(&task, tile, 0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, trillim_activation_task_neon, &task);
}

TRILLIM_KERNEL const char *trillim_neon_relu_squared_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _product, _contiguous_strides, _DTYPE_IDS\n"
           "def floating(*values):\n"
           "    for v in values:\n"
           "        if v.dtype not in ('fp16', 'bf16', 'fp32') or (v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape):\n"
           "            raise ValueError('kernel requires contiguous floating tensors')\n"
           "def bind(c):\n"
           "    a, b, o = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    floating(a,b,o)\n"
           "    if a.shape != b.shape or a.shape != o.shape or a.dtype != b.dtype or a.dtype != o.dtype:\n"
           "        raise ValueError('binary kernel requires matching floating operands')\n"
           "    return dict(count=_product(o.shape), dtype=_DTYPE_IDS[o.dtype])\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(ReluSquared(a),b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_relu_squared_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimActivationParameters, count), NULL},
        {"dtype", "uint32_t", offsetof(TrillimActivationParameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimActivationParameters), TRILLIM_ALIGNOF(TrillimActivationParameters),
        2, fields, 2, 1, trillim_activation_requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_relu_squared_mul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
