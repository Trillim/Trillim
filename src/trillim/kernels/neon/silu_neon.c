#include "common.h"
#include <arm_neon.h>
#include <math.h>

enum { TILE = 1024 };

typedef struct {
    size_t count;
    uint32_t dtype;
} Parameters;

typedef struct {
    float *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if ((p->dtype != TRILLIM_BF16 && p->dtype != TRILLIM_FP32) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->dtype);
    outputs[0] = p->count * sizeof(float);
    return TRILLIM_OK;
}

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float sigmoid(float value, uint32_t dtype) {
    float e = round_dtype(expf(-fabsf(value)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    return value >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
}

static float32x4_t exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

static float32x4_t sigmoid4(float32x4_t value) {
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(value)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    return vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0f)), positive, negative);
}

static TrillimStatus sigmoid_tile(
    const void *input, float *output, size_t count, uint32_t dtype,
    const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t i = 0;
    if (dtype == TRILLIM_FP32) {
        for (; count - i >= 4; i += 4) {
            vst1q_f32(output + i, sigmoid4(vld1q_f32((const float *)input + i)));
        }
    }
    for (; i < count; ++i) {
        output[i] = sigmoid(trillim_load(input, i, dtype).f, dtype);
    }
    return TRILLIM_OK;
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * TILE;
    size_t count = task->p->count - begin < TILE ? task->p->count - begin : TILE;
    uint32_t dtype = task->p->dtype;
    const uint8_t *input = (const uint8_t *)task->input + begin * trillim_itemsize(dtype);
    float *output = task->output + begin;
    TrillimStatus status = sigmoid_tile(input, output, count, dtype, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        float32x4_t value;
        if (dtype == TRILLIM_FP32) {
            value = vld1q_f32((const float *)input + i);
        } else {
            uint16x4_t bits = vld1_u16((const uint16_t *)input + i);
            value = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
        }
        vst1q_f32(output + i, vmulq_f32(value, vld1q_f32(output + i)));
    }
    for (; i < count; ++i) {
        output[i] = trillim_load(input, i, dtype).f * output[i];
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_silu(
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
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / TILE + (p->count % TILE != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_silu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    strides = _contiguous_strides(x.shape)\n"
           "    if x.dtype not in {'bf16', 'fp32'} or output.dtype != 'fp32' or x.shape != output.shape or (x.strides or strides) != strides:\n"
           "        raise ValueError('SiLU requires contiguous BF16 or FP32 input and FP32 output')\n"
           "    return {'count': _product(x.shape), 'dtype': _native_dtype(x.dtype)}\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Silu(x),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_silu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_silu_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        /* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
        TRILLIM_CAP_LIBM
#else
        TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM
#endif
    };
    return &capabilities;
}
