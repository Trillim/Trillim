#include "common.h"
#include <math.h>
#include <arm_neon.h>

typedef struct {
    size_t count;
    uint32_t dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->dtype);
    outputs[0] = p->count * trillim_itemsize(p->dtype);
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


static float apply(float value, uint32_t dtype) {
    float e = round_dtype(expf(-fabsf(value)), dtype);
    float added = round_dtype(e + 1.0f, dtype);
    float logarithm = round_dtype(logf(added), dtype);
    float positive = value >= 0.0f ? value : 0.0f;
    return round_dtype(positive + logarithm, dtype);
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


static float32x4_t apply4(float32x4_t value) {
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(value)));
    float lanes[4];
    vst1q_f32(lanes, vaddq_f32(e, vdupq_n_f32(1.0f)));
    for (size_t lane = 0; lane < 4; ++lane) {
        lanes[lane] = logf(lanes[lane]);
    }
    float32x4_t positive = vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0f)),
                                     value, vdupq_n_f32(0.0f));
    return vaddq_f32(positive, vld1q_f32(lanes));
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    size_t i = begin;
    if (p->dtype == TRILLIM_FP32) {
        for (; end - i >= 4; i += 4) {
            vst1q_f32((float *)task->output + i,
                       apply4(vld1q_f32((const float *)task->input + i)));
        }
    }
    for (; i < end; ++i) {
        TrillimValue value = {.f = apply(trillim_load(task->input, i, p->dtype).f, p->dtype)};
        trillim_store(task->output, i, p->dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_softplus(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_softplus_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('Softplus requires native floating tensors')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Softplus requires contiguous input')\n"
           "    return {'count': _product(x.shape), 'dtype': _native_dtype(x.dtype)}\n"
           "\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Softplus(x),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_softplus_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_softplus_capabilities(void) {
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
