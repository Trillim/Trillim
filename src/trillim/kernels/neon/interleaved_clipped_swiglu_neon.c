#include "common.h"
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t rows, width;
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
    if (p->width % 4) {
        return TRILLIM_ERROR;
    }
    if (!p->width || !p->rows) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / (2 * sizeof(float)) / p->width) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * sizeof(float);
    inputs[0] = outputs[0] * 2;
    return TRILLIM_OK;
}

/* Match the staged FP32 arithmetic in clipped_swiglu_neon.c. */
static float activate(float gate, float up) {
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    return (gate * probability) * (up + 1.0f);
}

static inline float32x4_t exp_negative(float32x4_t original) {
    float32x4_t x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(original, vdupq_n_f32(0)));
    float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(0.00021219444005469058f));
    float32x4_t value = vdupq_n_f32(1.0f / 5040.0f);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(0.5f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    value = vfmaq_f32(vdupq_n_f32(1.0f), value, r);
    int32x4_t exponent = vcvtq_s32_f32(n);
    int32x4_t low = vminq_s32(vaddq_s32(exponent, vdupq_n_s32(126)), vdupq_n_s32(0));
    int32x4_t high = vsubq_s32(exponent, low);
    float32x4_t main_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(high, vdupq_n_s32(127)), 23));
    float32x4_t tail_scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(low, vdupq_n_s32(127)), 23));
    value = vmulq_f32(vmulq_f32(value, main_scale), tail_scale);
    return vbslq_f32(vceqq_f32(original, original), value, original);
}

static float32x4_t activate4(float32x4_t gate, float32x4_t up) {
    float32x4_t low = vdupq_n_f32(-7.0f), high = vdupq_n_f32(7.0f);
    gate = vbslq_f32(vcleq_f32(gate, high), gate, high);
    up = vbslq_f32(vcgeq_f32(up, low), up, low);
    up = vbslq_f32(vcleq_f32(up, high), up, high);
    float32x4_t scaled = vmulq_n_f32(gate, 1.702f);
    float32x4_t e = exp_negative(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t denominator = vaddq_f32(e, one);
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0)),
        vdivq_f32(one, denominator), vdivq_f32(e, denominator));
    float32x4_t result = vmulq_f32(vmulq_f32(gate, probability), vaddq_f32(up, one));
    if (vminvq_f32(scaled) < -80.0f) {
        float gates[4], ups[4], values[4], arguments[4];
        vst1q_f32(gates, gate);
        vst1q_f32(ups, up);
        vst1q_f32(values, result);
        vst1q_f32(arguments, scaled);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (arguments[lane] < -80.0f) {
                values[lane] = activate(gates[lane], ups[lane]);
            }
        }
        result = vld1q_f32(values);
    }
    return result;
}

static TrillimStatus apply(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t width = task->p->width;
    const float *source = task->input + row * width * 2;
    float *output = task->output + row * width;
    for (size_t i = 0; i < width; i += 4) {
        if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4x2_t pair = vld2q_f32(source + 2 * i);
        vst1q_f32(output + i, activate4(pair.val[0], pair.val[1]));
    }
    return TRILLIM_OK;
}

static inline bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_interleaved_clipped_swiglu(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], p, context};
    bool decode_parallel = p->rows == 4 && p->width >= 2048 &&
        context->thread_pool->worker_count >= 4 && default_fpcr();
    if (decode_parallel || (p->rows > 4 && p->rows > 16384 / p->width)) {
        return context->thread_pool->run(context->thread_pool->context, p->rows, apply, &task);
    }
    for (size_t row = 0; row < p->rows; ++row) {
        TrillimStatus status = apply(&task, row, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_neon_interleaved_clipped_swiglu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if len(x.shape) != 3 or len(output.shape) != 3 or x.dtype != 'fp32' or output.dtype != 'fp32':\n"
           "        raise ValueError('interleaved clipped SwiGLU needs rank-three FP32 tensors')\n"
           "    width = output.shape[-1]\n"
           "    if width.value is None or width.value % 4 or x.shape[-1].value != width.value * 2 or x.shape[:-1] != output.shape[:-1]:\n"
           "        raise ValueError('interleaved clipped SwiGLU needs a gate and value per output channel')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('interleaved clipped SwiGLU needs contiguous projected rows')\n"
           "    return dict(rows=_product(output.shape[:-1]), width=width)\n"
           "def pattern(x: Tensor) -> tuple[Tensor, ...]:\n"
           "    width = Dim(x, -1)\n"
           "    gate_ids = Iota(Mul(width, 2), step=2)\n"
           "    up_ids = Iota(Mul(width, 2), start=1, step=2)\n"
           "    gate = Minimum(Take(x, gate_ids, axis=2, rank=3), 7.0)\n"
           "    up = Take(x, up_ids, axis=2, rank=3)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(up, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_interleaved_clipped_swiglu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_interleaved_clipped_swiglu_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
