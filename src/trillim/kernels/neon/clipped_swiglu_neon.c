#include "common.h"
#include <math.h>
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
} Parameters;

typedef struct {
    float *output;
    const float *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rank && (!p->shape || !p->a_strides || !p->b_strides)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (!p->shape[axis]) {
            return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
        }
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    size_t a, b;
    if (count != p->count || count > SIZE_MAX / sizeof(float) ||
        !trillim_span(p->rank, p->shape, p->a_strides, &a) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b) ||
        a > SIZE_MAX / sizeof(float) || b > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a * sizeof(float);
    inputs[1] = b * sizeof(float);
    outputs[0] = count * sizeof(float);
    return TRILLIM_OK;
}

static float activate(float gate, float up) {
    /* Compare/Select clamps send NaN gates to +7 and NaN up values to -7. */
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float activated = gate * probability;
    float shifted = up + 1.0f;
    return activated * shifted;
}

static inline float32x4_t exp_negative(float32x4_t x) {
    float32x4_t original = x;
    x = vmaxq_f32(vdupq_n_f32(-104.0f), vminq_f32(x, vdupq_n_f32(0)));
    float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.4426950408889634f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(0.00021219444005469058f));
    /* Range reduction bounds |r| by ln(2)/2 for the degree-seven polynomial. */
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


static float32x4_t load4(const float *data, size_t offset, size_t step) {
    if (step == 0) {
        return vdupq_n_f32(data[offset]);
    }
    if (step == 1) {
        return vld1q_f32(data + offset);
    }
    float32x4_t value = vdupq_n_f32(data[offset]);
    value = vsetq_lane_f32(data[offset + step], value, 1);
    value = vsetq_lane_f32(data[offset + step * 2], value, 2);
    return vsetq_lane_f32(data[offset + step * 3], value, 3);
}

static float32x4_t activate4(float32x4_t gate, float32x4_t up) {
    float32x4_t low = vdupq_n_f32(-7.0f), high = vdupq_n_f32(7.0f);
    gate = vbslq_f32(vcleq_f32(gate, high), gate, high);
    up = vbslq_f32(vcgeq_f32(up, low), up, low);
    up = vbslq_f32(vcleq_f32(up, high), up, high);
    float32x4_t scaled = vmulq_n_f32(gate, 1.702f);
    float32x4_t e = exp_negative(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0)),
        vdivq_f32(vdupq_n_f32(1.0f), denominator), vdivq_f32(e, denominator));
    float32x4_t activated = vmulq_f32(gate, probability);
    float32x4_t shifted = vaddq_f32(up, vdupq_n_f32(1.0f));
    float32x4_t result = vmulq_f32(activated, shifted);
    if (vminvq_f32(scaled) < -80.0f) {
        /* libm preserves subnormal rounding and the -Inf * 0 invalid result. */
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

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t index = tile * 2048;
    size_t end = p->count - index < 2048 ? p->count : index + 2048;
    while (index < end) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t a = trillim_offset(index, p->rank, p->shape, p->a_strides);
        size_t b = trillim_offset(index, p->rank, p->shape, p->b_strides);
        size_t a_step = p->rank ? p->a_strides[p->rank - 1] : 0;
        size_t b_step = p->rank ? p->b_strides[p->rank - 1] : 0;
        size_t width = p->rank ? p->shape[p->rank - 1] : 1;
        size_t remaining = width - index % width;
        size_t limit = end - index < remaining ? end : index + remaining;
        for (; limit - index >= 4; index += 4, a += a_step * 4, b += b_step * 4) {
            if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t gate, up;
            if (a_step == 2 && b_step == 2 &&
                (uintptr_t)(task->a + a) + sizeof(float) == (uintptr_t)(task->b + b)) {
                float32x4x2_t pair = vld2q_f32(task->a + a);
                gate = pair.val[0];
                up = pair.val[1];
            } else {
                gate = load4(task->a, a, a_step);
                up = load4(task->b, b, b_step);
            }
            vst1q_f32(task->output + index, activate4(gate, up));
        }
        for (; index < limit; ++index, a += a_step, b += b_step) {
            if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            task->output[index] = activate(task->a[a], task->b[b]);
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (output != 0 || input >= 2) {
        return false;
    }
    if (!parameters) {
        return true;
    }
    const Parameters *p = parameters;
    return trillim_contiguous(p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_clipped_swiglu(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->count) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->count / 2048 + (p->count % 2048 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_neon_clipped_swiglu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary\n"
           "def bind(context):\n"
           "    a, b, output = context.inputs['a'], context.inputs['b'], context.outputs[0]\n"
           "    if any(value.dtype != 'fp32' for value in (a, b, output)):\n"
           "        raise ValueError('clipped SwiGLU requires FP32 tensors')\n"
           "    parameters = _binary(context)\n"
           "    del parameters['dtype']\n"
           "    return parameters\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    gate = Minimum(a, 7.0)\n"
           "    return (Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(b, -7.0, 7.0), 1)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_clipped_swiglu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_clipped_swiglu_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
