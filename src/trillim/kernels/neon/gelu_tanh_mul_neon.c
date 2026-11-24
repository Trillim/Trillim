#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

enum { TILE = 1024 };

typedef struct {
    size_t count;
} Parameters;

typedef struct {
    float *output;
    const float *up;
    const float *gate;
    size_t count;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = inputs[1] = outputs[0] = p->count * sizeof(float);
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

static float gelu_tanh(float x) {
    float cube = (x * x) * x;
    float sum = x + cube * 0.044715f;
    float argument = sum * 0.7978845608028654f;
    float scaled = argument * 2.0f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float cdf = probability * 2.0f - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
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

static float32x4_t gelu_tanh4(float32x4_t x) {
    float32x4_t cube = vmulq_f32(vmulq_f32(x, x), x);
    float32x4_t sum = vaddq_f32(x, vmulq_n_f32(cube, 0.044715f));
    float32x4_t argument = vmulq_n_f32(sum, 0.7978845608028654f);
    float32x4_t scaled = vmulq_n_f32(argument, 2.0f);
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0.0f)), positive, negative);
    float32x4_t cdf = vsubq_f32(vmulq_n_f32(probability, 2.0f), vdupq_n_f32(1.0f));
    return vmulq_f32(vmulq_n_f32(x, 0.5f), vaddq_f32(cdf, vdupq_n_f32(1.0f)));
}

static TrillimStatus gelu_tanh_tile(
    const float *input, float *output, size_t count, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        vst1q_f32(output + i, gelu_tanh4(vld1q_f32(input + i)));
    }
    for (; i < count; ++i) {
        output[i] = gelu_tanh(input[i]);
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
    size_t count = task->count - begin < TILE ? task->count - begin : TILE;
    float activation[TILE];
    TrillimStatus status = gelu_tanh_tile(task->gate + begin, activation, count, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    const float *up = task->up + begin;
    float *output = task->output + begin;
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        vst1q_f32(output + i, vmulq_f32(vld1q_f32(up + i), vld1q_f32(activation + i)));
    }
    for (; i < count; ++i) {
        output[i] = up[i] * activation[i];
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gelu_tanh_mul(
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
    size_t tiles = p->count / TILE + (p->count % TILE != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_gelu_tanh_mul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    a, b, output = c.inputs['a'], c.inputs['b'], c.outputs[0]\n"
           "    for value in (a, b, output):\n"
           "        strides = _contiguous_strides(value.shape)\n"
           "        if value.dtype != 'fp32' or value.shape != output.shape or (value.strides or strides) != strides:\n"
           "            raise ValueError('GELU gate requires matching contiguous FP32 tensors')\n"
           "    return {'count': _product(output.shape)}\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(a, Gelu(b, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gelu_tanh_mul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL}
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 1, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gelu_tanh_mul_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM | TRILLIM_NEON_REQUIRED
    };
    return &capabilities;
}
