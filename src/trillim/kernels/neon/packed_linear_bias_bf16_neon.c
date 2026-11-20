#include "common.h"
#include <arm_neon.h>

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_cast(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear_bias(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, outputs;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, output_dtype, weight_dtype, bias_dtype;
} LinearParameters;

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} CastParameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 3 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->width % 256 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_DUBGRP_INT6, inputs + 1) ||
        !trillim_packed_payload_size(1, p->outputs, TRILLIM_BF16, inputs + 2) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_BF16, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} UnbiasedParameters;

static bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static inline uint16x4_t to_bf16(float32x4_t value) {
    const uint32x4_t bits = vreinterpretq_u32_f32(value);
    const uint32x4_t upper = vshrq_n_u32(bits, 16);
    const uint32x4_t tie_bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    const uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, tie_bias), 16);
    const uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    return vmovn_u32(vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded));
}

static TrillimStatus add_bias_and_cast(
    uint8_t *output, const float *projection, const uint8_t *bias, size_t count,
    const TrillimKernelContext *context
) {
    size_t column = 0;
    while (column < count) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t end = count - column < 256 ? count : column + 256;
        for (; end - column >= 4; column += 4) {
            uint16x4_t bias_bits;
            memcpy(&bias_bits, bias + column * sizeof(uint16_t), sizeof(bias_bits));
            const float32x4_t bias_values = vreinterpretq_f32_u32(vshll_n_u16(bias_bits, 16));
            const uint16x4_t result = to_bf16(vaddq_f32(vld1q_f32(projection + column), bias_values));
            memcpy(output + column * sizeof(uint16_t), &result, sizeof(result));
        }
        for (; column < end; ++column) {
            const float addend = trillim_load(bias, column, TRILLIM_BF16).f;
            const uint16_t result = trillim_to_bf16(projection[column] + addend);
            memcpy(output + column * sizeof(uint16_t), &result, sizeof(result));
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_linear_bias_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    void *temporary = scratch;
    void *projected[] = {temporary};
    if (p->tokens == 1 && p->width >= 256 && context->thread_pool->worker_count != 0 &&
        (uintptr_t)scratch % _Alignof(float) == 0 && default_fpcr()) {
        const UnbiasedParameters projection = {1, p->width, p->outputs, TRILLIM_BF16,
            TRILLIM_DUBGRP_INT6, TRILLIM_FP32};
        const TrillimStatus status = trillim_neon_packed_linear(projected, inputs, NULL, &projection, context);
        if (status != TRILLIM_OK) {
            return status;
        }
        return add_bias_and_cast(outputs[0], temporary, inputs[2], p->outputs, context);
    }
    const LinearParameters projection = {p->tokens, p->width, p->outputs, TRILLIM_BF16,
        TRILLIM_FP32, TRILLIM_DUBGRP_INT6, TRILLIM_BF16};
    const TrillimStatus status = trillim_neon_packed_linear_bias(projected, inputs, NULL, &projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }

    const size_t count = p->tokens * p->outputs, stride = 1;
    const CastParameters convert = {count, 1, &count, &stride, TRILLIM_FP32, TRILLIM_BF16};
    const void *cast_inputs[] = {temporary};
    return trillim_neon_cast(outputs, cast_inputs, NULL, &convert, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_linear_bias_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'weight', 'bias')\n"
           "def bind(c):\n"
           "    x, weight = (c.inputs[name] for name in ('x', 'weight'))\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16':\n"
           "        raise ValueError('requires BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if len(weight.shape) != 2 or weight.dtype != 'dubgrp_int6' or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete Q6 matrix')\n"
           "    bias = c.inputs['bias']\n"
           "    if bias.dtype != 'bf16' or bias.shape != weight.shape[:1]:\n"
           "        raise ValueError('requires a matching BF16 bias')\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or not width.value or width.value % 256:\n"
           "        raise ValueError('unaligned Q6 matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, outputs=outputs)\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if bias is None or dtype != 'bf16':\n"
           "        raise ValueError('requires a BF16 projection cast')\n"
           "    return (Cast(Linear(x, weight, bias=bias), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_linear_bias_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 3, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_linear_bias_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
