#include "common.h"
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


typedef struct { size_t count; uint32_t output_dtype; } Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->count > INT32_MAX || p->count > SIZE_MAX / sizeof(float) ||
        (p->output_dtype != TRILLIM_INT32 && p->output_dtype != TRILLIM_INT64)) return TRILLIM_ERROR;
    inputs[0] = p->count * sizeof(float);
    outputs[0] = trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_argmax(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    const float *input = inputs[0];
    int64_t selected = INT64_MAX;
    float best = 0.0f;
    size_t index = 0;
    bool nan = false;
    if (p->count >= 4) {
        float32x4_t maximum = vld1q_f32(input);
        const int32_t lane_indices[] = {0, 1, 2, 3};
        int32x4_t lanes = vld1q_s32(lane_indices);
        int32x4_t indices = lanes;
        nan = vminvq_u32(vceqq_f32(maximum, maximum)) != UINT32_MAX;
        for (index = 4; p->count - index >= 4; index += 4) {
            if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            float32x4_t value = vld1q_f32(input + index);
            nan |= vminvq_u32(vceqq_f32(value, value)) != UINT32_MAX;
            uint32x4_t choose = vcgtq_f32(value, maximum);
            maximum = vbslq_f32(choose, value, maximum);
            int32x4_t positions = vaddq_s32(vdupq_n_s32((int32_t)index), lanes);
            indices = vbslq_s32(choose, positions, indices);
        }
        float values[4];
        int32_t positions[4];
        vst1q_f32(values, maximum);
        vst1q_s32(positions, indices);
        best = values[0];
        selected = positions[0];
        for (size_t lane = 1; lane < 4; ++lane) {
            if (values[lane] > best || (values[lane] == best && positions[lane] < selected)) {
                best = values[lane];
                selected = positions[lane];
            }
        }
    }
    /* The scalar comparison also defines the ordering when a NaN occurs. */
    if (nan) { index = 0; selected = INT64_MAX; }
    for (; index < p->count; ++index) {
        if ((index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
        if (selected == INT64_MAX || !(best >= input[index])) {
            best = input[index];
            selected = (int64_t)index;
        }
    }
    TrillimValue result = {.i = selected};
    trillim_store(outputs[0], 0, p->output_dtype, result);
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_neon_argmax_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _vector_argmax as bind\n"
           "def pattern(x: Tensor, *, axis: int = -1, rank: int) -> tuple[Tensor, ...]:\n"
           "    return (Argmax(x, axis=axis, rank=rank),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_argmax_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_argmax_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED};
    return &capabilities;
}
