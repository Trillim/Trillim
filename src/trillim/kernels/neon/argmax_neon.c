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

typedef struct {
    float32x4_t maximum;
    int32x4_t indices;
    uint32x4_t ordered;
} MaxLanes;

static inline void track(MaxLanes *lanes, float32x4_t values, int32x4_t positions) {
    lanes->ordered = vandq_u32(lanes->ordered, vceqq_f32(values, values));
    uint32x4_t choose = vcgtq_f32(values, lanes->maximum);
    lanes->maximum = vbslq_f32(choose, values, lanes->maximum);
    lanes->indices = vbslq_s32(choose, positions, lanes->indices);
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
    if (p->count >= 4) {
        enum { vectors = 4, block = 4 * vectors };
        const int32_t first_indices[] = {0, 1, 2, 3};
        const int32x4_t first = vld1q_s32(first_indices);
        const int32x4_t stride = vdupq_n_s32(4);
        const MaxLanes initial = {vld1q_f32(input), first, vdupq_n_u32(UINT32_MAX)};
        MaxLanes maxima[vectors];
        for (size_t vector = 0; vector < vectors; ++vector) {
            maxima[vector] = initial;
        }
        for (; p->count - index >= block; index += block) {
            if (index != 0 && (index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            int32x4_t positions = vaddq_s32(vdupq_n_s32((int32_t)index), first);
            for (size_t vector = 0; vector < vectors; ++vector) {
                track(&maxima[vector], vld1q_f32(input + index + vector * 4), positions);
                positions = vaddq_s32(positions, stride);
            }
        }
        for (; p->count - index >= 4; index += 4) {
            if (index != 0 && (index & 1023) == 0 && context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
            track(&maxima[0], vld1q_f32(input + index), vaddq_s32(vdupq_n_s32((int32_t)index), first));
        }
        uint32x4_t ordered = maxima[0].ordered;
        for (size_t vector = 1; vector < vectors; ++vector) {
            ordered = vandq_u32(ordered, maxima[vector].ordered);
        }
        if (vminvq_u32(ordered) == UINT32_MAX) {
            for (size_t vector = 0; vector < vectors; ++vector) {
                float values[4];
                int32_t positions[4];
                vst1q_f32(values, maxima[vector].maximum);
                vst1q_s32(positions, maxima[vector].indices);
                for (size_t lane = 0; lane < 4; ++lane) {
                    if (selected == INT64_MAX || values[lane] > best ||
                        (values[lane] == best && positions[lane] < selected)) {
                        best = values[lane];
                        selected = positions[lane];
                    }
                }
            }
        } else {
            /* The original scalar comparison defines the ordering when a NaN occurs. */
            index = 0;
        }
    }
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
