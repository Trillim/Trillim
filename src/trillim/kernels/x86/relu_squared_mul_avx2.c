#include "common.h"
#include <immintrin.h>
#include <math.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

typedef struct { size_t count; } TrillimActivationParameters;

typedef struct {
    float *output;
    const float *gate;
    const float *up;
    const TrillimActivationParameters *parameters;
    const TrillimKernelContext *context;
    bool silu;
} TrillimActivationTask;

static inline TrillimStatus trillim_activation_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * sizeof(float);
    inputs[1] = inputs[0];
    outputs[0] = inputs[0];
    return TRILLIM_OK;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_activation_task_avx2(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimActivationTask *task = (const TrillimActivationTask *)opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = task->parameters->count - begin < 1024 ? task->parameters->count : begin + 1024;
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256 gate = _mm256_loadu_ps(task->gate + i);
        __m256 value;
        if (task->silu) {
            float probabilities[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                float scalar = task->gate[i + lane];
                float e = expf(-fabsf(scalar));
                probabilities[lane] = scalar >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
            }
            value = _mm256_mul_ps(gate, _mm256_loadu_ps(probabilities));
        } else {
            value = _mm256_max_ps(gate, _mm256_setzero_ps());
            value = _mm256_mul_ps(value, value);
        }
        _mm256_storeu_ps(task->output + i, _mm256_mul_ps(value, _mm256_loadu_ps(task->up + i)));
    }
    for (; i < end; ++i) {
        float gate = task->gate[i];
        float value;
        if (task->silu) {
            float e = expf(-fabsf(gate));
            value = gate * (gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
        } else {
            value = gate >= 0.0f ? gate : 0.0f;
            value *= value;
        }
        task->output[i] = value * task->up[i];
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input < 2;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_relu_squared_mul_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimActivationParameters *p = (const TrillimActivationParameters *)parameters;
    TrillimActivationTask task = {(float *)outputs[0], (const float *)inputs[0],
        (const float *)inputs[1], p, context, false};
    return context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_activation_task_avx2, &task);
}

TRILLIM_KERNEL const char *trillim_x86_relu_squared_mul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _activation as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(ReluSquared(a), b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_relu_squared_mul_avx2_parameters(void) {
    static const TrillimParameter fields[] = {{"count", "size_t", offsetof(TrillimActivationParameters, count), NULL}};
    static const size_t inputs[] = {4, 4};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimActivationParameters), TRILLIM_ALIGNOF(TrillimActivationParameters),
        1, fields, 2, 1, trillim_activation_requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_relu_squared_mul_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
