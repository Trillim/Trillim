#ifndef TRILLIM_KERNEL_ACTIVATION_AVX2_H
#define TRILLIM_KERNEL_ACTIVATION_AVX2_H

#include "activation.h"
#include "avx2_helpers.h"

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

#endif
