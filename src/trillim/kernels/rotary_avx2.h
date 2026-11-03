#ifndef TRILLIM_KERNEL_ROTARY_AVX2_H
#define TRILLIM_KERNEL_ROTARY_AVX2_H

#include "avx2_helpers.h"
#include "rotary.h"

TRILLIM_AVX2 static inline TrillimStatus trillim_rotary_task_avx2(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = (const TrillimRotaryTask *)opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t head = index / p->tokens;
    size_t token = index % p->tokens;
    size_t base = head * p->strides[0] + token * p->strides[1];
    size_t output = index * p->dim;
    size_t column = 0;
    for (; p->dim - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        _mm256_storeu_ps(task->output + output + column, _mm256_loadu_ps(task->input + base + column));
    }
    for (; column < p->dim; ++column) {
        task->output[output + column] = task->input[base + column];
    }
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype, 1, p->tokens);
    size_t pairs = p->rotary_dim / 2;
    size_t pair = 0;
    for (; pairs - pair >= 8; pair += 8) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float cosine[8], sine[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            float angle = position * expf((float)(pair + lane) * p->frequency_step);
            cosine[lane] = cosf(angle) * p->amplitude;
            sine[lane] = sinf(angle) * p->amplitude;
        }
        __m256 a, b;
        if (p->interleaved) {
            __m256i indices = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
            a = _mm256_i32gather_ps(task->input + base + pair * 2, indices, 4);
            b = _mm256_i32gather_ps(task->input + base + pair * 2 + 1, indices, 4);
        } else {
            a = _mm256_loadu_ps(task->input + base + pair);
            b = _mm256_loadu_ps(task->input + base + pair + pairs);
        }
        __m256 c = _mm256_loadu_ps(cosine);
        __m256 s = _mm256_loadu_ps(sine);
        __m256 first = _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s));
        __m256 second = _mm256_add_ps(_mm256_mul_ps(b, c), _mm256_mul_ps(a, s));
        if (p->interleaved) {
            __m256 low = _mm256_unpacklo_ps(first, second);
            __m256 high = _mm256_unpackhi_ps(first, second);
            _mm256_storeu_ps(task->output + output + pair * 2, _mm256_permute2f128_ps(low, high, 0x20));
            _mm256_storeu_ps(task->output + output + pair * 2 + 8, _mm256_permute2f128_ps(low, high, 0x31));
        } else {
            _mm256_storeu_ps(task->output + output + pair, first);
            _mm256_storeu_ps(task->output + output + pair + pairs, second);
        }
    }
    for (; pair < pairs; ++pair) {
        trillim_rotate_pair(task, base, output, position, pair);
    }
    return TRILLIM_OK;
}

#endif
