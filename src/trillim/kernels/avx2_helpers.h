#ifndef TRILLIM_KERNEL_AVX2_HELPERS_H
#define TRILLIM_KERNEL_AVX2_HELPERS_H

#include "codec.h"
#include <immintrin.h>

#define TRILLIM_AVX2 __attribute__((target("avx2")))

TRILLIM_AVX2 static inline __m256 trillim_avx2_load8(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    if (dtype == TRILLIM_BF16) {
        __m128i values = _mm_loadu_si128((const __m128i *)((const uint8_t *)data + index * 2));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(values), 16));
    }
    float values[8];
    for (size_t i = 0; i < 8; ++i) {
        values[i] = trillim_read_float(data, index + i, dtype, rows, columns);
    }
    return _mm256_loadu_ps(values);
}

TRILLIM_AVX2 static inline float trillim_avx2_sum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#endif
