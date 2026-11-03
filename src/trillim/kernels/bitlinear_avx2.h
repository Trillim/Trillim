#ifndef TRILLIM_KERNEL_BITLINEAR_AVX2_H
#define TRILLIM_KERNEL_BITLINEAR_AVX2_H

#include "bitlinear.h"

typedef struct {
    float *output;
    const float *input;
    const uint8_t *weight;
    float gamma;
    uint8_t *scratch;
    const TrillimBitLinearParameters *params;
    const TrillimKernelContext *context;
    size_t decode_columns;
} TrillimAvx2BitLinearTask;

#endif
