#ifndef TRILLIM_KERNEL_GENERATION_H
#define TRILLIM_KERNEL_GENERATION_H

#include "tensor.h"

typedef struct {
    size_t rank;
    const size_t *shape;
} TrillimShapeParameters;

typedef struct {
    size_t count;
    int64_t start;
    int64_t step;
    uint32_t dtype;
} TrillimIotaParameters;

typedef struct {
    size_t count;
    uint64_t seed;
    uint64_t offset;
    uint32_t distribution;
    uint32_t dtype;
} TrillimRandomParameters;

typedef struct {
    void *output;
    const void *parameters;
    const TrillimKernelContext *context;
} TrillimGenerationTask;

static inline uint64_t trillim_random_hash(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

#endif
