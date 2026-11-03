#ifndef TRILLIM_KERNEL_SELECTION_H
#define TRILLIM_KERNEL_SELECTION_H

#include "tensor.h"

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *a_strides;
    const size_t *b_strides;
    uint32_t dtype;
    uint32_t predicate;
} TrillimCompareParameters;

typedef struct {
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *a_strides;
    const size_t *b_strides;
    uint32_t dtype;
    const size_t *condition_strides;
} TrillimSelectParameters;

typedef struct {
    void *output;
    const void *const *inputs;
    const void *parameters;
    const TrillimKernelContext *context;
} TrillimSelectionTask;

static inline bool trillim_compare(TrillimValue a, TrillimValue b, uint32_t dtype, uint32_t predicate) {
    bool equal, less;
    if (trillim_float(dtype)) {
        equal = a.f == b.f;
        less = a.f < b.f;
        switch (predicate) {
        case 0: return equal;
        case 1: return !equal;
        case 2: return less;
        case 3: return a.f <= b.f;
        case 4: return a.f > b.f;
        default: return a.f >= b.f;
        }
    }
    equal = a.u == b.u;
    less = trillim_unsigned(dtype) ? a.u < b.u : a.i < b.i;
    switch (predicate) {
    case 0: return equal;
    case 1: return !equal;
    case 2: return less;
    case 3: return less || equal;
    case 4: return !less && !equal;
    default: return !less;
    }
}

#endif
