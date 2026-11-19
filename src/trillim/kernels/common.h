#ifndef TRILLIM_KERNEL_COMMON_H
#define TRILLIM_KERNEL_COMMON_H
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#define TRILLIM_KERNEL_ABI_VERSION 1

#define TRILLIM_CAP_X86_AVX2 (UINT64_C(1) << 0)
#define TRILLIM_CAP_X86_AVX512F (UINT64_C(1) << 1)
#define TRILLIM_CAP_X86_AVX512BW (UINT64_C(1) << 2)
#define TRILLIM_CAP_X86_AVX512DQ (UINT64_C(1) << 3)
#define TRILLIM_CAP_X86_AVX512VL (UINT64_C(1) << 4)
#define TRILLIM_CAP_X86_AVX_VNNI (UINT64_C(1) << 5)
#define TRILLIM_CAP_X86_AVX512_VNNI (UINT64_C(1) << 6)
#define TRILLIM_CAP_X86_FMA (UINT64_C(1) << 7)

#define TRILLIM_CAP_NEON (UINT64_C(1) << 0)
#define TRILLIM_CAP_NEON_DOTPROD (UINT64_C(1) << 1)

#define TRILLIM_CAP_LIBM (UINT64_C(1) << 32)
#define TRILLIM_CAP_BLAS (UINT64_C(1) << 33)

#if defined(_WIN32)
#define TRILLIM_EXPORT __declspec(dllexport)
#else
#define TRILLIM_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define TRILLIM_KERNEL extern "C" TRILLIM_EXPORT
#define TRILLIM_ALIGNOF(type) alignof(type)
extern "C" {
#else
#define TRILLIM_KERNEL TRILLIM_EXPORT
#define TRILLIM_ALIGNOF(type) _Alignof(type)
#endif
typedef int32_t TrillimStatus;
enum { TRILLIM_OK = 0, TRILLIM_CANCELLED = 1, TRILLIM_ERROR = 2 };
typedef struct {
    const char *name;
    const char *type;
    size_t offset;
    const char *length;
} TrillimParameter;
typedef struct {
    size_t size;
    size_t alignment;
    size_t field_count;
    const TrillimParameter *fields;
    size_t input_count;
    size_t output_count;
    TrillimStatus (*requirements)(const void *, size_t *, size_t *, size_t *);
    /* NULL parameters ask for size-independent alias permission after layout checks. */
    bool (*can_alias)(const void *, size_t, size_t);
    const size_t *input_alignments;
    const size_t *output_alignments;
    size_t scratch_alignment;
} TrillimParameterSchema;
typedef struct {
    uint32_t abi_version;
    uint64_t required;
} TrillimCapabilities;
typedef TrillimStatus (*TrillimTask)(void *, size_t, size_t);
typedef struct {
    void *context;
    size_t worker_count;
    TrillimStatus (*run)(void *, size_t, TrillimTask, void *);
} TrillimThreadPool;
typedef struct {
    const TrillimThreadPool *thread_pool;
    bool (*is_cancelled)(void *);
    void *cancellation_context;
} TrillimKernelContext;
typedef TrillimStatus (*TrillimKernel)(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
#ifdef __cplusplus
}
#endif

typedef enum {
    TRILLIM_BOOL,
    TRILLIM_INT8,
    TRILLIM_UINT8,
    TRILLIM_INT16,
    TRILLIM_UINT16,
    TRILLIM_INT32,
    TRILLIM_UINT32,
    TRILLIM_INT64,
    TRILLIM_UINT64,
    TRILLIM_FP16,
    TRILLIM_BF16,
    TRILLIM_FP32,
    TRILLIM_ROW_INT8,
    TRILLIM_MAT_TERNARY,
    TRILLIM_GRP_INT8,
    TRILLIM_GRP_INT4,
    TRILLIM_GRP_INT5,
    TRILLIM_DUBGRP_INT6,
    TRILLIM_GRP_BINARY,
    TRILLIM_GRP_TERNARY,
    TRILLIM_MXFP4,
} TrillimDType;

typedef union {
    uint64_t u;
    int64_t i;
    float f;
} TrillimValue;

static inline size_t trillim_itemsize(uint32_t dtype) {
    static const size_t sizes[] = {1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4};
    return dtype < sizeof(sizes) / sizeof(sizes[0]) ? sizes[dtype] : 0;
}

static inline bool trillim_float(uint32_t dtype) {
    return dtype >= TRILLIM_FP16 && dtype <= TRILLIM_FP32;
}

static inline bool trillim_unsigned(uint32_t dtype) {
    return dtype == TRILLIM_BOOL || dtype == TRILLIM_UINT8 || dtype == TRILLIM_UINT16 || dtype == TRILLIM_UINT32 || dtype == TRILLIM_UINT64;
}

static inline float trillim_bf16(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static inline uint16_t trillim_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
        return (uint16_t)((bits >> 16) | 64);
    }
    return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1)) >> 16);
}

static inline float trillim_fp16(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 31;
    uint32_t fraction = value & 1023;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            while ((fraction & 1024) == 0) {
                fraction <<= 1;
                --e;
            }
            bits = sign | (uint32_t)(e + 127) << 23 | (fraction & 1023) << 13;
        }
    } else if (exponent == 31) {
        bits = sign | UINT32_C(0x7f800000) | fraction << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | fraction << 13;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t trillim_to_fp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t fraction = bits & 0x7fffff;
    int32_t exponent = (int32_t)((bits >> 23) & 255) - 127;
    if (exponent == 128) {
        return (uint16_t)(sign | 0x7c00 | (fraction ? ((fraction >> 13) | 0x200) : 0));
    }
    if (exponent > 15) {
        return (uint16_t)(sign | 0x7c00);
    }
    if (exponent < -25) {
        return (uint16_t)sign;
    }
    if (exponent < -14) {
        uint32_t mantissa = fraction | 0x800000;
        uint32_t shift = (uint32_t)(-exponent - 1);
        uint32_t half = UINT32_C(1) << (shift - 1);
        return (uint16_t)(sign | ((mantissa + half - 1 + ((mantissa >> shift) & 1)) >> shift));
    }
    uint32_t rounded = fraction + 0xfff + ((fraction >> 13) & 1);
    return (uint16_t)(sign | (((uint32_t)(exponent + 15) << 10) + (rounded >> 13)));
}

static inline TrillimValue trillim_load(const void *data, size_t index, uint32_t dtype) {
    TrillimValue value;
    value.u = 0;
    const uint8_t *p = (const uint8_t *)data + index * trillim_itemsize(dtype);
    switch (dtype) {
        case TRILLIM_BOOL: case TRILLIM_UINT8: {
            uint8_t v;
            memcpy(&v, p, 1);
            value.u = v;
            break;
        }
        case TRILLIM_INT8: {
            int8_t v;
            memcpy(&v, p, 1);
            value.i = v;
            break;
        }
        case TRILLIM_UINT16: {
            uint16_t v;
            memcpy(&v, p, 2);
            value.u = v;
            break;
        }
        case TRILLIM_INT16: {
            int16_t v;
            memcpy(&v, p, 2);
            value.i = v;
            break;
        }
        case TRILLIM_UINT32: {
            uint32_t v;
            memcpy(&v, p, 4);
            value.u = v;
            break;
        }
        case TRILLIM_INT32: {
            int32_t v;
            memcpy(&v, p, 4);
            value.i = v;
            break;
        }
        case TRILLIM_UINT64: memcpy(&value.u, p, 8);
        break;
        case TRILLIM_INT64: memcpy(&value.i, p, 8);
        break;
        case TRILLIM_FP16: {
            uint16_t v;
            memcpy(&v, p, 2);
            value.f = trillim_fp16(v);
            break;
        }
        case TRILLIM_BF16: {
            uint16_t v;
            memcpy(&v, p, 2);
            value.f = trillim_bf16(v);
            break;
        }
        case TRILLIM_FP32: memcpy(&value.f, p, 4);
        break;
        default: break;
    }
    return value;
}

static inline void trillim_store(void *data, size_t index, uint32_t dtype, TrillimValue value) {
    uint8_t *p = (uint8_t *)data + index * trillim_itemsize(dtype);
    switch (dtype) {
        case TRILLIM_BOOL: {
            uint8_t v = value.u != 0;
            memcpy(p, &v, 1);
            break;
        }
        case TRILLIM_UINT8: case TRILLIM_INT8: {
            uint8_t v = (uint8_t)value.u;
            memcpy(p, &v, 1);
            break;
        }
        case TRILLIM_UINT16: case TRILLIM_INT16: {
            uint16_t v = (uint16_t)value.u;
            memcpy(p, &v, 2);
            break;
        }
        case TRILLIM_UINT32: case TRILLIM_INT32: {
            uint32_t v = (uint32_t)value.u;
            memcpy(p, &v, 4);
            break;
        }
        case TRILLIM_UINT64: case TRILLIM_INT64: memcpy(p, &value.u, 8);
        break;
        case TRILLIM_FP16: {
            uint16_t v = trillim_to_fp16(value.f);
            memcpy(p, &v, 2);
            break;
        }
        case TRILLIM_BF16: {
            uint16_t v = trillim_to_bf16(value.f);
            memcpy(p, &v, 2);
            break;
        }
        case TRILLIM_FP32: memcpy(p, &value.f, 4);
        break;
        default: break;
    }
}

static inline bool trillim_span(size_t rank, const size_t *shape, const size_t *strides, size_t *span) {
    if (rank != 0 && (shape == NULL || strides == NULL)) {
        return false;
    }
    size_t result = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            *span = 0;
            return true;
        }
        if (strides[i] != 0 && shape[i] - 1 > (SIZE_MAX - result) / strides[i]) {
            return false;
        }
        result += (shape[i] - 1) * strides[i];
    }
    *span = result;
    return true;
}

static inline size_t trillim_offset(size_t flat, size_t rank, const size_t *shape, const size_t *strides) {
    size_t result = 0;
    for (size_t axis = rank; axis > 0; --axis) {
        size_t i = axis - 1;
        result += (flat % shape[i]) * strides[i];
        flat /= shape[i];
    }
    return result;
}

static inline bool trillim_contiguous(size_t rank, const size_t *shape, const size_t *strides) {
    size_t stride = 1;
    for (size_t axis = rank; axis > 0; --axis) {
        size_t i = axis - 1;
        if ((shape[i] > 1 && strides[i] != stride) || (shape[i] != 0 && stride > SIZE_MAX / shape[i])) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

static inline bool trillim_packed_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t itemsize = trillim_itemsize(dtype);
    if (columns != 0 && rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (itemsize != 0) {
        if (count > SIZE_MAX / itemsize) {
            return false;
        }
        *bytes = count * itemsize;
        return true;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        if (rows > (SIZE_MAX - count) / 2) {
            return false;
        }
        *bytes = count + rows * 2;
        return true;
    }
    size_t group;
    size_t record;
    switch (dtype) {
    case TRILLIM_MAT_TERNARY: group = 128; record = 32; break;
    case TRILLIM_GRP_INT8: group = 32; record = 34; break;
    case TRILLIM_GRP_INT4: group = 32; record = 18; break;
    case TRILLIM_GRP_INT5: group = 32; record = 22; break;
    case TRILLIM_DUBGRP_INT6: group = 256; record = 210; break;
    case TRILLIM_GRP_BINARY: group = 128; record = 18; break;
    case TRILLIM_GRP_TERNARY: group = 128; record = 34; break;
    case TRILLIM_MXFP4: group = 32; record = 17; break;
    default: return false;
    }
    size_t groups = columns / group + (columns % group != 0);
    if (rows != 0 && groups > SIZE_MAX / record / rows) {
        return false;
    }
    *bytes = rows * groups * record;
    if (dtype == TRILLIM_MAT_TERNARY) {
        if (*bytes > SIZE_MAX - sizeof(float)) {
            return false;
        }
        *bytes += sizeof(float);
    }
    return true;
}

static inline float trillim_mxfp4_scale(uint8_t exponent) {
    uint32_t bits = exponent == 0 ? UINT32_C(0x00400000) : (uint32_t)exponent << 23;
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return exponent == 255 ? NAN : scale;
}

static inline float trillim_packed_read_float(
    const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns
) {
    if (trillim_itemsize(dtype) != 0) {
        TrillimValue value = trillim_load(data, index, dtype);
        if (trillim_float(dtype)) {
            return value.f;
        }
        return trillim_unsigned(dtype) ? (float)value.u : (float)value.i;
    }
    const uint8_t *values = (const uint8_t *)data;
    size_t row = index / columns;
    size_t column = index % columns;
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        int8_t code;
        memcpy(&bits, values + rows * columns + row * 2, 2);
        memcpy(&code, values + index, 1);
        return (float)code * trillim_bf16(bits);
    }
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_BINARY ||
          dtype == TRILLIM_GRP_TERNARY) ? 128 : 32);
    size_t groups = columns / group + (columns % group != 0);
    size_t total = rows * groups;
    size_t block = row * groups + column / group;
    size_t offset = column % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        size_t code = (values[block * 32 + offset % 32] >> (6 - (offset / 32) * 2)) & 3;
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
        return code == 3 ? NAN : (float)((int32_t)code - 1) * scale;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        uint8_t code = (values[block * 16 + offset / 2] >> ((offset % 2) * 4)) & 15;
        uint8_t exponent = values[total * 16 + block];
        float value = magnitudes[code & 7];
        if ((code & 8) != 0) {
            value = -value;
        }
        return value * trillim_mxfp4_scale(exponent);
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 :
        (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    uint16_t bits;
    memcpy(&bits, values + scale_start + block * 2, 2);
    float scale = trillim_bf16(bits);
    if (dtype == TRILLIM_GRP_INT8) {
        int8_t code;
        memcpy(&code, values + block * 32 + offset, 1);
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        int32_t code = ((values[block * 16 + offset / 8] >> (7 - offset % 8)) & 1) ? 1 : -1;
        return (float)code * scale;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t code = (values[block * 16 + offset % 16] >> ((offset / 16) * 4)) & 15;
        if (dtype == TRILLIM_GRP_INT5) {
            code |= ((values[scale_start + total * 2 + block * 4 + offset / 8] >> (offset % 8)) & 1) << 4;
        }
        return (float)(code - (dtype == TRILLIM_GRP_INT4 ? 8 : 16)) * scale;
    }
    size_t half = offset / 128;
    size_t lane = (offset % 128) / 32;
    size_t low = block * 128 + half * 64 + (lane / 2) * 32 + offset % 32;
    size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
    int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
    code |= ((values[high] >> (lane * 2)) & 3) << 4;
    int8_t subscale;
    memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
    return (float)((code - 32) * subscale) * scale;
}
#endif
