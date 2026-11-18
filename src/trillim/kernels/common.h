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

#ifdef TRILLIM_ENABLE_PACKED_AVX2
#include <float.h>
#include <immintrin.h>

static inline float trillim_packed_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

/* An eight-lane read within one logical row. */
__attribute__((target("avx2"))) static inline __m256 trillim_packed_decode8(const uint8_t *values, size_t rows, size_t columns, uint32_t dtype, size_t row, size_t k) {
    if ((k & 7) != 0) {
        float lane_values[8];
        for (size_t i = 0; i < 8; ++i) {
            lane_values[i] = trillim_packed_read_float(values, row * columns + k + i, dtype, rows, columns);
        }
        return _mm256_loadu_ps(lane_values);
    }
    __m256i ones = _mm256_set1_epi32(1), mask = _mm256_set1_epi32(15);
    __m256i codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = trillim_packed_bf16(values + rows * columns + row * 2);
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + row * columns + k)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
    }
    size_t group = (dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY ? 128 : 32)), groups = columns / group + (columns % group != 0);
    size_t total = rows * groups, block = row * groups + k / group, offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            scale = trillim_packed_bf16(values + total * 32 + block * 2);
        }
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset % 32)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(6 - offset / 32 * 2))), _mm256_set1_epi32(3));
        __m256 decoded = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(codes, ones)), _mm256_set1_ps(scale));
        return _mm256_blendv_ps(decoded, _mm256_set1_ps(NAN), _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, _mm256_set1_epi32(3))));
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float palette[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
        uint32_t bytes;
        memcpy(&bytes, values + block * 16 + offset / 2, 4);
        __m128i packed = _mm_cvtsi32_si128((int)bytes);
        __m128i low = _mm_and_si128(packed, _mm_set1_epi8(15));
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
        codes = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
        uint8_t exponent = values[total * 16 + block];
        scale = trillim_mxfp4_scale(exponent);
        return _mm256_mul_ps(_mm256_i32gather_ps(palette, codes, 4), _mm256_set1_ps(scale));
    }
    size_t width = dtype == TRILLIM_GRP_INT8 ? 32 : dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16;
    size_t scale_start = total * width;
    scale = trillim_packed_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset)));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        codes = _mm256_srlv_epi32(_mm256_set1_epi32(values[block * 16 + offset / 8]), _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
        codes = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(codes, ones), 1), ones);
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 16 + offset % 16)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(offset / 16 * 4))), mask);
        if (dtype == TRILLIM_GRP_INT5) {
            __m256i high = _mm256_srlv_epi32(_mm256_set1_epi32(values[scale_start + total * 2 + block * 4 + offset / 8]), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
            codes = _mm256_or_si256(codes, _mm256_slli_epi32(_mm256_and_si256(high, ones), 4));
        }
        codes = _mm256_sub_epi32(codes, _mm256_set1_epi32(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
    } else {
        size_t half = offset / 128, lane = offset % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + offset % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + low)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(lane % 2 * 4))), mask);
        __m256i upper = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + high)));
        upper = _mm256_and_si256(_mm256_srlv_epi32(upper, _mm256_set1_epi32((int)(lane * 2))), _mm256_set1_epi32(3));
        codes = _mm256_sub_epi32(_mm256_or_si256(codes, _mm256_slli_epi32(upper, 4)), _mm256_set1_epi32(32));
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + offset / 16, 1);
        codes = _mm256_mullo_epi32(codes, _mm256_set1_epi32(subscale));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
}

__attribute__((target("avx2"))) static inline bool trillim_packed_finite8(__m256 values) {
    __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values);
    return _mm256_movemask_ps(_mm256_cmp_ps(magnitude, _mm256_set1_ps(FLT_MAX), _CMP_LE_OQ)) == 255;
}

__attribute__((target("avx2"))) static inline void trillim_packed_store8(
    void *output, size_t index, uint32_t dtype, __m256 values
) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)output + index, values);
        return;
    }
    if (dtype == TRILLIM_BF16) {
        __m256i bits = _mm256_castps_si256(values);
        __m256i high = _mm256_srli_epi32(bits, 16);
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                      _mm256_and_si256(high, _mm256_set1_epi32(1)));
        __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
        __m256i quiet = _mm256_or_si256(high, _mm256_set1_epi32(0x40));
        __m256i nan = _mm256_castps_si256(_mm256_cmp_ps(values, values, _CMP_UNORD_Q));
        __m256i result = _mm256_blendv_epi8(rounded, quiet, nan);
        __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(result),
                                         _mm256_extracti128_si256(result, 1));
        _mm_storeu_si128((__m128i *)((uint16_t *)output + index), packed);
        return;
    }
    float lane_values[8];
    _mm256_storeu_ps(lane_values, values);
    for (size_t i = 0; i < 8; ++i) {
        trillim_store(output, index + i, dtype, (TrillimValue){.f = lane_values[i]});
    }
}
#endif

#ifdef TRILLIM_ENABLE_PACKED_ENCODER
static inline float trillim_packed_round_even(float value) {
    float integer;
    float fraction = fabsf(modff(value, &integer));
    if (fraction > 0.5f || (fraction == 0.5f && fmodf(integer, 2.0f) != 0.0f)) {
        integer += copysignf(1.0f, value);
    }
    return integer;
}

static inline int32_t trillim_packed_quantize(float value, float scale, int32_t bound) {
    float normalized = scale == 0.0f ? value : value / scale;
    normalized = fminf((float)(bound - 1), fmaxf((float)-bound, normalized));
    return (int32_t)trillim_packed_round_even(normalized);
}

static float trillim_packed_absolute_sum128(const float *input) {
    float lanes[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        lanes[lane] = fabsf(input[lane]);
    }
    for (size_t i = 8; i < 128; i += 8) {
        for (size_t lane = 0; lane < 8; ++lane) {
            lanes[lane] += fabsf(input[i + lane]);
        }
    }
    return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
           ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
}

#ifdef TRILLIM_ENABLE_PACKED_AVX2
__attribute__((target("avx2"))) static float trillim_packed_max_abs16(const float *input) {
    __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 low = _mm256_andnot_ps(sign, _mm256_loadu_ps(input));
    __m256 high = _mm256_andnot_ps(sign, _mm256_loadu_ps(input + 8));
    __m256 maximum = _mm256_max_ps(low, high);
    __m128 lanes = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
    lanes = _mm_max_ps(lanes, _mm_movehl_ps(lanes, lanes));
    return _mm_cvtss_f32(_mm_max_ss(lanes, _mm_shuffle_ps(lanes, lanes, _MM_SHUFFLE(1, 1, 1, 1))));
}
#else
static float trillim_packed_max_abs16(const float *input) {
    float maximum = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        maximum = fmaxf(maximum, fabsf(input[i]));
    }
    return maximum;
}
#endif

static float trillim_packed_encode_int6(const float *input, int8_t *subscales, uint8_t *codes) {
    float ideal[16];
    size_t anchor = 0;
    for (size_t sub = 0; sub < 16; ++sub) {
        ideal[sub] = trillim_packed_max_abs16(input + sub * 16) / 31.0f;
        if (ideal[sub] > ideal[anchor]) {
            anchor = sub;
        }
    }
    float master = trillim_bf16(trillim_to_bf16(ideal[anchor] / 127.0f));
    for (size_t sub = 0; sub < 16; ++sub) {
        const float *chunk = input + sub * 16;
        float ratio = master == 0.0f ? ideal[sub] : ideal[sub] / master;
        int32_t subscale = (int32_t)fminf(127.0f, trillim_packed_round_even(ratio));
        bool changed = false;
        for (size_t i = 0; i < 16; ++i) {
            int32_t code = trillim_packed_quantize(chunk[i], master * (float)subscale, 32);
            codes[sub * 16 + i] = (uint8_t)(code + 32);
            changed |= fabsf((float)(code * subscale) * master - chunk[i]) > 1e-5f;
        }
        subscales[sub] = (int8_t)subscale;
        if (!changed || sub == anchor) {
            continue;
        }
        int32_t products[16];
        bool on_lattice = true;
        for (size_t i = 0; i < 16; ++i) {
            float ratio_value = master == 0.0f ? chunk[i] : chunk[i] / master;
            products[i] = (int32_t)trillim_packed_round_even(ratio_value);
            on_lattice &= fabsf((float)products[i] * master - chunk[i]) <= 1e-5f;
        }
        if (!on_lattice) {
            continue;
        }
        /* Keep values already representable by a different subscale. */
        int32_t selected = 0;
        int32_t best_distance = 128;
        for (int32_t candidate = 1; candidate <= 127; ++candidate) {
            int32_t distance = candidate - subscale;
            distance = distance < 0 ? -distance : distance;
            if (distance >= best_distance) {
                continue;
            }
            bool valid = true;
            for (size_t i = 0; i < 16; ++i) {
                if (products[i] < -32 * candidate || products[i] > 31 * candidate ||
                    products[i] % candidate != 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                selected = candidate;
                best_distance = distance;
            }
        }
        if (selected != 0) {
            subscales[sub] = (int8_t)selected;
            for (size_t i = 0; i < 16; ++i) {
                codes[sub * 16 + i] = (uint8_t)(products[i] / selected + 32);
            }
        }
    }
    return master;
}

typedef struct {
    const float *source;
    size_t rows;
    size_t columns;
    uint32_t dtype;
    uint8_t *output;
    const TrillimKernelContext *context;
    size_t group_width;
    size_t groups;
    size_t value_width;
    float gamma;
} TrillimPackedEncodeTask;

static TrillimStatus trillim_packed_encode_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimPackedEncodeTask *task = opaque;
    const TrillimKernelContext *context = task->context;
    size_t rows = task->rows, columns = task->columns;
    const float *source = task->source;
    uint8_t *values = task->output;
    float maximum = 1e-5f;
    for (size_t column = 0; column < columns; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum = fmaxf(maximum, fabsf(source[row * columns + column]));
    }
    uint16_t scale = trillim_to_bf16(maximum / 127.0f);
    memcpy(values + rows * columns + row * 2, &scale, 2);
    float multiplier = 127.0f / maximum;
    for (size_t column = 0; column < columns; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float q = trillim_packed_round_even(source[row * columns + column] * multiplier);
        int8_t code = (int8_t)fminf(127.0f, fmaxf(-127.0f, q));
        memcpy(values + row * columns + column, &code, 1);
    }
    return TRILLIM_OK;
}

static void trillim_packed_encode_block(const TrillimPackedEncodeTask *task, size_t block) {
    size_t columns = task->columns, group = task->group_width, value_width = task->value_width;
    size_t total = task->rows * task->groups;
    size_t row = block / task->groups;
    size_t start = block % task->groups * group;
    size_t valid = columns - start < group ? columns - start : group;
    uint32_t dtype = task->dtype;
    const float *source = task->source;
    uint8_t *values = task->output;
    memset(values + block * value_width, 0, value_width);
    if (dtype == TRILLIM_GRP_INT5) {
        memset(values + total * (16 + 2) + block * 4, 0, 4);
    } else if (dtype == TRILLIM_DUBGRP_INT6) {
        memset(values + total * (128 + 2) + block * 64, 0, 64);
    }
    float input[256];
    float minimum = 0.0f, positive = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        input[i] = i < valid ? source[row * columns + start + i] : 0.0f;
        minimum = fminf(minimum, input[i]);
        positive = fmaxf(positive, input[i]);
    }
    positive = fabsf(positive);
    float negative = fabsf(minimum);
    float maximum = fmaxf(positive, negative);
    float scale = maximum;
    int32_t bound = dtype == TRILLIM_GRP_INT4 ? 8 :
        (dtype == TRILLIM_GRP_INT5 ? 16 : 128);
    uint8_t int6_codes[256];
    int exponent = 0;
    if (dtype == TRILLIM_GRP_INT8) {
        scale = maximum / 127.0f;
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        float ideal = fmaxf(positive / (float)(bound - 1), negative / (float)bound);
        scale = trillim_bf16(trillim_to_bf16(ideal));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        scale = trillim_packed_absolute_sum128(input) / (float)valid;
    } else if (dtype == TRILLIM_DUBGRP_INT6) {
        int8_t subscales[16];
        scale = trillim_packed_encode_int6(input, subscales, int6_codes);
        memcpy(values + total * (128 + 2 + 64) + block * 16, subscales, sizeof(subscales));
    } else if (dtype == TRILLIM_MXFP4) {
        if (maximum != 0.0f) {
            frexpf(maximum, &exponent);
            exponent -= 3;
            exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
        }
        values[total * 16 + block] = (uint8_t)(exponent + 127);
    }
    if (dtype != TRILLIM_MAT_TERNARY && dtype != TRILLIM_MXFP4) {
        uint16_t bits = trillim_to_bf16(scale);
        memcpy(values + total * value_width + block * 2, &bits, 2);
    }
    for (size_t i = 0; i < group; ++i) {
        if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
            float magnitude = dtype == TRILLIM_MAT_TERNARY ? task->gamma : maximum;
            float normalized = magnitude == 0.0f ? 0.0f : input[i] / magnitude;
            int32_t code;
            if (dtype == TRILLIM_MAT_TERNARY) {
                code = (int32_t)trillim_packed_round_even(fminf(1.0f, fmaxf(-1.0f, normalized))) + 1;
            } else {
                code = normalized >= 0.5f ? 2 : (normalized <= -0.5f ? 0 : 1);
            }
            values[block * 32 + i % 32] |= (uint8_t)(code << (6 - (i / 32) * 2));
        } else if (dtype == TRILLIM_GRP_BINARY) {
            values[block * 16 + i / 8] |= (uint8_t)((input[i] >= 0.0f) << (7 - i % 8));
        } else if (dtype == TRILLIM_MXFP4) {
            static const float midpoints[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
            float magnitude = fabsf(ldexpf(input[i], -exponent));
            uint8_t code = 0;
            while (code < 7 && magnitude > midpoints[code]) {
                ++code;
            }
            if (code < 7 && magnitude == midpoints[code] && (code & 1)) {
                ++code;
            }
            code |= signbit(input[i]) ? 8 : 0;
            values[block * 16 + i / 2] |= (uint8_t)(code << ((i % 2) * 4));
        } else if (dtype == TRILLIM_DUBGRP_INT6) {
            uint8_t code = int6_codes[i];
            size_t half = i / 128;
            size_t lane = (i % 128) / 32;
            values[block * 128 + half * 64 + (lane / 2) * 32 + i % 32] |= (uint8_t)((code & 15) << ((lane % 2) * 4));
            values[total * (128 + 2) + block * 64 + half * 32 + i % 32] |= (uint8_t)((code >> 4) << (lane * 2));
        } else {
            int32_t quantized = trillim_packed_quantize(input[i], scale, bound);
            if (dtype == TRILLIM_GRP_INT8) {
                int8_t code = (int8_t)(quantized < -127 ? -127 : quantized);
                memcpy(values + block * 32 + i, &code, 1);
            } else {
                uint8_t code = (uint8_t)(quantized + bound);
                values[block * 16 + i % 16] |= (uint8_t)((code & 15) << ((i / 16) * 4));
                if (dtype == TRILLIM_GRP_INT5) {
                    values[total * (16 + 2) + block * 4 + i / 8] |= (uint8_t)((code >> 4) << (i % 8));
                }
            }
        }
    }
}

static TrillimStatus trillim_packed_encode_blocks(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimPackedEncodeTask *task = opaque;
    size_t count = task->rows * task->groups, first = tile * 4;
    size_t end = count - first < 4 ? count : first + 4;
    for (size_t block = first; block < end; ++block) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        trillim_packed_encode_block(task, block);
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_packed_encode_quant(
    const float *source, size_t rows, size_t columns, uint32_t dtype,
    void *output, const TrillimKernelContext *context
) {
    TrillimPackedEncodeTask task = {
        .source = source, .rows = rows, .columns = columns,
        .dtype = dtype, .output = output, .context = context
    };
    TrillimTask operation = trillim_packed_encode_row;
    size_t tiles = rows;
    if (dtype != TRILLIM_ROW_INT8) {
        task.group_width = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
            ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY ||
              dtype == TRILLIM_GRP_BINARY) ? 128 : 32);
        task.groups = columns / task.group_width + (columns % task.group_width != 0);
        size_t count = rows * task.groups;
        task.value_width = dtype == TRILLIM_GRP_INT8 || dtype == TRILLIM_MAT_TERNARY ||
            dtype == TRILLIM_GRP_TERNARY ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
        if (dtype == TRILLIM_MAT_TERNARY) {
            size_t elements = rows * columns;
            for (size_t start = 0; start < elements;) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = elements - start < 1024 ? elements : start + 1024;
                for (; start < end; ++start) {
                    task.gamma = fmaxf(task.gamma, fabsf(source[start]));
                }
            }
            memcpy(task.output + count * 32, &task.gamma, sizeof(task.gamma));
        }
        tiles = count / 4 + (count % 4 != 0);
        operation = trillim_packed_encode_blocks;
    }
    return tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
}

#endif
#endif
