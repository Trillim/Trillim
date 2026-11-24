
#include "common.h"
#include <fenv.h>
#include <immintrin.h>
#include <math.h>

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
        const __m128i doubled = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12);
        uint32_t bytes;
        memcpy(&bytes, values + block * 16 + offset / 2, sizeof(bytes));
        __m128i packed = _mm_cvtsi32_si128((int32_t)bytes);
        __m128i low = _mm_and_si128(packed, _mm_set1_epi8(15));
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
        __m128i nibbles = _mm_unpacklo_epi8(low, high);
        codes = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(doubled, nibbles));
        scale = trillim_mxfp4_scale(values[total * 16 + block]);
        __m256 magnitudes = _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(0.5f));
        __m256 decoded = _mm256_mul_ps(magnitudes, _mm256_set1_ps(scale));
        __m256i signs = _mm256_slli_epi32(_mm256_cvtepu8_epi32(nibbles), 28);
        signs = _mm256_and_si256(signs, _mm256_set1_epi32(INT32_MIN));
        return _mm256_xor_ps(decoded, _mm256_castsi256_ps(signs));
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

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif
enum { PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024 };

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    size_t first_row;
    uint32_t dtype;
} PackedActivation;

static inline size_t packed_activation_group(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5:
    case TRILLIM_GRP_INT8:
        return 32;
    case TRILLIM_DUBGRP_INT6:
        return 256;
    default:
        return 0;
    }
}

static inline bool packed_activation_quantize(
    const void *input, uint32_t dtype, size_t start, size_t group, float *scale, int8_t *codes
) {
    float values[256];
    float peak = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        float value = trillim_load(input, start + i, dtype).f;
        if (!isfinite(value)) {
            return false;
        }
        if (fabsf(value) > fabsf(peak)) {
            peak = value;
        }
        values[i] = value;
    }
    if (peak == 0.0f) {
        *scale = 0.0f;
        memset(codes, 0, group);
        return true;
    }
    float inverse;
    if (group == 256) {
        inverse = -127.0f / peak;
        if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
            return false;
        }
        *scale = 1.0f / inverse;
    } else {
        float step = fabsf(peak) / 127.0f;
        if (step > 65504.0f || (*scale = trillim_fp16(trillim_to_fp16(step))) == 0.0f) {
            return false;
        }
        inverse = 1.0f / step;
    }
    for (size_t i = 0; i < group; ++i) {
        /* Bias the significand to round to nearest, ties to even. */
        float rounded = values[i] * inverse + 12582912.0f;
        uint32_t bits;
        memcpy(&bits, &rounded, sizeof(bits));
        codes[i] = (int8_t)((int32_t)(bits & UINT32_C(0x007fffff)) - INT32_C(0x00400000));
    }
    return true;
}

static inline TrillimStatus packed_activation_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const PackedActivation *a = opaque;
    size_t group = a->width / a->groups;
    size_t row_start = (a->first_row + row) * a->width;
    float *scales = a->scales + row * a->groups;
    int8_t *codes = a->codes + row * a->width;
    if (fegetround() != FE_TONEAREST) {
        scales[0] = NAN;
        return TRILLIM_OK;
    }
    for (size_t g = 0; g < a->groups; ++g) {
        if (g % 4 == 0 && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t start = g * group;
        if (!packed_activation_quantize(a->input, a->dtype, row_start + start,
                group, scales + g, codes + start)) {
            scales[0] = NAN;
            return TRILLIM_OK;
        }
    }
    return TRILLIM_OK;
}

static inline size_t packed_activation_space(uint32_t dtype, size_t rows, size_t width, size_t columns) {
    size_t group = packed_activation_group(dtype);
    if (!group || columns < 16) {
        return 0;
    }
    size_t bounded_width = width < 65536 ? width : 65536;
    size_t row_bytes = (bounded_width / group) * (group + sizeof(float));
    if (!row_bytes) {
        return 0;
    }
    return rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes ? PACKED_ACTIVATION_MAX_BYTES : rows * row_bytes;
}

static inline TrillimStatus packed_activation_prepare(
    PackedActivation *a, const void *input, uint32_t input_dtype, uint32_t weight_dtype,
    size_t first_row, size_t *rows, size_t width, size_t columns, const TrillimKernelContext *context, void *scratch
) {
    *a = (PackedActivation){0};
    size_t group = packed_activation_group(weight_dtype);
    if (scratch == NULL || !group || width < group || width > 65536 || width % group || columns < 16 ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    size_t capacity = PACKED_ACTIVATION_MAX_BYTES / (groups * sizeof(float) + width);
    size_t count = *rows < capacity ? *rows : capacity;
    *rows = count;
    float *scales = scratch;
    int8_t *codes = (int8_t *)(scales + count * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, first_row, input_dtype};
    const TrillimThreadPool *pool = context->thread_pool;
    return count == 1 ? packed_activation_row(a, 0, 0)
                      : pool->run(pool->context, count, packed_activation_row, a);
}

static inline __m256i packed_pair_products(__m256i weight, __m256i input) {
    // The activation codes fit in [-127, 127], so neither negation nor pair sums saturate.
    return _mm256_maddubs_epi16(_mm256_abs_epi8(weight), _mm256_sign_epi8(input, weight));
}

static inline __m256i packed_q8_weights(const uint8_t *low, const uint8_t *high, uint32_t dtype) {
    if (dtype == TRILLIM_GRP_INT8) {
        return _mm256_loadu_si256((const __m256i *)low);
    }
    size_t center = dtype == TRILLIM_GRP_INT4 ? 8 : 16;
    __m128i packed = _mm_loadu_si128((const __m128i *)low);
    __m256i codes = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
    codes = _mm256_and_si256(codes, _mm256_set1_epi8(15));
    if (dtype == TRILLIM_GRP_INT5) {
        uint32_t bits;
        memcpy(&bits, high, sizeof(bits));
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t)bits), _mm256_setr_epi64x(
            0, INT64_C(0x0101010101010101), INT64_C(0x0202020202020202), INT64_C(0x0303030303030303)));
        __m256i mask = _mm256_set1_epi64x((int64_t)UINT64_C(0x8040201008040201));
        __m256i fifth = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, mask), mask);
        codes = _mm256_or_si256(codes, _mm256_and_si256(fifth, _mm256_set1_epi8(16)));
    }
    return _mm256_sub_epi8(codes, _mm256_set1_epi8((char)center));
}

static inline float packed_sum_f32(__m256 sums) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static inline __m256i packed_q6_weights(const uint8_t *low, const uint8_t *high, size_t lane) {
    __m256i lower = _mm256_loadu_si256((const __m256i *)low);
    __m256i upper = _mm256_loadu_si256((const __m256i *)high);
    lower = _mm256_srl_epi16(lower, _mm_cvtsi32_si128((int)(lane % 2 * 4)));
    upper = _mm256_srl_epi16(upper, _mm_cvtsi32_si128((int)(lane * 2)));
    lower = _mm256_and_si256(lower, _mm256_set1_epi8(15));
    upper = _mm256_and_si256(upper, _mm256_set1_epi8(3));
    return _mm256_sub_epi8(_mm256_or_si256(lower, _mm256_slli_epi16(upper, 4)), _mm256_set1_epi8(32));
}

static inline bool packed_activation_ready(const PackedActivation *a, size_t row) {
    return a->scales != NULL && !isnan(a->scales[row * a->groups]);
}

static inline TrillimStatus packed_activation_dot_one(
    const PackedActivation *a, size_t row, const void *weight,
    uint32_t dtype, size_t outputs, size_t column, float *result, bool *used
) {
    *used = false;
    if (!packed_activation_ready(a, row)) {
        return TRILLIM_OK;
    }
    size_t total = outputs * a->groups;
    size_t block = column * a->groups;
    const uint8_t *bytes = weight;
    bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    size_t low_size = q6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *low = bytes + block * low_size;
    const uint8_t *scales = bytes + total * low_size + block * 2;
    const uint8_t *high = bytes + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (q6 ? total * 64 + block * 16 : 0));
    size_t high_size = q6 ? 64 : (dtype == TRILLIM_GRP_INT5 ? 4 : 0);
    high += block * high_size;
    const int8_t *codes = a->codes + row * a->width;
    const float *activation_scales = a->scales + row * a->groups;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 output = _mm256_setzero_ps();
    for (size_t g = 0; g < a->groups; ++g) {
        if ((q6 || g % 8 == 0) && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float scale = trillim_bf16(bits);
        if (!isfinite(scale) || fabsf(scale) > 0x1p40f) {
            return TRILLIM_OK;
        }
        const uint8_t *l = low + g * low_size;
        const uint8_t *h = high + g * high_size;
        __m256i products;
        if (q6) {
            products = _mm256_setzero_si256();
            for (size_t region = 0; region < 8; ++region) {
                size_t half = region / 4, lane = region % 4;
                __m256i w = packed_q6_weights(l + half * 64 + lane / 2 * 32, h + half * 32, lane);
                const int8_t *s = subscales + g * 16 + region * 2;
                __m256i scale_pair = _mm256_set_m128i(_mm_set1_epi16(s[1]), _mm_set1_epi16(s[0]));
                const int8_t *x = codes + g * 256 + region * 32;
                __m256i pairs = packed_pair_products(w, _mm256_loadu_si256((const __m256i *)x));
                products = _mm256_add_epi32(products, _mm256_madd_epi16(pairs, scale_pair));
            }
        } else {
            __m256i w = packed_q8_weights(l, h, dtype);
            const int8_t *x = codes + g * 32;
            __m256i pairs = packed_pair_products(w, _mm256_loadu_si256((const __m256i *)x));
            products = _mm256_madd_epi16(pairs, ones);
        }
        float combined_scale = scale * activation_scales[g];
        __m256 partial = _mm256_mul_ps(_mm256_cvtepi32_ps(products), _mm256_set1_ps(combined_scale));
        output = _mm256_add_ps(output, partial);
    }
    *result = packed_sum_f32(output);
    *used = true;
    return TRILLIM_OK;
}

enum { TILE_SIZE = 16 };

typedef struct {
    size_t tokens, width, outputs;
    size_t hidden_width;
    uint32_t input_dtype, output_dtype;
    uint32_t gate_dtype, up_dtype, down_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
    PackedActivation activation;
    size_t first_row, rows;
} Task;

static bool packed(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    *scratch = outputs[0] = 0;
    memset(inputs, 0, 4 * sizeof(*inputs));
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) ||
        !packed(p->gate_dtype) ||
        !packed(p->up_dtype) ||
        !packed(p->down_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (!trillim_packed_payload_size(p->tokens, p->outputs, p->output_dtype, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->width, p->input_dtype, inputs)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->up_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->gate_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    if (p->hidden_width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->hidden_width, p->down_dtype, inputs + 3)) {
        return TRILLIM_ERROR;
    }
    if (!trillim_packed_payload_size(p->tokens, p->hidden_width, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    if (p->output_dtype == TRILLIM_FP32) {
        uint32_t dtype = packed_activation_group(p->up_dtype) != 0 ? p->up_dtype : p->gate_dtype;
        size_t up = packed_activation_space(dtype, p->tokens, p->width, p->hidden_width);
        size_t down = packed_activation_space(p->down_dtype, p->tokens, p->hidden_width, p->outputs);
        size_t workspace = up > down ? up : down;
        if (workspace > SIZE_MAX - *scratch) {
            return TRILLIM_ERROR;
        }
        *scratch += workspace;
    }
    return TRILLIM_OK;
}

__attribute__((target("avx2"))) static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    const uint8_t *bytes = data;
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)(bytes + index * sizeof(float)));
    }
    const uint8_t *narrow = bytes + index * sizeof(uint16_t);
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)narrow);
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    uint16_t bits[8];
    float values[8];
    memcpy(bits, narrow, sizeof(bits));
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = trillim_fp16(bits[lane]);
    }
    return _mm256_loadu_ps(values);
}

__attribute__((target("avx2"))) static TrillimStatus dot(
    const PackedActivation *activation, const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, const TrillimKernelContext *context, float *result
) {
    if (activation->scales != NULL && width / activation->groups == packed_activation_group(weight_dtype)) {
        bool used;
        TrillimStatus status = packed_activation_dot_one(activation, row - activation->first_row, weight, weight_dtype,
            outputs, column, result, &used);
        if (status != TRILLIM_OK || used) {
            return status;
        }
    }
    size_t x_start = row * width;
    size_t w_start = column * width;
    size_t k = 0;
    float sum = 0.0f;
    __m256 sums = _mm256_setzero_ps();
    for (; width - k >= 8; k += 8) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 x = load8(input, x_start + k, input_dtype);
        __m256 w = trillim_packed_decode8(weight, outputs, width, weight_dtype, column, k);
        sums = _mm256_add_ps(sums, _mm256_mul_ps(x, w));
    }
    __m128 partial = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    partial = _mm_hadd_ps(partial, partial);
    sum = _mm_cvtss_f32(_mm_hadd_ps(partial, partial));
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        float w = trillim_packed_read_float(weight, w_start + k, weight_dtype, outputs, width);
        sum += x * w;
    }
    *result = sum;
    return TRILLIM_OK;
}

static float silu(float x) {
    float e = expf(-fabsf(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * sigmoid;
}

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t end = (task->first_row + task->rows) * p->hidden_width;
    size_t first = task->first_row * p->hidden_width + tile * TILE_SIZE;
    if (end - first > TILE_SIZE) {
        end = first + TILE_SIZE;
    }
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->hidden_width, column = index % p->hidden_width;
        float value;
        TrillimStatus status = dot(
            &task->activation, task->inputs[0], p->input_dtype, row, task->inputs[1], p->up_dtype,
            column, p->width, p->hidden_width, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        float gate;
        status = dot(
            &task->activation, task->inputs[0], p->input_dtype, row, task->inputs[2], p->gate_dtype,
            column, p->width, p->hidden_width, task->context, &gate
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        ((float *)task->hidden)[index] = value * silu(gate);
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t end = (task->first_row + task->rows) * p->outputs;
    size_t first = task->first_row * p->outputs + tile * TILE_SIZE;
    if (end - first > TILE_SIZE) {
        end = first + TILE_SIZE;
    }
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->outputs, column = index % p->outputs;
        float value;
        TrillimStatus status = dot(
            &task->activation, task->hidden, TRILLIM_FP32, row, task->inputs[3], p->down_dtype,
            column, p->hidden_width, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static inline TrillimStatus run_projection(
    Task *task, TrillimTask project_rows, const void *input,
    uint32_t input_dtype, uint32_t weight_dtype, size_t width, size_t columns, void *workspace
) {
    if (columns == 0) {
        return TRILLIM_OK;
    }
    const TrillimKernelContext *context = task->context;
    const TrillimThreadPool *pool = context->thread_pool;
    size_t tokens = task->p->tokens;
    for (size_t row = 0; row < tokens; row += task->rows) {
        task->first_row = row;
        task->rows = tokens - row;
        TrillimStatus status = packed_activation_prepare(&task->activation, input, input_dtype, weight_dtype,
            row, &task->rows, width, columns, context, workspace);
        if (status != TRILLIM_OK) {
            return status;
        }
        size_t count = task->rows * columns;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        status = tiles == 1 ? project_rows(task, 0, 0) : pool->run(pool->context, tiles, project_rows, task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[4], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    size_t hidden_count = p->tokens * p->hidden_width;
    void *workspace = NULL;
    if (scratch != NULL && p->output_dtype == TRILLIM_FP32) {
        workspace = (uint8_t *)scratch + hidden_count * sizeof(float);
    }
    uint32_t activation_dtype = packed_activation_group(p->up_dtype) != 0 ? p->up_dtype : p->gate_dtype;
    Task task = {.output = outputs[0], .inputs = inputs, .hidden = scratch, .p = p, .context = context};
    status = run_projection(&task, project_hidden, inputs[0], p->input_dtype, activation_dtype,
        p->width, p->hidden_width, workspace);
    if (status != TRILLIM_OK) {
        return status;
    }
    return run_projection(&task, project, scratch, TRILLIM_FP32, p->down_dtype, p->hidden_width, p->outputs, workspace);
}

TRILLIM_KERNEL const char *trillim_x86_packed_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = up, gate, down\n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if any(matrix.dtype not in packed for matrix in matrices):\n"
           "        raise ValueError('packed FFN requires compressed weights')\n"
           "    if gate.shape != up.shape or x.shape[-1] != up.shape[1] or down.shape[1] != up.shape[0]:\n"
           "        raise ValueError('incompatible packed FFN dimensions')\n"
           "    if output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=down.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        hidden_width=up.shape[0],\n"
           "        gate_dtype=_DTYPE_IDS[gate.dtype],\n"
           "        up_dtype=_DTYPE_IDS[up.dtype],\n"
           "        down_dtype=_DTYPE_IDS[down.dtype],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is not Silu:\n"
           "        raise ValueError('packed FFN requires an unbiased SiLU gate')\n"
           "    result = FFN(x, up=up, gate=gate, down=down)\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_ffn_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
