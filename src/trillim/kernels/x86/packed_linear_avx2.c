#include "common.h"
#include <fenv.h>
#include <immintrin.h>
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

enum {
    PACKED_ACTIVATION_TILE = 4,
    PACKED_Q5_PREFILL_TILE = 8,
    PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024,
};

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    size_t stride;
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
    if (dtype == TRILLIM_BF16 || dtype == TRILLIM_FP32) {
        const __m256i abs_mask = _mm256_set1_epi32(INT32_MAX);
        __m256 magnitude = _mm256_setzero_ps();
        const uint8_t *data = input;
        for (size_t i = 0; i < group; i += 8) {
            __m256 value;
            if (dtype == TRILLIM_BF16) {
                __m128i bits = _mm_loadu_si128((const __m128i *)(data + (start + i) * 2));
                value = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
            } else {
                value = _mm256_loadu_ps((const float *)(data + (start + i) * 4));
            }
            __m256i absolute = _mm256_and_si256(_mm256_castps_si256(value), abs_mask);
            __m256i nonfinite = _mm256_cmpgt_epi32(absolute, _mm256_set1_epi32(0x7f7fffff));
            if (_mm256_movemask_epi8(nonfinite)) {
                return false;
            }
            magnitude = _mm256_max_ps(magnitude, _mm256_castsi256_ps(absolute));
            _mm256_storeu_ps(values + i, value);
        }
        __m128 maximum = _mm_max_ps(_mm256_castps256_ps128(magnitude), _mm256_extractf128_ps(magnitude, 1));
        maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
        maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, _MM_SHUFFLE(1, 1, 1, 1)));
        peak = _mm_cvtss_f32(maximum);
        if (group == 256 && peak != 0.0f) {
            // Q6 uses the sign of the first maximum in input order.
            for (size_t i = 0; i < group; i += 8) {
                __m256 value = _mm256_loadu_ps(values + i);
                __m256 absolute = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(value), abs_mask));
                unsigned matches = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(absolute, _mm256_set1_ps(peak), _CMP_EQ_OQ));
                if (matches) {
                    peak = values[i + (size_t)__builtin_ctz(matches)];
                    break;
                }
            }
        }
    } else {
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
    const __m256 multiplier = _mm256_set1_ps(inverse);
    const __m256 bias = _mm256_set1_ps(12582912.0f);
    const __m256i bias_bits = _mm256_castps_si256(bias);
    for (size_t i = 0; i < group; i += 8) {
        __m256 values8 = _mm256_loadu_ps(values + i);
        __m256 rounded = _mm256_add_ps(_mm256_mul_ps(values8, multiplier), bias);
        __m256i integers = _mm256_sub_epi32(_mm256_castps_si256(rounded), bias_bits);
        __m128i words = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        _mm_storel_epi64((__m128i *)(codes + i), _mm_packs_epi16(words, words));
    }
    return true;
}

static inline TrillimStatus packed_activation_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const PackedActivation *a = opaque;
    size_t group = a->width / a->groups;
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
        if (!packed_activation_quantize(a->input, a->dtype, row * a->stride + start,
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
    size_t rows, size_t width, size_t stride, size_t columns, const TrillimKernelContext *context, void *scratch
) {
    size_t group = packed_activation_group(weight_dtype);
    if (scratch == NULL || !group || !rows || width < group || width > 65536 || width % group || columns < 16 ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    float *scales = scratch;
    int8_t *codes = (int8_t *)(scales + rows * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, stride, input_dtype};
    const TrillimThreadPool *pool = context->thread_pool;
    return rows == 1 ? packed_activation_row(a, 0, 0)
                     : pool->run(pool->context, rows, packed_activation_row, a);
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

static inline TrillimStatus packed_activation_dot(
    const PackedActivation *a, size_t first_row, size_t rows, const uint8_t *weight,
    uint32_t dtype, size_t outputs, size_t column,
    float sums[PACKED_Q5_PREFILL_TILE], bool *used
) {
    *used = false;
    for (size_t row = 0; row < rows; ++row) {
        if (!packed_activation_ready(a, first_row + row)) {
            return TRILLIM_OK;
        }
    }
    size_t total = outputs * a->groups;
    size_t block = column * a->groups;
    bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    size_t low_size = q6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *low = weight + block * low_size;
    const uint8_t *scales = weight + total * low_size + block * 2;
    const uint8_t *high = weight + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (q6 ? total * 64 + block * 16 : 0));
    size_t high_size = q6 ? 64 : (dtype == TRILLIM_GRP_INT5 ? 4 : 0);
    high += block * high_size;
    const int8_t *codes = a->codes + first_row * a->width;
    const float *activation_scales = a->scales + first_row * a->groups;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 output[PACKED_Q5_PREFILL_TILE];
    for (size_t row = 0; row < rows; ++row) {
        output[row] = _mm256_setzero_ps();
    }
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
        __m256i products[PACKED_Q5_PREFILL_TILE];
        if (q6) {
            for (size_t row = 0; row < rows; ++row) {
                products[row] = _mm256_setzero_si256();
            }
            for (size_t region = 0; region < 8; ++region) {
                size_t half = region / 4, lane = region % 4;
                __m256i w = packed_q6_weights(l + half * 64 + lane / 2 * 32, h + half * 32, lane);
                const int8_t *s = subscales + g * 16 + region * 2;
                __m256i scale_pair = _mm256_set_m128i(_mm_set1_epi16(s[1]), _mm_set1_epi16(s[0]));
                for (size_t row = 0; row < rows; ++row) {
                    const int8_t *x = codes + row * a->width + g * 256 + region * 32;
                    __m256i pairs = packed_pair_products(w, _mm256_loadu_si256((const __m256i *)x));
                    products[row] = _mm256_add_epi32(products[row], _mm256_madd_epi16(pairs, scale_pair));
                }
            }
        } else {
            __m256i w = packed_q8_weights(l, h, dtype);
            for (size_t row = 0; row < rows; ++row) {
                const int8_t *x = codes + row * a->width + g * 32;
                __m256i pairs = packed_pair_products(w, _mm256_loadu_si256((const __m256i *)x));
                products[row] = _mm256_madd_epi16(pairs, ones);
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float combined_scale = scale * activation_scales[row * a->groups + g];
            __m256 partial = _mm256_mul_ps(_mm256_cvtepi32_ps(products[row]), _mm256_set1_ps(combined_scale));
            output[row] = _mm256_add_ps(output[row], partial);
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = packed_sum_f32(output[row]);
    }
    *used = true;
    return TRILLIM_OK;
}

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *weight;
    const Parameters *p;
    size_t columns, rows_per_tile;
    const TrillimKernelContext *context;
    PackedActivation activation;
} Task;

static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum,sum));
}
static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits),16));
    }
    if (dtype == TRILLIM_FP16) {
        __m128i packed = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        __m256i bits = _mm256_cvtepu16_epi32(packed);
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x8000)),16);
        __m256i exponent = _mm256_and_si256(bits,_mm256_set1_epi32(0x7c00));
        __m256i mantissa = _mm256_and_si256(bits,_mm256_set1_epi32(0x03ff));
        __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(0x7fff)),13),_mm256_set1_epi32(0x38000000));
        __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa),_mm256_set1_ps(0x1p-24f));
        __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000),_mm256_slli_epi32(mantissa,13));
        __m256 result = _mm256_blendv_ps(_mm256_castsi256_ps(normal),subnormal,_mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent,_mm256_setzero_si256())));
        result = _mm256_blendv_ps(result,_mm256_castsi256_ps(special),_mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent,_mm256_set1_epi32(0x7c00))));
        return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(result),sign));
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) values[lane] = trillim_load(data,index + lane,dtype).f;
    return _mm256_loadu_ps(values);
}
static size_t group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        p->weight_dtype < TRILLIM_ROW_INT8 || p->weight_dtype > TRILLIM_MXFP4) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t input_bytes = trillim_itemsize(p->input_dtype), output_bytes = trillim_itemsize(p->output_dtype);
    if ((p->width && p->tokens > SIZE_MAX / input_bytes / p->width) ||
        p->tokens > SIZE_MAX / output_bytes / p->outputs) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * input_bytes;
    outputs[0] = p->tokens * p->outputs * output_bytes;
    if (!p->width) {
        return TRILLIM_OK;
    }
    if (p->weight_dtype == TRILLIM_ROW_INT8) {
        if (p->width > SIZE_MAX - 2 || p->outputs > SIZE_MAX / (p->width + 2)) {
            return TRILLIM_ERROR;
        }
        inputs[1] = p->outputs * (p->width + 2);
        return TRILLIM_OK;
    }
    static const size_t records[] = {32, 34, 18, 22, 210, 18, 34, 17};
    size_t group = group_width(p->weight_dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    size_t record = records[p->weight_dtype - TRILLIM_MAT_TERNARY];
    size_t extra = p->weight_dtype == TRILLIM_MAT_TERNARY ? sizeof(float) : 0;
    if (groups > SIZE_MAX / record || p->outputs > (SIZE_MAX - extra) / (groups * record)) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->outputs * groups * record + extra;
    if (p->output_dtype == TRILLIM_FP32) {
        *scratch = packed_activation_space(p->weight_dtype, p->tokens, p->width, p->outputs);
    }
    return TRILLIM_OK;
}

static float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static float load_fp32(const void *data, size_t index) {
    float value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return value;
}

static float load_fp16(const void *data, size_t index) {
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return trillim_fp16(value);
}

static float load_bf16(const void *data, size_t index) {
    return read_bf16((const uint8_t *)data + index * sizeof(uint16_t));
}

static inline __m256 decode8(const uint8_t *values, size_t rows, size_t columns, uint32_t dtype, size_t row, size_t k) {
    __m256i ones = _mm256_set1_epi32(1), mask = _mm256_set1_epi32(15);
    __m256i codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = read_bf16(values + rows * columns + row * 2);
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + row * columns + k)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(codes),_mm256_set1_ps(scale));
    }
    size_t group = group_width(dtype), groups = columns / group + (columns % group != 0);
    size_t total = rows * groups, block = row * groups + k / group, offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        if (dtype == TRILLIM_MAT_TERNARY) memcpy(&scale,values + total * 32,4);
        else scale = read_bf16(values + total * 32 + block * 2);
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset % 32)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(6 - offset / 32 * 2))),_mm256_set1_epi32(3));
        __m256 decoded = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(codes,ones)),_mm256_set1_ps(scale));
        return _mm256_blendv_ps(decoded,_mm256_set1_ps(NAN),_mm256_castsi256_ps(_mm256_cmpeq_epi32(codes,_mm256_set1_epi32(3))));
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
    scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        codes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 32 + offset)));
    } else if (dtype == TRILLIM_GRP_BINARY) {
        codes = _mm256_srlv_epi32(_mm256_set1_epi32(values[block * 16 + offset / 8]),_mm256_setr_epi32(7,6,5,4,3,2,1,0));
        codes = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(codes,ones),1),ones);
    } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + block * 16 + offset % 16)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(offset / 16 * 4))),mask);
        if (dtype == TRILLIM_GRP_INT5) {
            __m256i high = _mm256_srlv_epi32(_mm256_set1_epi32(values[scale_start + total * 2 + block * 4 + offset / 8]),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
            codes = _mm256_or_si256(codes,_mm256_slli_epi32(_mm256_and_si256(high,ones),4));
        }
        codes = _mm256_sub_epi32(codes,_mm256_set1_epi32(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
    } else {
        size_t half = offset / 128, lane = offset % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + offset % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
        codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + low)));
        codes = _mm256_and_si256(_mm256_srlv_epi32(codes,_mm256_set1_epi32((int)(lane % 2 * 4))),mask);
        __m256i upper = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + high)));
        upper = _mm256_and_si256(_mm256_srlv_epi32(upper,_mm256_set1_epi32((int)(lane * 2))),_mm256_set1_epi32(3));
        codes = _mm256_sub_epi32(_mm256_or_si256(codes,_mm256_slli_epi32(upper,4)),_mm256_set1_epi32(32));
        int8_t subscale;
        memcpy(&subscale,values + scale_start + total * 66 + block * 16 + offset / 16,1);
        codes = _mm256_mullo_epi32(codes,_mm256_set1_epi32(subscale));
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes),_mm256_set1_ps(scale));
}
static void decode_scalar(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
    const Parameters *p = task->p;
    const uint8_t *values = task->weight;
    uint32_t dtype = p->weight_dtype;
    if (dtype == TRILLIM_ROW_INT8) {
        float scale = read_bf16(values + p->outputs * p->width + column * 2);
        const int8_t *codes = (const int8_t *)values + column * p->width + k;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    size_t group = group_width(dtype);
    size_t groups = p->width / group + (p->width % group != 0);
    size_t total = p->outputs * groups;
    size_t block = column * groups + k / group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            scale = read_bf16(values + total * 32 + block * 2);
        }
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 32 + i % 32] >> (6 - (i / 32) * 2)) & 3;
            decoded[i] = code == 3 ? NAN : (float)(code - 1) * scale;
        }
        return;
    }
    if (dtype == TRILLIM_MXFP4) {
        static const float magnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        uint8_t exponent = values[total * 16 + block];
        uint32_t bits = exponent != 0 ? (uint32_t)exponent << 23 : UINT32_C(0x00400000);
        float scale;
        memcpy(&scale, &bits, sizeof(scale));
        float palette[16];
        for (size_t i = 0; i < 8; ++i) {
            palette[i] = exponent == 255 ? NAN : magnitudes[i] * scale;
            palette[i + 8] = exponent == 255 ? NAN : -palette[i];
        }
        for (size_t i = 0; i < count; ++i) {
            uint8_t code = (values[block * 16 + i / 2] >> ((i % 2) * 4)) & 15;
            decoded[i] = palette[code];
        }
        return;
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    float scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)values + block * 32;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        for (size_t i = 0; i < count; ++i) {
            bool positive = (values[block * 16 + i / 8] >> (7 - i % 8)) & 1;
            decoded[i] = positive ? scale : -scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t offset = dtype == TRILLIM_GRP_INT4 ? 8 : 16;
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 16 + i % 16] >> ((i / 16) * 4)) & 15;
            if (dtype == TRILLIM_GRP_INT5) {
                code |= ((values[scale_start + total * 2 + block * 4 + i / 8] >> (i % 8)) & 1) << 4;
            }
            decoded[i] = (float)(code - offset) * scale;
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t half = i / 128;
        size_t lane = i % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + i % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + i % 32;
        int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
        code |= ((values[high] >> (lane * 2)) & 3) << 4;
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + i / 16, sizeof(subscale));
        decoded[i] = (float)((code - 32) * subscale) * scale;
    }
}
static void decode_block(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
    const Parameters *p = task->p;
    if (count % 8 || k % 8) { decode_scalar(task,column,k,count,decoded); return; }
    for (size_t i = 0; i < count; i += 8) _mm256_storeu_ps(decoded + i,decode8(task->weight,p->outputs,p->width,p->weight_dtype,column,k+i));
}

#define TRILLIM_PACKED_PROJECT(NAME, LOAD) \
static TrillimStatus project_##NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const Parameters *p = task->p; \
    size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0); \
    size_t row = tile / tiles * task->rows_per_tile; \
    size_t rows = p->tokens - row < task->rows_per_tile ? p->tokens - row : task->rows_per_tile; \
    size_t first = tile % tiles * task->columns; \
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns; \
    size_t group = group_width(p->weight_dtype); \
    for (size_t column = first; column < end; ++column) { \
        float sums[PACKED_Q5_PREFILL_TILE] = {0}; \
        bool quantized = false; \
        if (task->activation.scales != NULL) { \
            TrillimStatus status = packed_activation_dot(&task->activation, row, rows, task->weight, \
                p->weight_dtype, p->outputs, column, sums, &quantized); \
            if (status != TRILLIM_OK) { \
                return status; \
            } \
        } \
        for (size_t k = 0; !quantized && k < p->width; k += group) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t count = p->width - k < group ? p->width - k : group; \
            float decoded[256]; \
            decode_block(task, column, k, count, decoded); \
            size_t i = 0; \
            __m256 accumulators[PACKED_Q5_PREFILL_TILE] = {0}; \
            for (; count - i >= 8; i += 8) { \
                __m256 weight = _mm256_loadu_ps(decoded + i); \
                for (size_t r = 0; r < rows; ++r) { \
                    __m256 x = load8(task->input,row * p->width + k + r * p->width + i,p->input_dtype); \
                    accumulators[r] = _mm256_add_ps(accumulators[r],_mm256_mul_ps(x,weight)); \
                } \
            } \
            for (size_t r = 0; r < rows; ++r) sums[r] += sum8(accumulators[r]); \
            for (; i < count; ++i) { \
                float weight = load_fp32(decoded, i); \
                for (size_t r = 0; r < rows; ++r) { \
                    sums[r] = multiply_add(LOAD(task->input, (row + r) * p->width + k + i), weight, sums[r]); \
                } \
            } \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue result = {.f = sums[r]}; \
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, result); \
        } \
    } \
    return TRILLIM_OK; \
}

TRILLIM_PACKED_PROJECT(fp32, load_fp32)
TRILLIM_PACKED_PROJECT(bf16, load_bf16)
TRILLIM_PACKED_PROJECT(fp16, load_fp16)

#undef TRILLIM_PACKED_PROJECT

TRILLIM_KERNEL TrillimStatus trillim_x86_block_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t columns = p->outputs / workers / 4;
    columns = columns / 16 * 16;
    if (columns < 16) {
        columns = 16;
    }
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    TrillimTask project;
    switch (p->input_dtype) {
    case TRILLIM_FP32: project = project_fp32; break;
    case TRILLIM_BF16: project = project_bf16; break;
    case TRILLIM_FP16: project = project_fp16; break;
    default: return TRILLIM_ERROR;
    }
    size_t capacity = p->tokens;
    size_t rows_per_tile = PACKED_ACTIVATION_TILE;
    size_t group = packed_activation_group(p->weight_dtype);
    if (p->output_dtype == TRILLIM_FP32 && scratch != NULL && group != 0 &&
        p->width >= group && p->width <= 65536 && p->width % group == 0 && p->outputs >= 16 &&
        fegetround() == FE_TONEAREST) {
        capacity = PACKED_ACTIVATION_MAX_BYTES / (p->width / group * sizeof(float) + p->width);
        if (p->weight_dtype == TRILLIM_GRP_INT5 && p->tokens >= PACKED_Q5_PREFILL_TILE && tiles >= workers) {
            rows_per_tile = PACKED_Q5_PREFILL_TILE;
        }
        if (capacity < p->tokens) {
            capacity = capacity / rows_per_tile * rows_per_tile;
        }
    }
    for (size_t first = 0; first < p->tokens;) {
        Parameters panel = *p;
        panel.tokens = p->tokens - first < capacity ? p->tokens - first : capacity;
        Task task = {outputs[0], inputs[0], inputs[1], &panel, columns, rows_per_tile, context, {0}};
        if (first != 0) {
            task.input = (const uint8_t *)inputs[0] + first * p->width * trillim_itemsize(p->input_dtype);
            task.output = (uint8_t *)outputs[0] + first * p->outputs * trillim_itemsize(p->output_dtype);
        }
        TrillimStatus status = TRILLIM_OK;
        if (p->output_dtype == TRILLIM_FP32) {
            status = packed_activation_prepare(&task.activation, task.input, p->input_dtype, p->weight_dtype,
                panel.tokens, p->width, p->width, p->outputs, context, scratch);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t panel_rows = panel.tokens / rows_per_tile + (panel.tokens % rows_per_tile != 0);
        status = panel_rows * tiles == 1 ? project(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, panel_rows * tiles, project, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        first += panel.tokens;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_block_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides\n"
           "def bind(context):\n"
           "    source, weight = context.inputs['x'], context.inputs['weight']\n"
           "    if source.dtype not in {'fp16', 'bf16', 'fp32'} or weight.dtype not in {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}:\n"
           "        raise ValueError('packed linear requires floating activations and compressed weights')\n"
           "    for tensor in (source, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('packed linear requires contiguous operands')\n"
           "    values = _linear(context)\n"
           "    if values['b_storage_rows'] != values['n'] or values['b_storage_columns'] != values['k']:\n"
           "        raise ValueError('packed linear requires a complete canonical weight matrix')\n"
           "    return dict(tokens=values['m'], width=values['k'], outputs=values['n'], input_dtype=values['a_dtype'], weight_dtype=values['b_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_block_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_block_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
