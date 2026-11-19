#include "common.h"
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

__attribute__((target("avx2"))) static float trillim_packed_max_abs16(const float *input) {
    __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 low = _mm256_andnot_ps(sign, _mm256_loadu_ps(input));
    __m256 high = _mm256_andnot_ps(sign, _mm256_loadu_ps(input + 8));
    __m256 maximum = _mm256_max_ps(low, high);
    __m128 lanes = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
    lanes = _mm_max_ps(lanes, _mm_movehl_ps(lanes, lanes));
    return _mm_cvtss_f32(_mm_max_ss(lanes, _mm_shuffle_ps(lanes, lanes, _MM_SHUFFLE(1, 1, 1, 1))));
}

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

typedef struct {
    size_t rows;
    size_t width;
    size_t tokens;
    size_t output_rows;
    size_t output_columns;
    uint32_t dtype;
    uint32_t index_dtype;
} Parameters;

typedef struct {
    float *decoded;
    const uint8_t *weight;
    const void *indices;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static bool index_dtype(uint32_t dtype) {
    return dtype == TRILLIM_INT32 || dtype == TRILLIM_UINT32 ||
           dtype == TRILLIM_INT64 || dtype == TRILLIM_UINT64;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0 ||
        p->dtype < TRILLIM_ROW_INT8 || p->dtype > TRILLIM_MXFP4 ||
        !index_dtype(p->index_dtype) ||
        !trillim_packed_payload_size(p->rows, p->width, p->dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t id_size = trillim_itemsize(p->index_dtype);
    if (p->tokens > SIZE_MAX / id_size || p->tokens > SIZE_MAX / sizeof(float) / p->width ||
        !p->output_columns || p->output_rows > SIZE_MAX / p->output_columns ||
        p->output_rows * p->output_columns != p->tokens * p->width ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    inputs[1] = p->tokens * id_size;
    *scratch = p->tokens * p->width * sizeof(float);
    return TRILLIM_OK;
}

static size_t tiles_per_row(size_t width) {
    return width / 1024 + (width % 1024 != 0);
}

__attribute__((target("avx2"))) static TrillimStatus gather(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = tiles_per_row(p->width);
    size_t token = tile / tiles;
    size_t column = (tile % tiles) * 1024;
    size_t end = p->width - column < 1024 ? p->width : column + 1024;
    TrillimValue coordinate = trillim_load(task->indices, token, p->index_dtype);
    uint64_t row = trillim_unsigned(p->index_dtype) ? coordinate.u : (uint64_t)coordinate.i;
    if (row >= p->rows) {
        return TRILLIM_ERROR;
    }
    size_t source = (size_t)row * p->width;
    size_t destination = token * p->width;
    for (; end - column >= 8; column += 8) {
        __m256 values = trillim_packed_decode8(task->weight, p->rows, p->width, p->dtype,
                                               (size_t)row, column);
        if (!trillim_packed_finite8(values)) {
            return TRILLIM_ERROR;
        }
        _mm256_storeu_ps(task->decoded + destination + column, values);
    }
    for (; column < end; ++column) {
        float value = trillim_packed_read_float(task->weight, source + column, p->dtype, p->rows, p->width);
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        task->decoded[destination + column] = value;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_gather_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    Task task = {scratch, inputs[0], inputs[1], p, context};
    size_t tiles = p->tokens * tiles_per_row(p->width);
    TrillimStatus status = tiles == 1 ? gather(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, gather, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    return trillim_packed_encode_quant(task.decoded, p->output_rows, p->output_columns,
                                      p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_x86_packed_gather_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'indices')\n"
           "def bind(context):\n"
           "    weight, indices = context.inputs['x'], context.inputs['indices']\n"
           "    output = context.outputs[0]\n"
           "    if _DTYPE_IDS[weight.dtype] < _DTYPE_IDS['row_int8'] or output.dtype != weight.dtype:\n"
           "        raise ValueError('packed gather requires the same packed input and output dtype')\n"
           "    if not weight.shape:\n"
           "        raise ValueError('packed gather requires a source row axis')\n"
           "    rows, width = _storage_shape(weight)\n"
           "    if (rows, width) != (weight.shape[0], _product(weight.shape[1:])):\n"
           "        raise ValueError('packed gather requires a complete canonical source')\n"
           "    if not indices.shape or indices.shape[-1].value != 1 or indices.dtype not in ('int32', 'uint32', 'int64', 'uint64'):\n"
           "        raise ValueError('packed gather requires single integer coordinates')\n"
           "    for tensor in (weight, indices):\n"
           "        if tensor.strides and tensor.strides != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('packed gather requires contiguous inputs')\n"
           "    output_rows, output_columns = _storage_shape(output)\n"
           "    return dict(rows=rows, width=width, tokens=_product(indices.shape[:-1]), output_rows=output_rows, output_columns=output_columns, dtype=_DTYPE_IDS[weight.dtype], index_dtype=_DTYPE_IDS[indices.dtype])\n"
           "def pattern(x: Tensor, indices: Tensor, *, fill=0) -> tuple[Tensor, ...]:\n"
           "    return (Gather(x, indices=indices, fill=fill),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_gather_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"output_rows", "size_t", offsetof(Parameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(Parameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"index_dtype", "uint32_t", offsetof(Parameters, index_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_gather_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
