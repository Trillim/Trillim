#ifndef TRILLIM_KERNEL_QUANTIZE_H
#define TRILLIM_KERNEL_QUANTIZE_H

#include "codec.h"

static inline float trillim_round_even(float value) {
    float lower = floorf(value);
    float fraction = value - lower;
    if (fraction < 0.5f || (fraction == 0.5f && fmodf(lower, 2.0f) == 0.0f)) {
        return lower;
    }
    return lower + 1.0f;
}

static inline int32_t trillim_quantize(float value, float scale, int32_t bound) {
    float normalized = scale == 0.0f ? value : value / scale;
    normalized = fminf((float)(bound - 1), fmaxf((float)-bound, normalized));
    return (int32_t)trillim_round_even(normalized);
}

static inline TrillimStatus trillim_encode_quant(
    const float *source, size_t rows, size_t columns, uint32_t dtype,
    void *output, const TrillimKernelContext *context
) {
    size_t bytes;
    if (!trillim_payload_size(rows, columns, dtype, &bytes) || trillim_itemsize(dtype) != 0) {
        return TRILLIM_ERROR;
    }
    uint8_t *values = (uint8_t *)output;
    for (size_t offset = 0; offset < bytes; offset += 1024) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = bytes - offset < 1024 ? bytes - offset : 1024;
        memset(values + offset, 0, count);
    }
    float gamma = 0.0f;
    for (size_t start = 0; start < rows * columns; start += 1024) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = rows * columns - start < 1024 ? rows * columns : start + 1024;
        for (size_t index = start; index < end; ++index) {
            if (!isfinite(source[index])) {
                return TRILLIM_ERROR;
            }
            gamma = fmaxf(gamma, fabsf(source[index]));
        }
    }
    if (dtype == TRILLIM_ROW_INT8) {
        for (size_t row = 0; row < rows; ++row) {
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
                float q = trillim_round_even(source[row * columns + column] * multiplier);
                int8_t code = (int8_t)fminf(127.0f, fmaxf(-127.0f, q));
                memcpy(values + row * columns + column, &code, 1);
            }
        }
        return TRILLIM_OK;
    }
    size_t group = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        ((dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY ||
          dtype == TRILLIM_GRP_BINARY) ? 128 : 32);
    size_t groups = columns / group + (columns % group != 0);
    size_t total = rows * groups;
    size_t value_width = dtype == TRILLIM_GRP_INT8 || dtype == TRILLIM_MAT_TERNARY ||
        dtype == TRILLIM_GRP_TERNARY ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    if (dtype == TRILLIM_MAT_TERNARY) {
        memcpy(values + total * 32, &gamma, sizeof(gamma));
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t part = 0; part < groups; ++part) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t block = row * groups + part;
            float input[256];
            float maximum = 0.0f;
            float absolute_sum = 0.0f;
            for (size_t i = 0; i < group; ++i) {
                size_t column = part * group + i;
                input[i] = column < columns ? source[row * columns + column] : 0.0f;
                maximum = fmaxf(maximum, fabsf(input[i]));
                absolute_sum += fabsf(input[i]);
            }
            float scale = maximum;
            int32_t bound = dtype == TRILLIM_GRP_INT4 ? 8 :
                (dtype == TRILLIM_GRP_INT5 ? 16 : 128);
            int8_t subscales[16];
            int exponent = 0;
            if (dtype == TRILLIM_GRP_INT8 || dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
                scale = maximum / (dtype == TRILLIM_GRP_INT8 ? 127.0f : (float)bound);
            } else if (dtype == TRILLIM_GRP_BINARY) {
                scale = absolute_sum / 128.0f;
            } else if (dtype == TRILLIM_DUBGRP_INT6) {
                float ideal[16];
                float master = 0.0f;
                for (size_t sub = 0; sub < 16; ++sub) {
                    float submax = 0.0f;
                    for (size_t i = 0; i < 16; ++i) {
                        submax = fmaxf(submax, fabsf(input[sub * 16 + i]));
                    }
                    ideal[sub] = submax / 31.0f;
                    master = fmaxf(master, ideal[sub]);
                }
                scale = master / 127.0f;
                for (size_t sub = 0; sub < 16; ++sub) {
                    float value = scale == 0.0f ? ideal[sub] : ideal[sub] / scale;
                    subscales[sub] = (int8_t)fminf(127.0f, trillim_round_even(value));
                }
                memcpy(values + total * (128 + 2 + 64) + block * 16, subscales, 16);
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
                    float magnitude = dtype == TRILLIM_MAT_TERNARY ? gamma : maximum;
                    float normalized = magnitude == 0.0f ? 0.0f : input[i] / magnitude;
                    int32_t code;
                    if (dtype == TRILLIM_MAT_TERNARY) {
                        code = (int32_t)trillim_round_even(fminf(1.0f, fmaxf(-1.0f, normalized))) + 1;
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
                    float effective = scale * subscales[i / 16];
                    uint8_t code = (uint8_t)(trillim_quantize(input[i], effective, 32) + 32);
                    size_t half = i / 128;
                    size_t lane = (i % 128) / 32;
                    values[block * 128 + half * 64 + (lane / 2) * 32 + i % 32] |= (uint8_t)((code & 15) << ((lane % 2) * 4));
                    values[total * (128 + 2) + block * 64 + half * 32 + i % 32] |= (uint8_t)((code >> 4) << (lane * 2));
                } else {
                    int32_t quantized = trillim_quantize(input[i], scale, bound);
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
    }
    return TRILLIM_OK;
}

#endif
