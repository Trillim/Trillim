#include "common.h"
#include <math.h>
#include <string.h>

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
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

static inline float trillim_read_float(
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
        return exponent == 255 ? NAN : ldexpf(value, (int)exponent - 127);
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

typedef enum {
    TRILLIM_ADD, TRILLIM_SUB, TRILLIM_MUL, TRILLIM_DIV,
    TRILLIM_FLOOR_DIV, TRILLIM_MOD, TRILLIM_EXP, TRILLIM_LOG,
    TRILLIM_SQRT, TRILLIM_SIN, TRILLIM_COS, TRILLIM_ERF
} TrillimElementwiseOperation;

static inline TrillimStatus trillim_binary_value(
    TrillimElementwiseOperation operation, TrillimValue a, TrillimValue b,
    uint32_t dtype, TrillimValue *result
) {
    if (trillim_float(dtype)) {
        switch (operation) {
        case TRILLIM_ADD: result->f = a.f + b.f; break;
        case TRILLIM_SUB: result->f = a.f - b.f; break;
        case TRILLIM_MUL: result->f = a.f * b.f; break;
        case TRILLIM_DIV: result->f = a.f / b.f; break;
        default: return TRILLIM_ERROR;
        }
    } else {
        switch (operation) {
        case TRILLIM_ADD: result->u = a.u + b.u; break;
        case TRILLIM_SUB: result->u = a.u - b.u; break;
        case TRILLIM_MUL: result->u = a.u * b.u; break;
        case TRILLIM_FLOOR_DIV: case TRILLIM_MOD:
            if (b.u == 0) {
                return TRILLIM_ERROR;
            }
            if (trillim_unsigned(dtype)) {
                result->u = operation == TRILLIM_MOD ? a.u % b.u : a.u / b.u;
            } else if (a.i == INT64_MIN && b.i == -1) {
                result->i = operation == TRILLIM_MOD ? 0 : INT64_MIN;
            } else {
                int64_t quotient = a.i / b.i;
                int64_t remainder = a.i % b.i;
                if (remainder != 0 && (remainder < 0) != (b.i < 0)) {
                    --quotient;
                    remainder += b.i;
                }
                result->i = operation == TRILLIM_MOD ? remainder : quotient;
            }
            break;
        default: return TRILLIM_ERROR;
        }
    }
    return TRILLIM_OK;
}

typedef struct {
    size_t count;
    size_t items;
    size_t coordinates;
    size_t index_rank;
    size_t slice_rank;
    size_t slice_size;
    const size_t *index_shape;
    const size_t *index_strides;
    const size_t *mask_strides;
    size_t coordinate_stride;
    const size_t *indexed_shape;
    const size_t *indexed_strides;
    const size_t *output_indexed_strides;
    const size_t *slice_shape;
    const size_t *slice_strides;
    size_t values_rank;
    const size_t *values_shape;
    const size_t *values_strides;
    size_t storage_rows;
    size_t storage_columns;
    size_t output_rows;
    size_t output_columns;
    uint32_t dtype;
    uint32_t masked;
    uint32_t mode;
    uint64_t fill_bits;
    float fill_float;
} TrillimIndexParameters;

typedef struct {
    void *output;
    const void *const *inputs;
    float *scratch;
    const TrillimIndexParameters *parameters;
    const TrillimKernelContext *context;
    bool scatter;
} TrillimIndexTask;

static inline bool trillim_index_table_size(size_t items, size_t *capacity) {
    size_t result = 1;
    if (items > SIZE_MAX / 2) {
        return false;
    }
    while (result < items * 2) {
        if (result > SIZE_MAX / 2) {
            return false;
        }
        result *= 2;
    }
    *capacity = result;
    return result <= SIZE_MAX / sizeof(size_t);
}

static inline TrillimStatus trillim_index_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch,
    bool scatter, bool masked
) {
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    size_t count = scatter ? (masked ? 4 : 3) : (masked ? 3 : 2);
    for (size_t i = 0; i < count; ++i) {
        inputs[i] = 0;
    }
    outputs[0] = 0;
    *scratch = 0;
    if (p->masked != (uint32_t)masked || p->mode > 1 ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->dtype, &inputs[0]) ||
        !trillim_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (p->items != 0) {
        size_t span;
        if (!trillim_span(p->index_rank, p->index_shape, p->index_strides, &span)) {
            return TRILLIM_ERROR;
        }
        if (p->coordinates != 0) {
            if (p->coordinate_stride != 0 && p->coordinates - 1 > (SIZE_MAX - span) / p->coordinate_stride) {
                return TRILLIM_ERROR;
            }
            span += (p->coordinates - 1) * p->coordinate_stride;
            if (span > SIZE_MAX / sizeof(int64_t)) {
                return TRILLIM_ERROR;
            }
            inputs[1] = span * sizeof(int64_t);
        }
        if (masked) {
            if (!trillim_span(p->index_rank, p->index_shape, p->mask_strides, &span)) {
                return TRILLIM_ERROR;
            }
            inputs[scatter ? 3 : 2] = span;
        }
        if (scatter && p->slice_size != 0) {
            size_t bytes = trillim_itemsize(p->dtype);
            if (bytes == 0 || !trillim_span(p->values_rank, p->values_shape, p->values_strides, &span) || span > SIZE_MAX / bytes) {
                return TRILLIM_ERROR;
            }
            inputs[2] = span * bytes;
        }
    }
    if (scatter && p->mode == 0 && p->items != 0) {
        size_t capacity;
        if (!trillim_index_table_size(p->items, &capacity)) {
            return TRILLIM_ERROR;
        }
        *scratch = capacity * sizeof(size_t);
    }
    if (trillim_itemsize(p->dtype) == 0) {
        if (p->count > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->count * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_coordinates(
    const TrillimIndexTask *task, size_t item, size_t *source, size_t *destination, size_t *key, bool *active
) {
    const TrillimIndexParameters *p = task->parameters;
    *source = 0;
    *destination = 0;
    *key = 0;
    *active = true;
    if (p->masked != 0) {
        size_t index = trillim_offset(item, p->index_rank, p->index_shape, p->mask_strides);
        if (((const uint8_t *)task->inputs[task->scatter ? 3 : 2])[index] == 0) {
            *active = false;
            return TRILLIM_OK;
        }
    }
    size_t base = trillim_offset(item, p->index_rank, p->index_shape, p->index_strides);
    for (size_t axis = 0; axis < p->coordinates; ++axis) {
        if ((axis & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t coordinate;
        memcpy(&coordinate, (const uint8_t *)task->inputs[1] + (base + axis * p->coordinate_stride) * sizeof(int64_t), sizeof(coordinate));
        if (coordinate < 0 || (uint64_t)coordinate >= p->indexed_shape[axis]) {
            return TRILLIM_ERROR;
        }
        if (p->indexed_shape[axis] != 0 && *key > (SIZE_MAX - (size_t)coordinate) / p->indexed_shape[axis]) {
            return TRILLIM_ERROR;
        }
        *key = *key * p->indexed_shape[axis] + (size_t)coordinate;
        *source += (size_t)coordinate * p->indexed_strides[axis];
        *destination += (size_t)coordinate * p->output_indexed_strides[axis];
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_gather_task(void *opaque, size_t item, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = (const TrillimIndexTask *)opaque;
    const TrillimIndexParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t source, destination, key;
    bool active;
    TrillimStatus status = trillim_coordinates(task, item, &source, &destination, &key, &active);
    if (status != TRILLIM_OK) {
        return status;
    }
    for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t end = p->slice_size - begin < 1024 ? p->slice_size : begin + 1024;
        for (size_t i = begin; i < end; ++i) {
            size_t output_index = item * p->slice_size + i;
            size_t input_index = source + trillim_offset(i, p->slice_rank, p->slice_shape, p->slice_strides);
            if (trillim_itemsize(p->dtype) == 0) {
                task->scratch[output_index] = active ?
                    trillim_read_float(task->inputs[0], input_index, p->dtype, p->storage_rows, p->storage_columns) : p->fill_float;
            } else {
                TrillimValue value;
                if (active) {
                    value = trillim_load(task->inputs[0], input_index, p->dtype);
                } else if (trillim_float(p->dtype)) {
                    value.f = p->fill_float;
                } else {
                    value.u = p->fill_bits;
                }
                trillim_store(task->output, output_index, p->dtype, value);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    return trillim_index_requirements(parameters, inputs, outputs, scratch, false, false);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_gather(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    TrillimIndexTask task = {outputs[0], inputs, (float *)scratch, p, context, false};
    TrillimStatus status;
    if (!false) {
        status = context->thread_pool->run(context->thread_pool->context, p->items, trillim_gather_task, &task);
    } else {
        if (p->slice_size != 0) {
            for (size_t begin = 0; begin < p->count; begin += 1024) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
                for (size_t i = begin; i < end; ++i) {
                    size_t source = trillim_offset(i / p->slice_size, p->coordinates, p->indexed_shape, p->indexed_strides) +
                        trillim_offset(i % p->slice_size, p->slice_rank, p->slice_shape, p->slice_strides);
                    TrillimValue value = trillim_load(inputs[0], source, p->dtype);
                    trillim_store(outputs[0], i, p->dtype, value);
                }
            }
        }
        size_t table_capacity = 0;
        size_t *table = (size_t *)scratch;
        if (p->mode == 0 && p->items != 0) {
            if (!trillim_index_table_size(p->items, &table_capacity)) {
                return TRILLIM_ERROR;
            }
            for (size_t begin = 0; begin < table_capacity; begin += 128) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t count = table_capacity - begin < 128 ? table_capacity - begin : 128;
                memset(table + begin, 255, count * sizeof(size_t));
            }
        }
        for (size_t item = 0; item < p->items; ++item) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t source, destination, key;
            bool active;
            status = trillim_coordinates(&task, item, &source, &destination, &key, &active);
            if (status != TRILLIM_OK) {
                return status;
            }
            if (!active) {
                continue;
            }
            if (p->mode == 0) {
                if (key == SIZE_MAX) {
                    return TRILLIM_ERROR;
                }
                size_t slot = (key * (size_t)UINT64_C(0x9e3779b97f4a7c15)) & (table_capacity - 1);
                size_t probes = 0;
                while (table[slot] != SIZE_MAX) {
                    if ((probes++ & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    if (table[slot] == key) {
                        return TRILLIM_ERROR;
                    }
                    slot = (slot + 1) & (table_capacity - 1);
                }
                table[slot] = key;
            }
            for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = p->slice_size - begin < 1024 ? p->slice_size : begin + 1024;
                for (size_t i = begin; i < end; ++i) {
                    size_t index = trillim_offset(item * p->slice_size + i, p->values_rank, p->values_shape, p->values_strides);
                    TrillimValue value = trillim_load(inputs[2], index, p->dtype);
                    if (p->mode != 0) {
                        TrillimValue previous = trillim_load(outputs[0], destination + i, p->dtype);
                        status = trillim_binary_value(TRILLIM_ADD, previous, value, p->dtype, &value);
                        if (status != TRILLIM_OK) {
                            return status;
                        }
                    }
                    trillim_store(outputs[0], destination + i, p->dtype, value);
                }
            }
        }
        status = TRILLIM_OK;
    }
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->dtype) != 0) {
        return status;
    }
    return trillim_encode_quant((const float *)scratch, p->output_rows, p->output_columns, p->dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_gather_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _gather as bind\n"
           "physical_inputs = ('x', 'indices')\n"
           "def pattern(x: Tensor, indices: Tensor, *, fill = 0) -> tuple[Tensor, ...]:\n"
           "    return (Gather(x, indices=indices, fill=fill),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gather_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimIndexParameters, count), NULL},
        {"items", "size_t", offsetof(TrillimIndexParameters, items), NULL},
        {"coordinates", "size_t", offsetof(TrillimIndexParameters, coordinates), NULL},
        {"index_rank", "size_t", offsetof(TrillimIndexParameters, index_rank), NULL},
        {"slice_rank", "size_t", offsetof(TrillimIndexParameters, slice_rank), NULL},
        {"slice_size", "size_t", offsetof(TrillimIndexParameters, slice_size), NULL},
        {"index_shape", "size_t", offsetof(TrillimIndexParameters, index_shape), "index_rank"},
        {"index_strides", "size_t", offsetof(TrillimIndexParameters, index_strides), "index_rank"},
        {"mask_strides", "size_t", offsetof(TrillimIndexParameters, mask_strides), "index_rank"},
        {"coordinate_stride", "size_t", offsetof(TrillimIndexParameters, coordinate_stride), NULL},
        {"indexed_shape", "size_t", offsetof(TrillimIndexParameters, indexed_shape), "coordinates"},
        {"indexed_strides", "size_t", offsetof(TrillimIndexParameters, indexed_strides), "coordinates"},
        {"output_indexed_strides", "size_t", offsetof(TrillimIndexParameters, output_indexed_strides), "coordinates"},
        {"slice_shape", "size_t", offsetof(TrillimIndexParameters, slice_shape), "slice_rank"},
        {"slice_strides", "size_t", offsetof(TrillimIndexParameters, slice_strides), "slice_rank"},
        {"values_rank", "size_t", offsetof(TrillimIndexParameters, values_rank), NULL},
        {"values_shape", "size_t", offsetof(TrillimIndexParameters, values_shape), "values_rank"},
        {"values_strides", "size_t", offsetof(TrillimIndexParameters, values_strides), "values_rank"},
        {"storage_rows", "size_t", offsetof(TrillimIndexParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimIndexParameters, storage_columns), NULL},
        {"output_rows", "size_t", offsetof(TrillimIndexParameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(TrillimIndexParameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(TrillimIndexParameters, dtype), NULL},
        {"masked", "uint32_t", offsetof(TrillimIndexParameters, masked), NULL},
        {"mode", "uint32_t", offsetof(TrillimIndexParameters, mode), NULL},
        {"fill_bits", "uint64_t", offsetof(TrillimIndexParameters, fill_bits), NULL},
        {"fill_float", "float", offsetof(TrillimIndexParameters, fill_float), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimIndexParameters), TRILLIM_ALIGNOF(TrillimIndexParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(size_t)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gather_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
