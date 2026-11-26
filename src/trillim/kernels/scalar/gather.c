#include "common.h"
#include <math.h>
#include <string.h>

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

static float trillim_packed_max_abs16(const float *input) {
    float maximum = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        maximum = fmaxf(maximum, fabsf(input[i]));
    }
    return maximum;
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

static TrillimStatus trillim_packed_encode_fp8(void *opaque, size_t block, size_t worker) {
    (void)worker;
    const TrillimPackedEncodeTask *task = opaque;
    const TrillimKernelContext *context = task->context;
    const size_t rows = task->rows, columns = task->columns;
    const size_t row_start = block / task->groups * 128;
    const size_t column_start = block % task->groups * 128;
    const size_t row_end = rows - row_start < 128 ? rows : row_start + 128;
    const size_t column_end = columns - column_start < 128 ? columns : column_start + 128;
    float maximum = 0.0f;
    for (size_t row = row_start; row < row_end; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t column = column_start; column < column_end; ++column) {
            maximum = fmaxf(maximum, fabsf(task->source[row * columns + column]));
        }
    }
    float scale = trillim_fp8_e4m3_scale(maximum);
    memcpy(task->output + rows * columns + block * sizeof(scale), &scale, sizeof(scale));
    for (size_t row = row_start; row < row_end; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t column = column_start; column < column_end; ++column) {
            size_t index = row * columns + column;
            task->output[index] = trillim_to_fp8_e4m3(task->source[index] / scale);
        }
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
    if (dtype == TRILLIM_FP8_E4M3) {
        task.groups = columns / 128 + (columns % 128 != 0);
        size_t row_groups = rows / 128 + (rows % 128 != 0);
        tiles = row_groups * task.groups;
        operation = trillim_packed_encode_fp8;
    } else if (dtype != TRILLIM_ROW_INT8) {
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
        !trillim_packed_payload_size(p->storage_rows, p->storage_columns, p->dtype, &inputs[0]) ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->dtype, &outputs[0])) {
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
        if (((const uint8_t *)task->inputs[2])[index] == 0) {
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
    if (active && (p->dtype == TRILLIM_FP16 || p->dtype == TRILLIM_BF16)) {
        size_t stride = 1;
        if (!trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides)) {
            size_t axes = 0;
            for (size_t axis = 0; axis < p->slice_rank; ++axis) {
                if (p->slice_shape[axis] > 1) {
                    stride = p->slice_strides[axis];
                    ++axes;
                }
            }
            if (axes != 1) {
                goto generic_gather;
            }
        }
        uint16_t infinity = p->dtype == TRILLIM_FP16 ? 0x7c00 : 0x7f80;
        uint16_t quiet = p->dtype == TRILLIM_FP16 ? 0x0200 : 0x0040;
        for (size_t begin = 0; begin < p->slice_size; begin += 1024) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->slice_size - begin < 1024 ? p->slice_size - begin : 1024;
            const uint8_t *input = (const uint8_t *)task->inputs[0] + (source + begin * stride) * 2;
            uint8_t *output = (uint8_t *)task->output + (item * p->slice_size + begin) * 2;
            for (size_t i = 0; i < count; ++i) {
                uint16_t bits;
                memcpy(&bits, input + i * stride * 2, sizeof(bits));
                /* Native gather preserves payloads and quiets signaling NaNs. */
                bits |= (bits & 0x7fff) > infinity ? quiet : 0;
                memcpy(output + i * 2, &bits, sizeof(bits));
            }
        }
        return TRILLIM_OK;
    }
generic_gather:;
    size_t bytes = trillim_itemsize(p->dtype);
    if (active && bytes && p->dtype > TRILLIM_BOOL &&
        p->dtype != TRILLIM_FP16 && p->dtype != TRILLIM_BF16) {
        size_t width = 0, stride = 0;
        if (trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides)) {
            width = p->slice_size;
        } else if (p->slice_rank == 2 && p->slice_strides[1] == 1 &&
                   p->slice_shape[1] >= 16 && p->slice_size % p->slice_shape[1] == 0 &&
                   p->slice_size / p->slice_shape[1] == p->slice_shape[0]) {
            width = p->slice_shape[1];
            stride = p->slice_strides[0];
        }
        if (width) {
            for (size_t offset = 0; offset < p->slice_size; offset += width, source += stride) {
                for (size_t begin = 0; begin < width; begin += 1024) {
                    if (task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    size_t count = width - begin < 1024 ? width - begin : 1024;
                    uint8_t *output = (uint8_t *)task->output +
                        (item * p->slice_size + offset + begin) * bytes;
                    const uint8_t *input = (const uint8_t *)task->inputs[0] + (source + begin) * bytes;
                    memcpy(output, input, count * bytes);
                }
            }
            return TRILLIM_OK;
        }
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
                float value = active ?
                    trillim_packed_read_float(task->inputs[0], input_index, p->dtype, p->storage_rows, p->storage_columns) : p->fill_float;
                if (!isfinite(value)) {
                    return TRILLIM_ERROR;
                }
                task->scratch[output_index] = value;
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

static size_t gather_batch_size(const TrillimIndexParameters *p) {
    return p->slice_size && p->slice_size < 1024 ? 1024 / p->slice_size : 1;
}

static TrillimStatus gather_batch(void *opaque, size_t batch, size_t worker) {
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    size_t size = gather_batch_size(p), first = batch * size;
    size_t end = p->items - first < size ? p->items : first + size;
    for (size_t item = first; item < end; ++item) {
        TrillimStatus status = trillim_gather_task(opaque, item, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static bool trillim_mxfp4_copyable(const TrillimIndexParameters *p) {
    if (p->dtype != TRILLIM_MXFP4 || p->masked || p->coordinates != 1 ||
        !p->storage_columns || p->storage_columns % 32 || p->output_columns % 32 ||
        p->slice_size != p->storage_columns || p->indexed_shape[0] != p->storage_rows ||
        p->indexed_strides[0] != p->storage_columns ||
        !trillim_contiguous(p->index_rank, p->index_shape, p->index_strides) ||
        !trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides) ||
        p->items > SIZE_MAX / p->slice_size || p->items * p->slice_size != p->count ||
        (p->output_columns && p->output_rows > SIZE_MAX / p->output_columns) ||
        p->output_rows * p->output_columns != p->count) {
        return false;
    }
    size_t items = 1;
    for (size_t axis = 0; axis < p->index_rank; ++axis) {
        if (p->index_shape[axis] && items > SIZE_MAX / p->index_shape[axis]) {
            return false;
        }
        items *= p->index_shape[axis];
    }
    return items == p->items;
}

enum { MXFP4_GROUPS_PER_TILE = 32768 };

static TrillimStatus trillim_mxfp4_validate_tile(
    const uint8_t *codes, const uint8_t *scales, size_t count, const TrillimKernelContext *context
) {
    unsigned exceptional = 0;
    for (size_t i = 0; i < count; ++i) {
        exceptional |= scales[i] >= 253;
    }
    if (exceptional) {
        for (size_t i = 0; i < count; ++i) {
            if ((i & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint8_t exponent = scales[i];
            if (exponent == 255) {
                return TRILLIM_ERROR;
            }
            if (exponent < 253) {
                continue;
            }
            uint8_t limit = exponent == 253 ? 6 : 4;
            const uint8_t *block = codes + i * 16;
            for (size_t j = 0; j < 16; ++j) {
                if ((block[j] & 7) >= limit || ((block[j] >> 4) & 7) >= limit) {
                    return TRILLIM_ERROR;
                }
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_mxfp4_copy_slice(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    const uint8_t *input = task->inputs[0];
    const uint8_t *indices = task->inputs[1];
    uint8_t *output = task->output;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t groups = p->storage_columns / 32;
    size_t tiles = groups / MXFP4_GROUPS_PER_TILE + (groups % MXFP4_GROUPS_PER_TILE != 0);
    size_t item = tile / tiles;
    size_t first = tile % tiles * MXFP4_GROUPS_PER_TILE;
    size_t count = groups - first < MXFP4_GROUPS_PER_TILE ? groups - first : MXFP4_GROUPS_PER_TILE;
    int64_t selected;
    memcpy(&selected, indices + item * sizeof(selected), sizeof(selected));
    size_t source = (size_t)selected * groups + first;
    size_t destination = item * groups + first;
    const uint8_t *codes = input + source * 16;
    const uint8_t *scales = input + p->storage_rows * groups * 16 + source;
    TrillimStatus status = trillim_mxfp4_validate_tile(codes, scales, count, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    memcpy(output + destination * 16, codes, count * 16);
    memcpy(output + p->items * groups * 16 + destination, scales, count);
    return TRILLIM_OK;
}

static TrillimStatus trillim_mxfp4_copy_repeated(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    const uint8_t *input = task->inputs[0];
    const uint8_t *indices = task->inputs[1];
    uint8_t *output = task->output;
    size_t groups = p->storage_columns / 32;
    size_t tiles = groups / MXFP4_GROUPS_PER_TILE + (groups % MXFP4_GROUPS_PER_TILE != 0);
    size_t row = tile / tiles;
    size_t first = tile % tiles * MXFP4_GROUPS_PER_TILE;
    size_t count = groups - first < MXFP4_GROUPS_PER_TILE ? groups - first : MXFP4_GROUPS_PER_TILE;
    size_t source = row * groups + first;
    const uint8_t *codes = input + source * 16;
    const uint8_t *scales = input + p->storage_rows * groups * 16 + source;
    uint8_t *output_scales = output + p->items * groups * 16;

    bool validated = false;
    for (size_t item = 0; item < p->items; ++item) {
        if ((item & 255) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t selected;
        memcpy(&selected, indices + item * sizeof(selected), sizeof(selected));
        if ((size_t)selected != row) {
            continue;
        }
        if (!validated) {
            TrillimStatus status = trillim_mxfp4_validate_tile(codes, scales, count, context);
            if (status != TRILLIM_OK) {
                return status;
            }
            validated = true;
        }
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t destination = item * groups + first;
        memcpy(output + destination * 16, codes, count * 16);
        memcpy(output_scales + destination, scales, count);
    }
    return TRILLIM_OK;
}

static TrillimStatus trillim_mxfp4_gather(TrillimIndexTask *task) {
    const TrillimIndexParameters *p = task->parameters;
    const TrillimThreadPool *pool = task->context->thread_pool;
    size_t groups = p->storage_columns / 32;
    size_t tiles = groups / MXFP4_GROUPS_PER_TILE + (groups % MXFP4_GROUPS_PER_TILE != 0);
    bool reuse_tiles = p->storage_rows <= 64 && p->items >= 4 * p->storage_rows && groups >= 8192;
    uint64_t selected_rows = 0;
    size_t active_rows = 0;
    for (size_t item = 0; item < p->items; ++item) {
        if ((item & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t selected;
        memcpy(&selected, (const uint8_t *)task->inputs[1] + item * sizeof(selected), sizeof(selected));
        if (selected < 0 || (uint64_t)selected >= p->storage_rows) {
            return TRILLIM_ERROR;
        }
        if (reuse_tiles) {
            uint64_t row_bit = UINT64_C(1) << selected;
            active_rows += (selected_rows & row_bit) == 0;
            selected_rows |= row_bit;
        }
    }
    if (reuse_tiles && active_rows * tiles >= pool->worker_count) {
        /* Reuse a cached source tile across outputs while keeping the workers busy. */
        return pool->run(pool->context, p->storage_rows * tiles, trillim_mxfp4_copy_repeated, task);
    }
    size_t tasks = p->items * tiles;
    return tasks == 1 ? trillim_mxfp4_copy_slice(task, 0, 0)
                      : pool->run(pool->context, tasks, trillim_mxfp4_copy_slice, task);
}

enum { TRANSPOSE_ITEMS = 32, TRANSPOSE_ROWS = 32 };

static bool transpose_gather(const TrillimIndexParameters *p) {
    return !p->masked && p->dtype == TRILLIM_FP32 && p->coordinates == 1 && p->index_rank == 1 &&
        p->slice_rank == 1 && p->index_shape[0] == p->items && p->index_strides[0] == 1 &&
        p->indexed_strides[0] == 1 && p->slice_strides[0] > 1 &&
        p->slice_size == p->slice_shape[0] && p->items >= TRANSPOSE_ITEMS &&
        p->slice_size >= TRANSPOSE_ROWS;
}

static TrillimStatus gather_transpose_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    const uint8_t *input = task->inputs[0], *indices = task->inputs[1];
    uint8_t *output = task->output;
    size_t begin = index * TRANSPOSE_ITEMS;
    size_t width = p->items - begin < TRANSPOSE_ITEMS ? p->items - begin : TRANSPOSE_ITEMS;
    size_t columns[TRANSPOSE_ITEMS];
    bool contiguous = true;
    for (size_t i = 0; i < width; ++i) {
        int64_t column;
        memcpy(&column, indices + (begin + i) * sizeof(column), sizeof(column));
        if (column < 0 || (uint64_t)column >= p->indexed_shape[0]) {
            return TRILLIM_ERROR;
        }
        columns[i] = (size_t)column;
        contiguous &= columns[i] >= columns[0] && columns[i] - columns[0] == i;
    }
    for (size_t row = 0; row < p->slice_size; row += TRANSPOSE_ROWS) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t height = p->slice_size - row < TRANSPOSE_ROWS ? p->slice_size - row : TRANSPOSE_ROWS;
        /* Read each slice row together before writing the gathered columns. */
        uint32_t tile[TRANSPOSE_ROWS][TRANSPOSE_ITEMS];
        for (size_t j = 0; j < height; ++j) {
            const uint8_t *src = input + (row + j) * p->slice_strides[0] * sizeof(uint32_t);
            if (contiguous) {
                memcpy(tile[j], src + columns[0] * sizeof(uint32_t), width * sizeof(uint32_t));
                continue;
            }
            for (size_t i = 0; i < width; ++i) {
                memcpy(&tile[j][i], src + columns[i] * sizeof(uint32_t), sizeof(uint32_t));
            }
        }
        for (size_t i = 0; i < width; ++i) {
            uint8_t *dst = output + ((begin + i) * p->slice_size + row) * sizeof(uint32_t);
            for (size_t j = 0; j < height; ++j) {
                memcpy(dst + j * sizeof(uint32_t), &tile[j][i], sizeof(uint32_t));
            }
        }
    }
    return TRILLIM_OK;
}

enum { SINGLETON_TILE = 4096, SINGLETON_COPY = 256 };

static bool singleton_gather(const TrillimIndexParameters *p) {
    return !p->masked && p->dtype == TRILLIM_FP32 && p->coordinates == 1 &&
        p->index_rank == 1 && p->index_shape[0] == p->items && p->index_strides[0] == 1 &&
        p->indexed_strides[0] == 1 && p->slice_size == 1 && p->items >= 16;
}

static bool consecutive_indices(const uint8_t *indices, size_t count, size_t bound, size_t *first) {
    int64_t start;
    memcpy(&start, indices, sizeof(start));
    if (start < 0 || (uint64_t)start >= bound || count > bound - (size_t)start ||
        count - 1 > (uint64_t)INT64_MAX - (uint64_t)start) {
        return false;
    }
    for (size_t i = 1; i < count; ++i) {
        int64_t index;
        memcpy(&index, indices + i * sizeof(index), sizeof(index));
        if (index != start + (int64_t)i) {
            return false;
        }
    }
    *first = (size_t)start;
    return true;
}

static bool nonoverlapping_spans(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t first = (uintptr_t)a, second = (uintptr_t)b;
    return first <= second ? second - first >= a_bytes : first - second >= b_bytes;
}

static bool consecutive_slices(const TrillimIndexTask *task, size_t *source) {
    const TrillimIndexParameters *p = task->parameters;
    if (p->masked || p->dtype != TRILLIM_FP32 || p->coordinates != 1 ||
        p->index_rank != 1 || p->index_shape[0] != p->items || p->index_strides[0] != 1 ||
        p->items < 2 || p->items > 32 || p->slice_size < 1024 ||
        p->indexed_strides[0] != p->slice_size ||
        p->items > SIZE_MAX / p->slice_size || p->items * p->slice_size != p->count ||
        p->count > SIZE_MAX / sizeof(float) ||
        !trillim_contiguous(p->slice_rank, p->slice_shape, p->slice_strides) ||
        !nonoverlapping_spans(task->output, p->count * sizeof(float),
                              task->inputs[1], p->items * sizeof(int64_t))) {
        return false;
    }
    size_t start;
    if (!consecutive_indices(task->inputs[1], p->items, p->indexed_shape[0], &start) ||
        start > SIZE_MAX / sizeof(float) / p->slice_size) {
        return false;
    }
    *source = start * p->slice_size;
    const size_t bytes = p->count * sizeof(float);
    return nonoverlapping_spans(task->output, bytes,
        (const uint8_t *)task->inputs[0] + *source * sizeof(float), bytes);
}

static TrillimStatus gather_singletons(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimIndexTask *task = opaque;
    const TrillimIndexParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    const uint8_t *input = task->inputs[0], *indices = task->inputs[1];
    uint8_t *output = task->output;
    const bool independent_indices = p->items <= SIZE_MAX / sizeof(int64_t) &&
        nonoverlapping_spans(output, p->items * sizeof(float), indices, p->items * sizeof(int64_t));
    size_t first = tile * SINGLETON_TILE;
    size_t end = p->items - first < SINGLETON_TILE ? p->items : first + SINGLETON_TILE;
    for (; first < end;) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = end - first < SINGLETON_COPY ? end - first : SINGLETON_COPY;
        size_t width = p->indexed_shape[0];
        size_t selected_start;
        if (independent_indices &&
            consecutive_indices(indices + first * sizeof(int64_t), count, width, &selected_start)) {
            uint8_t *dst = output + first * sizeof(float);
            const uint8_t *src = input + selected_start * sizeof(float);
            size_t bytes = count * sizeof(float);
            if (nonoverlapping_spans(dst, bytes, src, bytes)) {
                memcpy(dst, src, bytes);
                first += count;
                continue;
            }
        }
        for (size_t i = 0; i < count; ++i) {
            int64_t selected;
            memcpy(&selected, indices + (first + i) * sizeof(selected), sizeof(selected));
            if (selected < 0 || (uint64_t)selected >= width) {
                return TRILLIM_ERROR;
            }
            memcpy(output + (first + i) * sizeof(float), input + (size_t)selected * sizeof(float), sizeof(float));
        }
        first += count;
    }
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    TrillimStatus status = trillim_index_requirements(parameters, inputs, outputs, scratch, false, false);
    if (status == TRILLIM_OK && trillim_mxfp4_copyable(parameters)) {
        *scratch = 0;
    }
    return status;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_gather(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimIndexParameters *p = (const TrillimIndexParameters *)parameters;
    TrillimIndexTask task = {outputs[0], inputs, (float *)scratch, p, context};
    if (trillim_mxfp4_copyable(p)) {
        return trillim_mxfp4_gather(&task);
    }
    size_t source;
    if (consecutive_slices(&task, &source)) {
        const uint8_t *input = (const uint8_t *)inputs[0] + source * sizeof(float);
        uint8_t *output = outputs[0];
        for (size_t first = 0; first < p->count;) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->count - first < 4096 ? p->count - first : 4096;
            memcpy(output + first * sizeof(float), input + first * sizeof(float), count * sizeof(float));
            first += count;
        }
        return TRILLIM_OK;
    }
    if (transpose_gather(p)) {
        const TrillimThreadPool *pool = context->thread_pool;
        size_t tasks = p->items / TRANSPOSE_ITEMS + (p->items % TRANSPOSE_ITEMS != 0);
        return pool->run(pool->context, tasks, gather_transpose_task, &task);
    }
    if (singleton_gather(p)) {
        size_t tiles = p->items / SINGLETON_TILE + (p->items % SINGLETON_TILE != 0);
        const TrillimThreadPool *pool = context->thread_pool;
        return tiles == 1 ? gather_singletons(&task, 0, 0) :
            pool->run(pool->context, tiles, gather_singletons, &task);
    }
    size_t size = gather_batch_size(p);
    size_t batches = p->items / size + (p->items % size != 0);
    TrillimStatus status = TRILLIM_OK;
    if (trillim_itemsize(p->dtype) && p->count <= 8192) {
        for (size_t batch = 0; batch < batches && status == TRILLIM_OK; ++batch) {
            status = gather_batch(&task, batch, 0);
        }
    } else if (batches == 1) {
        status = gather_batch(&task, 0, 0);
    } else {
        status = context->thread_pool->run(context->thread_pool->context, batches, gather_batch, &task);
    }
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->dtype) != 0) {
        return status;
    }
    return trillim_packed_encode_quant((const float *)scratch, p->output_rows, p->output_columns, p->dtype, outputs[0], context);
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
