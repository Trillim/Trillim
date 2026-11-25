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
    size_t count;
    size_t rank;
    const size_t *shape;
    const size_t *strides;
    size_t source_offset;
    size_t source_rows;
    size_t source_columns;
    size_t output_rows;
    size_t output_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimCastParameters;

typedef struct {
    void *output;
    const void *input;
    float *scratch;
    const TrillimCastParameters *parameters;
    const TrillimKernelContext *context;
} TrillimCastTask;

static inline TrillimStatus trillim_convert(TrillimValue input, uint32_t from, uint32_t to, TrillimValue *output) {
    if (trillim_float(to)) {
        output->f = trillim_float(from) ? input.f :
            (trillim_unsigned(from) ? (float)input.u : (float)input.i);
    } else if (to == TRILLIM_BOOL) {
        output->u = trillim_float(from) ? input.f != 0.0f : input.u != 0;
    } else if (trillim_float(from)) {
        if (!isfinite(input.f)) {
            return TRILLIM_ERROR;
        }
        size_t bits = trillim_itemsize(to) * 8;
        long double modulus = ldexpl(1.0L, (int)bits);
        long double value = fmodl(truncl((long double)input.f), modulus);
        if (value < 0.0L) {
            value += modulus;
        }
        output->u = (uint64_t)value;
    } else {
        output->u = input.u;
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_cast_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimCastParameters *p = (const TrillimCastParameters *)parameters;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    size_t span;
    if (p->source_columns == 0 || p->output_columns == 0 ||
        p->source_rows > SIZE_MAX / p->source_columns ||
        p->output_rows > SIZE_MAX / p->output_columns ||
        p->count != p->output_rows * p->output_columns ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        span > p->source_rows * p->source_columns ||
        p->source_offset > p->source_rows * p->source_columns - span ||
        !trillim_packed_payload_size(p->source_rows, p->source_columns, p->input_dtype, &inputs[0]) ||
        !trillim_packed_payload_size(p->output_rows, p->output_columns, p->output_dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (trillim_itemsize(p->output_dtype) == 0) {
        if (p->count > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = p->count * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline TrillimStatus trillim_cast_task(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimCastTask *task = (const TrillimCastTask *)opaque;
    const TrillimCastParameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    for (size_t index = begin; index < end; ++index) {
        size_t source_index = p->source_offset + trillim_offset(index, p->rank, p->shape, p->strides);
        if (trillim_itemsize(p->output_dtype) == 0) {
            float value = trillim_packed_read_float(task->input, source_index, p->input_dtype,
                                                    p->source_rows, p->source_columns);
            if (!isfinite(value)) {
                return TRILLIM_ERROR;
            }
            task->scratch[index] = value;
        } else {
            TrillimValue input;
            uint32_t input_dtype = p->input_dtype;
            if (trillim_itemsize(input_dtype) == 0) {
                input.f = trillim_packed_read_float(task->input, source_index, input_dtype, p->source_rows, p->source_columns);
                input_dtype = TRILLIM_FP32;
            } else {
                input = trillim_load(task->input, source_index, input_dtype);
            }
            TrillimValue output;
            TrillimStatus status = trillim_convert(input, input_dtype, p->output_dtype, &output);
            if (status != TRILLIM_OK) {
                return status;
            }
            trillim_store(task->output, index, p->output_dtype, output);
        }
    }
    return TRILLIM_OK;
}

enum { LAYOUT_ROWS = 8, LAYOUT_COLUMNS = 32, LAYOUT_CONTIGUOUS = 32768 };

typedef struct {
    TrillimCastTask task;
    size_t rows, columns, stripe;
} TrillimLayoutTask;

static bool dense_layout(const TrillimCastParameters *p, size_t *rows, size_t *columns) {
    size_t span;
    if (trillim_contiguous(p->rank, p->shape, p->strides) &&
        trillim_span(p->rank, p->shape, p->strides, &span) && span == p->count) {
        *rows = 1;
        *columns = p->count;
        return true;
    }
    if (p->rank < 2) {
        return false;
    }
    /* Flatten the dense leading axes into the rows of a transposed matrix. */
    size_t outer_rank = p->rank - 1;
    *columns = p->shape[outer_rank];
    if (*columns < 2 || p->count % *columns != 0 ||
        !trillim_contiguous(outer_rank, p->shape, p->strides) ||
        !trillim_span(outer_rank, p->shape, p->strides, &span) ||
        span != p->count / *columns || p->strides[outer_rank] != span) {
        return false;
    }
    *rows = span;
    return true;
}

static inline void cast_dense_run(uint8_t *output, const uint8_t *input, size_t count, bool narrow) {
    if (narrow) {
        for (size_t i = 0; i < count; ++i) {
            float value;
            memcpy(&value, input + i * sizeof(value), sizeof(value));
            uint16_t bits = trillim_to_bf16(value);
            memcpy(output + i * sizeof(bits), &bits, sizeof(bits));
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            memcpy(&bits, input + i * sizeof(bits), sizeof(bits));
            uint32_t value = (uint32_t)bits << 16;
            memcpy(output + i * sizeof(value), &value, sizeof(value));
        }
    }
}

static TrillimStatus cast_dense(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimLayoutTask *layout = opaque;
    const TrillimCastTask *task = &layout->task;
    const TrillimCastParameters *p = task->parameters;
    size_t in_bytes = trillim_itemsize(p->input_dtype);
    size_t out_bytes = trillim_itemsize(p->output_dtype);
    const uint8_t *input = (const uint8_t *)task->input + p->source_offset * in_bytes;
    uint8_t *output = task->output;
    size_t first = tile * LAYOUT_CONTIGUOUS;
    size_t remaining = p->count - first;
    size_t length = remaining < LAYOUT_CONTIGUOUS ? remaining : LAYOUT_CONTIGUOUS;
    for (size_t offset = 0; offset < length;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t begin = first + offset;
        size_t count = length - offset < 4096 ? length - offset : 4096;
        cast_dense_run(output + begin * out_bytes, input + begin * in_bytes, count,
                       p->input_dtype == TRILLIM_FP32);
        offset += count;
    }
    return TRILLIM_OK;
}

static TrillimStatus cast_transpose(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimLayoutTask *layout = opaque;
    const TrillimCastTask *task = &layout->task;
    const TrillimCastParameters *p = task->parameters;
    size_t rows = layout->rows, columns = layout->columns;
    size_t in_bytes = trillim_itemsize(p->input_dtype);
    const uint8_t *input = (const uint8_t *)task->input + p->source_offset * in_bytes;
    uint8_t *output = task->output;
    bool narrow = p->input_dtype == TRILLIM_FP32;
    size_t first = tile * layout->stripe;
    size_t row_end = rows - first < layout->stripe ? rows : first + layout->stripe;
    for (size_t row = first; row < row_end;) {
        size_t height = row_end - row < LAYOUT_ROWS ? row_end - row : LAYOUT_ROWS;
        for (size_t column = 0; column < columns;) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t width = columns - column < LAYOUT_COLUMNS ? columns - column : LAYOUT_COLUMNS;
            if (narrow) {
                uint16_t values[LAYOUT_COLUMNS][LAYOUT_ROWS];
                for (size_t c = 0; c < width; ++c) {
                    const uint8_t *source = input + ((column + c) * rows + row) * sizeof(float);
                    cast_dense_run((uint8_t *)values[c], source, height, true);
                }
                for (size_t r = 0; r < height; ++r) {
                    uint8_t *target = output + ((row + r) * columns + column) * sizeof(uint16_t);
                    for (size_t c = 0; c < width; ++c) {
                        memcpy(target + c * sizeof(uint16_t), &values[c][r], sizeof(uint16_t));
                    }
                }
            } else {
                uint32_t values[LAYOUT_COLUMNS][LAYOUT_ROWS];
                for (size_t c = 0; c < width; ++c) {
                    const uint8_t *source = input + ((column + c) * rows + row) * sizeof(uint16_t);
                    cast_dense_run((uint8_t *)values[c], source, height, false);
                }
                for (size_t r = 0; r < height; ++r) {
                    uint8_t *target = output + ((row + r) * columns + column) * sizeof(uint32_t);
                    for (size_t c = 0; c < width; ++c) {
                        memcpy(target + c * sizeof(uint32_t), &values[c][r], sizeof(uint32_t));
                    }
                }
            }
            column += width;
        }
        row += height;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_cast(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimCastParameters *p = (const TrillimCastParameters *)parameters;
    TrillimCastTask task = {outputs[0], inputs[0], (float *)scratch, p, context};
    size_t rows, columns;
    if (p->count >= 16 && ((p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_BF16) ||
         (p->input_dtype == TRILLIM_BF16 && p->output_dtype == TRILLIM_FP32)) &&
        dense_layout(p, &rows, &columns)) {
        /* Keep narrow transposes from dispatching a task per small tile. */
        size_t row_tiles = 1 + (4096 / LAYOUT_ROWS - 1) / columns;
        size_t stripe = p->count <= 4096 ? rows : LAYOUT_ROWS * row_tiles;
        size_t work = rows == 1 ? p->count : rows;
        size_t step = rows == 1 ? LAYOUT_CONTIGUOUS : stripe;
        size_t tiles = work / step + (work % step != 0);
        TrillimLayoutTask layout = {task, rows, columns, stripe};
        TrillimTask operation = rows == 1 ? cast_dense : cast_transpose;
        if (tiles == 1) {
            return operation(&layout, 0, 0);
        }
        const TrillimThreadPool *pool = context->thread_pool;
        return pool->run(pool->context, tiles, operation, &layout);
    }
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_cast_task, &task);
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->output_dtype) != 0) {
        return status;
    }
    return trillim_packed_encode_quant((const float *)scratch, p->output_rows, p->output_columns,
        p->output_dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_cast_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _cast as bind\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_cast_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimCastParameters, count), NULL},
        {"rank", "size_t", offsetof(TrillimCastParameters, rank), NULL},
        {"shape", "size_t", offsetof(TrillimCastParameters, shape), "rank"},
        {"strides", "size_t", offsetof(TrillimCastParameters, strides), "rank"},
        {"source_offset", "size_t", offsetof(TrillimCastParameters, source_offset), NULL},
        {"source_rows", "size_t", offsetof(TrillimCastParameters, source_rows), NULL},
        {"source_columns", "size_t", offsetof(TrillimCastParameters, source_columns), NULL},
        {"output_rows", "size_t", offsetof(TrillimCastParameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(TrillimCastParameters, output_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimCastParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimCastParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimCastParameters), TRILLIM_ALIGNOF(TrillimCastParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1, trillim_cast_requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_cast_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
