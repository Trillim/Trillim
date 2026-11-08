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
        !trillim_payload_size(p->source_rows, p->source_columns, p->input_dtype, &inputs[0]) ||
        !trillim_payload_size(p->output_rows, p->output_columns, p->output_dtype, &outputs[0])) {
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
            task->scratch[index] = trillim_read_float(task->input, source_index, p->input_dtype, p->source_rows, p->source_columns);
        } else {
            TrillimValue input;
            uint32_t input_dtype = p->input_dtype;
            if (trillim_itemsize(input_dtype) == 0) {
                input.f = trillim_read_float(task->input, source_index, input_dtype, p->source_rows, p->source_columns);
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

TRILLIM_KERNEL TrillimStatus trillim_scalar_permute(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimCastParameters *p = (const TrillimCastParameters *)parameters;
    TrillimCastTask task = {outputs[0], inputs[0], (float *)scratch, p, context};
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context,
        p->count / 1024 + (p->count % 1024 != 0), trillim_cast_task, &task);
    if (status != TRILLIM_OK || p->count == 0 || trillim_itemsize(p->output_dtype) != 0) {
        return status;
    }
    return trillim_encode_quant((const float *)scratch, p->output_rows, p->output_columns,
        p->output_dtype, outputs[0], context);
}

TRILLIM_KERNEL const char *trillim_scalar_permute_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _permute as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...]) -> tuple[Tensor, ...]:\n"
           "    return (Permute(x, axes),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_permute_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_permute_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
