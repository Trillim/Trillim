#include "common.h"
#include <fenv.h>
#include <math.h>
#include <stdlib.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} Parameters;

enum { ACTIVATION_MAX_WIDTH = 65536, ACTIVATION_Q8 = 32, ACTIVATION_Q8_K = 256 };

typedef struct {
    float *scales;
    int8_t *codes;
    size_t groups;
} TrillimScalarActivation;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *weight;
    const Parameters *p;
    size_t columns;
    const TrillimKernelContext *context;
    const TrillimScalarActivation *activation;
} Task;

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

static size_t activation_group_width(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5:
    case TRILLIM_GRP_INT8:
        return ACTIVATION_Q8;
    case TRILLIM_DUBGRP_INT6:
        return ACTIVATION_Q8_K;
    default:
        return 0;
    }
}

static int32_t activation_nearest(float value) {
    /* Adding 1.5 * 2^23 rounds the low bits to integers in nearest-even mode. */
    float rounded = value + 12582912.0f;
    uint32_t bits;
    memcpy(&bits, &rounded, sizeof(bits));
    return (int32_t)(bits & UINT32_C(0x007fffff)) - INT32_C(0x00400000);
}

static bool activation_quantize_group(
    const void *input, uint32_t input_dtype, size_t start, size_t group, float *scale, int8_t *codes
) {
    float values[ACTIVATION_Q8_K];
    float peak = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        float value = trillim_load(input, start + i, input_dtype).f;
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
        memset(codes, 0, group * sizeof(*codes));
        return true;
    }
    float inverse;
    if (group == ACTIVATION_Q8_K) {
        inverse = -127.0f / peak;
        if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
            return false;
        }
        *scale = 1.0f / inverse;
    } else {
        float step = fabsf(peak) / 127.0f;
        if (step > 65504.0f) {
            return false;
        }
        *scale = trillim_fp16(trillim_to_fp16(step));
        if (*scale == 0.0f) {
            return false;
        }
        inverse = 1.0f / step;
    }
    for (size_t i = 0; i < group; ++i) {
        codes[i] = (int8_t)activation_nearest(values[i] * inverse);
    }
    return true;
}

static TrillimStatus activation_prepare(
    TrillimScalarActivation *result, const void *input, size_t width,
    uint32_t input_dtype, uint32_t weight_dtype, const TrillimKernelContext *context
) {
    size_t group = activation_group_width(weight_dtype);
    if (!group || width < group || width > ACTIVATION_MAX_WIDTH || width % group ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    float *scales = malloc(groups * sizeof(*scales) + width);
    if (scales == NULL) {
        return TRILLIM_OK;
    }
    int8_t *codes = (int8_t *)(scales + groups);
    for (size_t start = 0; start < width; start += group) {
        if (start % 1024 == 0 && context->is_cancelled(context->cancellation_context)) {
            free(scales);
            return TRILLIM_CANCELLED;
        }
        if (!activation_quantize_group(input, input_dtype, start, group, scales + start / group, codes + start)) {
            free(scales);
            return TRILLIM_OK;
        }
    }
    *result = (TrillimScalarActivation){scales, codes, groups};
    return TRILLIM_OK;
}

static void activation_dot_q8(
    const uint8_t *low, const uint8_t *high, const int8_t *codes, uint32_t dtype, int32_t sums[4]
) {
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *weight = (const int8_t *)low;
        for (size_t i = 0; i < ACTIVATION_Q8; ++i) {
            sums[(i % 16) / 4] += weight[i] * codes[i];
        }
        return;
    }
    uint32_t fifth = 0;
    int32_t center = 8;
    if (dtype == TRILLIM_GRP_INT5) {
        fifth = (uint32_t)high[0] | (uint32_t)high[1] << 8 |
            (uint32_t)high[2] << 16 | (uint32_t)high[3] << 24;
        center = 16;
    }
    for (size_t i = 0; i < 16; ++i) {
        int32_t lo = (int32_t)((low[i] & 15) | (((fifth >> i) & 1) << 4)) - center;
        int32_t hi = (int32_t)((low[i] >> 4) | (((fifth >> (i + 16)) & 1) << 4)) - center;
        sums[i / 4] += lo * codes[i] + hi * codes[i + 16];
    }
}

static int32_t activation_dot_q6(
    const uint8_t *low, const uint8_t *high, const int8_t *subscales, const int8_t *codes
) {
    int32_t total = 0;
    for (size_t region = 0; region < 8; ++region) {
        size_t half = region / 4;
        size_t lane = region % 4;
        const uint8_t *l = low + half * 64 + lane / 2 * 32;
        const uint8_t *h = high + half * 32;
        for (size_t sub = 0; sub < 2; ++sub) {
            int32_t sum = 0;
            size_t first = sub * 16;
            for (size_t i = first; i < first + 16; ++i) {
                int32_t lo = (l[i] >> ((lane % 2) * 4)) & 15;
                int32_t hi = (h[i] >> (lane * 2)) & 3;
                int32_t w = (lo | hi << 4) - 32;
                sum += w * codes[region * 32 + i];
            }
            total += sum * subscales[region * 2 + sub];
        }
    }
    return total;
}

static float activation_sum4(const float sums[4]) {
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

static TrillimStatus activation_dot(
    const TrillimScalarActivation *activation, const void *weight, uint32_t dtype, size_t outputs,
    size_t column, const TrillimKernelContext *context, float *result, bool *used
) {
    size_t groups = activation->groups;
    size_t total = groups * outputs;
    bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    size_t low_size = q6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *bytes = weight;
    const uint8_t *low = bytes + column * groups * low_size;
    const uint8_t *scales = bytes + total * low_size + column * groups * 2;
    const uint8_t *high = bytes + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (q6 ? total * 64 + column * groups * 16 : 0));
    high += column * groups * (q6 ? 64 : (dtype == TRILLIM_GRP_INT5 ? 4 : 0));
    float sums[2][4] = {{0}};
    float output = 0.0f, tail_scale = 0.0f;
    int32_t tail = 0;
    for (size_t g = 0; g < groups; ++g) {
        if ((q6 || g % 8 == 0) && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float weight_scale = trillim_bf16(bits);
        if (!isfinite(weight_scale) || fabsf(weight_scale) > 0x1p40f) {
            *used = false;
            return TRILLIM_OK;
        }
        float scale = activation->scales[g] * weight_scale;
        if (q6) {
            int32_t dot = activation_dot_q6(low + g * 128, high + g * 64, subscales + g * 16,
                activation->codes + g * ACTIVATION_Q8_K);
            output = multiply_add(scale, (float)dot, output);
        } else {
            int32_t dot[4] = {0};
            const uint8_t *upper = high + (dtype == TRILLIM_GRP_INT5 ? g * 4 : 0);
            activation_dot_q8(low + g * low_size, upper, activation->codes + g * ACTIVATION_Q8, dtype, dot);
            if ((groups & 1) && g == groups - 1) {
                tail = dot[0] + dot[1] + dot[2] + dot[3];
                tail_scale = scale;
            } else {
                for (size_t lane = 0; lane < 4; ++lane) {
                    sums[g & 1][lane] = multiply_add((float)dot[lane], scale, sums[g & 1][lane]);
                }
            }
        }
    }
    if (!q6) {
        output = activation_sum4(sums[0]) + activation_sum4(sums[1]);
        if (groups & 1) {
            output = multiply_add(tail_scale, (float)tail, output);
        }
    }
    *result = output;
    *used = true;
    return TRILLIM_OK;
}

static void decode_block(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
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

#define TRILLIM_PACKED_PROJECT(NAME, LOAD) \
static TrillimStatus project_##NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const Parameters *p = task->p; \
    size_t tiles = p->outputs / task->columns + (p->outputs % task->columns != 0); \
    size_t row = tile / tiles * 4; \
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4; \
    size_t first = tile % tiles * task->columns; \
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns; \
    size_t group = group_width(p->weight_dtype); \
    for (size_t column = first; column < end; ++column) { \
        float sums[4] = {0}; \
        bool quantized = false; \
        if (task->activation != NULL) { \
            TrillimStatus status = activation_dot(task->activation, task->weight, \
                p->weight_dtype, p->outputs, column, task->context, &sums[0], &quantized); \
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
            for (size_t i = 0; i < count; ++i) { \
                float weight = load_fp32(decoded, i); \
                sums[0] = multiply_add(LOAD(task->input, row * p->width + k + i), weight, sums[0]); \
                if (rows > 1) { \
                    sums[1] = multiply_add(LOAD(task->input, (row + 1) * p->width + k + i), weight, sums[1]); \
                } \
                if (rows > 2) { \
                    sums[2] = multiply_add(LOAD(task->input, (row + 2) * p->width + k + i), weight, sums[2]); \
                } \
                if (rows > 3) { \
                    sums[3] = multiply_add(LOAD(task->input, (row + 3) * p->width + k + i), weight, sums[3]); \
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

TRILLIM_KERNEL TrillimStatus trillim_scalar_block_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
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
    size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    TrillimTask project;
    switch (p->input_dtype) {
    case TRILLIM_FP32: project = project_fp32; break;
    case TRILLIM_BF16: project = project_bf16; break;
    case TRILLIM_FP16: project = project_fp16; break;
    default: return TRILLIM_ERROR;
    }
    TrillimScalarActivation activation = {0};
    if (p->tokens == 1 && p->outputs >= 16 && p->output_dtype == TRILLIM_FP32) {
        TrillimStatus status = activation_prepare(&activation, inputs[0], p->width,
            p->input_dtype, p->weight_dtype, context);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, columns, context,
        activation.scales == NULL ? NULL : &activation};
    TrillimStatus status = rows * tiles == 1 ? project(&task, 0, 0)
        : context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
    free(activation.scales);
    return status;
}

TRILLIM_KERNEL const char *trillim_scalar_block_linear_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_block_linear_parameters(void) {
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
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_block_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
