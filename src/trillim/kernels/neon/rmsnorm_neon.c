#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index, vreinterpret_u16_f16(vcvt_f16_f32(value)));
        return;
    }
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

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

static inline float32x4_t load_weight4(const void *data, size_t index, uint32_t dtype, size_t rows, size_t columns) {
    if (trillim_float(dtype)) {
        return load4(data, index, dtype);
    }
    float values[4];
    for (size_t lane = 0; lane < 4; ++lane) {
        values[lane] = trillim_read_float(data, index + lane, dtype, rows, columns);
    }
    return vld1q_f32(values);
}

typedef struct {
    size_t rows;
    size_t width;
    size_t row_stride;
    size_t column_stride;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    float eps;
    float weight_offset;
} TrillimRMSNormParameters;

typedef struct {
    void *output;
    const void *input;
    const void *weight;
    const TrillimRMSNormParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRMSNormTask;

static inline TrillimStatus trillim_rms_requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const TrillimRMSNormParameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->output_dtype) || !trillim_float(p->input_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if ((p->row_stride != p->width || p->column_stride != 1) &&
        (p->input_dtype != TRILLIM_FP32 || p->weight_dtype != TRILLIM_FP32 ||
         p->output_dtype != TRILLIM_FP32)) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->output_dtype);
    if (p->rows > SIZE_MAX / p->width || p->rows * p->width > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    size_t span = 1;
    if (p->row_stride != 0 && p->rows - 1 > (SIZE_MAX - span) / p->row_stride) {
        return TRILLIM_ERROR;
    }
    span += (p->rows - 1) * p->row_stride;
    if (p->column_stride != 0 && p->width - 1 > (SIZE_MAX - span) / p->column_stride) {
        return TRILLIM_ERROR;
    }
    span += (p->width - 1) * p->column_stride;
    size_t input_bytes = trillim_itemsize(p->input_dtype);
    if (input_bytes != 0) {
        if (span > SIZE_MAX / input_bytes) {
            return TRILLIM_ERROR;
        }
        inputs[0] = span * input_bytes;
    } else {
        if (p->storage_columns == 0 || p->storage_rows > SIZE_MAX / p->storage_columns ||
            span > p->storage_rows * p->storage_columns ||
            !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
            return TRILLIM_ERROR;
        }
    }
    if (!trillim_payload_size(1, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * bytes;
    return TRILLIM_OK;
}

static inline float32x4_t load_input4(const TrillimRMSNormTask *task, size_t start) {
    const TrillimRMSNormParameters *p = task->parameters;
    if (p->column_stride == 1) {
        return load4(task->input, start, p->input_dtype);
    }
    float values[4];
    for (size_t lane = 0; lane < 4; ++lane) {
        values[lane] = trillim_load(
            task->input, start + lane * p->column_stride, p->input_dtype
        ).f;
    }
    return vld1q_f32(values);
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = (const TrillimRMSNormTask *)opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    size_t start = row * p->row_stride;
    size_t output_start = row * p->width;
    float sum = 0.0f;
    size_t column = 0;
    if (p->output_dtype != TRILLIM_FP32) {
        for (; p->width - column >= 4; column += 4) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            float lanes[4];
            vst1q_f32(lanes, load_input4(task, start + column * p->column_stride));
            for (size_t lane = 0; lane < 4; ++lane) sum += lanes[lane] * lanes[lane];
        }
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column * p->column_stride, p->input_dtype).f;
            sum += value * value;
        }
    } else {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        column = 0;
        for (; p->width - column >= 16; column += 16) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a = load_input4(task, start + column * p->column_stride);
            float32x4_t b = load_input4(task, start + (column + 4) * p->column_stride);
            float32x4_t c = load_input4(task, start + (column + 8) * p->column_stride);
            float32x4_t d = load_input4(task, start + (column + 12) * p->column_stride);
            sum0 = vaddq_f32(sum0, vmulq_f32(a, a));
            sum1 = vaddq_f32(sum1, vmulq_f32(b, b));
            sum2 = vaddq_f32(sum2, vmulq_f32(c, c));
            sum3 = vaddq_f32(sum3, vmulq_f32(d, d));
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column * p->column_stride, p->input_dtype).f;
            sum += value * value;
        }
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    float32x4_t divisor = vdupq_n_f32(denominator);
    float32x4_t offset = vdupq_n_f32(p->weight_offset);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vdivq_f32(load_input4(task, start + column * p->column_stride), divisor);
        float32x4_t weight = load_weight4(task->weight, column, p->weight_dtype, 1, p->width);
        value = vmulq_f32(value, vaddq_f32(weight, offset));
        store4(task->output, output_start + column, p->output_dtype, value);
    }
    for (; column < p->width; ++column) {
        float weight = trillim_read_float(task->weight, column, p->weight_dtype, 1, p->width);
        TrillimValue value = {.f = (trillim_load(task->input, start + column * p->column_stride, p->input_dtype).f / denominator) * (weight + p->weight_offset)};
        trillim_store(task->output, output_start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_pair(void *opaque, size_t pair, size_t worker) {
    const TrillimRMSNormTask *task = opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    size_t row = pair * 2;
    if (p->rows - row == 1) {
        return normalize(opaque, row, worker);
    }
    const uint16_t *input0 = (const uint16_t *)task->input + row * p->width;
    const uint16_t *input1 = input0 + p->width;
    uint16_t *output0 = (uint16_t *)task->output + row * p->width;
    uint16_t *output1 = output0 + p->width;
    float32x2_t sum = vdup_n_f32(0.0f);
    for (size_t column = 0; column < p->width; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x4x2_t columns = vzip_u16(vld1_u16(input0 + column), vld1_u16(input1 + column));
        float32x4_t first = vreinterpretq_f32_u32(vshll_n_u16(columns.val[0], 16));
        float32x4_t second = vreinterpretq_f32_u32(vshll_n_u16(columns.val[1], 16));
        float32x2_t a = vget_low_f32(first);
        float32x2_t b = vget_high_f32(first);
        float32x2_t c = vget_low_f32(second);
        float32x2_t d = vget_high_f32(second);
        // Each lane adds its row's squares in column order, as in normalize.
        sum += a * a;
        sum += b * b;
        sum += c * c;
        sum += d * d;
    }
    float denominator0 = sqrtf(vget_lane_f32(sum, 0) / (float)p->width + p->eps);
    float denominator1 = sqrtf(vget_lane_f32(sum, 1) / (float)p->width + p->eps);
    if (!isfinite(denominator0) || !isfinite(denominator1) || denominator0 == 0.0f || denominator1 == 0.0f) {
        TrillimStatus status = normalize(opaque, row, worker);
        return status == TRILLIM_OK ? normalize(opaque, row + 1, worker) : status;
    }
    float32x4_t divisor0 = vdupq_n_f32(denominator0);
    float32x4_t divisor1 = vdupq_n_f32(denominator1);
    float32x4_t offset = vdupq_n_f32(p->weight_offset);
    for (size_t column = 0; column < p->width; column += 4) {
        if ((column & 511) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t weight = vaddq_f32(load4(task->weight, column, TRILLIM_BF16), offset);
        float32x4_t value0 = vdivq_f32(load4(input0, column, TRILLIM_BF16), divisor0);
        float32x4_t value1 = vdivq_f32(load4(input1, column, TRILLIM_BF16), divisor1);
        store4(output0, column, TRILLIM_BF16, vmulq_f32(value0, weight));
        store4(output1, column, TRILLIM_BF16, vmulq_f32(value1, weight));
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_transposed(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    size_t row = tile * 4;
    if (p->rows - row < 4) {
        for (; row < p->rows; ++row) {
            TrillimStatus status = normalize(opaque, row, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    const float *input = task->input;
    const float *weight = task->weight;
    float *output = task->output;
    float32x4_t sum = vdupq_n_f32(0);
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vld1q_f32(input + row + column * p->column_stride);
        sum = vaddq_f32(sum, vmulq_f32(value, value));
    }
    float32x4_t mean = vdivq_f32(sum, vdupq_n_f32((float)p->width));
    float32x4_t denominator = vsqrtq_f32(vaddq_f32(mean, vdupq_n_f32(p->eps)));
    size_t column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float weights[4];
        vst1q_f32(weights, vaddq_f32(vld1q_f32(weight + column), vdupq_n_f32(p->weight_offset)));
        float32x4_t values[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            float32x4_t input_rows = vld1q_f32(input + row + (column + lane) * p->column_stride);
            values[lane] = vmulq_n_f32(vdivq_f32(input_rows, denominator), weights[lane]);
        }
        float32x4_t a = vtrn1q_f32(values[0], values[1]);
        float32x4_t b = vtrn2q_f32(values[0], values[1]);
        float32x4_t c = vtrn1q_f32(values[2], values[3]);
        float32x4_t d = vtrn2q_f32(values[2], values[3]);
        vst1q_f32(output + row * p->width + column, vcombine_f32(vget_low_f32(a), vget_low_f32(c)));
        vst1q_f32(output + (row + 1) * p->width + column, vcombine_f32(vget_low_f32(b), vget_low_f32(d)));
        vst1q_f32(output + (row + 2) * p->width + column, vcombine_f32(vget_high_f32(a), vget_high_f32(c)));
        vst1q_f32(output + (row + 3) * p->width + column, vcombine_f32(vget_high_f32(b), vget_high_f32(d)));
    }
    for (; column < p->width; ++column) {
        float32x4_t input_rows = vld1q_f32(input + row + column * p->column_stride);
        float32x4_t value = vmulq_n_f32(
            vdivq_f32(input_rows, denominator), weight[column] + p->weight_offset
        );
        float lanes[4];
        vst1q_f32(lanes, value);
        for (size_t lane = 0; lane < 4; ++lane) {
            output[(row + lane) * p->width + column] = lanes[lane];
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRMSNormParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype &&
           p->column_stride == 1 && p->row_stride == p->width;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_rmsnorm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], inputs[1], p, context};
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    bool transposed = p->row_stride == 1 && p->column_stride != 1;
    TrillimTask worker = transposed ? normalize_transposed : normalize;
    size_t tasks = transposed ? p->rows / 4 + (p->rows % 4 != 0) : p->rows;
    if (p->input_dtype == TRILLIM_BF16 && p->weight_dtype == TRILLIM_BF16 &&
        p->output_dtype == TRILLIM_BF16 && p->column_stride == 1 && p->row_stride == p->width &&
        p->width % 4 == 0 && p->rows >= 2 && isfinite(p->weight_offset) &&
        p->rows / 2 >= context->thread_pool->worker_count) {
        worker = normalize_pair;
        tasks = p->rows / 2 + (p->rows % 2 != 0);
    }
    size_t serial_limit = p->output_dtype == TRILLIM_FP32 ? 4096 : 2048;
    if (transposed) {
        serial_limit = 8192;
    }
    if (p->rows == 1 || p->rows <= serial_limit / p->width) {
        for (size_t tile = 0; tile < tasks; ++tile) {
            TrillimStatus status = worker(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tasks, worker, &task);
}

TRILLIM_KERNEL const char *trillim_neon_rmsnorm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _reshape_strides, _storage_shape, _DTYPE_IDS, _native_dtype\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('RMSNorm requires a matching channel vector')\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    rows, width = _product(x.shape[:-1]), x.shape[-1]\n"
           "    matrix = _reshape_strides(x.shape, strides, (rows, width))\n"
           "    if matrix is None:\n"
           "        axes = [(d, s) for d, s in zip(x.shape[:-1], strides[:-1]) if d.op != 'const' or d.value != 1]\n"
           "        row_stride = axes[-1][1] if axes else width\n"
           "        expected = row_stride\n"
           "        for dimension, stride in reversed(axes):\n"
           "            if stride != expected:\n"
           "                raise ValueError('RMSNorm rows must flatten regularly')\n"
           "            expected = _expression('mul', expected, dimension)\n"
           "        matrix = (row_stride, strides[-1])\n"
           "    canonical = matrix == (width, _constant_expression(1))\n"
           "    if not canonical and any(v.dtype != 'fp32' for v in (x, weight, output)):\n"
           "        raise ValueError('strided RMSNorm currently requires FP32 arithmetic')\n"
           "    if (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "        raise ValueError('RMSNorm weight must be contiguous')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    storage_rows, storage_columns = _storage_shape(x)\n"
           "    return dict(rows=rows, width=width, row_stride=matrix[0], column_stride=matrix[1],\n"
           "                storage_rows=storage_rows, storage_columns=storage_columns,\n"
           "                input_dtype=_DTYPE_IDS[x.dtype], weight_dtype=_DTYPE_IDS[weight.dtype],\n"
           "                output_dtype=_native_dtype(output.dtype), eps=float(eps), weight_offset=float(offset))\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_rmsnorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(TrillimRMSNormParameters, rows), NULL},
        {"width", "size_t", offsetof(TrillimRMSNormParameters, width), NULL},
        {"row_stride", "size_t", offsetof(TrillimRMSNormParameters, row_stride), NULL},
        {"column_stride", "size_t", offsetof(TrillimRMSNormParameters, column_stride), NULL},
        {"storage_rows", "size_t", offsetof(TrillimRMSNormParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimRMSNormParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, output_dtype), NULL},
        {"eps", "float", offsetof(TrillimRMSNormParameters, eps), NULL},
        {"weight_offset", "float", offsetof(TrillimRMSNormParameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, trillim_rms_requirements,
        can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_rmsnorm_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
