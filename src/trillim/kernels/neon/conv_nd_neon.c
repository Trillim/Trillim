#include "common.h"
#include <limits.h>

#define TRANSPOSED 0
#define BIASED 0
#define USE_NEON 1

#if USE_NEON
#include <arm_neon.h>
#endif

typedef struct {
    size_t dimensions, channels, outputs, groups;
    const size_t *input_shape, *output_shape, *kernel, *stride, *padding, *dilation, *output_padding;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
    size_t channel_stride;
    const size_t *input_strides;
    size_t output_channel_stride;
    const size_t *output_strides;
} Parameters;

typedef struct {
    void *output;
    const void *input, *weight, *bias;
    const Parameters *p;
    size_t input_volume, output_volume, kernel_volume;
    bool spatial_contiguous;
    const TrillimKernelContext *context;
    bool output_contiguous;
} Task;

static bool product(size_t a, size_t b, size_t *result) {
    if (b && a > SIZE_MAX / b) {
        return false;
    }
    *result = a * b;
    return true;
}

static bool volume(size_t dimensions, const size_t *shape, size_t *result) {
    *result = 1;
    for (size_t axis = 0; axis < dimensions; ++axis) {
        if (!product(*result, shape[axis], result)) {
            return false;
        }
    }
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
#if BIASED
    inputs[2] = 0;
#endif
    if (!p->dimensions || p->dimensions > 3 || !p->channels || !p->groups ||
        p->channels % p->groups || p->outputs % p->groups ||
        !trillim_float(p->input_dtype) || p->weight_dtype < TRILLIM_FP16 ||
        p->weight_dtype > TRILLIM_MXFP4 ||
        !trillim_float(p->output_dtype)
#if BIASED
        || !trillim_float(p->bias_dtype)
#endif
    ) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        size_t input = p->input_shape[axis], output = p->output_shape[axis];
        size_t kernel = p->kernel[axis], stride = p->stride[axis], padding = p->padding[axis];
        size_t dilation = p->dilation[axis], extra = p->output_padding[axis], effective, spread;
        if (!input || !output || !kernel || !stride || !dilation ||
            input > INT64_MAX || output > INT64_MAX || stride > INT64_MAX || dilation > INT64_MAX ||
            padding > INT64_MAX / 2 ||
            !product(kernel - 1, dilation, &effective) || effective >= INT64_MAX) {
            return TRILLIM_ERROR;
        }
        ++effective;
#if TRANSPOSED
        if (!product(input - 1, stride, &spread) || spread > INT64_MAX - effective ||
            extra > INT64_MAX - spread - effective || spread + effective + extra <= 2 * padding ||
            output != spread + effective + extra - 2 * padding ||
            output - 1 > INT64_MAX - padding) {
            return TRILLIM_ERROR;
        }
#else
        (void)extra;
        if (input > INT64_MAX - 2 * padding || input + 2 * padding < effective ||
            output != (input + 2 * padding - effective) / stride + 1 ||
            !product(output - 1, stride, &spread) || spread > INT64_MAX - effective) {
            return TRILLIM_ERROR;
        }
#endif
    }
    size_t input_volume, output_volume, kernel_volume, count;
    if (!volume(p->dimensions, p->input_shape, &input_volume) ||
        !volume(p->dimensions, p->output_shape, &output_volume) ||
        !volume(p->dimensions, p->kernel, &kernel_volume) ||
#if TRANSPOSED
        !product(p->outputs / p->groups, kernel_volume, &count) ||
        !trillim_packed_payload_size(p->channels, count, p->weight_dtype, &inputs[1]) ||
#else
        !product(p->channels / p->groups, kernel_volume, &count) ||
        !trillim_packed_payload_size(p->outputs, count, p->weight_dtype, &inputs[1]) ||
#endif
        !product(p->outputs, output_volume, &count)) {
        return TRILLIM_ERROR;
    }
    size_t span;
    if (!product(p->channels - 1, p->channel_stride, &span)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        size_t extent;
        if (!product(p->input_shape[axis] - 1, p->input_strides[axis], &extent) || extent > SIZE_MAX - span) {
            return TRILLIM_ERROR;
        }
        span += extent;
    }
    if (span == SIZE_MAX || !product(span + 1, trillim_itemsize(p->input_dtype), &inputs[0])) {
        return TRILLIM_ERROR;
    }
    size_t output_span = 0;
    if (p->outputs) {
        if (!product(p->outputs - 1, p->output_channel_stride, &output_span)) {
            return TRILLIM_ERROR;
        }
        for (size_t axis = 0; axis < p->dimensions; ++axis) {
            size_t extent;
            if (!product(p->output_shape[axis] - 1, p->output_strides[axis], &extent) || extent > SIZE_MAX - output_span) {
                return TRILLIM_ERROR;
            }
            output_span += extent;
        }
        if (output_span == SIZE_MAX || !product(output_span + 1, trillim_itemsize(p->output_dtype), &outputs[0])) {
            return TRILLIM_ERROR;
        }
    }
#if BIASED
    if (!product(p->outputs, trillim_itemsize(p->bias_dtype), &inputs[2])) {
        return TRILLIM_ERROR;
    }
#endif
    return TRILLIM_OK;
}

static float rounded(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static void coordinates(size_t position, size_t dimensions, const size_t *shape, size_t *result) {
    if (dimensions == 1) {
        result[0] = position;
        return;
    }
    for (size_t axis = dimensions; axis-- > 0;) {
        result[axis] = position % shape[axis];
        position /= shape[axis];
    }
}

static bool source_position(const Parameters *p, const size_t *out, const size_t *offset, size_t lane, size_t *position) {
    *position = 0;
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        size_t coordinate = out[axis] + (axis + 1 == p->dimensions ? lane : 0);
#if TRANSPOSED
        int64_t index = (int64_t)(coordinate + p->padding[axis]) -
            (int64_t)(offset[axis] * p->dilation[axis]);
        if (index < 0 || (uint64_t)index % p->stride[axis]) {
            return false;
        }
        size_t input = (uint64_t)index / p->stride[axis];
#else
        int64_t index = (int64_t)(coordinate * p->stride[axis] + offset[axis] * p->dilation[axis]) -
            (int64_t)p->padding[axis];
        if (index < 0) {
            return false;
        }
        size_t input = (size_t)index;
#endif
        if (input >= p->input_shape[axis]) {
            return false;
        }
        *position += input * p->input_strides[axis];
    }
    return true;
}

static size_t weight_position(const Task *task, size_t output, size_t channel, size_t offset) {
    const Parameters *p = task->p;
#if TRANSPOSED
    size_t input_group = p->channels / p->groups, output_group = p->outputs / p->groups;
    size_t input = output / output_group * input_group + channel;
    return (input * output_group + output % output_group) * task->kernel_volume + offset;
#else
    return (output * (p->channels / p->groups) + channel) * task->kernel_volume + offset;
#endif
}

static inline float weight_value(const Task *task, size_t index) {
    const Parameters *p = task->p;
    if (trillim_float(p->weight_dtype)) {
        return trillim_load(task->weight, index, p->weight_dtype).f;
    }
#if TRANSPOSED
    size_t rows = p->channels, columns = p->outputs / p->groups * task->kernel_volume;
#else
    size_t rows = p->outputs, columns = p->channels / p->groups * task->kernel_volume;
#endif
    return trillim_packed_read_float(task->weight, index, p->weight_dtype, rows, columns);
}

static void store(const Task *task, size_t output, size_t position, float sum) {
    const Parameters *p = task->p;
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    sum = rounded(sum, compute);
#if BIASED
    sum += trillim_load(task->bias, output, p->bias_dtype).f;
    sum = rounded(sum, compute == p->bias_dtype ? compute : TRILLIM_FP32);
#endif
    TrillimValue value = {.f = sum};
    size_t index;
    if (task->output_contiguous) {
        index = output * task->output_volume + position;
    } else {
        index = output * p->output_channel_stride;
        for (size_t axis = p->dimensions; axis-- > 0;) {
            index += position % p->output_shape[axis] * p->output_strides[axis];
            position /= p->output_shape[axis];
        }
    }
    trillim_store(task->output, index, p->output_dtype, value);
}

#if USE_NEON
static float32x4_t load4(const void *data, size_t position, uint32_t dtype, size_t step) {
    if (dtype == TRILLIM_FP32) {
        const float *values = (const float *)data + position;
        return step == 1 ? vld1q_f32(values) : vld2q_f32(values).val[0];
    }
    const uint16_t *source = (const uint16_t *)data + position;
    uint16x4_t values = step == 1 ? vld1_u16(source) : vld2_u16(source).val[0];
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(values));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(values, 16));
}

#if TRANSPOSED
static float32x4_t round4(float32x4_t values, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return values;
    }
    float lanes[4];
    vst1q_f32(lanes, values);
    for (size_t lane = 0; lane < 4; ++lane) {
        lanes[lane] = rounded(lanes[lane], dtype);
    }
    return vld1q_f32(lanes);
}

#endif

static TrillimStatus convolve_vectors(const Task *task, size_t output, size_t position, const size_t *out, size_t step, size_t lanes) {
    const Parameters *p = task->p;
    size_t input_group = p->channels / p->groups, offset[3], indices[16];
    size_t group_base = output / (p->outputs / p->groups) * input_group;
#if TRANSPOSED
    float32x4_t sum = vdupq_n_f32(0);
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
        if ((kernel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        coordinates(kernel, p->dimensions, p->kernel, offset);
        uint32_t valid[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            valid[lane] = source_position(p, out, offset, lane * step, &indices[lane]) ? UINT32_MAX : 0;
        }
        if (!(valid[0] | valid[1] | valid[2] | valid[3])) {
            continue;
        }
        float32x4_t partial = vdupq_n_f32(0);
        for (size_t channel = 0; channel < input_group; ++channel) {
            if ((channel & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t base = (group_base + channel) * p->channel_stride;
            float32x4_t input;
            if (valid[0] && valid[1] && valid[2] && valid[3]) {
                input = load4(task->input, base + indices[0], p->input_dtype, 1);
            } else {
                float values[4];
                for (size_t lane = 0; lane < 4; ++lane) {
                    values[lane] = valid[lane] ? trillim_load(task->input, base + indices[lane], p->input_dtype).f : 0;
                }
                input = vld1q_f32(values);
            }
            float weight = weight_value(task, weight_position(task, output, channel, kernel));
            partial = vfmaq_n_f32(partial, input, weight);
        }
        partial = vbslq_f32(vld1q_u32(valid), round4(partial, compute), vdupq_n_f32(0));
        sum = round4(vaddq_f32(sum, partial), compute);
    }
#else
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    size_t cached_indices[64][16];
    unsigned cached_valid[64];
    bool cached = task->kernel_volume <= 64;
    if (cached) {
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            coordinates(kernel, p->dimensions, p->kernel, offset);
            cached_valid[kernel] = 0;
            for (size_t lane = 0; lane < lanes; ++lane) {
                if (source_position(p, out, offset, lane * step, &cached_indices[kernel][lane])) {
                    cached_valid[kernel] |= 1u << lane;
                }
            }
        }
    }
    for (size_t channel = 0; channel < input_group; ++channel) {
        if ((channel & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            unsigned valid;
            const size_t *source;
            if (cached) {
                valid = cached_valid[kernel];
                source = cached_indices[kernel];
            } else {
                if ((kernel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                coordinates(kernel, p->dimensions, p->kernel, offset);
                valid = 0;
                for (size_t lane = 0; lane < lanes; ++lane) {
                    if (source_position(p, out, offset, lane * step, &indices[lane])) {
                        valid |= 1u << lane;
                    }
                }
                source = indices;
            }
            size_t base = (group_base + channel) * p->channel_stride;
            size_t stride = p->stride[p->dimensions - 1] * p->input_strides[p->dimensions - 1];
            float weight = weight_value(task, weight_position(task, output, channel, kernel));
            for (size_t vector = 0; vector < lanes / 4; ++vector) {
                const size_t *indices = source + vector * 4;
                unsigned mask = (valid >> (vector * 4)) & 15;
                float32x4_t values;
                if (mask == 15 && (stride == 1 || (stride == 2 && task->spatial_contiguous && indices[3] + 1 < task->input_volume))) {
                    values = load4(task->input, base + indices[0], p->input_dtype, stride);
                } else {
                    float values4[4];
                    for (size_t lane = 0; lane < 4; ++lane) {
                        values4[lane] = (mask & (1u << lane)) ?
                            trillim_load(task->input, base + indices[lane], p->input_dtype).f : 0;
                    }
                    values = vld1q_f32(values4);
                }
                sums[vector] = vfmaq_n_f32(sums[vector], values, weight);
            }
        }
    }
#endif
    float values[16];
#if TRANSPOSED
    vst1q_f32(values, sum);
#else
    for (size_t vector = 0; vector < lanes / 4; ++vector) {
        vst1q_f32(values + vector * 4, sums[vector]);
    }
#endif
    for (size_t lane = 0; lane < lanes; ++lane) {
        store(task, output, position + lane * step, values[lane]);
    }
    return TRILLIM_OK;
}
#endif

static TrillimStatus convolve1(const Task *task, size_t output, size_t position, const size_t *out) {
    const Parameters *p = task->p;
    size_t input_group = p->channels / p->groups;
    size_t group_base = output / (p->outputs / p->groups) * input_group;
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    size_t offset[3];
    float sum = 0;
#if TRANSPOSED
    for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
        coordinates(kernel, p->dimensions, p->kernel, offset);
        size_t index;
        if (!source_position(p, out, offset, 0, &index)) {
            continue;
        }
        float partial = 0;
        for (size_t channel = 0; channel < input_group; ++channel) {
            if ((channel & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float input = trillim_load(task->input, (group_base + channel) * p->channel_stride + index, p->input_dtype).f;
            float weight = weight_value(task, weight_position(task, output, channel, kernel));
            partial += input * weight;
        }
        sum = rounded(sum + rounded(partial, compute), compute);
    }
#else
    (void)compute;
    size_t cached_indices[64];
    bool cached_valid[64];
    bool cached = task->kernel_volume <= 64;
    if (cached) {
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            coordinates(kernel, p->dimensions, p->kernel, offset);
            cached_valid[kernel] = source_position(p, out, offset, 0, &cached_indices[kernel]);
        }
    }
    for (size_t channel = 0; channel < input_group; ++channel) {
        if ((channel & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            size_t index;
            bool valid;
            if (cached) {
                valid = cached_valid[kernel];
                index = cached_indices[kernel];
            } else {
                if ((kernel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                coordinates(kernel, p->dimensions, p->kernel, offset);
                valid = source_position(p, out, offset, 0, &index);
            }
            float input = valid ? trillim_load(task->input, (group_base + channel) * p->channel_stride + index, p->input_dtype).f : 0;
            float weight = weight_value(task, weight_position(task, output, channel, kernel));
            sum += input * weight;
        }
    }
#endif
    store(task, output, position, sum);
    return TRILLIM_OK;
}

static TrillimStatus convolve(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = task->output_volume / 64 + (task->output_volume % 64 != 0);
    size_t output = tile / tiles, begin = tile % tiles * 64;
    size_t end = task->output_volume - begin < 64 ? task->output_volume : begin + 64;
#if USE_NEON && TRANSPOSED
    size_t step = p->stride[p->dimensions - 1];
    for (size_t row = begin; row < end;) {
        size_t row_end = row + p->output_shape[p->dimensions - 1] - row % p->output_shape[p->dimensions - 1];
        if (row_end > end) {
            row_end = end;
        }
        size_t residues = row_end - row < step ? row_end - row : step;
        for (size_t residue = 0; residue < residues; ++residue) {
            for (size_t position = row + residue; position < row_end;) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t out[3];
                coordinates(position, p->dimensions, p->output_shape, out);
                TrillimStatus status;
                if ((row_end - position - 1) / step >= 3) {
                    status = convolve_vectors(task, output, position, out, step, 4);
                    position += 4 * step;
                } else {
                    status = convolve1(task, output, position, out);
                    position += step;
                }
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
        }
        row = row_end;
    }
#else
    for (size_t position = begin; position < end;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t out[3];
        coordinates(position, p->dimensions, p->output_shape, out);
#if USE_NEON
        if (end - position >= 4 && p->output_shape[p->dimensions - 1] - out[p->dimensions - 1] >= 4) {
            size_t lanes = end - position >= 16 &&
                p->output_shape[p->dimensions - 1] - out[p->dimensions - 1] >= 16 ? 16 : 4;
            TrillimStatus status = convolve_vectors(task, output, position, out, 1, lanes);
            if (status != TRILLIM_OK) {
                return status;
            }
            position += lanes;
            continue;
        }
#endif
        TrillimStatus status = convolve1(task, output, position, out);
        if (status != TRILLIM_OK) {
            return status;
        }
        ++position;
    }
#endif
    return TRILLIM_OK;
}

static TrillimStatus convolve_batch(void *opaque, size_t batch, size_t worker) {
    const Task *task = opaque;
    size_t tiles = task->output_volume / 64 + (task->output_volume % 64 != 0);
    size_t count = task->p->outputs * tiles, first = batch * 32;
    size_t end = count - first < 32 ? count : first + 32;
    for (size_t tile = first; tile < end; ++tile) {
        TrillimStatus status = convolve(opaque, tile, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static void depthwise_weights(const Task *task, size_t channel, float32x4_t weights[4]) {
    const uint32_t dtype = task->p->weight_dtype;
    if (dtype == TRILLIM_FP32) {
        const float32x4x4_t packed = vld4q_f32((const float *)task->weight + channel * 4);
        for (size_t k = 0; k < 4; ++k) {
            weights[k] = packed.val[k];
        }
    } else {
        const uint16x4x4_t packed = vld4_u16((const uint16_t *)task->weight + channel * 4);
        for (size_t k = 0; k < 4; ++k) {
            weights[k] = dtype == TRILLIM_FP16 ?
                vcvt_f32_f16(vreinterpret_f16_u16(packed.val[k])) :
                vreinterpretq_f32_u32(vshll_n_u16(packed.val[k], 16));
        }
    }
}

enum { DEPTHWISE_SINGLE_CHANNELS = 1024 };

static TrillimStatus depthwise_single(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const size_t channels = task->p->channels;
    const float *input = task->input;
    float *output = task->output;
    size_t channel = tile * DEPTHWISE_SINGLE_CHANNELS;
    const size_t remaining = channels - channel;
    const size_t end = channel + (remaining < DEPTHWISE_SINGLE_CHANNELS ? remaining : DEPTHWISE_SINGLE_CHANNELS);
    for (; end - channel >= 4; channel += 4) {
        if ((channel & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t w[4];
        depthwise_weights(task, channel, w);
        const float *x = input + channel;
        float32x4_t sum = vdupq_n_f32(0);
        sum = vfmaq_f32(sum, vld1q_f32(x), w[0]);
        sum = vfmaq_f32(sum, vld1q_f32(x + channels), w[1]);
        sum = vfmaq_f32(sum, vld1q_f32(x + 2 * channels), w[2]);
        sum = vfmaq_f32(sum, vld1q_f32(x + 3 * channels), w[3]);
        vst1q_f32(output + channel, sum);
    }
    const size_t out[] = {0};
    for (; channel < end; ++channel) {
        TrillimStatus status = convolve1(task, channel, 0, out);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus depthwise_short(void *opaque, size_t tile, size_t worker) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t channel = tile * 32, end = p->outputs - channel < 32 ? p->outputs : channel + 32;
    for (; end - channel >= 4; channel += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t weights[4];
        depthwise_weights(task, channel, weights);
        float32x4_t first = vdupq_n_f32(0), second = vdupq_n_f32(0);
        for (size_t k = 0; k < 4; ++k) {
            const float *input = (const float *)task->input + channel + k * p->channels;
            first = vfmaq_f32(first, vld1q_f32(input), weights[k]);
            second = vfmaq_f32(second, vld1q_f32(input + p->channels), weights[k]);
        }
#if BIASED
        float32x4_t bias = load4(task->bias, channel, p->bias_dtype, 1);
        first = vaddq_f32(first, bias);
        second = vaddq_f32(second, bias);
#endif
        float32x4x2_t output = {{first, second}};
        vst2q_f32((float *)task->output + channel * 2, output);
    }
    for (; channel < end; ++channel) {
        TrillimStatus status = convolve(opaque, channel, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}


enum { DEPTHWISE_CHANNELS = 64, DEPTHWISE_POSITIONS = 32 };

static float32x4_t depthwise_combine_pairs(float32x4_t lower, float32x4_t upper, bool high) {
    uint64x2_t a = vreinterpretq_u64_f32(lower), b = vreinterpretq_u64_f32(upper);
    return vreinterpretq_f32_u64(high ? vzip2q_u64(a, b) : vzip1q_u64(a, b));
}

static TrillimStatus depthwise_time_major(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    const size_t rows = task->output_volume;
    const size_t temporal_tiles = rows / DEPTHWISE_POSITIONS + (rows % DEPTHWISE_POSITIONS != 0);
    const size_t first = tile % temporal_tiles * DEPTHWISE_POSITIONS;
    const size_t last = rows - first < DEPTHWISE_POSITIONS ? rows : first + DEPTHWISE_POSITIONS;
    size_t channel = tile / temporal_tiles * DEPTHWISE_CHANNELS;
    const size_t end = p->channels - channel < DEPTHWISE_CHANNELS ? p->channels : channel + DEPTHWISE_CHANNELS;
    const float *input = task->input;
    float *output = task->output;
    for (; end - channel >= 4; channel += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t w[4];
        depthwise_weights(task, channel, w);
        size_t position = first;
        for (; last - position >= 4; position += 4) {
            const float *x = input + position * p->channels + channel;
            float32x4_t centers[7];
            for (size_t j = 0; j < 7; ++j) {
                centers[j] = vld1q_f32(x + j * p->channels);
            }
            float32x4_t sums[4];
            for (size_t j = 0; j < 4; ++j) {
                float32x4_t s = vdupq_n_f32(0.0f);
                s = vfmaq_f32(s, centers[j], w[0]);
                s = vfmaq_f32(s, centers[j + 1], w[1]);
                s = vfmaq_f32(s, centers[j + 2], w[2]);
                sums[j] = vfmaq_f32(s, centers[j + 3], w[3]);
            }
            const float32x4_t first_low = vzip1q_f32(sums[0], sums[1]);
            const float32x4_t first_high = vzip2q_f32(sums[0], sums[1]);
            const float32x4_t second_low = vzip1q_f32(sums[2], sums[3]);
            const float32x4_t second_high = vzip2q_f32(sums[2], sums[3]);
            float *dst = output + channel * rows + position;
            vst1q_f32(dst, depthwise_combine_pairs(first_low, second_low, false));
            vst1q_f32(dst + rows, depthwise_combine_pairs(first_low, second_low, true));
            vst1q_f32(dst + 2 * rows, depthwise_combine_pairs(first_high, second_high, false));
            vst1q_f32(dst + 3 * rows, depthwise_combine_pairs(first_high, second_high, true));
        }
        for (; position < last; ++position) {
            const float *x = input + position * p->channels + channel;
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (size_t k = 0; k < 4; ++k) {
                sum = vfmaq_f32(sum, vld1q_f32(x + k * p->channels), w[k]);
            }
            float *dst = output + channel * rows + position;
            vst1q_lane_f32(dst, sum, 0);
            vst1q_lane_f32(dst + rows, sum, 1);
            vst1q_lane_f32(dst + 2 * rows, sum, 2);
            vst1q_lane_f32(dst + 3 * rows, sum, 3);
        }
    }
    for (; channel < end; ++channel) {
        for (size_t position = first; position < last; ++position) {
            const size_t out[] = {position};
            TrillimStatus status = convolve1(task, channel, position, out);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_conv_nd(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {
        .output = outputs[0], .input = inputs[0], .weight = inputs[1], .p = p,
        .context = context, .output_contiguous = true
    };
#if BIASED
    task.bias = inputs[2];
#endif
    if (!volume(p->dimensions, p->input_shape, &task.input_volume) ||
        !volume(p->dimensions, p->output_shape, &task.output_volume) ||
        !volume(p->dimensions, p->kernel, &task.kernel_volume)) {
        return TRILLIM_ERROR;
    }
    size_t output_spacing = 1;
    for (size_t axis = p->dimensions; axis-- > 0;) {
        if (p->output_shape[axis] > 1 && p->output_strides[axis] != output_spacing) {
            task.output_contiguous = false;
        }
        output_spacing *= p->output_shape[axis];
    }
    task.output_contiguous = task.output_contiguous &&
        (p->outputs <= 1 || p->output_channel_stride == output_spacing);
    task.spatial_contiguous = true;
    size_t spacing = 1;
    for (size_t axis = p->dimensions; axis-- > 0;) {
        if (p->input_shape[axis] > 1 && p->input_strides[axis] != spacing) {
            task.spatial_contiguous = false;
        }
        spacing *= p->input_shape[axis];
    }
    if (p->dimensions == 1 && p->groups == p->channels && p->outputs == p->channels &&
        task.output_contiguous && p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_FP32 &&
        p->input_shape[0] > 3 && task.output_volume == p->input_shape[0] - 3 && p->kernel[0] == 4 &&
        p->stride[0] == 1 && p->padding[0] == 0 && p->dilation[0] == 1 &&
        p->channel_stride == 1 && p->input_strides[0] == p->channels &&
        (uintptr_t)task.input % sizeof(float) == 0 && (uintptr_t)task.output % sizeof(float) == 0 &&
        trillim_float(p->weight_dtype) &&
        (uintptr_t)task.weight % trillim_itemsize(p->weight_dtype) == 0) {
        const TrillimThreadPool *pool = context->thread_pool;
        if (task.output_volume == 1) {
            const size_t tiles = p->channels / DEPTHWISE_SINGLE_CHANNELS +
                (p->channels % DEPTHWISE_SINGLE_CHANNELS != 0);
            return tiles == 1 ? depthwise_single(&task, 0, 0) :
                pool->run(pool->context, tiles, depthwise_single, &task);
        }
        if (task.output_volume == 2) {
            return pool->run(pool->context, p->outputs / 32 + (p->outputs % 32 != 0),
                             depthwise_short, &task);
        }
        const size_t temporal = task.output_volume / DEPTHWISE_POSITIONS +
            (task.output_volume % DEPTHWISE_POSITIONS != 0);
        const size_t channels = p->channels / DEPTHWISE_CHANNELS +
            (p->channels % DEPTHWISE_CHANNELS != 0);
        return pool->run(pool->context, temporal * channels, depthwise_time_major, &task);
    }
    size_t tiles = task.output_volume / 64 + (task.output_volume % 64 != 0);
    size_t count = p->outputs * tiles;
    if (task.output_volume <= 4 && p->channels == p->groups) {
        return context->thread_pool->run(context->thread_pool->context,
            count / 32 + (count % 32 != 0), convolve_batch, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, count, convolve, &task);
}

TRILLIM_KERNEL const char *trillim_neon_conv_nd_tdl(void) {
    return            "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _storage_shape, _product\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression\n"
           "def pattern(x: Tensor, weight: Tensor, *, kernel: tuple[int, ...] = (3,), stride=1, padding=0, dilation=1, groups: int=1, dtype: DType | None=None) -> tuple[Tensor, ...]:\n"
           "    result=Conv(x, weight, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n"
           "def bind(c):\n"
           "    x,w,o=c.inputs['x'],c.inputs['weight'],c.outputs[0]\n"
           "    k=c.attrs.get('kernel',(3,))\n"
           "    if not isinstance(k,tuple) or not 1<=len(k)<=3 or any(not isinstance(v,int) or v<=0 for v in k):\n"
           "        raise ValueError('convolution requires one to three positive kernel dimensions')\n"
           "    n=len(k)\n"
           "    def axes(name,default):\n"
           "        v=c.attrs.get(name,default)\n"
           "        v=(v,)*n if isinstance(v,int) else v\n"
           "        if not isinstance(v,tuple) or len(v)!=n or any(not isinstance(a,int) or a<(0 if name in ('padding','output_padding') else 1) for a in v):\n"
           "            raise ValueError('invalid convolution '+name)\n"
           "        return v\n"
           "    s,p,d,e=axes('stride',1),axes('padding',0),axes('dilation',1),axes('output_padding',0)\n"
           "    g=c.attrs.get('groups',1)\n"
           "    if not isinstance(g,int) or g<=0 or len(x.shape)!=n+1 or len(w.shape)!=n+2 or len(o.shape)!=n+1:\n"
           "        raise ValueError('convolution ranks or groups differ')\n"
           "    if x.shape[0].op!='const' or x.shape[0].value is None or x.shape[0].value%g:\n"
           "        raise ValueError('convolution requires fixed divisible channels')\n"
           "    if any(a!=_constant_expression(b) for a,b in zip(w.shape[2:],k)):\n"
           "        raise ValueError('convolution weight kernel differs')\n"
           "    out=w.shape[0]\n"
           "    if w.shape[1]!=_constant_expression(x.shape[0].value//g) or o.shape[0]!=out or out.op!='const' or out.value is None or out.value%g:\n"
           "        raise ValueError('convolution channel shapes differ')\n"
           "    if x.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating input')\n"
           "    ws=w.strides or _contiguous_strides(w.shape)\n"
           "    if _DTYPE_IDS[w.dtype]<_DTYPE_IDS['fp16'] or any(size!=_constant_expression(1) and stride!=expected for size,stride,expected in zip(w.shape,ws,_contiguous_strides(w.shape),strict=True)):\n"
           "        raise ValueError('convolution requires contiguous floating or packed weights')\n"
           "    if _DTYPE_IDS[w.dtype]>=_DTYPE_IDS['row_int8'] and _storage_shape(w)!=(w.shape[0],_product(w.shape[1:])):\n"
           "        raise ValueError('convolution requires a complete packed weight tensor')\n"
           "    if o.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating output')\n"
           "    xs=x.strides or _contiguous_strides(x.shape)\n"
           "    os=o.strides or _contiguous_strides(o.shape)\n"
           "    return dict(output_channel_stride=os[0],output_strides=os[1:],channel_stride=xs[0],input_strides=xs[1:],dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS['fp32'])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_conv_nd_parameters(void) {
    static const TrillimParameter fields[] = {
        {"dimensions", "size_t", offsetof(Parameters, dimensions), NULL},
        {"channels", "size_t", offsetof(Parameters, channels), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"groups", "size_t", offsetof(Parameters, groups), NULL},
        {"input_shape", "size_t", offsetof(Parameters, input_shape), "dimensions"},
        {"output_shape", "size_t", offsetof(Parameters, output_shape), "dimensions"},
        {"kernel", "size_t", offsetof(Parameters, kernel), "dimensions"},
        {"stride", "size_t", offsetof(Parameters, stride), "dimensions"},
        {"padding", "size_t", offsetof(Parameters, padding), "dimensions"},
        {"dilation", "size_t", offsetof(Parameters, dilation), "dimensions"},
        {"output_padding", "size_t", offsetof(Parameters, output_padding), "dimensions"},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"channel_stride", "size_t", offsetof(Parameters, channel_stride), NULL},
        {"input_strides", "size_t", offsetof(Parameters, input_strides), "dimensions"},
        {"output_channel_stride", "size_t", offsetof(Parameters, output_channel_stride), NULL},
        {"output_strides", "size_t", offsetof(Parameters, output_strides), "dimensions"},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        BIASED ? 3 : 2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_conv_nd_capabilities(void) {
    /* NEON is part of the AArch64 baseline. Optional instructions are not used. */
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &caps;
}
