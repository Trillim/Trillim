#include "common.h"
#include <cpuid.h>
#include <limits.h>

#define TRANSPOSED 0
#define BIASED 1
#define USE_SIMD 1
#include <immintrin.h>

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
    const TrillimKernelContext *context;
    bool f16c;
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

/* F(4,3): transform 6x6 input patches, multiply in blocks of six channels,
   and reconstruct 4x4 output tiles. The sixteen tile lanes share each weight. */
typedef struct {
    size_t tile_rows, tile_columns, tiles, tile_capacity;
    size_t input_channels, output_channels, padded_height, padded_width;
    size_t weights, inputs, products, padded, count;
} WinogradLayout;

typedef struct {
    const Task *task;
    WinogradLayout layout;
    float *weights, *inputs, *products, *padded;
} WinogradTask;

static bool fma_available(void) {
    unsigned eax, ebx, ecx, edx;
    return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 12));
}

static bool winograd_layout(const Parameters *p, WinogradLayout *w) {
    if (p->dimensions != 2 || p->groups != 1 ||
        p->input_dtype != TRILLIM_FP32 || p->weight_dtype != TRILLIM_FP32 ||
        p->bias_dtype != TRILLIM_FP32 || p->output_dtype != TRILLIM_FP32 ||
        p->kernel[0] != 3 || p->kernel[1] != 3 ||
        p->stride[0] != 1 || p->stride[1] != 1 ||
        p->padding[0] != 1 || p->padding[1] != 1 ||
        p->dilation[0] != 1 || p->dilation[1] != 1 ||
        p->input_strides[0] != p->input_shape[1] || p->input_strides[1] != 1 ||
        p->output_strides[0] != p->output_shape[1] || p->output_strides[1] != 1 ||
        p->input_shape[0] > SIZE_MAX - 5 || p->input_shape[1] > SIZE_MAX - 5 ||
        p->channels > SIZE_MAX - 7 || p->outputs > SIZE_MAX - 5 ||
        p->channels < 3 || p->outputs < 16) {
        return false;
    }
    size_t image;
    if (!product(p->input_shape[0], p->input_shape[1], &image) ||
        p->channel_stride != image || p->output_channel_stride != image || image < 64) {
        return false;
    }
    w->tile_rows = (p->input_shape[0] + 3) / 4;
    w->tile_columns = (p->input_shape[1] + 3) / 4;
    if (!product(w->tile_rows, w->tile_columns, &w->tiles) || w->tiles > SIZE_MAX - 15) {
        return false;
    }
    w->tile_capacity = (w->tiles + 15) / 16 * 16;
    w->input_channels = (p->channels + 7) / 8 * 8;
    w->output_channels = (p->outputs + 5) / 6 * 6;
    w->padded_height = w->tile_rows * 4 + 2;
    w->padded_width = w->tile_columns * 4 + 2;
    size_t padded_image;
    if (!product(w->input_channels, w->output_channels, &w->weights) ||
        !product(w->weights, 36, &w->weights) ||
        !product(w->input_channels, w->tile_capacity, &w->inputs) ||
        !product(w->inputs, 36, &w->inputs) ||
        !product(w->output_channels, w->tile_capacity, &w->products) ||
        !product(w->products, 36, &w->products) ||
        !product(w->padded_height, w->padded_width, &padded_image) || padded_image > INT32_MAX ||
        !product(p->channels, padded_image, &w->padded)) {
        return false;
    }
    w->count = w->weights;
    const size_t pieces[] = {w->inputs, w->products, w->padded};
    for (size_t i = 0; i < 3; ++i) {
        if (pieces[i] > SIZE_MAX - w->count) {
            return false;
        }
        w->count += pieces[i];
    }
    return w->count <= SIZE_MAX / sizeof(float);
}


static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
#if BIASED
    inputs[2] = 0;
#endif
    if (!p->dimensions || p->dimensions > 3 || !p->channels || !p->groups ||
        p->channels % p->groups || p->outputs % p->groups ||
        !trillim_float(p->input_dtype) || p->weight_dtype < TRILLIM_FP16 || p->weight_dtype > TRILLIM_MXFP4 ||
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
        !product(p->outputs, output_volume, &count) ||
        !product(count, trillim_itemsize(p->output_dtype), &outputs[0])) {
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
    WinogradLayout layout;
    if (winograd_layout(p, &layout)) {
        *scratch = layout.count * sizeof(float);
    }
    return TRILLIM_OK;
}

static TrillimStatus winograd_pad(void *opaque, size_t channel, size_t worker) {
    (void)worker;
    const WinogradTask *w = opaque;
    const Parameters *p = w->task->p;
    size_t width = w->layout.padded_width, count = w->layout.padded_height * width;
    float *destination = w->padded + channel * count;
    for (size_t position = 0; position < count;) {
        if (w->task->context->is_cancelled(w->task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t chunk = count - position < 16384 ? count - position : 16384;
        memset(destination + position, 0, chunk * sizeof(float));
        position += chunk;
    }
    const float *source = (const float *)w->task->input + channel * p->channel_stride;
    size_t copied = 0;
    for (size_t row = 0; row < p->input_shape[0]; ++row) {
        for (size_t column = 0; column < p->input_shape[1];) {
            if (!copied && w->task->context->is_cancelled(w->task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t chunk = p->input_shape[1] - column;
            if (chunk > 16384 - copied) {
                chunk = 16384 - copied;
            }
            memcpy(destination + (row + 1) * width + 1 + column,
                   source + row * p->input_shape[1] + column, chunk * sizeof(float));
            column += chunk;
            copied = (copied + chunk) % 16384;
        }
    }
    return TRILLIM_OK;
}

static void winograd_weight_line(float a, float b, float c, float out[6]) {
    out[0] = a * 0.25f;
    out[1] = -(a + b + c) * (1.0f / 6.0f);
    out[2] = (-a + b - c) * (1.0f / 6.0f);
    out[3] = (a + 2.0f * b + 4.0f * c) * (1.0f / 24.0f);
    out[4] = (a - 2.0f * b + 4.0f * c) * (1.0f / 24.0f);
    out[5] = c;
}

static TrillimStatus winograd_weights(void *opaque, size_t block, size_t worker) {
    (void)worker;
    const WinogradTask *w = opaque;
    const Parameters *p = w->task->p;
    size_t stride = w->layout.input_channels * w->layout.output_channels;
    for (size_t input = 0; input < w->layout.input_channels; ++input) {
        if ((input & 31) == 0 && w->task->context->is_cancelled(w->task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t lane = 0; lane < 6; ++lane) {
            size_t output = block * 6 + lane;
            float *destination = w->weights + block * w->layout.input_channels * 6 + input * 6 + lane;
            if (output >= p->outputs || input >= p->channels) {
                for (size_t plane = 0; plane < 36; ++plane) {
                    destination[plane * stride] = 0;
                }
                continue;
            }
            const float *source = (const float *)w->task->weight + (output * p->channels + input) * 9;
            float rows[3][6];
            for (size_t row = 0; row < 3; ++row) {
                winograd_weight_line(source[row * 3], source[row * 3 + 1], source[row * 3 + 2], rows[row]);
            }
            for (size_t column = 0; column < 6; ++column) {
                float values[6];
                winograd_weight_line(rows[0][column], rows[1][column], rows[2][column], values);
                for (size_t row = 0; row < 6; ++row) {
                    destination[(row * 6 + column) * stride] = values[row];
                }
            }
        }
    }
    return TRILLIM_OK;
}

__attribute__((target("avx2,fma"))) static inline void winograd_input_line(const __m256 d[6], __m256 out[6]) {
    __m256 a = _mm256_mul_ps(d[2], _mm256_set1_ps(4));
    __m256 b = _mm256_mul_ps(d[1], _mm256_set1_ps(4));
    __m256 c = _mm256_mul_ps(d[1], _mm256_set1_ps(2));
    __m256 e = _mm256_mul_ps(d[3], _mm256_set1_ps(2));
    out[0] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(d[0], _mm256_set1_ps(4)), _mm256_mul_ps(d[2], _mm256_set1_ps(5))), d[4]);
    out[1] = _mm256_add_ps(_mm256_sub_ps(d[3], _mm256_add_ps(a, b)), d[4]);
    out[2] = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(b, a), d[3]), d[4]);
    out[3] = _mm256_add_ps(_mm256_sub_ps(e, _mm256_add_ps(c, d[2])), d[4]);
    out[4] = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(c, d[2]), e), d[4]);
    out[5] = _mm256_add_ps(_mm256_sub_ps(b, _mm256_mul_ps(d[3], _mm256_set1_ps(5))), d[5]);
}

__attribute__((target("avx2,fma"))) static TrillimStatus winograd_inputs(void *opaque, size_t job, size_t worker) {
    (void)worker;
    const WinogradTask *w = opaque;
    const WinogradLayout *l = &w->layout;
    const Parameters *p = w->task->p;
    size_t blocks = (l->tile_capacity + 63) / 64;
    size_t channel = job / blocks, first = job % blocks * 64;
    size_t end = l->tile_capacity - first < 64 ? l->tile_capacity : first + 64;
    size_t stride = l->input_channels * l->tile_capacity;
    if (w->task->context->is_cancelled(w->task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (channel >= p->channels) {
        for (size_t tile = first; tile < end; tile += 16) {
            float *destination = w->inputs + tile / 16 * l->input_channels * 16 + channel * 16;
            for (size_t plane = 0; plane < 36; ++plane) {
                memset(destination + plane * stride, 0, 16 * sizeof(float));
            }
        }
        return TRILLIM_OK;
    }
    const float *source = w->padded + channel * l->padded_height * l->padded_width;
    for (size_t tile = first; tile < end; tile += 8) {
        int32_t indices[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            size_t t = tile + lane < l->tiles ? tile + lane : 0;
            indices[lane] = (int32_t)((t / l->tile_columns * l->padded_width + t % l->tile_columns) * 4);
        }
        __m256i index = _mm256_loadu_si256((const __m256i *)indices);
        __m256 rows[6][6];
        for (size_t row = 0; row < 6; ++row) {
            __m256 values[6];
            for (size_t column = 0; column < 6; ++column) {
                values[column] = _mm256_i32gather_ps(source + row * l->padded_width + column, index, 4);
            }
            winograd_input_line(values, rows[row]);
        }
        float *destination = w->inputs + tile / 16 * l->input_channels * 16 + channel * 16 + tile % 16;
        for (size_t column = 0; column < 6; ++column) {
            __m256 values[6], result[6];
            for (size_t row = 0; row < 6; ++row) {
                values[row] = rows[row][column];
            }
            winograd_input_line(values, result);
            for (size_t row = 0; row < 6; ++row) {
                _mm256_storeu_ps(destination + (row * 6 + column) * stride, result[row]);
            }
        }
    }
    return TRILLIM_OK;
}

__attribute__((target("avx2,fma"))) static inline void winograd_multiply(const float *u, const float *v, float *m, size_t channels) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0;
    __m256 d0 = a0, d1 = a0, e0 = a0, e1 = a0, f0 = a0, f1 = a0;
    for (size_t input = 0; input < channels; ++input) {
        __m256 x = _mm256_loadu_ps(v), y = _mm256_loadu_ps(v + 8), weight;
        weight = _mm256_broadcast_ss(u);
        a0 = _mm256_fmadd_ps(x, weight, a0);
        a1 = _mm256_fmadd_ps(y, weight, a1);
        weight = _mm256_broadcast_ss(u + 1);
        b0 = _mm256_fmadd_ps(x, weight, b0);
        b1 = _mm256_fmadd_ps(y, weight, b1);
        weight = _mm256_broadcast_ss(u + 2);
        c0 = _mm256_fmadd_ps(x, weight, c0);
        c1 = _mm256_fmadd_ps(y, weight, c1);
        weight = _mm256_broadcast_ss(u + 3);
        d0 = _mm256_fmadd_ps(x, weight, d0);
        d1 = _mm256_fmadd_ps(y, weight, d1);
        weight = _mm256_broadcast_ss(u + 4);
        e0 = _mm256_fmadd_ps(x, weight, e0);
        e1 = _mm256_fmadd_ps(y, weight, e1);
        weight = _mm256_broadcast_ss(u + 5);
        f0 = _mm256_fmadd_ps(x, weight, f0);
        f1 = _mm256_fmadd_ps(y, weight, f1);
        v += 16;
        u += 6;
    }
    _mm256_storeu_ps(m, a0);
    _mm256_storeu_ps(m + 8, a1);
    _mm256_storeu_ps(m + 16, b0);
    _mm256_storeu_ps(m + 24, b1);
    _mm256_storeu_ps(m + 32, c0);
    _mm256_storeu_ps(m + 40, c1);
    _mm256_storeu_ps(m + 48, d0);
    _mm256_storeu_ps(m + 56, d1);
    _mm256_storeu_ps(m + 64, e0);
    _mm256_storeu_ps(m + 72, e1);
    _mm256_storeu_ps(m + 80, f0);
    _mm256_storeu_ps(m + 88, f1);
}

__attribute__((target("avx2,fma"))) static TrillimStatus winograd_products(void *opaque, size_t job, size_t worker) {
    (void)worker;
    const WinogradTask *w = opaque;
    const WinogradLayout *l = &w->layout;
    size_t blocks = (l->tile_capacity + 63) / 64;
    size_t plane = job / blocks, first = job % blocks * 64;
    size_t end = l->tile_capacity - first < 64 ? l->tile_capacity : first + 64;
    if (w->task->context->is_cancelled(w->task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t tile = first; tile < end; tile += 16) {
        const float *input = w->inputs + plane * l->input_channels * l->tile_capacity + tile / 16 * l->input_channels * 16;
        for (size_t output = 0; output < l->output_channels; output += 6) {
            if (output % 48 == 0 && w->task->context->is_cancelled(w->task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const float *weight = w->weights + plane * l->output_channels * l->input_channels + output * l->input_channels;
            float *product = w->products + plane * l->output_channels * l->tile_capacity + tile / 16 * l->output_channels * 16 + output * 16;
            winograd_multiply(weight, input, product, l->input_channels);
        }
    }
    return TRILLIM_OK;
}

__attribute__((target("avx2,fma"))) static inline void winograd_output_line(const __m256 d[6], __m256 out[4]) {
    __m256 a = _mm256_add_ps(d[1], d[2]), b = _mm256_sub_ps(d[1], d[2]);
    __m256 c = _mm256_add_ps(d[3], d[4]), e = _mm256_sub_ps(d[3], d[4]);
    out[0] = _mm256_add_ps(_mm256_add_ps(d[0], a), c);
    out[1] = _mm256_fmadd_ps(e, _mm256_set1_ps(2), b);
    out[2] = _mm256_fmadd_ps(c, _mm256_set1_ps(4), a);
    out[3] = _mm256_add_ps(_mm256_fmadd_ps(e, _mm256_set1_ps(8), b), d[5]);
}

__attribute__((target("avx2,fma"))) static TrillimStatus winograd_outputs(void *opaque, size_t job, size_t worker) {
    (void)worker;
    const WinogradTask *w = opaque;
    const WinogradLayout *l = &w->layout;
    const Parameters *p = w->task->p;
    size_t blocks = (l->tiles + 63) / 64;
    size_t output = job / blocks, first = job % blocks * 64;
    size_t end = l->tiles - first < 64 ? l->tiles : first + 64;
    size_t stride = l->output_channels * l->tile_capacity;
    if (w->task->context->is_cancelled(w->task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    float bias = ((const float *)w->task->bias)[output];
    float *destination = (float *)w->task->output + output * w->task->output_volume;
    for (size_t tile = first; tile < end; tile += 8) {
        __m256 rows[6][4];
        const float *source = w->products + tile / 16 * l->output_channels * 16 + output * 16 + tile % 16;
        for (size_t row = 0; row < 6; ++row) {
            __m256 values[6];
            for (size_t column = 0; column < 6; ++column) {
                values[column] = _mm256_loadu_ps(source + (row * 6 + column) * stride);
            }
            winograd_output_line(values, rows[row]);
        }
        float values[4][4][8];
        for (size_t column = 0; column < 4; ++column) {
            __m256 input[6], result[4];
            for (size_t row = 0; row < 6; ++row) {
                input[row] = rows[row][column];
            }
            winograd_output_line(input, result);
            for (size_t row = 0; row < 4; ++row) {
                _mm256_storeu_ps(values[row][column], _mm256_add_ps(result[row], _mm256_set1_ps(bias)));
            }
        }
        size_t lanes = end - tile < 8 ? end - tile : 8;
        for (size_t lane = 0; lane < lanes; ++lane) {
            size_t y = (tile + lane) / l->tile_columns * 4, x = (tile + lane) % l->tile_columns * 4;
            for (size_t row = 0; row < 4 && y + row < p->output_shape[0]; ++row) {
                for (size_t column = 0; column < 4 && x + column < p->output_shape[1]; ++column) {
                    destination[(y + row) * p->output_shape[1] + x + column] = values[row][column][lane];
                }
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus winograd(const Task *task, void *scratch, const WinogradLayout *layout) {
    WinogradTask w = {task, *layout, scratch, NULL, NULL, NULL};
    w.inputs = w.weights + layout->weights;
    w.products = w.inputs + layout->inputs;
    w.padded = w.products + layout->products;
    const TrillimThreadPool *pool = task->context->thread_pool;
    size_t blocks = (layout->tile_capacity + 63) / 64;
    TrillimStatus status = pool->run(pool->context, task->p->channels, winograd_pad, &w);
    if (status == TRILLIM_OK) status = pool->run(pool->context, layout->output_channels / 6, winograd_weights, &w);
    if (status == TRILLIM_OK) status = pool->run(pool->context, layout->input_channels * blocks, winograd_inputs, &w);
    if (status == TRILLIM_OK) status = pool->run(pool->context, 36 * blocks, winograd_products, &w);
    if (status == TRILLIM_OK) status = pool->run(pool->context, task->p->outputs * ((layout->tiles + 63) / 64), winograd_outputs, &w);
    return status;
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

__attribute__((target("avx2"))) static void store(const Task *task, size_t output, size_t position, float sum) {
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


static bool f16c_available(void) {
    unsigned eax, ebx, ecx, edx;
    return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 29));
}

__attribute__((target("avx2,f16c"))) static __m256 load_half(const uint16_t *data) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
}

__attribute__((target("avx2"))) static __m256 load8(const void *data, size_t position, uint32_t dtype, bool f16c) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + position);
    }
    const uint16_t *source = (const uint16_t *)data + position;
    if (dtype == TRILLIM_BF16) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)source)), 16));
    }
    if (f16c) {
        return load_half(source);
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = trillim_fp16(source[lane]);
    }
    return _mm256_loadu_ps(values);
}

#if TRANSPOSED
__attribute__((target("avx2"))) static __m256 round8(__m256 values, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return values;
    }
    if (dtype == TRILLIM_BF16) {
        __m256i bits = _mm256_castps_si256(values);
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
        __m256 rounded_values = _mm256_castsi256_ps(_mm256_and_si256(_mm256_add_epi32(bits, bias), _mm256_set1_epi32((int)0xffff0000)));
        return _mm256_blendv_ps(values, rounded_values, _mm256_cmp_ps(values, values, _CMP_ORD_Q));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, values);
    for (size_t lane = 0; lane < 8; ++lane) {
        lanes[lane] = rounded(lanes[lane], dtype);
    }
    return _mm256_loadu_ps(lanes);
}
#endif

__attribute__((target("avx2"))) static __m256 source8(const Task *task, size_t base, const size_t *indices, unsigned mask) {
    if (mask == 255) {
        if (indices[7] == indices[0] + 7) {
            return load8(task->input, base + indices[0], task->p->input_dtype, task->f16c);
        }
        if (task->p->input_dtype == TRILLIM_FP32) {
            int32_t offsets[8];
            bool fits = true;
            for (size_t lane = 0; lane < 8; ++lane) {
                fits = fits && indices[lane] <= INT32_MAX;
                offsets[lane] = (int32_t)indices[lane];
            }
            if (fits) {
                return _mm256_i32gather_ps((const float *)task->input + base, _mm256_loadu_si256((const __m256i *)offsets), 4);
            }
        }
    }
    float values[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = mask & (1u << lane) ? trillim_load(task->input, base + indices[lane], task->p->input_dtype).f : 0;
    }
    return _mm256_loadu_ps(values);
}

__attribute__((target("avx2"))) static TrillimStatus convolve_vectors(const Task *task, size_t output, size_t position, const size_t *out, size_t step, size_t lanes) {
    const Parameters *p = task->p;
    size_t input_group = p->channels / p->groups, offset[3], indices[16];
    size_t group_base = output / (p->outputs / p->groups) * input_group;
#if TRANSPOSED
    (void)lanes;
    __m256 sum = _mm256_setzero_ps();
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
        if ((kernel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        coordinates(kernel, p->dimensions, p->kernel, offset);
        unsigned mask = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            if (source_position(p, out, offset, lane * step, &indices[lane])) {
                mask |= 1u << lane;
            }
        }
        if (!mask) {
            continue;
        }
        __m256 partial = _mm256_setzero_ps();
        for (size_t channel = 0; channel < input_group; ++channel) {
            if ((channel & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 input = source8(task, (group_base + channel) * p->channel_stride, indices, mask);
            float weight = weight_value(task, weight_position(task, output, channel, kernel));
            partial = _mm256_add_ps(partial, _mm256_mul_ps(input, _mm256_set1_ps(weight)));
        }
        int32_t valid[8];
        for (size_t lane = 0; lane < 8; ++lane) {
            valid[lane] = mask & (1u << lane) ? -1 : 0;
        }
        partial = _mm256_and_ps(round8(partial, compute), _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)valid)));
        sum = round8(_mm256_add_ps(sum, partial), compute);
    }
    float values[8];
    _mm256_storeu_ps(values, sum);
#else
    __m256 sums[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
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
            __m256 weight = _mm256_set1_ps(weight_value(task, weight_position(task, output, channel, kernel)));
            for (size_t vector = 0; vector < lanes / 8; ++vector) {
                __m256 values = source8(task, base, source + vector * 8, valid >> (vector * 8) & 255);
                sums[vector] = _mm256_add_ps(sums[vector], _mm256_mul_ps(values, weight));
            }
        }
    }
    float values[16];
    for (size_t vector = 0; vector < lanes / 8; ++vector) {
        _mm256_storeu_ps(values + vector * 8, sums[vector]);
    }
#endif
    if (step == 1 && task->output_contiguous && p->output_dtype == TRILLIM_FP32 &&
        (p->input_dtype != p->weight_dtype || p->input_dtype == TRILLIM_FP32)) {
        __m256 bias = _mm256_setzero_ps();
#if BIASED
        bias = _mm256_set1_ps(trillim_load(task->bias, output, p->bias_dtype).f);
#endif
        float *destination = (float *)task->output + output * task->output_volume + position;
        for (size_t lane = 0; lane < lanes; lane += 8) {
            __m256 result = _mm256_loadu_ps(values + lane);
#if BIASED
            result = _mm256_add_ps(result, bias);
#else
            (void)bias;
#endif
            _mm256_storeu_ps(destination + lane, result);
        }
        return TRILLIM_OK;
    }
    for (size_t lane = 0; lane < lanes; ++lane) {
        store(task, output, position + lane * step, values[lane]);
    }
    return TRILLIM_OK;
}

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
#if USE_SIMD && TRANSPOSED
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
                if ((row_end - position - 1) / step >= 7) {
                    status = convolve_vectors(task, output, position, out, step, 8);
                    position += 8 * step;
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
#if USE_SIMD
        if (end - position >= 8 && p->output_shape[p->dimensions - 1] - out[p->dimensions - 1] >= 8) {
            size_t lanes = end - position >= 16 &&
                p->output_shape[p->dimensions - 1] - out[p->dimensions - 1] >= 16 ? 16 : 8;
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



__attribute__((target("avx2"))) static TrillimStatus depthwise(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = task->output_volume / 64 + (task->output_volume % 64 != 0);
    size_t output = tile / tiles, position = tile % tiles * 64;
    size_t end = task->output_volume - position < 64 ? task->output_volume : position + 64;
    size_t base = output * p->channel_stride;
    __m256 bias = _mm256_setzero_ps();
#if BIASED
    bias = _mm256_set1_ps(trillim_load(task->bias, output, p->bias_dtype).f);
#endif
    for (; end - position >= 8; position += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            if ((kernel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int64_t index = (int64_t)(position + kernel * p->dilation[0]) - (int64_t)p->padding[0];
            __m256 input;
            if (index >= 0 && (uint64_t)index + 8 <= p->input_shape[0]) {
                input = load8(task->input, base + (size_t)index, p->input_dtype, task->f16c);
            } else {
                float lanes[8];
                for (size_t lane = 0; lane < 8; ++lane) {
                    int64_t source = index + (int64_t)lane;
                    lanes[lane] = source >= 0 && (uint64_t)source < p->input_shape[0] ?
                        trillim_load(task->input, base + (size_t)source, p->input_dtype).f : 0;
                }
                input = _mm256_loadu_ps(lanes);
            }
            float weight = weight_value(task, output * task->kernel_volume + kernel);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(input, _mm256_set1_ps(weight)));
        }
#if BIASED
        sum = _mm256_add_ps(sum, bias);
#else
        (void)bias;
#endif
        _mm256_storeu_ps((float *)task->output + output * task->output_volume + position, sum);
    }
    for (; position < end; ++position) {
        size_t out[] = {position};
        TrillimStatus status = convolve1(task, output, position, out);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

enum { DEPTHWISE_TIME_TILE = 32, DEPTHWISE_CHANNEL_TILE = 64, DEPTHWISE_DECODE_CHANNELS = 512 };

__attribute__((target("avx2"))) static TrillimStatus depthwise_time_major(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t rows = task->output_volume, channels = p->channels;
    size_t temporal = rows / DEPTHWISE_TIME_TILE + (rows % DEPTHWISE_TIME_TILE != 0);
    size_t first = tile % temporal * DEPTHWISE_TIME_TILE;
    size_t last = rows - first < DEPTHWISE_TIME_TILE ? rows : first + DEPTHWISE_TIME_TILE;
    size_t width = rows == 1 ? DEPTHWISE_DECODE_CHANNELS : DEPTHWISE_CHANNEL_TILE;
    size_t channel = tile / temporal * width;
    size_t end = channels - channel < width ? channels : channel + width;
    for (; end - channel >= 4; channel += 4) {
        if ((channel & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 lo = load8(task->weight, channel * 4, p->weight_dtype, task->f16c);
        __m256 hi = load8(task->weight, channel * 4 + 8, p->weight_dtype, task->f16c);
        __m128 w0 = _mm256_castps256_ps128(lo), w1 = _mm256_extractf128_ps(lo, 1);
        __m128 w2 = _mm256_castps256_ps128(hi), w3 = _mm256_extractf128_ps(hi, 1);
        _MM_TRANSPOSE4_PS(w0, w1, w2, w3);
#if BIASED
        __m128 bias = _mm_setr_ps(trillim_load(task->bias, channel, p->bias_dtype).f,
                                  trillim_load(task->bias, channel + 1, p->bias_dtype).f,
                                  trillim_load(task->bias, channel + 2, p->bias_dtype).f,
                                  trillim_load(task->bias, channel + 3, p->bias_dtype).f);
#endif
        size_t position = first;
        while (position < last) {
            size_t count = last - position < 4 ? last - position : 4;
            const float *input = (const float *)task->input + position * channels + channel;
            __m128 x[7], sums[4];
            for (size_t t = 0; t < count + 3; ++t) {
                x[t] = _mm_loadu_ps(input + t * channels);
            }
            for (size_t t = 0; t < count; ++t) {
                __m128 sum = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(x[t], w0));
                sum = _mm_add_ps(sum, _mm_mul_ps(x[t + 1], w1));
                sum = _mm_add_ps(sum, _mm_mul_ps(x[t + 2], w2));
                sum = _mm_add_ps(sum, _mm_mul_ps(x[t + 3], w3));
#if BIASED
                sum = _mm_add_ps(sum, bias);
#endif
                sums[t] = sum;
            }
            float *output = (float *)task->output + channel * rows + position;
            if (count == 4) {
                _MM_TRANSPOSE4_PS(sums[0], sums[1], sums[2], sums[3]);
                for (size_t lane = 0; lane < 4; ++lane) {
                    _mm_storeu_ps(output + lane * rows, sums[lane]);
                }
            } else if (rows == 1) {
                _mm_storeu_ps(output, sums[0]);
            } else if (rows == 2) {
                _mm_storeu_ps(output, _mm_unpacklo_ps(sums[0], sums[1]));
                _mm_storeu_ps(output + 4, _mm_unpackhi_ps(sums[0], sums[1]));
            } else {
                for (size_t t = 0; t < count; ++t) {
                    float values[4];
                    _mm_storeu_ps(values, sums[t]);
                    for (size_t lane = 0; lane < 4; ++lane) {
                        output[lane * rows + t] = values[lane];
                    }
                }
            }
            position += count;
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

TRILLIM_KERNEL TrillimStatus trillim_x86_conv_nd_bias(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], inputs[1], NULL, p, 0, 0, 0, context, f16c_available(), true};
#if BIASED
    task.bias = inputs[2];
#endif
    if (!volume(p->dimensions, p->input_shape, &task.input_volume) ||
        !volume(p->dimensions, p->output_shape, &task.output_volume) ||
        !volume(p->dimensions, p->kernel, &task.kernel_volume)) {
        return TRILLIM_ERROR;
    }
    size_t output_stride = 1;
    for (size_t axis = p->dimensions; axis-- > 0;) {
        if (p->output_strides[axis] != output_stride) {
            task.output_contiguous = false;
        }
        output_stride *= p->output_shape[axis];
    }
    task.output_contiguous = task.output_contiguous && p->output_channel_stride == output_stride;
    WinogradLayout layout;
    if (fma_available() && winograd_layout(p, &layout)) {
        return winograd(&task, scratch, &layout);
    }
    if (p->dimensions == 1 && p->groups == p->channels && p->outputs == p->channels &&
        p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_FP32 && trillim_float(p->weight_dtype) &&
        p->input_shape[0] > 3 && task.output_volume == p->input_shape[0] - 3 && task.kernel_volume == 4 &&
        p->stride[0] == 1 && p->padding[0] == 0 && p->dilation[0] == 1 &&
        p->channel_stride == 1 && p->input_strides[0] == p->channels &&
        (task.output_volume == 1 || p->output_strides[0] == 1) &&
        (p->outputs == 1 || p->output_channel_stride == task.output_volume) &&
        (uintptr_t)task.input % sizeof(float) == 0 && (uintptr_t)task.output % sizeof(float) == 0 &&
        (uintptr_t)task.weight % trillim_itemsize(p->weight_dtype) == 0) {
        size_t temporal = task.output_volume / DEPTHWISE_TIME_TILE + (task.output_volume % DEPTHWISE_TIME_TILE != 0);
        size_t width = task.output_volume == 1 ? DEPTHWISE_DECODE_CHANNELS : DEPTHWISE_CHANNEL_TILE;
        size_t channels = p->channels / width + (p->channels % width != 0);
        size_t tiles = temporal * channels;
        return tiles == 1 ? depthwise_time_major(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tiles, depthwise_time_major, &task);
    }
    size_t tiles = task.output_volume / 64 + (task.output_volume % 64 != 0);
    size_t count = p->outputs * tiles;
    if (p->dimensions == 1 && p->groups == p->channels && p->outputs == p->channels &&
        p->stride[0] == 1 && p->input_strides[0] == 1 && task.output_contiguous &&
        p->output_dtype == TRILLIM_FP32 && (p->input_dtype != p->weight_dtype || p->input_dtype == TRILLIM_FP32)) {
        return context->thread_pool->run(context->thread_pool->context, count, depthwise, &task);
    }
    if (task.output_volume <= 4 && p->channels == p->groups) {
        return context->thread_pool->run(context->thread_pool->context,
            count / 32 + (count % 32 != 0), convolve_batch, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, count, convolve, &task);
}

TRILLIM_KERNEL const char *trillim_x86_conv_nd_bias_tdl(void) {
    return            "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _storage_shape, _product\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, kernel: tuple[int, ...] = (3,), stride=1, padding=0, dilation=1, groups: int=1, dtype: DType | None=None) -> tuple[Tensor, ...]:\n"
           "    result=Conv(x, weight, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)\n"
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
           "    if c.inputs['bias'].shape!=(out,):\n"
           "        raise ValueError('convolution bias shape differs')\n"
           "    if x.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating input')\n"
           "    if _DTYPE_IDS[w.dtype]<_DTYPE_IDS['fp16'] or o.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating outputs and floating or packed weights')\n"
           "    ws=w.strides or _contiguous_strides(w.shape)\n"
           "    if any(size!=_constant_expression(1) and stride!=expected for size,stride,expected in zip(w.shape,ws,_contiguous_strides(w.shape),strict=True)) or (_DTYPE_IDS[w.dtype]>=_DTYPE_IDS['row_int8'] and _storage_shape(w)!=(w.shape[0],_product(w.shape[1:]))):\n"
           "        raise ValueError('convolution requires a complete contiguous weight tensor')\n"
           "    for t in (o,c.inputs['bias']):\n"
           "        if t.dtype not in ('fp16','bf16','fp32') or (t.strides or _contiguous_strides(t.shape))!=_contiguous_strides(t.shape):\n"
           "            raise ValueError('convolution requires contiguous floating tensors')\n"
           "    xs=x.strides or _contiguous_strides(x.shape)\n"
           "    os=o.strides or _contiguous_strides(o.shape)\n"
           "    return dict(output_channel_stride=os[0],output_strides=os[1:],channel_stride=xs[0],input_strides=xs[1:],dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS[c.inputs['bias'].dtype])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_conv_nd_bias_parameters(void) {
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
        BIASED ? 3 : 2, 1, requirements, NULL, inputs, outputs, 32
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_conv_nd_bias_capabilities(void) {
    
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
