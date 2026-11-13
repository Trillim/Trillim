#include "common.h"
#include <limits.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__FLT16_MANT_DIG__)
__extension__ typedef _Float16 TrillimFloat16;
#endif

#define BLOCK_M 64
#define BLOCK_N 256
#define BLOCK_K 1024
#define COLUMN_LIMIT (16 * 1024 * 1024)
#define PACK_LANES 32
#define PACK_POSITIONS 4096
#define WORKSPACE_LIMIT (64 * 1024 * 1024)

static bool initialize_blas(void);

typedef struct {
    size_t dimensions, channels, outputs, groups;
    const size_t *input_shape, *output_shape, *kernel, *stride, *padding, *dilation, *output_padding;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
    size_t channel_stride;
    const size_t *input_strides;
} Parameters;

typedef struct {
    void *output;
    const void *input, *weight, *bias;
    float *columns, *weights;
    const Parameters *p;
    size_t input_volume, output_volume, kernel_volume, width, column_stride, lanes, pack_positions;
    const TrillimKernelContext *context;
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

static bool pointwise(const Parameters *p) {
    if (p->input_dtype != TRILLIM_FP32) {
        return false;
    }
    size_t expected = 1;
    for (size_t axis = p->dimensions; axis-- > 0;) {
        if (p->input_shape[axis] > 1 && p->input_strides[axis] != expected) {
            return false;
        }
        if (!product(expected, p->input_shape[axis], &expected)) {
            return false;
        }
    }
    if (p->channels > 1 && p->channel_stride != expected) {
        return false;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        if (p->kernel[axis] != 1 || p->stride[axis] != 1 || p->padding[axis] != 0) {
            return false;
        }
    }
    return true;
}

static bool workspace(const Parameters *p, size_t width, size_t positions,
                      size_t *columns, size_t *weights, size_t *stride, size_t *lanes) {
    *weights = 0;
    if (p->weight_dtype != TRILLIM_FP32 && !product(p->outputs, width, weights)) {
        return false;
    }
    *columns = 0;
    *stride = positions;
    *lanes = 0;
    if (*weights >= WORKSPACE_LIMIT) {
        return false;
    }
    if (pointwise(p)) {
        return true;
    }
    if (!product(p->groups, width, columns) || !product(*columns, positions, columns)) {
        return false;
    }
    size_t budget = WORKSPACE_LIMIT - *weights;
    if (*columns <= COLUMN_LIMIT && *columns <= budget) {
        return true;
    }
    if (width > budget) {
        return false;
    }
    *stride = positions < BLOCK_N ? positions : BLOCK_N;
    if (*stride > budget / width) {
        *stride = budget / width;
    }
    size_t per_lane = width * *stride;
    size_t tiles = positions / *stride + (positions % *stride != 0);
    if (!product(p->groups, tiles, lanes)) {
        return false;
    }
    if (*lanes > PACK_LANES) {
        *lanes = PACK_LANES;
    }
    if (*lanes > budget / per_lane) {
        *lanes = budget / per_lane;
    }
    return product(*lanes, per_lane, columns);
}

static size_t capped_product(size_t a, size_t b, size_t limit) {
    return b != 0 && a > limit / b ? limit : a * b;
}

static size_t reserved_columns(const Parameters *p, size_t width, size_t positions, size_t weights) {
    if (pointwise(p)) {
        return 0;
    }
    size_t budget = WORKSPACE_LIMIT - weights;
    size_t full_limit = budget < COLUMN_LIMIT ? budget : COLUMN_LIMIT;
    size_t full = capped_product(p->groups, width, full_limit + 1);
    full = capped_product(full, positions, full_limit + 1);
    if (full <= full_limit) {
        return full;
    }
    /* Smaller shapes can use more memory before packing switches to blocks. */
    full = full_limit;
    size_t stride = positions < BLOCK_N ? positions : BLOCK_N;
    size_t tiles = positions / BLOCK_N + (positions % BLOCK_N != 0);
    size_t lanes = capped_product(p->groups, tiles, PACK_LANES);
    size_t blocked = capped_product(width, stride, budget);
    blocked = capped_product(blocked, lanes, budget);
    return full > blocked ? full : blocked;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    inputs[2] = 0;
    if (!p->dimensions || p->dimensions > 3 || !p->channels || !p->groups ||
        p->channels % p->groups || p->outputs % p->groups ||
        !trillim_float(p->input_dtype) || !trillim_float(p->weight_dtype) ||
        !trillim_float(p->output_dtype)
        || !trillim_float(p->bias_dtype)
    ) {
        return TRILLIM_ERROR;
    }
    bool empty = false;
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        size_t input = p->input_shape[axis], output = p->output_shape[axis];
        size_t kernel = p->kernel[axis], stride = p->stride[axis], padding = p->padding[axis];
        size_t dilation = p->dilation[axis], extra = p->output_padding[axis], effective, spread;
        if (!kernel || !stride || !dilation ||
            input > INT64_MAX || output > INT64_MAX || stride > INT64_MAX || dilation > INT64_MAX ||
            padding > INT64_MAX / 2 ||
            !product(kernel - 1, dilation, &effective) || effective >= INT64_MAX) {
            return TRILLIM_ERROR;
        }
        ++effective;
        if (!output) {
            empty = true;
            continue;
        }
        (void)extra;
        if (input > INT64_MAX - 2 * padding || input + 2 * padding < effective ||
            output != (input + 2 * padding - effective) / stride + 1 ||
            !product(output - 1, stride, &spread) || spread > INT64_MAX - effective) {
            return TRILLIM_ERROR;
        }
    }
    if (empty) {
        return TRILLIM_OK;
    }
    size_t input_volume, output_volume, kernel_volume, count;
    if (!volume(p->dimensions, p->input_shape, &input_volume) ||
        !volume(p->dimensions, p->output_shape, &output_volume) ||
        !volume(p->dimensions, p->kernel, &kernel_volume) ||
        !product(p->channels / p->groups, p->outputs, &count) ||
        !product(count, kernel_volume, &count) ||
        !product(count, trillim_itemsize(p->weight_dtype), &inputs[1]) ||
        !product(p->outputs, output_volume, &count) ||
        !product(count, trillim_itemsize(p->output_dtype), &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (!product(p->outputs, trillim_itemsize(p->bias_dtype), &inputs[2])) {
        return TRILLIM_ERROR;
    }
    size_t span = 0;
    if (input_volume) {
        if (!product(p->channels - 1, p->channel_stride, &span)) {
            return TRILLIM_ERROR;
        }
        for (size_t axis = 0; axis < p->dimensions; ++axis) {
            size_t extent;
            if (!product(p->input_shape[axis] - 1, p->input_strides[axis], &extent) ||
                extent > SIZE_MAX - span) {
                return TRILLIM_ERROR;
            }
            span += extent;
        }
        if (span == SIZE_MAX) {
            return TRILLIM_ERROR;
        }
        ++span;
    }
    if (!product(span, trillim_itemsize(p->input_dtype), &inputs[0])) {
        return TRILLIM_ERROR;
    }
    size_t width, count_columns, count_weights, column_stride, lanes;
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    if (compute != TRILLIM_FP32 || p->output_dtype != TRILLIM_FP32 ||
        !product(p->channels / p->groups, kernel_volume, &width) ||
        width > INT32_MAX || output_volume > INT32_MAX ||
        !workspace(p, width, output_volume, &count_columns, &count_weights, &column_stride, &lanes) ||
        count_columns > SIZE_MAX / sizeof(float) ||
        count_weights > SIZE_MAX / sizeof(float) - count_columns ||
        !initialize_blas()) {
        return TRILLIM_ERROR;
    }
    size_t reservation = reserved_columns(p, width, output_volume, count_weights);
    *scratch = (reservation + count_weights) * sizeof(float);
    return TRILLIM_OK;
}

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
#ifdef __APPLE__
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
static GetThreading get_threading;
static SetThreading set_threading;
#endif
static atomic_uint blas_state = 0;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                FreeLibrary(library);
            }
        }
#else
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
        static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
#endif
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            void *library = dlopen(names[i], RTLD_LOCAL | RTLD_LAZY);
            if (library != NULL) {
                void *address = dlsym(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
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
        int64_t index = (int64_t)(coordinate * p->stride[axis] + offset[axis] * p->dilation[axis]) -
            (int64_t)p->padding[axis];
        if (index < 0) {
            return false;
        }
        size_t input = (size_t)index;
        if (input >= p->input_shape[axis]) {
            return false;
        }
        *position += input * p->input_strides[axis];
    }
    return true;
}

static float load(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)data + index * sizeof(bits), sizeof(bits));
    return dtype == TRILLIM_BF16 ? trillim_bf16(bits) : trillim_fp16(bits);
}

static void unpack(const void *data, uint32_t dtype, size_t index, size_t count,
                   size_t stride, float *target) {
    if (dtype == TRILLIM_FP32) {
        const float *source = (const float *)data + index;
        if (stride == 1) {
            memcpy(target, source, count * sizeof(float));
        } else {
            for (size_t i = 0; i < count; ++i) {
                target[i] = source[i * stride];
            }
        }
    } else if (dtype == TRILLIM_BF16) {
        const uint16_t *source = (const uint16_t *)data + index;
        for (size_t i = 0; i < count; ++i) {
            target[i] = trillim_bf16(source[i * stride]);
        }
    } else {
#if defined(__FLT16_MANT_DIG__)
        const TrillimFloat16 *source = (const TrillimFloat16 *)data + index;
        for (size_t i = 0; i < count; ++i) {
            target[i] = (float)source[i * stride];
        }
#else
        const uint16_t *source = (const uint16_t *)data + index;
        for (size_t i = 0; i < count; ++i) {
            target[i] = trillim_fp16(source[i * stride]);
        }
#endif
    }
}

static TrillimStatus pack_columns(const Task *task, size_t input, size_t kernel,
                                  size_t first, size_t count, float *target) {
    const Parameters *p = task->p;
    size_t offset[3], end = first + count, axis = p->dimensions - 1;
    coordinates(kernel, p->dimensions, p->kernel, offset);
    for (size_t position = first; position < end;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t out[3], index;
        coordinates(position, p->dimensions, p->output_shape, out);
        bool valid = source_position(p, out, offset, 0, &index);
        if (valid) {
            size_t length = end - position;
            size_t row = p->output_shape[axis] - out[axis];
            size_t coordinate = out[axis] * p->stride[axis] + offset[axis] * p->dilation[axis] - p->padding[axis];
            size_t available = (p->input_shape[axis] - 1 - coordinate) / p->stride[axis] + 1;
            if (length > row) {
                length = row;
            }
            if (length > available) {
                length = available;
            }
            unpack(task->input, p->input_dtype, input * p->channel_stride + index,
                   length, p->stride[axis] * p->input_strides[axis], target + position - first);
            position += length;
        } else {
            target[position - first] = 0;
            ++position;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus pack_weight(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t first = tile * 1024, count = task->p->outputs * task->width;
    size_t end = count - first < 1024 ? count : first + 1024;
    for (size_t index = first; index < end; index += 256) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t length = end - index < 256 ? end - index : 256;
        unpack(task->weight, task->p->weight_dtype, index, length, 1, task->weights + index);
    }
    return TRILLIM_OK;
}

static TrillimStatus im2col(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t width = task->pack_positions;
    size_t tiles = task->output_volume / width + (task->output_volume % width != 0);
    size_t row = tile / tiles, first = tile % tiles * width;
    size_t count = task->output_volume - first < width ? task->output_volume - first : width;
    size_t kernel = row % task->kernel_volume;
    size_t input = row / task->kernel_volume;
    return pack_columns(task, input, kernel, first, count,
                        task->columns + row * task->output_volume + first);
}

static TrillimStatus finish(const Task *task, size_t output, size_t rows, size_t position, size_t columns) {
    const Parameters *p = task->p;
    for (size_t row = 0; row < rows; ++row) {
        if ((row & 3) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float bias = load(task->bias, output + row, p->bias_dtype);
        float *target = (float *)task->output + (output + row) * task->output_volume + position;
        for (size_t column = 0; column < columns; ++column) {
            target[column] += bias;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus convolve_impl(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t group_outputs = p->outputs / p->groups;
    size_t columns = task->output_volume / BLOCK_N + (task->output_volume % BLOCK_N != 0);
    size_t rows = group_outputs / BLOCK_M + (group_outputs % BLOCK_M != 0);
    size_t group = tile / columns / rows;
    size_t output = group * group_outputs + tile / columns % rows * BLOCK_M;
    size_t position = tile % columns * BLOCK_N;
    size_t m = group_outputs - (output - group * group_outputs);
    size_t n = task->output_volume - position;
    if (m > BLOCK_M) {
        m = BLOCK_M;
    }
    if (n > BLOCK_N) {
        n = BLOCK_N;
    }
    const float *weights = p->weight_dtype == TRILLIM_FP32 ? (const float *)task->weight : task->weights;
    const float *values = pointwise(p) ? (const float *)task->input : task->columns;
    for (size_t first = 0; first < task->width; first += BLOCK_K) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t k = task->width - first < BLOCK_K ? task->width - first : BLOCK_K;
        sgemm(101, 111, 111, (int)m, (int)n, (int)k, 1.0f,
              weights + output * task->width + first, (int)task->width,
              values + (group * task->width + first) * task->output_volume + position,
              (int)task->output_volume, first == 0 ? 0.0f : 1.0f,
              (float *)task->output + output * task->output_volume + position, (int)task->output_volume);
    }
    return finish(task, output, m, position, n);
}


static TrillimStatus convolve_blocked_impl(void *opaque, size_t lane, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t stride = task->column_stride;
    size_t tiles = task->output_volume / stride + (task->output_volume % stride != 0);
    size_t count = p->groups * tiles, group_outputs = p->outputs / p->groups;
    const float *weights = p->weight_dtype == TRILLIM_FP32 ? (const float *)task->weight : task->weights;
    float *columns = task->columns + lane * task->width * stride;
    for (size_t tile = lane; tile < count; tile += task->lanes) {
        size_t group = tile / tiles, first = tile % tiles * stride;
        size_t n = task->output_volume - first < stride ? task->output_volume - first : stride;
        for (size_t row = 0; row < task->width; ++row) {
            size_t input = group * (p->channels / p->groups) + row / task->kernel_volume;
            TrillimStatus status = pack_columns(task, input, row % task->kernel_volume,
                                               first, n, columns + row * stride);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        for (size_t row = 0; row < group_outputs; row += BLOCK_M) {
            size_t output = group * group_outputs + row;
            size_t m = group_outputs - row < BLOCK_M ? group_outputs - row : BLOCK_M;
            for (size_t k = 0; k < task->width; k += BLOCK_K) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t width = task->width - k < BLOCK_K ? task->width - k : BLOCK_K;
                sgemm(101, 111, 111, (int)m, (int)n, (int)width, 1.0f,
                      weights + output * task->width + k, (int)task->width,
                      columns + k * stride, (int)stride, k == 0 ? 0.0f : 1.0f,
                      (float *)task->output + output * task->output_volume + first, (int)task->output_volume);
            }
            TrillimStatus status = finish(task, output, m, first, n);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus convolve(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = convolve_impl(opaque, index, worker);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return convolve_impl(opaque, index, worker);
}

static TrillimStatus convolve_blocked(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = convolve_blocked_impl(opaque, index, worker);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return convolve_blocked_impl(opaque, index, worker);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_conv_bias_blas(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->outputs) {
        return TRILLIM_OK;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        if (!p->output_shape[axis]) {
            return TRILLIM_OK;
        }
    }
    if (!initialize_blas()) {
        return TRILLIM_ERROR;
    }
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], (float *)scratch, NULL, p, 0, 0, 0, 0, 0, 0, 0, context};
    if (!volume(p->dimensions, p->input_shape, &task.input_volume) ||
        !volume(p->dimensions, p->output_shape, &task.output_volume) ||
        !volume(p->dimensions, p->kernel, &task.kernel_volume) ||
        !product(p->channels / p->groups, task.kernel_volume, &task.width)) {
        return TRILLIM_ERROR;
    }
    size_t column_count, weight_count;
    if (!workspace(p, task.width, task.output_volume, &column_count, &weight_count,
                   &task.column_stride, &task.lanes)) {
        return TRILLIM_ERROR;
    }
    task.weights = column_count ? task.columns + column_count : task.columns;
    task.pack_positions = p->channels * task.kernel_volume / 4 >= context->thread_pool->worker_count ? PACK_POSITIONS : 256;
    TrillimStatus status;
    if (p->weight_dtype != TRILLIM_FP32) {
        size_t tiles = weight_count / 1024 + (weight_count % 1024 != 0);
        status = context->thread_pool->run(context->thread_pool->context, tiles, pack_weight, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (task.lanes) {
        status = context->thread_pool->run(context->thread_pool->context, task.lanes, convolve_blocked, &task);
    } else {
        if (!pointwise(p)) {
            size_t positions = task.output_volume / task.pack_positions + (task.output_volume % task.pack_positions != 0);
            status = context->thread_pool->run(context->thread_pool->context, p->channels * task.kernel_volume * positions, im2col, &task);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t columns = task.output_volume / BLOCK_N + (task.output_volume % BLOCK_N != 0);
        size_t outputs_per_group = p->outputs / p->groups;
        size_t rows = outputs_per_group / BLOCK_M + (outputs_per_group % BLOCK_M != 0);
        status = context->thread_pool->run(context->thread_pool->context, p->groups * rows * columns, convolve, &task);
    }
    return status;
}

TRILLIM_KERNEL const char *trillim_scalar_conv_bias_blas_tdl(void) {
    return            "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS\n"
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
           "    width=x.shape[0].value//g\n"
           "    for item in k:\n"
           "        width*=item\n"
           "    group_outputs=out.value//g\n"
           "    if width<16 or (group_outputs<4 and width<64):\n"
           "        raise ValueError('BLAS convolution requires sufficiently large fixed channels')\n"
           "    if all(v.op=='const' for v in o.shape[1:]):\n"
           "        positions=1\n"
           "        for v in o.shape[1:]:\n"
           "            positions*=v.value\n"
           "        direct=x.dtype=='fp32' and all(v==1 for v in k+s) and all(v==0 for v in p)\n"
           "        if positions<4 or (not direct and ((positions<128 and group_outputs<512) or (group_outputs<16 and positions<512))):\n"
           "            raise ValueError('BLAS convolution requires a larger fixed output')\n"
           "    if x.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating input')\n"
           "    for t in (w,o,c.inputs['bias']):\n"
           "        if t.dtype not in ('fp16','bf16','fp32') or (t.strides or _contiguous_strides(t.shape))!=_contiguous_strides(t.shape):\n"
           "            raise ValueError('convolution requires contiguous floating tensors')\n"
           "    xs=x.strides or _contiguous_strides(x.shape)\n"
           "    return dict(channel_stride=xs[0],input_strides=xs[1:],dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS[c.inputs['bias'].dtype])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_conv_bias_blas_parameters(void) {
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
    };
    static const size_t inputs[] = {4, 4, 1}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        3, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_conv_bias_blas_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_BLAS};
    return &caps;
}
