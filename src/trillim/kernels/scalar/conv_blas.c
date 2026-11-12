#include "common.h"
#include <limits.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define BLOCK_M 32
#define BLOCK_N 128
#define BLOCK_K 256
#define COLUMN_LIMIT (16 * 1024 * 1024)
#define PACK_LANES 32
#define WORKSPACE_LIMIT (64 * 1024 * 1024)

static bool initialize_blas(void);

typedef struct {
    size_t dimensions, channels, outputs, groups;
    const size_t *input_shape, *output_shape, *kernel, *stride, *padding, *dilation, *output_padding;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input, *weight, *bias;
    float *columns, *weights;
    const Parameters *p;
    size_t input_volume, output_volume, kernel_volume, width, column_stride, lanes;
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

static bool workspace(const Parameters *p, size_t width, size_t positions,
                      size_t *columns, size_t *weights, size_t *stride, size_t *lanes) {
    *weights = 0;
    if (p->weight_dtype != TRILLIM_FP32 && !product(p->outputs, width, weights)) {
        return false;
    }
    if (*weights >= WORKSPACE_LIMIT || !product(p->groups, width, columns) ||
        !product(*columns, positions, columns)) {
        return false;
    }
    size_t budget = WORKSPACE_LIMIT - *weights;
    *stride = positions;
    *lanes = 0;
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

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!p->dimensions || p->dimensions > 3 || !p->channels || !p->groups ||
        p->channels % p->groups || p->outputs % p->groups ||
        !trillim_float(p->input_dtype) || !trillim_float(p->weight_dtype) ||
        !trillim_float(p->output_dtype)
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
        (void)extra;
        if (input > INT64_MAX - 2 * padding || input + 2 * padding < effective ||
            output != (input + 2 * padding - effective) / stride + 1 ||
            !product(output - 1, stride, &spread) || spread > INT64_MAX - effective) {
            return TRILLIM_ERROR;
        }
    }
    size_t input_volume, output_volume, kernel_volume, count;
    if (!volume(p->dimensions, p->input_shape, &input_volume) ||
        !volume(p->dimensions, p->output_shape, &output_volume) ||
        !volume(p->dimensions, p->kernel, &kernel_volume) ||
        !product(p->channels, input_volume, &count) ||
        !product(count, trillim_itemsize(p->input_dtype), &inputs[0]) ||
        !product(p->channels / p->groups, p->outputs, &count) ||
        !product(count, kernel_volume, &count) ||
        !product(count, trillim_itemsize(p->weight_dtype), &inputs[1]) ||
        !product(p->outputs, output_volume, &count) ||
        !product(count, trillim_itemsize(p->output_dtype), &outputs[0])) {
        return TRILLIM_ERROR;
    }
    size_t width, count_columns, count_weights, column_stride, lanes;
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    if (compute != TRILLIM_FP32 || p->output_dtype != TRILLIM_FP32 ||
        !product(p->channels / p->groups, kernel_volume, &width) ||
        width < 16 || p->outputs / p->groups < 4 || output_volume < 4 ||
        width > INT32_MAX || output_volume > INT32_MAX ||
        !workspace(p, width, output_volume, &count_columns, &count_weights, &column_stride, &lanes) ||
        count_columns > SIZE_MAX / sizeof(float) ||
        count_weights > SIZE_MAX / sizeof(float) - count_columns ||
        !initialize_blas()) {
        return TRILLIM_ERROR;
    }
    *scratch = (count_columns + count_weights) * sizeof(float);
    return TRILLIM_OK;
}

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
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
        *position = *position * p->input_shape[axis] + input;
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

static TrillimStatus pack_weight(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t first = tile * 1024, count = task->p->outputs * task->width;
    size_t end = count - first < 1024 ? count : first + 1024;
    for (size_t index = first; index < end; ++index) {
        if ((index & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        task->weights[index] = load(task->weight, index, task->p->weight_dtype);
    }
    return TRILLIM_OK;
}

static TrillimStatus im2col(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = task->output_volume / 256 + (task->output_volume % 256 != 0);
    size_t row = tile / tiles, first = tile % tiles * 256;
    size_t end = task->output_volume - first < 256 ? task->output_volume : first + 256;
    size_t kernel = row % task->kernel_volume;
    size_t input = row / task->kernel_volume;
    size_t offset[3];
    coordinates(kernel, p->dimensions, p->kernel, offset);
    for (size_t position = first; position < end; ++position) {
        if ((position & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t out[3], index;
        coordinates(position, p->dimensions, p->output_shape, out);
        bool valid = source_position(p, out, offset, 0, &index);
        task->columns[row * task->output_volume + position] =
            valid ? load(task->input, input * task->input_volume + index, p->input_dtype) : 0;
    }
    return TRILLIM_OK;
}

static TrillimStatus convolve(void *opaque, size_t tile, size_t worker) {
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
    for (size_t first = 0; first < task->width; first += BLOCK_K) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t k = task->width - first < BLOCK_K ? task->width - first : BLOCK_K;
        sgemm(101, 111, 111, (int)m, (int)n, (int)k, 1.0f,
              weights + output * task->width + first, (int)task->width,
              task->columns + (group * task->width + first) * task->output_volume + position,
              (int)task->output_volume, first == 0 ? 0.0f : 1.0f,
              (float *)task->output + output * task->output_volume + position, (int)task->output_volume);
    }
    return TRILLIM_OK;
}

static TrillimStatus convolve_blocked(void *opaque, size_t lane, size_t worker) {
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
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t input = group * (p->channels / p->groups) + row / task->kernel_volume;
            size_t offset[3];
            coordinates(row % task->kernel_volume, p->dimensions, p->kernel, offset);
            for (size_t position = 0; position < n; ++position) {
                size_t out[3], index;
                coordinates(first + position, p->dimensions, p->output_shape, out);
                bool valid = source_position(p, out, offset, 0, &index);
                columns[row * stride + position] =
                    valid ? load(task->input, input * task->input_volume + index, p->input_dtype) : 0;
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
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_conv_blas(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->outputs) {
        return TRILLIM_OK;
    }
    if (!initialize_blas()) {
        return TRILLIM_ERROR;
    }
    Task task = {outputs[0], inputs[0], inputs[1], NULL, (float *)scratch, NULL, p, 0, 0, 0, 0, 0, 0, context};
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
    task.weights = task.columns + column_count;
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
        size_t positions = task.output_volume / 256 + (task.output_volume % 256 != 0);
        status = context->thread_pool->run(context->thread_pool->context, p->channels * task.kernel_volume * positions, im2col, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        size_t columns = task.output_volume / BLOCK_N + (task.output_volume % BLOCK_N != 0);
        size_t outputs_per_group = p->outputs / p->groups;
        size_t rows = outputs_per_group / BLOCK_M + (outputs_per_group % BLOCK_M != 0);
        status = context->thread_pool->run(context->thread_pool->context, p->groups * rows * columns, convolve, &task);
    }

    return status;
}

TRILLIM_KERNEL const char *trillim_scalar_conv_blas_tdl(void) {
    return            "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS\n"
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
           "    for t in (*c.inputs.values(),o):\n"
           "        if t.dtype not in ('fp16','bf16','fp32') or (t.strides or _contiguous_strides(t.shape))!=_contiguous_strides(t.shape):\n"
           "            raise ValueError('convolution requires contiguous floating tensors')\n"
           "    return dict(dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS['fp32'])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_conv_blas_parameters(void) {
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
    };
    static const size_t inputs[] = {1, 4, 1}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        2, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_conv_blas_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_BLAS};
    return &caps;
}
