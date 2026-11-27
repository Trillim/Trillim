#include "common.h"
#include <limits.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define ROW_TILE 32
#define POSITION_TILE 256
#define REDUCTION_TILE 1024
#define MAX_LANES 32
#define WORKSPACE_LIMIT (64 * 1024 * 1024)

static bool initialize_blas(void);

typedef struct {
    size_t dimensions, channels, outputs, groups;
    const size_t *input_shape, *output_shape, *kernel, *stride, *padding, *dilation, *output_padding;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
} Parameters;

typedef struct {
    float *output;
    const float *input;
    const void *weight, *bias;
    float *packed, *partials;
    const Parameters *p;
    size_t input_group, output_group, phases, row_tiles, lanes, count;
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
        if (!input) {
            if (extra > INT64_MAX - effective) {
                return TRILLIM_ERROR;
            }
            size_t extent = effective + extra;
            if (stride > extent || 2 * padding > extent - stride ||
                output != extent - stride - 2 * padding) {
                return TRILLIM_ERROR;
            }
            continue;
        }
        if (!product(input - 1, stride, &spread) || spread > INT64_MAX - effective ||
            extra > INT64_MAX - spread - effective || spread + effective + extra <= 2 * padding ||
            output != spread + effective + extra - 2 * padding ||
            output - 1 > INT64_MAX - padding) {
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
        !product(p->channels, input_volume, &count) ||
        !product(count, trillim_itemsize(p->input_dtype), &inputs[0]) ||
        !product(p->channels / p->groups, p->outputs, &count) ||
        !product(count, kernel_volume, &count) ||
        !product(count, trillim_itemsize(p->weight_dtype), &inputs[1]) ||
        !product(p->outputs, output_volume, &count) ||
        !product(count, trillim_itemsize(p->output_dtype), &outputs[0])) {
        return TRILLIM_ERROR;
    }
    size_t group_outputs = p->outputs / p->groups;
    size_t input_group = p->channels / p->groups;
    if (p->dimensions != 1 || p->input_dtype != TRILLIM_FP32 ||
        p->output_dtype != TRILLIM_FP32 || group_outputs < 4 || input_group < 16 ||
        input_group > INT32_MAX || input_volume > INT32_MAX) {
        return TRILLIM_ERROR;
    }
    size_t phases = p->stride[0] < output_volume ? p->stride[0] : output_volume;
    size_t rows = group_outputs / ROW_TILE + (group_outputs % ROW_TILE != 0);
    count = p->groups * rows * phases;
    size_t lanes = count < MAX_LANES ? count : MAX_LANES;
    size_t weights;
    if (!product(p->outputs, input_group, &weights) ||
        !product(weights, kernel_volume, &weights) || weights >= WORKSPACE_LIMIT ||
        lanes > (WORKSPACE_LIMIT - weights) / ROW_TILE / POSITION_TILE || !initialize_blas()) {
        return TRILLIM_ERROR;
    }
    *scratch = (weights + lanes * ROW_TILE * POSITION_TILE) * sizeof(float);
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
        static const char *const names[] = {"libopenblas.dll", "openblas.dll"};
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
        static const char *const names[] = {"libopenblas.so.0", "libopenblas.so"};
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
    const Parameters *p = task->p;
    size_t input_tiles = task->input_group / 32 + (task->input_group % 32 != 0);
    size_t offset = tile / input_tiles / task->row_tiles / p->groups;
    size_t group = tile / input_tiles / task->row_tiles % p->groups;
    size_t output = tile / input_tiles % task->row_tiles * ROW_TILE;
    size_t input = tile % input_tiles * 32;
    size_t rows = task->output_group - output < ROW_TILE ? task->output_group - output : ROW_TILE;
    size_t columns = task->input_group - input < 32 ? task->input_group - input : 32;
    size_t kernel = p->kernel[0];
    uint32_t dtype = p->weight_dtype;
    for (size_t column = 0; column < columns; ++column) {
        if ((column & 7) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t source = ((group * task->input_group + input + column) * task->output_group + output) * kernel + offset;
        size_t target = (offset * p->outputs + group * task->output_group + output) * task->input_group + input + column;
        for (size_t row = 0; row < rows; ++row) {
            task->packed[target + row * task->input_group] = load(task->weight, source + row * kernel, dtype);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus initialize_output(const Task *task, size_t first, size_t rows, size_t phase, bool bias) {
    const Parameters *p = task->p;
    size_t length = p->output_shape[0], stride = p->stride[0];
    for (size_t row = 0; row < rows; ++row) {
        float value = 0;
        for (size_t position = phase, visits = 0;; ++visits) {
            if ((visits & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float *output = task->output + (first + row) * length + position;
            if (bias) {
                *output += value;
            } else {
                *output = 0;
            }
            if (length - position <= stride) {
                break;
            }
            position += stride;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus convolve_impl(void *opaque, size_t lane, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t input_length = p->input_shape[0], output_length = p->output_shape[0], stride = p->stride[0];
    float *partial = task->partials + lane * ROW_TILE * POSITION_TILE;
    for (size_t tile = lane; tile < task->count; tile += task->lanes) {
        size_t group = tile / task->phases / task->row_tiles;
        size_t output = group * task->output_group + tile / task->phases % task->row_tiles * ROW_TILE;
        size_t phase = tile % task->phases;
        size_t m = task->output_group - (output - group * task->output_group);
        if (m > ROW_TILE) {
            m = ROW_TILE;
        }
        TrillimStatus status = initialize_output(task, output, m, phase, false);
        if (status != TRILLIM_OK) {
            return status;
        }
        for (size_t offset = 0; offset < p->kernel[0]; ++offset) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t residue = offset * p->dilation[0] % stride, padding = p->padding[0] % stride;
            residue = residue >= padding ? residue - padding : stride - (padding - residue);
            if (residue != phase) {
                continue;
            }
            int64_t delta = (int64_t)(offset * p->dilation[0]) - (int64_t)p->padding[0];
            if (delta >= (int64_t)output_length) {
                continue;
            }
            size_t first = delta < 0 ? (uint64_t)(-delta) / stride + ((uint64_t)(-delta) % stride != 0) : 0;
            uint64_t last_position = (uint64_t)((int64_t)(output_length - 1) - delta) / stride + 1;
            size_t end = last_position < input_length ? (size_t)last_position : input_length;
            for (; first < end; first += POSITION_TILE) {
                size_t n = end - first < POSITION_TILE ? end - first : POSITION_TILE;
                for (size_t k = 0; k < task->input_group; k += REDUCTION_TILE) {
                    if (task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    size_t width = task->input_group - k < REDUCTION_TILE ? task->input_group - k : REDUCTION_TILE;
                    sgemm(101, 111, 111, (int)m, (int)n, (int)width, 1.0f,
                          task->packed + (offset * p->outputs + output) * task->input_group + k, (int)task->input_group,
                          task->input + (group * task->input_group + k) * input_length + first, (int)input_length,
                          k == 0 ? 0.0f : 1.0f, partial, POSITION_TILE);
                }
                for (size_t row = 0; row < m; ++row) {
                    if ((row & 3) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    size_t position = (size_t)((int64_t)(first * stride) + delta);
                    float *target = task->output + (output + row) * output_length + position;
                    for (size_t column = 0; column < n; ++column) {
                        target[column * stride] += partial[row * POSITION_TILE + column];
                    }
                }
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

TRILLIM_KERNEL TrillimStatus trillim_scalar_conv_transpose_blas(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        if (!p->output_shape[axis]) {
            return TRILLIM_OK;
        }
    }
    if (!initialize_blas()) {
        return TRILLIM_ERROR;
    }
    Task task = {(float *)outputs[0], (const float *)inputs[0], inputs[1], NULL,
                 (float *)scratch, NULL, p, p->channels / p->groups, p->outputs / p->groups, 0, 0, 0, 0, context};
    task.phases = p->stride[0] < p->output_shape[0] ? p->stride[0] : p->output_shape[0];
    task.row_tiles = task.output_group / ROW_TILE + (task.output_group % ROW_TILE != 0);
    task.count = p->groups * task.row_tiles * task.phases;
    task.lanes = task.count < MAX_LANES ? task.count : MAX_LANES;
    size_t weights = p->outputs * task.input_group * p->kernel[0];
    task.partials = task.packed + weights;
    size_t input_tiles = task.input_group / 32 + (task.input_group % 32 != 0);
    size_t tiles = p->kernel[0] * p->groups * task.row_tiles * input_tiles;
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context, tiles, pack_weight, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    return context->thread_pool->run(context->thread_pool->context, task.lanes, convolve, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_conv_transpose_blas_tdl(void) {
    return            "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS\n"
           "from trillim.compiler.passes.infer import _constant_expression, _expression\n"
           "def pattern(x: Tensor, weight: Tensor, *, kernel: tuple[int, ...] = (3,), stride=1, padding=0, dilation=1, groups: int=1, dtype: DType | None=None, output_padding=0) -> tuple[Tensor, ...]:\n"
           "    result=ConvTranspose(x, weight, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=0)\n"
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
           "    out=_expression('mul',w.shape[1],_constant_expression(g))\n"
           "    if w.shape[0]!=x.shape[0] or o.shape[0]!=out:\n"
           "        raise ValueError('transposed convolution channel shapes differ')\n"
           "    for t in (*c.inputs.values(),o):\n"
           "        if t.dtype not in ('fp16','bf16','fp32') or (t.strides or _contiguous_strides(t.shape))!=_contiguous_strides(t.shape):\n"
           "            raise ValueError('convolution requires contiguous floating tensors')\n"
           "    return dict(dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS['fp32'])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_conv_transpose_blas_parameters(void) {
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
    static const size_t inputs[] = {4, 1, 1}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        2, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_conv_transpose_blas_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_PLATFORM_BLAS};
    return &caps;
}
