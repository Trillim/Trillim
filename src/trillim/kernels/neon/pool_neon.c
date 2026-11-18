#include "common.h"
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t dimensions, channels, channel_stride;
    const size_t *input_shape, *input_strides, *output_shape;
    const size_t *kernel, *stride, *padding, *dilation;
    uint32_t average, ceil_mode, count_include_pad;
} Parameters;

typedef struct {
    const float *input;
    float *output;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t output_volume, kernel_volume;
} Task;

static bool product(size_t count, const size_t *shape, size_t *result) {
    *result = 1;
    for (size_t axis = 0; axis < count; ++axis) {
        if (!shape[axis] || *result > SIZE_MAX / shape[axis]) {
            return false;
        }
        *result *= shape[axis];
    }
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t input, output, kernel;
    if (!p->dimensions || p->dimensions > 3 || !p->channels || p->average > 1 ||
        p->ceil_mode > 1 || p->count_include_pad > 1 ||
        !product(p->dimensions, p->input_shape, &input) ||
        !product(p->dimensions, p->output_shape, &output) ||
        !product(p->dimensions, p->kernel, &kernel)) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->dimensions; ++axis) {
        size_t length = p->input_shape[axis], pad = p->padding[axis];
        size_t dilation = p->dilation[axis], stride = p->stride[axis], k = p->kernel[axis];
        if (!stride || !dilation || stride > INT64_MAX || dilation > INT64_MAX ||
            length > INT64_MAX || pad > INT64_MAX / 2 ||
            length > INT64_MAX - 2 * pad || k - 1 > (INT64_MAX - 1) / dilation) {
            return TRILLIM_ERROR;
        }
        size_t effective = dilation * (k - 1) + 1;
        size_t padded = length + 2 * pad;
        size_t rounding = p->ceil_mode ? stride - 1 : 0;
        if (padded > INT64_MAX - rounding || padded + rounding < effective) {
            return TRILLIM_ERROR;
        }
        size_t expected = (padded + rounding - effective) / stride + 1;
        if (p->ceil_mode && (expected - 1) * stride >= length + pad) {
            --expected;
        }
        if (expected != p->output_shape[axis] || expected - 1 > INT64_MAX / stride ||
            (expected - 1) * stride > INT64_MAX - effective) {
            return TRILLIM_ERROR;
        }
    }
    size_t span;
    if (!trillim_span(p->dimensions, p->input_shape, p->input_strides, &span) ||
        (p->channel_stride && p->channels - 1 > (SIZE_MAX - span) / p->channel_stride)) {
        return TRILLIM_ERROR;
    }
    span += (p->channels - 1) * p->channel_stride;
    if (span > SIZE_MAX / sizeof(float) || output > SIZE_MAX / sizeof(float) / p->channels) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * sizeof(float);
    outputs[0] = p->channels * output * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus generic(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t count = p->channels * task->output_volume;
    size_t first = tile * 128, end = count - first < 128 ? count : first + 128;
    for (size_t output = first; output < end; ++output) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t position = output % task->output_volume;
        int64_t origin[3];
        for (size_t axis = p->dimensions; axis-- > 0;) {
            origin[axis] = (int64_t)(position % p->output_shape[axis] * p->stride[axis]) - (int64_t)p->padding[axis];
            position /= p->output_shape[axis];
        }
        size_t base = output / task->output_volume * p->channel_stride, divisor = 0;
        float result = p->average ? 0.0f : -INFINITY;
        for (size_t k = 0; k < task->kernel_volume; ++k) {
            if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t kernel = k, offset = base;
            bool valid = true, declared = true;
            for (size_t axis = p->dimensions; axis-- > 0;) {
                int64_t index = origin[axis] + (int64_t)(kernel % p->kernel[axis] * p->dilation[axis]);
                kernel /= p->kernel[axis];
                valid = valid && index >= 0 && (uint64_t)index < p->input_shape[axis];
                declared = declared && index >= -(int64_t)p->padding[axis] &&
                    index < (int64_t)(p->input_shape[axis] + p->padding[axis]);
                if (index >= 0 && (uint64_t)index < p->input_shape[axis]) {
                    offset += (size_t)index * p->input_strides[axis];
                }
            }
            float value = valid ? task->input[offset] : (p->average ? 0.0f : -INFINITY);
            if (p->average) {
                result += value;
                divisor += p->count_include_pad ? declared : valid;
            } else {
                result = result >= value ? result : value;
            }
        }
        task->output[output] = p->average ? result / (float)divisor : result;
    }
    return TRILLIM_OK;
}

static float32x4_t maximum(float32x4_t a, float32x4_t b) {
    return vbslq_f32(vcgeq_f32(a, b), a, b);
}

static TrillimStatus horizontal(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t axis = p->dimensions - 1, columns = p->output_shape[axis], k = p->kernel[axis];
    size_t rows = p->channels * task->output_volume / columns;
    size_t first = tile * 128, end = rows - first < 128 ? rows : first + 128;
    size_t spatial_rows = task->output_volume / columns;
    for (size_t row = first; row < end; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t base = row / spatial_rows * p->channel_stride +
            trillim_offset(row % spatial_rows, axis, p->input_shape, p->input_strides);
        const float *input = task->input + base;
        float *output = task->output + row * columns;
        size_t column = 0;
        for (; columns - column >= 4; column += 4) {
            if ((column & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const float *source = input + column * k;
            float32x4_t result;
            if (k == 2) {
                float32x4x2_t values = vld2q_f32(source);
                result = p->average ? vmulq_n_f32(vaddq_f32(values.val[0], values.val[1]), 0.5f)
                                    : maximum(values.val[0], values.val[1]);
            } else {
                float32x4x3_t values = vld3q_f32(source);
                result = p->average ? vdivq_f32(vaddq_f32(vaddq_f32(values.val[0], values.val[1]), values.val[2]), vdupq_n_f32(3.0f))
                                    : maximum(maximum(values.val[0], values.val[1]), values.val[2]);
            }
            vst1q_f32(output + column, result);
        }
        for (; column < columns; ++column) {
            float result = input[column * k];
            for (size_t index = 1; index < k; ++index) {
                float value = input[column * k + index];
                result = p->average ? result + value : (result >= value ? result : value);
            }
            output[column] = p->average ? result / (float)k : result;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_pool(void *const *outputs, const void *const *inputs, void *scratch,
                                                 const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {inputs[0], outputs[0], p, context, 0, 0};
    if (!product(p->dimensions, p->output_shape, &task.output_volume) ||
        !product(p->dimensions, p->kernel, &task.kernel_volume)) {
        return TRILLIM_ERROR;
    }
    size_t axis = p->dimensions - 1, k = p->kernel[axis];
    bool fast = (k == 2 || k == 3) && p->input_strides[axis] == 1 &&
        p->stride[axis] == k && !p->padding[axis] && p->dilation[axis] == 1 &&
        p->input_shape[axis] % k == 0;
    for (size_t i = 0; i < axis; ++i) {
        fast = fast && p->kernel[i] == 1 && p->stride[i] == 1 && !p->padding[i] && p->dilation[i] == 1;
    }
    size_t count = p->channels * task.output_volume;
    if (fast) {
        count /= p->output_shape[axis];
        return context->thread_pool->run(context->thread_pool->context,
            count / 128 + (count % 128 != 0), horizontal, &task);
    }
    return context->thread_pool->run(context->thread_pool->context, count / 128 + (count % 128 != 0), generic, &task);
}

TRILLIM_KERNEL const char *trillim_neon_pool_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "def pattern(x: Tensor, *, kernel: tuple[int, ...], mode: str = 'max', stride = None, padding = 0, dilation = 1, ceil_mode: bool = False, count_include_pad: bool = True) -> tuple[Tensor, ...]:\n"
           "    return (Pool(x, kernel=kernel, mode=mode, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, count_include_pad=count_include_pad),)\n"
           "def bind(c):\n"
           "    x,o=c.inputs['x'],c.outputs[0]\n"
           "    k=c.attrs['kernel']; n=len(k)\n"
           "    if not 1<=n<=3 or any(not isinstance(v,int) or v<=0 for v in k):\n"
           "        raise ValueError('pooling requires one to three positive kernel dimensions')\n"
           "    if x.dtype!='fp32' or o.dtype!='fp32' or len(x.shape)!=n+1 or len(o.shape)!=n+1:\n"
           "        raise ValueError('pooling requires channel-first fp32 input/output')\n"
           "    if o.shape[0]!=x.shape[0] or (o.strides or _contiguous_strides(o.shape))!=_contiguous_strides(o.shape):\n"
           "        raise ValueError('pooling requires matching channels and contiguous output')\n"
           "    def axes(name,default):\n"
           "        v=c.attrs.get(name,default); v=default if v is None else v\n"
           "        v=(v,)*n if isinstance(v,int) else v\n"
           "        if not isinstance(v,tuple) or len(v)!=n or any(not isinstance(a,int) or a<(0 if name=='padding' else 1) for a in v):\n"
           "            raise ValueError('invalid pooling '+name)\n"
           "        return v\n"
           "    mode=c.attrs.get('mode','max')\n"
           "    if mode not in ('max','avg'):\n"
           "        raise ValueError('invalid pooling mode')\n"
           "    xs=x.strides or _contiguous_strides(x.shape)\n"
           "    return dict(dimensions=n,channels=x.shape[0],channel_stride=xs[0],input_shape=x.shape[1:],input_strides=xs[1:],output_shape=o.shape[1:],kernel=k,stride=axes('stride',k),padding=axes('padding',0),dilation=axes('dilation',1),average=int(mode=='avg'),ceil_mode=int(c.attrs.get('ceil_mode',False)),count_include_pad=int(c.attrs.get('count_include_pad',True)))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_pool_parameters(void) {
    static const TrillimParameter fields[] = {
        {"dimensions", "size_t", offsetof(Parameters, dimensions), NULL},
        {"channels", "size_t", offsetof(Parameters, channels), NULL},
        {"channel_stride", "size_t", offsetof(Parameters, channel_stride), NULL},
        {"input_shape", "size_t", offsetof(Parameters, input_shape), "dimensions"},
        {"input_strides", "size_t", offsetof(Parameters, input_strides), "dimensions"},
        {"output_shape", "size_t", offsetof(Parameters, output_shape), "dimensions"},
        {"kernel", "size_t", offsetof(Parameters, kernel), "dimensions"},
        {"stride", "size_t", offsetof(Parameters, stride), "dimensions"},
        {"padding", "size_t", offsetof(Parameters, padding), "dimensions"},
        {"dilation", "size_t", offsetof(Parameters, dilation), "dimensions"},
        {"average", "uint32_t", offsetof(Parameters, average), NULL},
        {"ceil_mode", "uint32_t", offsetof(Parameters, ceil_mode), NULL},
        {"count_include_pad", "uint32_t", offsetof(Parameters, count_include_pad), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields)/sizeof(fields[0]), fields,
        1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_pool_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED};
    return &caps;
}
