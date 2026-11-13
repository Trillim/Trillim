#include "common.h"
#include <limits.h>

#define TRANSPOSED 1
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
} Parameters;

typedef struct {
    void *output;
    const void *input, *weight, *bias;
    const Parameters *p;
    size_t input_volume, output_volume, kernel_volume;
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
#if BIASED
    inputs[2] = 0;
#endif
    if (!p->dimensions || p->dimensions > 3 || !p->channels || !p->groups ||
        p->channels % p->groups || p->outputs % p->groups ||
        !trillim_float(p->input_dtype) || !trillim_float(p->weight_dtype) ||
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
        !product(p->channels / p->groups, p->outputs, &count) ||
        !product(count, kernel_volume, &count) ||
        !product(count, trillim_itemsize(p->weight_dtype), &inputs[1]) ||
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
        if (!product(p->input_shape[axis] - 1, p->input_strides[axis], &extent) ||
            extent > SIZE_MAX - span) {
            return TRILLIM_ERROR;
        }
        span += extent;
    }
    if (span == SIZE_MAX || !product(span + 1, trillim_itemsize(p->input_dtype), &inputs[0])) {
        return TRILLIM_ERROR;
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

static void store(const Task *task, size_t output, size_t position, float sum) {
    const Parameters *p = task->p;
    uint32_t compute = p->input_dtype == p->weight_dtype ? p->input_dtype : TRILLIM_FP32;
    sum = rounded(sum, compute);
#if BIASED
    sum += trillim_load(task->bias, output, p->bias_dtype).f;
    sum = rounded(sum, compute == p->bias_dtype ? compute : TRILLIM_FP32);
#endif
    TrillimValue value = {.f = sum};
    trillim_store(task->output, output * task->output_volume + position, p->output_dtype, value);
}

#if USE_NEON
static float32x4_t load4(const void *data, size_t position, uint32_t dtype, size_t step) {
    if (step != 1) {
        float values[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            values[lane] = trillim_load(data, position + lane * step, dtype).f;
        }
        return vld1q_f32(values);
    }
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

static TrillimStatus convolve4(const Task *task, size_t output, size_t position, const size_t *out, size_t step) {
    const Parameters *p = task->p;
    size_t input_group = p->channels / p->groups, offset[3], indices[4];
    size_t group_base = output / (p->outputs / p->groups) * input_group;
    float32x4_t sum = vdupq_n_f32(0);
#if TRANSPOSED
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
                input = load4(task->input, base + indices[0], p->input_dtype, p->input_strides[p->dimensions - 1]);
            } else {
                float values[4];
                for (size_t lane = 0; lane < 4; ++lane) {
                    values[lane] = valid[lane] ? trillim_load(task->input, base + indices[lane], p->input_dtype).f : 0;
                }
                input = vld1q_f32(values);
            }
            float weight = trillim_load(task->weight, weight_position(task, output, channel, kernel), p->weight_dtype).f;
            partial = vfmaq_n_f32(partial, input, weight);
        }
        partial = vbslq_f32(vld1q_u32(valid), round4(partial, compute), vdupq_n_f32(0));
        sum = round4(vaddq_f32(sum, partial), compute);
    }
#else
    size_t cached_indices[64][4];
    unsigned cached_valid[64];
    bool cached = task->kernel_volume <= 64;
    if (cached) {
        for (size_t kernel = 0; kernel < task->kernel_volume; ++kernel) {
            coordinates(kernel, p->dimensions, p->kernel, offset);
            cached_valid[kernel] = 0;
            for (size_t lane = 0; lane < 4; ++lane) {
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
                for (size_t lane = 0; lane < 4; ++lane) {
                    if (source_position(p, out, offset, lane * step, &indices[lane])) {
                        valid |= 1u << lane;
                    }
                }
                source = indices;
            }
            size_t base = (group_base + channel) * p->channel_stride;
            float32x4_t values;
            size_t stride = p->stride[p->dimensions - 1];
            if (valid == 15 && (stride == 1 || (stride == 2 && source[3] + 1 < task->input_volume))) {
                values = load4(task->input, base + source[0], p->input_dtype, stride);
            } else {
                float lanes[4];
                for (size_t lane = 0; lane < 4; ++lane) {
                    lanes[lane] = (valid & (1u << lane)) ?
                        trillim_load(task->input, base + source[lane], p->input_dtype).f : 0;
                }
                values = vld1q_f32(lanes);
            }
            float weight = trillim_load(task->weight, weight_position(task, output, channel, kernel), p->weight_dtype).f;
            sum = vfmaq_n_f32(sum, values, weight);
        }
    }
#endif
    float values[4];
    vst1q_f32(values, sum);
    for (size_t lane = 0; lane < 4; ++lane) {
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
            float weight = trillim_load(task->weight, weight_position(task, output, channel, kernel), p->weight_dtype).f;
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
            float weight = trillim_load(task->weight, weight_position(task, output, channel, kernel), p->weight_dtype).f;
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
                    status = convolve4(task, output, position, out, step);
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
            TrillimStatus status = convolve4(task, output, position, out, 1);
            if (status != TRILLIM_OK) {
                return status;
            }
            position += 4;
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

TRILLIM_KERNEL TrillimStatus trillim_neon_conv_transpose_nd(void *const *outputs, const void *const *inputs, void *scratch,
                                                  const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], inputs[1], NULL, p, 0, 0, 0, context};
#if BIASED
    task.bias = inputs[2];
#endif
    if (!volume(p->dimensions, p->input_shape, &task.input_volume) ||
        !volume(p->dimensions, p->output_shape, &task.output_volume) ||
        !volume(p->dimensions, p->kernel, &task.kernel_volume)) {
        return TRILLIM_ERROR;
    }
    size_t tiles = task.output_volume / 64 + (task.output_volume % 64 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->outputs * tiles, convolve, &task);
}

TRILLIM_KERNEL const char *trillim_neon_conv_transpose_nd_tdl(void) {
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
           "    if x.dtype not in ('fp16','bf16','fp32'):\n"
           "        raise ValueError('convolution requires floating input')\n"
           "    for t in (w,o):\n"
           "        if t.dtype not in ('fp16','bf16','fp32') or (t.strides or _contiguous_strides(t.shape))!=_contiguous_strides(t.shape):\n"
           "            raise ValueError('convolution requires contiguous floating tensors')\n"
           "    xs=x.strides or _contiguous_strides(x.shape)\n"
           "    return dict(channel_stride=xs[0],input_strides=xs[1:],dimensions=n,channels=x.shape[0],outputs=out,groups=g,input_shape=x.shape[1:],output_shape=o.shape[1:],kernel=k,stride=s,padding=p,dilation=d,output_padding=e,input_dtype=_DTYPE_IDS[x.dtype],weight_dtype=_DTYPE_IDS[w.dtype],output_dtype=_DTYPE_IDS[o.dtype],bias_dtype=_DTYPE_IDS['fp32'])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_conv_transpose_nd_parameters(void) {
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
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        BIASED ? 3 : 2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_conv_transpose_nd_capabilities(void) {
    /* NEON is part of the AArch64 baseline. Optional instructions are not used. */
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &caps;
}
