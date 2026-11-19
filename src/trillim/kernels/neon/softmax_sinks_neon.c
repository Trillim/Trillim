#include "common.h"
#include <math.h>
#include <arm_neon.h>

typedef struct {
    size_t rows, width, rank;
    const size_t *shape, *sink_strides;
    uint32_t dtype, sink_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input, *sinks;
    float *scratch;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t rows_per_task;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    size_t span;
    if (!trillim_float(p->dtype) || !trillim_float(p->sink_dtype) ||
        !trillim_float(p->output_dtype) || !p->width ||
        p->rows > SIZE_MAX / sizeof(float) / p->width ||
        !trillim_span(p->rank, p->shape, p->sink_strides, &span) ||
        span > SIZE_MAX / trillim_itemsize(p->sink_dtype)) {
        return TRILLIM_ERROR;
    }
    size_t rows = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] && rows > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        rows *= p->shape[axis];
    }
    if (rows != p->rows) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * trillim_itemsize(p->dtype);
    inputs[1] = span * trillim_itemsize(p->sink_dtype);
    outputs[0] = p->rows * p->width * trillim_itemsize(p->output_dtype);
    *scratch = p->rows * p->width * sizeof(float);
    return TRILLIM_OK;
}

static float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
    } else if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index,
                 vreinterpret_u16_f16(vcvt_f16_f32(value)));
    } else {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
    }
}


static float32x4_t exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        if (vminvq_u32(vceqq_f32(x, vdupq_n_f32(-INFINITY))) == UINT32_MAX) {
            return vdupq_n_f32(0.0f);
        }
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = lanes[lane] == -INFINITY ? 0.0f : expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static TrillimStatus softmax_precise(const Task *task, size_t row) {
    const Parameters *p = task->p;
    size_t start = row * p->width;
    float maximum = -INFINITY;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float value = trillim_load(task->input, start + column, p->dtype).f;
        maximum = maximum >= value ? maximum : value;
    }
    float sink = trillim_load(task->sinks,
                              trillim_offset(row, p->rank, p->shape, p->sink_strides),
                              p->sink_dtype).f;
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float difference = round_dtype(trillim_load(task->input, start + column, p->dtype).f - maximum, p->output_dtype);
        float numerator = round_dtype(expf(difference), p->output_dtype);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    sum = round_dtype(sum, p->output_dtype);
    float sink_difference = round_dtype(sink - maximum, p->output_dtype);
    float sink_numerator = round_dtype(expf(sink_difference), p->output_dtype);
    sum = round_dtype(sum + sink_numerator, p->output_dtype);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}


static TrillimStatus softmax_row(const Task *task, size_t row) {
    const Parameters *p = task->p;
    if (p->dtype != TRILLIM_FP32) {
        /* Preserve low-precision rounding at scalar-reference boundaries. */
        return softmax_precise(task, row);
    }
    size_t start = row * p->width, column = 0;
    float32x4_t maximum4 = vdupq_n_f32(-INFINITY);
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum4 = vmaxq_f32(maximum4, load4(task->input, start + column, p->dtype));
    }
    float maximum = vmaxvq_f32(maximum4);
    if (isnan(maximum)) {
        return softmax_precise(task, row);
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, p->dtype).f;
        if (isnan(value)) {
            return softmax_precise(task, row);
        }
        maximum = maximum >= value ? maximum : value;
    }
    float sink = trillim_load(task->sinks,
                              trillim_offset(row, p->rank, p->shape, p->sink_strides),
                              p->sink_dtype).f;
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float32x4_t sum4 = vdupq_n_f32(0.0f), maximum_vector = vdupq_n_f32(maximum);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t numerator = exp4(vsubq_f32(load4(task->input, start + column, p->dtype),
                                              maximum_vector));
        vst1q_f32(task->scratch + start + column, numerator);
        sum4 = vaddq_f32(sum4, numerator);
    }
    float sum = vaddvq_f32(sum4) + expf(sink - maximum);
    for (; column < p->width; ++column) {
        float numerator = expf(trillim_load(task->input, start + column, p->dtype).f - maximum);
        task->scratch[start + column] = numerator;
        sum += numerator;
    }
    float denominator = sum > 0.0f ? sum : 1.0f;
    float32x4_t denominator_vector = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, start + column, p->output_dtype,
               vdivq_f32(vld1q_f32(task->scratch + start + column), denominator_vector));
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = task->scratch[start + column] / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus softmax_tile(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t first = tile * task->rows_per_task;
    size_t count = task->p->rows - first;
    if (count > task->rows_per_task) {
        count = task->rows_per_task;
    }
    for (size_t row = first; row < first + count; ++row) {
        TrillimStatus status = softmax_row(task, row);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_normalize_softmax_sinks(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], inputs[1], scratch, p, context, 1};
    if (p->rows == 1) {
        return softmax_row(&task, 0);
    }
    /* Batch short rows while leaving enough tasks to occupy the thread pool. */
    size_t rows_per_task = p->width <= 1024 ? 1024 / p->width : 1;
    size_t rows_per_worker = p->rows / context->thread_pool->worker_count;
    if (rows_per_worker != 0 && rows_per_worker < rows_per_task) {
        rows_per_task = rows_per_worker;
    }
    task.rows_per_task = rows_per_task;
    size_t tiles = p->rows / rows_per_task + (p->rows % rows_per_task != 0);
    return context->thread_pool->run(context->thread_pool->context, tiles, softmax_tile, &task);
}

TRILLIM_KERNEL const char *trillim_neon_normalize_softmax_sinks_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _native_dtype, _product\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "\n"
           "def bind(context):\n"
           "    x, sinks, output = context.inputs['x'], context.inputs['sinks'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('Softmax requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or sinks.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'} or output.shape != x.shape:\n"
           "        raise ValueError('Softmax requires floating tensors')\n"
           "    if x.strides and x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('Softmax requires contiguous input')\n"
           "    shape = x.shape[:-1] + (_constant_expression(1),)\n"
           "    strides = _broadcast_strides(sinks.shape, shape, sinks.strides)\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'rank': len(shape) - 1, 'shape': shape[:-1], 'sink_strides': strides[:-1], 'dtype': _native_dtype(x.dtype), 'sink_dtype': _native_dtype(sinks.dtype), 'output_dtype': _native_dtype(output.dtype)}\n"
           "\n"
           "def pattern(x: Tensor, sinks: Tensor, *, axis: int = -1) -> tuple[Tensor, ...]:\n"
           "    return (Softmax(x, axis=axis, sinks=sinks),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_normalize_softmax_sinks_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"sink_strides", "size_t", offsetof(Parameters, sink_strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"sink_dtype", "uint32_t", offsetof(Parameters, sink_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 8, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_normalize_softmax_sinks_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        /* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
        TRILLIM_CAP_LIBM
#else
        TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM
#endif
    };
    return &capabilities;
}
