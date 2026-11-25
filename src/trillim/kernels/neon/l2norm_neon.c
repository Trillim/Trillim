#include "common.h"
#include <math.h>
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#define ROWS_PER_TASK 8

typedef struct {
    size_t rank, rows, width;
    const size_t *shape, *strides;
    uint32_t dtype;
    float eps;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t span;
    if (!trillim_float(p->dtype) || !p->rank || !p->width ||
        !isfinite(p->eps) || p->eps < 0.0f ||
        !trillim_span(p->rank, p->shape, p->strides, &span) ||
        p->width != p->shape[p->rank - 1]) {
        return TRILLIM_ERROR;
    }
    size_t itemsize = trillim_itemsize(p->dtype);
    if (span > SIZE_MAX / itemsize || p->rows > SIZE_MAX / itemsize / p->width) {
        return TRILLIM_ERROR;
    }
    size_t rows = 1;
    for (size_t axis = 0; axis < p->rank - 1; ++axis) {
        size_t dim = p->shape[axis];
        if (dim && rows > SIZE_MAX / dim) {
            return TRILLIM_ERROR;
        }
        rows *= dim;
    }
    if (p->rows != rows) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * itemsize;
    outputs[0] = p->rows * p->width * itemsize;
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

static float32x4_t load_row4(const void *data, size_t index, size_t stride, uint32_t dtype) {
    if (stride == 1) {
        return load4(data, index, dtype);
    }
    if (stride == 0) {
        return vdupq_n_f32(trillim_load(data, index, dtype).f);
    }
    float lanes[4];
    for (size_t lane = 0; lane < 4; ++lane) {
        lanes[lane] = trillim_load(data, index + lane * stride, dtype).f;
    }
    return vld1q_f32(lanes);
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

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float32x4_t round4(float32x4_t value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vcvt_f16_f32(value));
    }
    if (dtype == TRILLIM_BF16) {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        return vreinterpretq_f32_u32(vshlq_n_u32(rounded, 16));
    }
    return value;
}

static TrillimStatus normalize_row(const Task *task, size_t row) {
    const Parameters *p = task->p;
    const TrillimKernelContext *context = task->context;
    size_t start = trillim_offset(row, p->rank - 1, p->shape, p->strides);
    size_t output_start = row * p->width;
    size_t stride = p->strides[p->rank - 1];
    size_t column = 0;
    float sum = 0.0f;
    /* Match the primitive Mul and the left-to-right Sum. */
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t x = load_row4(task->input, start + column * stride, stride, p->dtype);
        float32x4_t squared = round4(vmulq_f32(x, x), p->dtype);
        sum += vgetq_lane_f32(squared, 0);
        sum += vgetq_lane_f32(squared, 1);
        sum += vgetq_lane_f32(squared, 2);
        sum += vgetq_lane_f32(squared, 3);
    }
    for (; column < p->width; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(task->input, start + column * stride, p->dtype).f;
        sum += round_dtype(x * x, p->dtype);
    }
    sum = round_dtype(sum, p->dtype);
    float added = round_dtype(sum + round_dtype(p->eps, p->dtype), p->dtype);
    float denominator = round_dtype(sqrtf(added), p->dtype);
    float32x4_t denominator4 = vdupq_n_f32(denominator);
    column = 0;
    for (; stride == 1 && p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store4(task->output, output_start + column, p->dtype,
               vdivq_f32(load4(task->input, start + column, p->dtype), denominator4));
    }
    for (; column < p->width; ++column) {
        if ((column & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        TrillimValue value = {
            .f = trillim_load(task->input, start + column * stride, p->dtype).f / denominator
        };
        trillim_store(task->output, output_start + column, p->dtype, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_rows(const Task *task, size_t start, size_t count) {
    for (size_t offset = 0; offset < count; ++offset) {
        TrillimStatus status = normalize_row(task, start + offset);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t start = index * ROWS_PER_TASK;
    size_t count = task->p->rows - start;
    return normalize_rows(task, start, count < ROWS_PER_TASK ? count : ROWS_PER_TASK);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_l2norm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    Task task = {outputs[0], inputs[0], p, context};
    if (p->rows <= 1 || p->rows <= 4096 / p->width) {
        return normalize_rows(&task, 0, p->rows);
    }
    size_t tasks = (p->rows - 1) / ROWS_PER_TASK + 1;
    return context->thread_pool->run(context->thread_pool->context, tasks,
                                     normalize_task, &task);
}

TRILLIM_KERNEL const char *trillim_neon_l2norm_tdl(void) {
    return "from trillim.tdl import *\n"
           "from struct import pack, unpack\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _native_dtype, _product\n"
           "def bind(context):\n"
           "    x, output = context.inputs['x'], context.outputs[0]\n"
           "    axis = context.attrs.get('axis', -1)\n"
           "    if not x.shape or not isinstance(axis, int) or axis % len(x.shape) != len(x.shape) - 1:\n"
           "        raise ValueError('L2Norm requires the final axis')\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != x.dtype or output.shape != x.shape:\n"
           "        raise ValueError('L2Norm requires matching floating input and output')\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    eps = context.attrs.get('eps', 1e-6)\n"
           "    if not isinstance(eps, (int, float)):\n"
           "        raise ValueError('invalid L2Norm epsilon')\n"
           "    # Round the literal directly to FP16 before packing the float field.\n"
           "    if x.dtype == 'fp16':\n"
           "        try:\n"
           "            eps = unpack('e', pack('e', eps))[0]\n"
           "        except OverflowError:\n"
           "            pass\n"
           "    return {'rows': _product(x.shape[:-1]), 'width': x.shape[-1], 'rank': len(x.shape), 'shape': x.shape, 'strides': strides, 'dtype': _native_dtype(x.dtype), 'eps': float(eps)}\n"
           "def pattern(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> tuple[Tensor, ...]:\n"
           "    return (L2Norm(x, axis=axis, eps=eps),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const Parameters *p = parameters;
    return output == 0 && input == 0 &&
        (!p || trillim_contiguous(p->rank, p->shape, p->strides));
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_l2norm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL}
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_l2norm_capabilities(void) {
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
