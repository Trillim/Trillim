#include "common.h"
#include <arm_neon.h>

#define ROWS_PER_TASK 8
#if defined(__APPLE__)
#define HADAMARD_CAPS 0
#else
#define HADAMARD_CAPS TRILLIM_CAP_NEON
#endif

typedef struct { size_t rows, row_stride; } Parameters;
typedef struct {
    const float *input;
    float *output;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->rows == 0) return TRILLIM_OK;
    size_t shape[] = {p->rows, 1024}, strides[] = {p->row_stride, 1}, span;
    if (!trillim_span(2, shape, strides, &span) || span > SIZE_MAX / sizeof(float) ||
        p->rows > SIZE_MAX / 1024 / sizeof(float)) return TRILLIM_ERROR;
    inputs[0] = span * sizeof(float);
    outputs[0] = p->rows * 1024 * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus transform(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t first = tile * ROWS_PER_TASK;
    size_t count = task->p->rows - first;
    if (count > ROWS_PER_TASK) count = ROWS_PER_TASK;
    for (size_t row = first; row < first + count; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        const float *x = task->input + row * task->p->row_stride;
        float *y = task->output + row * 1024;
        for (size_t lane = 0; lane < 1024; lane += 4) {
            float32x4_t input = vld1q_f32(x + lane);
            float32x4_t even = vtrn1q_f32(input, input);
            float32x4_t odd = vtrn2q_f32(input, input);
            float32x4_t pairs = vtrn1q_f32(vaddq_f32(even, odd), vsubq_f32(even, odd));
            float32x2_t a = vget_low_f32(pairs);
            float32x2_t b = vget_high_f32(pairs);
            vst1q_f32(y + lane, vcombine_f32(vadd_f32(a, b), vsub_f32(a, b)));
        }
        for (size_t half = 4; half < 1024; half *= 4) {
            if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            for (size_t group = 0; group < 1024; group += half * 4) {
                for (size_t lane = group; lane < group + half; lane += 4) {
                    float32x4_t a = vld1q_f32(y + lane);
                    float32x4_t b = vld1q_f32(y + lane + half);
                    float32x4_t c = vld1q_f32(y + lane + 2 * half);
                    float32x4_t d = vld1q_f32(y + lane + 3 * half);
                    float32x4_t ab_sum = vaddq_f32(a, b), ab_difference = vsubq_f32(a, b);
                    float32x4_t cd_sum = vaddq_f32(c, d), cd_difference = vsubq_f32(c, d);
                    float32x4_t y0 = vaddq_f32(ab_sum, cd_sum), y1 = vaddq_f32(ab_difference, cd_difference);
                    float32x4_t y2 = vsubq_f32(ab_sum, cd_sum), y3 = vsubq_f32(ab_difference, cd_difference);
                    if (half == 256) {
                        y0 = vmulq_n_f32(y0, 0.03125f);
                        y1 = vmulq_n_f32(y1, 0.03125f);
                        y2 = vmulq_n_f32(y2, 0.03125f);
                        y3 = vmulq_n_f32(y3, 0.03125f);
                    }
                    vst1q_f32(y + lane, y0);
                    vst1q_f32(y + lane + half, y1);
                    vst1q_f32(y + lane + 2 * half, y2);
                    vst1q_f32(y + lane + 3 * half, y3);
                }
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_hadamard_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (p->rows == 0) return TRILLIM_OK;
    Task task = {inputs[0], outputs[0], p, context};
    size_t tiles = p->rows / ROWS_PER_TASK + (p->rows % ROWS_PER_TASK != 0);
    return tiles == 1 ? transform(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, transform, &task);
}

TRILLIM_KERNEL const char *trillim_neon_hadamard_matmul_tdl(void) {
    return "from trillim.tdl import Tensor, Constant, Matmul, Mul\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "import numpy as np\n"
           "_H = np.ones((1, 1), dtype=np.float32)\n"
           "while _H.shape[0] < 1024:\n"
           "    _H = np.block([[_H, _H], [_H, -_H]])\n"
           "def pattern(a: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Mul(Matmul(a, Constant(_H)), 1.0 / 32),)\n"
           "def bind(context):\n"
           "    x, y = context.inputs['a'], context.outputs[0]\n"
           "    one = _constant_expression(1)\n"
           "    width = _constant_expression(1024)\n"
           "    if x.dtype != 'fp32' or y.dtype != 'fp32' or len(x.shape) != 2 or x.shape[1] != width or y.shape != x.shape:\n"
           "        raise ValueError('requires FP32 normalized Sylvester-1024 multiplication')\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    if strides[-1] != one:\n"
           "        raise ValueError('requires contiguous transform blocks')\n"
           "    return dict(rows=x.shape[0], row_stride=strides[0])\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const Parameters *p = parameters;
    return output == 0 && input == 0 && (!p || p->rows <= 1 || p->row_stride == 1024);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_hadamard_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"row_stride", "size_t", offsetof(Parameters, row_stride), NULL},
    };
    static const size_t inputs[] = {4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 2, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_hadamard_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, HADAMARD_CAPS};
    return &capabilities;
}
