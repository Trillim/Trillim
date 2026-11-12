#include "common.h"
#include <arm_neon.h>
#if defined(__APPLE__)
#define HADAMARD_CAPS 0
#else
#define HADAMARD_CAPS TRILLIM_CAP_NEON
#endif
/* Every butterfly stage is a separate rounded primitive operation. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t rows, width, row_stride, storage_rows, storage_columns, block_size;
    float scale;
} Parameters;

typedef struct {
    float *output;
    const void *x, *signs;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!p->block_size || (p->block_size & (p->block_size - 1)) ||
        p->width % p->block_size) {
        return TRILLIM_ERROR;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    if (p->width > SIZE_MAX / sizeof(float) ||
        p->rows > SIZE_MAX / sizeof(float) / p->width ||
        p->storage_rows > SIZE_MAX / (p->storage_columns ? p->storage_columns : 1) ||
        (p->row_stride && p->rows - 1 > (SIZE_MAX - p->width) / p->row_stride)) {
        return TRILLIM_ERROR;
    }
    size_t span = (p->rows - 1) * p->row_stride + p->width;
    if (span > SIZE_MAX / sizeof(float) || span > p->storage_rows * p->storage_columns) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * sizeof(float);
    inputs[1] = p->width * sizeof(float);
    outputs[0] = p->rows * p->width * sizeof(float);
    return TRILLIM_OK;
}

static bool cancelled(const Task *task) {
    return task->context->is_cancelled(task->context->cancellation_context);
}

static float32x4_t butterfly4(float32x4_t x) {
    float32x4_t a = vuzp1q_f32(x, x), b = vuzp2q_f32(x, x);
    float32x4_t pairs = vzip1q_f32(vaddq_f32(a, b), vsubq_f32(a, b));
    float32x2_t left = vget_low_f32(pairs), right = vget_high_f32(pairs);
    return vcombine_f32(vadd_f32(left, right), vsub_f32(left, right));
}

static TrillimStatus transform(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t blocks = p->width / p->block_size;
    size_t row = index / blocks, column = index % blocks * p->block_size;
    size_t source = row * p->row_stride + column;
    float *y = task->output + row * p->width + column;
    if (cancelled(task)) {
        return TRILLIM_CANCELLED;
    }
    size_t copied = 0;
    if (p->block_size >= 8) {
        const float *x = (const float *)task->x + source;
        const float *signs = (const float *)task->signs + column;
        for (; copied < p->block_size; copied += 8) {
            if ((copied & 255) == 0 && cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a = butterfly4(vmulq_f32(vld1q_f32(x + copied), vld1q_f32(signs + copied)));
            float32x4_t b = butterfly4(vmulq_f32(vld1q_f32(x + copied + 4), vld1q_f32(signs + copied + 4)));
            vst1q_f32(y + copied, vaddq_f32(a, b));
            vst1q_f32(y + copied + 4, vsubq_f32(a, b));
        }
    }
    for (; copied < p->block_size; ++copied) {
        float x = trillim_load(task->x, source + copied, TRILLIM_FP32).f;
        float sign = trillim_load(task->signs, column + copied, TRILLIM_FP32).f;
        y[copied] = x * sign;
    }
    for (size_t half = (p->block_size >= 8 ? 8 : 1); half < p->block_size; half *= 2) {
        for (size_t group = 0; group < p->block_size; group += 2 * half) {
            size_t lane = 0;
            for (; lane + 4 <= half; lane += 4) {
                if (((group / 2 + lane) & 255) == 0 && cancelled(task)) {
                    return TRILLIM_CANCELLED;
                }
                float32x4_t a = vld1q_f32(y + group + lane);
                float32x4_t b = vld1q_f32(y + group + half + lane);
                vst1q_f32(y + group + lane, vaddq_f32(a, b));
                vst1q_f32(y + group + half + lane, vsubq_f32(a, b));
            }
            for (; lane < half; ++lane) {
                if (((group / 2 + lane) & 255) == 0 && cancelled(task)) {
                    return TRILLIM_CANCELLED;
                }
                float a = y[group + lane], b = y[group + half + lane];
                y[group + lane] = a + b;
                y[group + half + lane] = a - b;
            }
        }
    }
    size_t lane = 0;
    float32x4_t scale = vdupq_n_f32(p->scale);
    for (; lane + 4 <= p->block_size; lane += 4) {
        if ((lane & 255) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        vst1q_f32(y + lane, vmulq_f32(vld1q_f32(y + lane), scale));
    }
    for (; lane < p->block_size; ++lane) {
        if ((lane & 255) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        y[lane] *= p->scale;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_hadamard(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows || !p->width) {
        return TRILLIM_OK;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t count = p->rows * (p->width / p->block_size);
    return count == 1 ? transform(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, count, transform, &task);
}

TRILLIM_KERNEL const char *trillim_neon_hadamard_tdl(void) {
    return "\"\"\"Faithful signed normalized Sylvester transform for kernel replay.\"\"\"\n"
           "from __future__ import annotations\n"
           "import math\n"
           "from trillim.tdl import Add, Constant, Construct, Mul, Reshape, Dim, Stack, Sub, Take, Tensor\n"
           "\n"
           "@Construct\n"
           "def SignedHadamard(x, signs, *, block_size):\n"
           "    if block_size <= 0 or block_size & (block_size - 1):\n"
           "        raise ValueError(\"Hadamard block size must be a positive power of two\")\n"
           "    value = Reshape(Mul(x, signs), (-1, block_size))\n"
           "    half = 1\n"
           "    while half < block_size:\n"
           "        pairs = Reshape(value, (-1, 2, half))\n"
           "        left = Take(pairs, Constant(0), axis=1, rank=3, index_rank=0)\n"
           "        right = Take(pairs, Constant(1), axis=1, rank=3, index_rank=0)\n"
           "        value = Reshape(Stack((Add(left, right), Sub(left, right)), axis=1, rank=2), (-1, block_size))\n"
           "        half *= 2\n"
           "    return Mul(Reshape(value, (Dim(x, 0), Dim(x, 1))), 1.0 / math.sqrt(block_size))\n"
           "\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _storage_shape\n"
           "from trillim.compiler.passes.infer import _constant_expression\n"
           "\n"
           "def bind(context):\n"
           "    x, signs, output = context.inputs['x'], context.inputs['signs'], context.outputs[0]\n"
           "    block = context.attrs.get('block_size')\n"
           "    if not isinstance(block, int) or isinstance(block, bool) or block <= 0 or block & (block - 1):\n"
           "        raise ValueError('Hadamard requires a positive power-of-two block')\n"
           "    if len(x.shape) != 2 or x.dtype != 'fp32' or signs.dtype != 'fp32' or output.dtype != 'fp32' or output.shape != x.shape:\n"
           "        raise ValueError('Hadamard requires matching FP32 activations and output')\n"
           "    if signs.shape != x.shape[-1:]:\n"
           "        raise ValueError('Hadamard requires a full-channel sign vector')\n"
           "    one = _constant_expression(1)\n"
           "    if (signs.strides or (one,)) != (one,):\n"
           "        raise ValueError('Hadamard signs must be contiguous')\n"
           "    rows, width = x.shape\n"
           "    strides = x.strides or _contiguous_strides(x.shape)\n"
           "    if strides[-1] != one and width != one:\n"
           "        raise ValueError('Hadamard requires contiguous channels')\n"
           "    storage_rows, storage_columns = _storage_shape(x)\n"
           "    return {'rows': rows, 'width': width, 'row_stride': strides[0], 'storage_rows': storage_rows, 'storage_columns': storage_columns, 'block_size': block, 'scale': 1.0 / math.sqrt(block)}\n"
           "\n"
           "def pattern(x: Tensor, signs: Tensor, *, block_size: int) -> tuple[Tensor, ...]:\n"
           "    return (SignedHadamard(x, signs, block_size=block_size),)\n";
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    const Parameters *p = parameters;
    return output == 0 && input == 0 &&
        (!p || p->rows <= 1 || p->row_stride == p->width);
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_hadamard_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"row_stride", "size_t", offsetof(Parameters, row_stride), NULL},
        {"storage_rows", "size_t", offsetof(Parameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(Parameters, storage_columns), NULL},
        {"block_size", "size_t", offsetof(Parameters, block_size), NULL},
        {"scale", "float", offsetof(Parameters, scale), NULL},
    };
    static const size_t inputs[] = {4, 4}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 7, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_hadamard_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, HADAMARD_CAPS};
    return &capabilities;
}
