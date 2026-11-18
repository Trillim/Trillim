#define TRILLIM_ENABLE_PACKED_NEON

#include "common.h"
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { TILE_SIZE = 16 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, output_dtype;
    uint32_t gate_dtype, up_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static bool packed(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    *scratch = outputs[0] = 0;
    memset(inputs, 0, 3 * sizeof(*inputs));
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) ||
        !packed(p->gate_dtype) ||
        !packed(p->up_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (!trillim_packed_payload_size(p->tokens, p->outputs, p->output_dtype, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->width, p->input_dtype, inputs)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->width, p->gate_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->width, p->up_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

static TrillimStatus dot_pair(
    const void *input, uint32_t input_dtype, size_t row,
    const void *first_weight, uint32_t first_dtype, const void *second_weight, uint32_t second_dtype,
    size_t column, size_t width, size_t outputs, const TrillimKernelContext *context,
    float *first_result, float *second_result
) {
    if (width == 0) {
        *first_result = *second_result = 0.0f;
        return TRILLIM_OK;
    }
    size_t x_start = row * width, w_start = column * width;
    TrillimPackedNeonRow first = trillim_packed_neon_row(first_weight, first_dtype, column, width, outputs);
    TrillimPackedNeonRow second = trillim_packed_neon_row(second_weight, second_dtype, column, width, outputs);
    size_t k = 0;
    float32x4_t a[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    float32x4_t b[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; width - k >= 16; k += 16) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t aw[4], bw[4];
        trillim_packed_neon_decode16(&first, k, aw);
        trillim_packed_neon_decode16(&second, k, bw);
        for (size_t i = 0; i < 4; ++i) {
            float32x4_t x = trillim_neon_load4(input, x_start + k + i * 4, input_dtype);
            a[i] = vaddq_f32(a[i], vmulq_f32(x, aw[i]));
            b[i] = vaddq_f32(b[i], vmulq_f32(x, bw[i]));
        }
    }
    float first_sum = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3])));
    float second_sum = vaddvq_f32(vaddq_f32(vaddq_f32(b[0], b[2]), vaddq_f32(b[1], b[3])));
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        first_sum += x * trillim_packed_read_float(first_weight, w_start + k, first_dtype, outputs, width);
        second_sum += x * trillim_packed_read_float(second_weight, w_start + k, second_dtype, outputs, width);
    }
    *first_result = first_sum;
    *second_result = second_sum;
    return TRILLIM_OK;
}

static float silu(float x) {
    float e = expf(-fabsf(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * sigmoid;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->outputs;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->outputs, column = index % p->outputs;
        float value, up;
        TrillimStatus status = dot_pair(
            task->inputs[0], p->input_dtype, row,
            task->inputs[1], p->gate_dtype, task->inputs[2], p->up_dtype,
            column, p->width, p->outputs, task->context, &value, &up
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        value = silu(value) * up;
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_glu(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[3], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    (void)scratch;
    Task task = {outputs[0], inputs, p, context};
    size_t count = p->tokens * p->outputs;
    size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
    return tiles == 1 ? project(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_glu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'gate_weight', 'up_weight')\n"
           "\n"
           "def bind(context):\n"
           "    x, gate_weight, up_weight = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = gate_weight, up_weight\n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if gate_weight.dtype not in packed or up_weight.dtype not in packed:\n"
           "        raise ValueError('packed GLU requires compressed gate and up weights')\n"
           "    if gate_weight.shape != up_weight.shape or x.shape[-1] != gate_weight.shape[1]:\n"
           "        raise ValueError('incompatible packed GLU dimensions')\n"
           "    if output.shape != (*x.shape[:-1], gate_weight.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=gate_weight.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        gate_dtype=_DTYPE_IDS[gate_weight.dtype],\n"
           "        up_dtype=_DTYPE_IDS[up_weight.dtype],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, gate_weight: Tensor, up_weight: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    if dtype != 'fp32':\n"
           "        return (SiluMul(Linear(x, gate_weight), Linear(x, up_weight), dtype=dtype),)\n"
           "    result = Mul(Silu(Linear(x, gate_weight)), Linear(x, up_weight))\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_glu_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        3, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_glu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
    return &caps;
}
