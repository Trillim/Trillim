#define TRILLIM_ENABLE_PACKED_AVX2

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

__attribute__((target("avx2"))) static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    const uint8_t *bytes = data;
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)(bytes + index * sizeof(float)));
    }
    const uint8_t *narrow = bytes + index * sizeof(uint16_t);
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)narrow);
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    uint16_t bits[8];
    float values[8];
    memcpy(bits, narrow, sizeof(bits));
    for (size_t lane = 0; lane < 8; ++lane) {
        values[lane] = trillim_fp16(bits[lane]);
    }
    return _mm256_loadu_ps(values);
}

__attribute__((target("avx2"))) static TrillimStatus dot(
    const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, const TrillimKernelContext *context, float *result
) {
    size_t x_start = row * width;
    size_t w_start = column * width;
    size_t k = 0;
    float sum = 0.0f;
    __m256 sums = _mm256_setzero_ps();
    for (; width - k >= 8; k += 8) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 x = load8(input, x_start + k, input_dtype);
        __m256 w = trillim_packed_decode8(weight, outputs, width, weight_dtype, column, k);
        sums = _mm256_add_ps(sums, _mm256_mul_ps(x, w));
    }
    __m128 partial = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    partial = _mm_hadd_ps(partial, partial);
    sum = _mm_cvtss_f32(_mm_hadd_ps(partial, partial));
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        float w = trillim_packed_read_float(weight, w_start + k, weight_dtype, outputs, width);
        sum += x * w;
    }
    *result = sum;
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
        float value;
        TrillimStatus status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[1], p->gate_dtype,
            column, p->width, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        float up;
        status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[2], p->up_dtype,
            column, p->width, p->outputs, task->context, &up
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

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_glu(
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

TRILLIM_KERNEL const char *trillim_x86_packed_glu_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_glu_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_glu_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
