#include "common.h"
#include <math.h>
#if !defined(_WIN32)
#include <fenv.h>
#include <stdlib.h>
#endif
#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { TILE_SIZE = 16 };

typedef struct {
    size_t tokens, width, outputs;
    size_t hidden_width;
    uint32_t input_dtype, output_dtype;
    uint32_t gate_dtype, up_dtype, down_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static bool packed(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    *scratch = outputs[0] = 0;
    memset(inputs, 0, 4 * sizeof(*inputs));
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) ||
        !packed(p->gate_dtype) ||
        !packed(p->up_dtype) ||
        !packed(p->down_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (!trillim_packed_payload_size(p->tokens, p->outputs, p->output_dtype, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->width, p->input_dtype, inputs)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->up_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->gate_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    if (p->hidden_width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->hidden_width, p->down_dtype, inputs + 3)) {
        return TRILLIM_ERROR;
    }
    if (!trillim_packed_payload_size(p->tokens, p->hidden_width, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    return TRILLIM_OK;
}

static TrillimStatus dot(
    const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, const TrillimKernelContext *context, float *result
) {
    size_t x_start = row * width;
    size_t w_start = column * width;
    size_t k = 0;
    float sum = 0.0f;
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

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->hidden_width;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->hidden_width, column = index % p->hidden_width;
        float value;
        TrillimStatus status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[1], p->up_dtype,
            column, p->width, p->hidden_width, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        float gate;
        status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[2], p->gate_dtype,
            column, p->width, p->hidden_width, task->context, &gate
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        ((float *)task->hidden)[index] = value * silu(gate);
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
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
            task->hidden, TRILLIM_FP32, row, task->inputs[3], p->down_dtype,
            column, p->hidden_width, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

#if !defined(_WIN32)
/* A build containing only this source can continue using the scalar FFN. */
#if !defined(__APPLE__)
__attribute__((weak)) extern TrillimStatus trillim_scalar_block_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
#endif

static TrillimKernel linked_projection(void) {
#if defined(__APPLE__)
#define TRILLIM_KERNEL_NAME(symbol) #symbol
    void *symbol = dlsym(RTLD_SELF, TRILLIM_KERNEL_NAME(trillim_scalar_block_linear));
#undef TRILLIM_KERNEL_NAME
    TrillimKernel kernel;
    memcpy(&kernel, &symbol, sizeof(kernel));
    return kernel;
#else
    return trillim_scalar_block_linear;
#endif
}

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

static size_t quantized_group_width(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5:
    case TRILLIM_GRP_INT8:
        return 32;
    case TRILLIM_DUBGRP_INT6:
        return 256;
    default:
        return 0;
    }
}

static TrillimStatus check_activation(
    const void *input, uint32_t dtype, size_t width, size_t group,
    const TrillimKernelContext *context, bool *ordinary
) {
    *ordinary = false;
    for (size_t start = 0; start < width; start += group) {
        if (start % 1024 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float peak = 0.0f;
        for (size_t i = 0; i < group; ++i) {
            float value = trillim_load(input, start + i, dtype).f;
            if (!isfinite(value)) {
                return TRILLIM_OK;
            }
            if (fabsf(value) > peak) {
                peak = fabsf(value);
            }
        }
        if (peak != 0.0f) {
            if (group == 256) {
                if (peak > 0x1p40f || !isfinite(127.0f / peak)) {
                    return TRILLIM_OK;
                }
            } else {
                float step = peak / 127.0f;
                if (step > 65504.0f || trillim_to_fp16(step) == 0) {
                    return TRILLIM_OK;
                }
            }
        }
    }
    *ordinary = true;
    return TRILLIM_OK;
}

static TrillimStatus try_int8_ffn(const Task *task, bool *handled) {
    *handled = false;
    const Parameters *p = task->p;
    if (p->tokens != 1 ||
        task->context->thread_pool->worker_count == 0 ||
        p->width == 0 || p->width > 65536 || p->hidden_width < 16 || p->hidden_width > 65536 ||
        p->outputs < 16 || p->output_dtype != TRILLIM_FP32 || p->up_dtype != p->gate_dtype ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    TrillimKernel projection = linked_projection();
    if (projection == NULL) {
        return TRILLIM_OK;
    }
    size_t up_group = quantized_group_width(p->up_dtype);
    size_t down_group = quantized_group_width(p->down_dtype);
    if (up_group == 0 || down_group == 0 || p->width % up_group || p->hidden_width % down_group) {
        return TRILLIM_OK;
    }
    const TrillimKernelContext *context = task->context;
    bool ordinary;
    TrillimStatus status = check_activation(task->inputs[0], p->input_dtype, p->width, up_group, context, &ordinary);
    if (status != TRILLIM_OK || !ordinary) {
        return status;
    }
    float *gate = malloc(p->hidden_width * sizeof(*gate));
    if (gate == NULL) {
        return TRILLIM_OK;
    }
    LinearParameters expand = {1, p->width, p->hidden_width, p->input_dtype, p->up_dtype, TRILLIM_FP32};
    void *up_outputs[] = {task->hidden};
    status = projection(up_outputs, task->inputs, NULL, &expand, context);
    if (status == TRILLIM_OK) {
        const void *gate_inputs[] = {task->inputs[0], task->inputs[2]};
        void *gate_outputs[] = {gate};
        status = projection(gate_outputs, gate_inputs, NULL, &expand, context);
    }
    if (status == TRILLIM_OK) {
        float *hidden = task->hidden;
        for (size_t i = 0; i < p->hidden_width; ++i) {
            if (i % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
                status = TRILLIM_CANCELLED;
                break;
            }
            hidden[i] *= silu(gate[i]);
        }
    }
    free(gate);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = check_activation(task->hidden, TRILLIM_FP32, p->hidden_width, down_group, context, &ordinary);
    if (status != TRILLIM_OK || !ordinary) {
        return status;
    }
    LinearParameters down = {1, p->hidden_width, p->outputs, TRILLIM_FP32, p->down_dtype, p->output_dtype};
    const void *down_inputs[] = {task->hidden, task->inputs[3]};
    void *down_outputs[] = {task->output};
    *handled = true;
    return projection(down_outputs, down_inputs, NULL, &down, context);
}
#endif

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[4], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs, scratch, p, context};
#if !defined(_WIN32)
    bool handled;
    status = try_int8_ffn(&task, &handled);
    if (status != TRILLIM_OK || handled) {
        return status;
    }
#endif
    if (scratch_size != 0) {
        size_t count = p->tokens * p->hidden_width;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        status = tiles == 1 ? project_hidden(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tiles, project_hidden, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    size_t count = p->tokens * p->outputs;
    size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
    return tiles == 1 ? project(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_packed_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = up, gate, down\n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if any(matrix.dtype not in packed for matrix in matrices):\n"
           "        raise ValueError('packed FFN requires compressed weights')\n"
           "    if gate.shape != up.shape or x.shape[-1] != up.shape[1] or down.shape[1] != up.shape[0]:\n"
           "        raise ValueError('incompatible packed FFN dimensions')\n"
           "    if output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=down.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        hidden_width=up.shape[0],\n"
           "        gate_dtype=_DTYPE_IDS[gate.dtype],\n"
           "        up_dtype=_DTYPE_IDS[up.dtype],\n"
           "        down_dtype=_DTYPE_IDS[down.dtype],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is not Silu:\n"
           "        raise ValueError('packed FFN requires an unbiased SiLU gate')\n"
           "    result = FFN(x, up=up, gate=gate, down=down)\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_ffn_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
