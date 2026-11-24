#include "common.h"
#include <math.h>
#include <fenv.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024 };

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    uint32_t dtype;
} PackedActivation;

static inline size_t packed_activation_group(uint32_t dtype) {
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

static inline bool packed_activation_quantize(
    const void *input, uint32_t dtype, size_t start, size_t group, float *scale, int8_t *codes
) {
    float values[256];
    float peak = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        float value = trillim_load(input, start + i, dtype).f;
        if (!isfinite(value)) {
            return false;
        }
        if (fabsf(value) > fabsf(peak)) {
            peak = value;
        }
        values[i] = value;
    }
    if (peak == 0.0f) {
        *scale = 0.0f;
        memset(codes, 0, group);
        return true;
    }
    float inverse;
    if (group == 256) {
        inverse = -127.0f / peak;
        if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
            return false;
        }
        *scale = 1.0f / inverse;
    } else {
        float step = fabsf(peak) / 127.0f;
        if (step > 65504.0f || (*scale = trillim_fp16(trillim_to_fp16(step))) == 0.0f) {
            return false;
        }
        inverse = 1.0f / step;
    }
    for (size_t i = 0; i < group; ++i) {
        /* Bias the significand to round to nearest, ties to even. */
        float rounded = values[i] * inverse + 12582912.0f;
        uint32_t bits;
        memcpy(&bits, &rounded, sizeof(bits));
        codes[i] = (int8_t)((int32_t)(bits & UINT32_C(0x007fffff)) - INT32_C(0x00400000));
    }
    return true;
}

static inline TrillimStatus packed_activation_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const PackedActivation *a = opaque;
    size_t group = a->width / a->groups;
    float *scales = a->scales + row * a->groups;
    int8_t *codes = a->codes + row * a->width;
    if (fegetround() != FE_TONEAREST) {
        scales[0] = NAN;
        return TRILLIM_OK;
    }
    for (size_t g = 0; g < a->groups; ++g) {
        if (g % 4 == 0 && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t start = g * group;
        if (!packed_activation_quantize(a->input, a->dtype, row * a->width + start,
                group, scales + g, codes + start)) {
            scales[0] = NAN;
            return TRILLIM_OK;
        }
    }
    return TRILLIM_OK;
}

static inline size_t packed_activation_space(uint32_t dtype, size_t rows, size_t width, size_t columns) {
    size_t group = packed_activation_group(dtype);
    if (!group || columns < 16) {
        return 0;
    }
    size_t bounded_width = width < 65536 ? width : 65536;
    size_t row_bytes = (bounded_width / group) * (group + sizeof(float));
    if (!row_bytes) {
        return 0;
    }
    return rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes ? PACKED_ACTIVATION_MAX_BYTES : rows * row_bytes;
}

static inline TrillimStatus packed_activation_prepare(
    PackedActivation *a, const void *input, uint32_t input_dtype, size_t rows,
    size_t width, size_t groups, const TrillimKernelContext *context, void *scratch
) {
    float *scales = scratch;
    int8_t *codes = (int8_t *)(scales + rows * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, input_dtype};
    const TrillimThreadPool *pool = context->thread_pool;
    return rows == 1 ? packed_activation_row(a, 0, 0)
                     : pool->run(pool->context, rows, packed_activation_row, a);
}

static inline int32_t packed_q6_dot(
    const uint8_t *low, const uint8_t *high, const int8_t *subscales, const int8_t *codes
) {
    int32_t total = 0;
    for (size_t region = 0; region < 8; ++region) {
        size_t half = region / 4;
        size_t lane = region % 4;
        const uint8_t *l = low + half * 64 + lane / 2 * 32;
        const uint8_t *h = high + half * 32;
        for (size_t sub = 0; sub < 2; ++sub) {
            int32_t sum = 0;
            size_t first = sub * 16;
            for (size_t i = first; i < first + 16; ++i) {
                int32_t lo = (l[i] >> ((lane % 2) * 4)) & 15;
                int32_t hi = (h[i] >> (lane * 2)) & 3;
                int32_t w = (lo | hi << 4) - 32;
                sum += w * codes[region * 32 + i];
            }
            total += sum * subscales[region * 2 + sub];
        }
    }
    return total;
}

static inline bool packed_activation_ready(const PackedActivation *a, size_t row) {
    return a->scales != NULL && !isnan(a->scales[row * a->groups]);
}

static inline TrillimStatus packed_activation_dot_one(
    const PackedActivation *a, size_t row, const void *weight,
    uint32_t dtype, size_t outputs, size_t column, float *result, bool *used
) {
    *used = false;
    if (!packed_activation_ready(a, row)) {
        return TRILLIM_OK;
    }
    size_t total = outputs * a->groups;
    size_t block = column * a->groups;
    const uint8_t *bytes = weight;
    bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    size_t low_size = q6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *low = bytes + block * low_size;
    const uint8_t *scales = bytes + total * low_size + block * 2;
    const uint8_t *high = bytes + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (q6 ? total * 64 + block * 16 : 0));
    size_t high_size = q6 ? 64 : (dtype == TRILLIM_GRP_INT5 ? 4 : 0);
    high += block * high_size;
    const int8_t *codes = a->codes + row * a->width;
    const float *activation_scales = a->scales + row * a->groups;
    float output = 0.0f;
    for (size_t g = 0; g < a->groups; ++g) {
        if ((q6 || g % 8 == 0) && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float scale = trillim_bf16(bits);
        if (!isfinite(scale) || fabsf(scale) > 0x1p40f) {
            return TRILLIM_OK;
        }
        const uint8_t *l = low + g * low_size;
        const uint8_t *h = high + g * high_size;
        int32_t dot = 0;
        if (q6) {
            dot = packed_q6_dot(l, h, subscales + g * 16, codes + g * 256);
        } else {
            int8_t decoded[32];
            const int8_t *w = (const int8_t *)l;
            if (dtype != TRILLIM_GRP_INT8) {
                uint32_t fifth = 0;
                int32_t center = 8;
                if (dtype == TRILLIM_GRP_INT5) {
                    fifth = (uint32_t)h[0] | (uint32_t)h[1] << 8 |
                        (uint32_t)h[2] << 16 | (uint32_t)h[3] << 24;
                    center = 16;
                }
                for (size_t i = 0; i < 16; ++i) {
                    decoded[i] = (int8_t)((int32_t)((l[i] & 15) | ((fifth >> i & 1) << 4)) - center);
                    decoded[i + 16] = (int8_t)((int32_t)((l[i] >> 4) | ((fifth >> (i + 16) & 1) << 4)) - center);
                }
                w = decoded;
            }
            const int8_t *x = codes + g * 32;
            for (size_t i = 0; i < 32; ++i) {
                dot += w[i] * x[i];
            }
        }
        float combined_scale = scale * activation_scales[g];
        output += (float)dot * combined_scale;
    }
    *result = output;
    *used = true;
    return TRILLIM_OK;
}

enum { TILE_SIZE = 16 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, output_dtype;
    uint32_t weight_dtype, bias_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    const Parameters *p;
    const TrillimKernelContext *context;
    PackedActivation activation;
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
        !packed(p->weight_dtype) ||
        !trillim_float(p->bias_dtype)) {
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
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (!trillim_packed_payload_size(1, p->outputs, p->bias_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    if (p->output_dtype == TRILLIM_FP32) {
        *scratch = packed_activation_space(p->weight_dtype, p->tokens, p->width, p->outputs);
    }
    return TRILLIM_OK;
}

static TrillimStatus dot(
    const PackedActivation *activation, const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, const TrillimKernelContext *context, float *result
) {
    if (activation->scales != NULL && width / activation->groups == packed_activation_group(weight_dtype)) {
        bool used;
        TrillimStatus status = packed_activation_dot_one(activation, row, weight, weight_dtype,
            outputs, column, result, &used);
        if (status != TRILLIM_OK || used) {
            return status;
        }
    }
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
            &task->activation, task->inputs[0], p->input_dtype, row, task->inputs[1], p->weight_dtype,
            column, p->width, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        value += trillim_load(task->inputs[2], column, p->bias_dtype).f;
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_linear_bias(
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
    size_t group = packed_activation_group(p->weight_dtype);
    size_t groups = 0;
    size_t capacity = p->tokens;
    if (p->output_dtype == TRILLIM_FP32 && scratch != NULL && group != 0 &&
        p->width >= group && p->width <= 65536 && p->width % group == 0 && p->outputs >= 16 &&
        fegetround() == FE_TONEAREST) {
        groups = p->width / group;
        capacity = scratch_size / (groups * sizeof(float) + p->width);
    }
    for (size_t first = 0; first < p->tokens;) {
        Parameters panel = *p;
        panel.tokens = p->tokens - first < capacity ? p->tokens - first : capacity;
        const void *panel_inputs[] = {inputs[0], inputs[1], inputs[2]};
        void *output = outputs[0];
        if (first != 0) {
            panel_inputs[0] = (const uint8_t *)inputs[0] + first * p->width * trillim_itemsize(p->input_dtype);
            output = (uint8_t *)outputs[0] + first * p->outputs * trillim_itemsize(p->output_dtype);
        }
        Task task = {output, panel_inputs, &panel, context, {0}};
        if (groups != 0) {
            status = packed_activation_prepare(&task.activation, panel_inputs[0], p->input_dtype,
                panel.tokens, p->width, groups, context, scratch);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t count = panel.tokens * p->outputs;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        status = tiles == 1 ? project(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        first += panel.tokens;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_packed_linear_bias_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'weight', 'bias')\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, bias = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = weight, \n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if weight.dtype not in packed or x.shape[-1] != weight.shape[1]:\n"
           "        raise ValueError('packed fusion requires compatible compressed base weights')\n"
           "    if bias.dtype not in floating or bias.shape != weight.shape[:1]:\n"
           "        raise ValueError('packed projection requires a floating channel bias')\n"
           "    if output.shape != (*x.shape[:-1], weight.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=weight.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        weight_dtype=_DTYPE_IDS[weight.dtype],\n"
           "        bias_dtype=_DTYPE_IDS[bias.dtype],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    result = Linear(x, weight, bias=bias)\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_linear_bias_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        3, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_linear_bias_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
