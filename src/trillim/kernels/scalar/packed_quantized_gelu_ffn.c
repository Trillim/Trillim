#include "common.h"
#include <fenv.h>
#include <math.h>

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
    size_t stride;
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
        if (!packed_activation_quantize(a->input, a->dtype, row * a->stride + start,
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
    PackedActivation *a, const void *input, uint32_t input_dtype, uint32_t weight_dtype,
    size_t rows, size_t width, size_t stride, size_t columns, const TrillimKernelContext *context, void *scratch
) {
    size_t group = packed_activation_group(weight_dtype);
    if (scratch == NULL || !group || !rows || width < group || width > 65536 || width % group || columns < 16 ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    if (rows > PACKED_ACTIVATION_MAX_BYTES / (groups * sizeof(float) + width)) {
        return TRILLIM_OK;
    }
    float *scales = scratch;
    int8_t *codes = (int8_t *)(scales + rows * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, stride, input_dtype};
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
    size_t tokens, width, hidden, outputs;
    uint32_t up_dtype, down_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
    PackedActivation activation;
} Task;

static bool grouped(uint32_t dtype) {
    return dtype >= TRILLIM_GRP_INT8 && dtype <= TRILLIM_DUBGRP_INT6;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->hidden == 0 || !grouped(p->up_dtype) || !grouped(p->down_dtype) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, p->up_dtype, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, p->down_dtype, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    size_t up = packed_activation_space(p->up_dtype, p->tokens, p->width, p->hidden);
    size_t down = packed_activation_space(p->down_dtype, p->tokens, p->hidden, p->outputs);
    size_t workspace = up > down ? up : down;
    if (workspace > SIZE_MAX - *scratch) {
        return TRILLIM_ERROR;
    }
    *scratch += workspace;
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

static float gelu_tanh(float x) {
    float cube = (x * x) * x;
    float correction = cube * 0.044715f;
    float argument = (x + correction) * 0.7978845608028654f;
    float scaled = argument * 2.0f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    float cdf = probability * 2.0f - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->hidden;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->hidden, column = index % p->hidden;
        float value;
        TrillimStatus status = dot(
            &task->activation, task->inputs[0], TRILLIM_BF16, row, task->inputs[1], p->up_dtype,
            column, p->width, p->hidden, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        float gate;
        status = dot(
            &task->activation, task->inputs[0], TRILLIM_BF16, row, task->inputs[2], p->up_dtype,
            column, p->width, p->hidden, task->context, &gate
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        ((float *)task->hidden)[index] = value * gelu_tanh(gate);
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
            &task->activation, task->hidden, TRILLIM_FP32, row, task->inputs[3], p->down_dtype,
            column, p->hidden, p->outputs, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->output, index, TRILLIM_FP32, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_quantized_gelu_ffn(
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
    size_t hidden_count = p->tokens * p->hidden;
    void *workspace = scratch == NULL ? NULL : (uint8_t *)scratch + hidden_count * sizeof(float);
    Task task = {outputs[0], inputs, scratch, p, context, {0}};
    if (scratch == NULL || context->thread_pool->worker_count == 0) {
        return TRILLIM_ERROR;
    }
    status = packed_activation_prepare(&task.activation, inputs[0], TRILLIM_BF16, p->up_dtype,
        p->tokens, p->width, p->width, p->hidden, context, workspace);
    if (status == TRILLIM_OK && hidden_count != 0) {
        size_t tiles = hidden_count / TILE_SIZE + (hidden_count % TILE_SIZE != 0);
        status = tiles == 1 ? project_hidden(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tiles, project_hidden, &task);
    }
    task.activation = (PackedActivation){0};
    if (status == TRILLIM_OK) {
        status = packed_activation_prepare(&task.activation, scratch, TRILLIM_FP32, p->down_dtype,
            p->tokens, p->hidden, p->hidden, p->outputs, context, workspace);
    }
    size_t count = p->tokens * p->outputs;
    size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
    if (status == TRILLIM_OK) {
        status = tiles == 1 ? project(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
    }
    return status;
}

TRILLIM_KERNEL const char *trillim_scalar_packed_quantized_gelu_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(context):\n"
           "    x, up, gate, down = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != 'bf16' or output.dtype != 'fp32':\n"
           "        raise ValueError('packed GELU FFN requires BF16 input and FP32 output')\n"
           "    for tensor in (x, up, gate, down, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('packed GELU FFN requires contiguous operands')\n"
           "    grouped = {'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_int8'}\n"
           "    if gate.dtype != up.dtype or up.dtype not in grouped or down.dtype not in grouped:\n"
           "        raise ValueError('packed GELU FFN requires matching grouped up and gate weights and grouped down weights')\n"
           "    if any(len(t.shape) != 2 or _storage_shape(t) != t.shape for t in (up, gate, down)):\n"
           "        raise ValueError('packed GELU FFN requires full canonical weight matrices')\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible packed GELU FFN inputs')\n"
           "    if output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed GELU FFN output')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden, outputs=down.shape[0],\n"
           "                up_dtype=_DTYPE_IDS[up.dtype], down_dtype=_DTYPE_IDS[down.dtype])\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is Silu:\n"
           "        raise ValueError('packed GELU FFN requires an unbiased GELU gate')\n"
           "    return (FFN(x, up=up, gate=gate, down=down, activation=lambda value: Gelu(value, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_quantized_gelu_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"up_dtype", "uint32_t", offsetof(Parameters, up_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_quantized_gelu_ffn_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
