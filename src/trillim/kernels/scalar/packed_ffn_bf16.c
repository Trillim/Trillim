#include "common.h"
#include <fenv.h>
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { PACKED_ACTIVATION_TILE = 4, PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024 };

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    uint32_t dtype;
} PackedActivation;

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
    float inverse = -127.0f / peak;
    if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
        return false;
    }
    *scale = 1.0f / inverse;
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

static size_t packed_activation_space(size_t rows, size_t width, size_t columns) {
    size_t group = 256;
    if (width < group || columns < 16) {
        return 0;
    }
    if (rows > 1 && rows < 4) {
        rows = 1;
    }
    /* The compiler reserves scratch at the maximum shape. */
    size_t bounded_width = width < 65536 ? width : 65536;
    size_t groups = bounded_width / group;
    size_t row_bytes = groups * (group + sizeof(float));
    return rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes
        ? PACKED_ACTIVATION_MAX_BYTES : rows * row_bytes;
}

static TrillimStatus packed_activation_prepare(
    PackedActivation *a, void *workspace, const void *input, uint32_t input_dtype,
    size_t rows, size_t width, size_t columns, const TrillimKernelContext *context
) {
    size_t group = 256;
    if (workspace == NULL || (rows != 1 && rows < 4) ||
        width < group || width > 65536 || width % group != 0 || columns < 16 ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    size_t row_bytes = width + groups * sizeof(float);
    if (rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes) {
        return TRILLIM_OK;
    }
    float *scales = workspace;
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

static inline TrillimStatus packed_activation_dot(
    const PackedActivation *a, size_t first_row, size_t rows, const uint8_t *weight,
    size_t outputs, size_t column,
    float sums[PACKED_ACTIVATION_TILE], bool *used
) {
    *used = false;
    for (size_t row = 0; row < rows; ++row) {
        if (!packed_activation_ready(a, first_row + row)) {
            return TRILLIM_OK;
        }
    }
    size_t total = outputs * a->groups;
    size_t block = column * a->groups;
    size_t low_size = 128;
    const uint8_t *low = weight + block * low_size;
    const uint8_t *scales = weight + total * low_size + block * 2;
    const uint8_t *high = weight + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (total * 64 + block * 16));
    size_t high_size = 64;
    high += block * high_size;
    const int8_t *codes = a->codes + first_row * a->width;
    const float *activation_scales = a->scales + first_row * a->groups;
    float output[PACKED_ACTIVATION_TILE] = {0};
    for (size_t g = 0; g < a->groups; ++g) {
        if (a->context->is_cancelled(a->context->cancellation_context)) {
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
        int32_t dots[PACKED_ACTIVATION_TILE] = {0};
        if (rows == 1) {
            dots[0] = packed_q6_dot(l, h, subscales + g * 16, codes + g * 256);
        } else {
            for (size_t region = 0; region < 8; ++region) {
                size_t half = region / 4, lane = region % 4;
                const uint8_t *lower = l + half * 64 + lane / 2 * 32;
                const uint8_t *upper = h + half * 32;
                const int8_t *s = subscales + g * 16 + region * 2;
                int16_t decoded[32];
                for (size_t i = 0; i < 32; ++i) {
                    int32_t lo = (lower[i] >> (lane % 2 * 4)) & 15;
                    int32_t hi = (upper[i] >> (lane * 2)) & 3;
                    decoded[i] = (int16_t)(((lo | hi << 4) - 32) * s[i / 16]);
                }
                for (size_t row = 0; row < rows; ++row) {
                    const int8_t *x = codes + row * a->width + g * 256 + region * 32;
                    for (size_t i = 0; i < 32; ++i) {
                        dots[row] += decoded[i] * x[i];
                    }
                }
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float combined_scale = scale * activation_scales[row * a->groups + g];
            output[row] += (float)dots[row] * combined_scale;
        }
    }
    memcpy(sums, output, rows * sizeof(*sums));
    *used = true;
    return TRILLIM_OK;
}

typedef struct {
    size_t tokens, width, hidden;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->hidden == 0 || p->width % 256 != 0 || p->hidden % 256 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, TRILLIM_DUBGRP_INT6, inputs + 1) ||
        !trillim_packed_payload_size(p->width, p->hidden, TRILLIM_DUBGRP_INT6, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch)) {
        return TRILLIM_ERROR;
    }
    size_t expand_space = packed_activation_space(p->tokens, p->width, p->hidden);
    size_t down_space = packed_activation_space(p->tokens, p->hidden, p->width);
    size_t activation_space = expand_space > down_space ? expand_space : down_space;
    if (*scratch > SIZE_MAX - activation_space) {
        return TRILLIM_ERROR;
    }
    *scratch += activation_space;
    inputs[2] = inputs[1];
    return TRILLIM_OK;
}

static inline float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static void decode_block(
    const uint8_t *values, size_t total, size_t block, size_t count, float *decoded
) {
    size_t scale_start = total * 128;
    float scale = read_bf16(values + scale_start + block * 2);
    for (size_t i = 0; i < count; ++i) {
        size_t half = i / 128;
        size_t lane = i % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + i % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + i % 32;
        int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
        code |= ((values[high] >> (lane * 2)) & 3) << 4;
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + i / 16, sizeof(subscale));
        decoded[i] = (float)((code - 32) * subscale) * scale;
    }
}
static inline float load_activation(const void *input, size_t index, bool bf16_input) {
    if (bf16_input) {
        return read_bf16((const uint8_t *)input + index * sizeof(uint16_t));
    }
    return ((const float *)input)[index];
}

static TrillimStatus accumulate(
    const void *input, const uint8_t *weight, size_t width, size_t outputs,
    size_t row, size_t rows, size_t column, bool bf16_input,
    const TrillimKernelContext *context, const PackedActivation *activation, float sums[4]
) {
    if (activation->scales != NULL) {
        bool used;
        TrillimStatus status = packed_activation_dot(activation, row, rows, weight,
            outputs, column, sums, &used);
        if (status != TRILLIM_OK || used) {
            return status;
        }
    }
    size_t group = 256;
    size_t groups = width / group + (width % group != 0);
    size_t total = outputs * groups;
    for (size_t k = 0; k < width; k += group) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = width - k < group ? width - k : group;
        float decoded[256];
        decode_block(weight, total, column * groups + k / group, count, decoded);
        for (size_t r = 0; r < rows; ++r) {
            float sum = sums[r];
            for (size_t i = 0; i < count; ++i) {
                float x = load_activation(input, (row + r) * width + k + i, bf16_input);
                sum += x * decoded[i];
            }
            sums[r] = sum;
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static float activate(float x) {
    float e = expf(-fabsf(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * sigmoid;
}

typedef struct {
    const void *input;
    const void *weight;
    const void *gate;
    void *output;
    size_t tokens, width, outputs, columns;
    bool hidden;
    const TrillimKernelContext *context;
    PackedActivation activation;
} Task;

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t tiles = task->outputs / task->columns + (task->outputs % task->columns != 0);
    size_t row = tile / tiles * 4;
    size_t rows = task->tokens - row < 4 ? task->tokens - row : 4;
    size_t first = tile % tiles * task->columns;
    size_t end = task->outputs - first < task->columns ? task->outputs : first + task->columns;
    for (size_t column = first; column < end; ++column) {
        float sums[4] = {0};
        TrillimStatus status = accumulate(task->input, task->weight, task->width,
            task->outputs, row, rows, column, task->hidden, task->context, &task->activation, sums);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (task->hidden) {
            float gate[4] = {0};
            status = accumulate(task->input, task->gate, task->width,
                task->outputs, row, rows, column, true, task->context, &task->activation, gate);
            if (status != TRILLIM_OK) {
                return status;
            }
            for (size_t r = 0; r < rows; ++r) {
                size_t index = (row + r) * task->outputs + column;
                ((float *)task->output)[index] = sums[r] * activate(gate[r]);
            }
            continue;
        }
        for (size_t r = 0; r < rows; ++r) {
            size_t index = (row + r) * task->outputs + column;
            trillim_store(task->output, index, TRILLIM_BF16, (TrillimValue){.f = sums[r]});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus run(Task *task, void *workspace) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    if (pool->worker_count == 0) {
        return TRILLIM_ERROR;
    }
    size_t columns = task->outputs / pool->worker_count / 4;
    columns = columns / 16 * 16;
    task->columns = columns < 16 ? 16 : columns;
    size_t rows = task->tokens / 4 + (task->tokens % 4 != 0);
    size_t tiles = task->outputs / task->columns + (task->outputs % task->columns != 0);
    TrillimStatus status = packed_activation_prepare(&task->activation, workspace, task->input,
        task->hidden ? TRILLIM_BF16 : TRILLIM_FP32, task->tokens, task->width, task->outputs, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    return rows * tiles == 1 ? project(task, 0, 0)
        : pool->run(pool->context, rows * tiles, project, task);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_ffn_bf16(
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
    size_t hidden_bytes = p->tokens * p->hidden * sizeof(float);
    void *activation_workspace = (uint8_t *)scratch + hidden_bytes;
    Task expand = {
        .input = inputs[0], .weight = inputs[1], .gate = inputs[2], .output = scratch,
        .tokens = p->tokens, .width = p->width, .outputs = p->hidden,
        .hidden = true, .context = context
    };
    status = run(&expand, activation_workspace);
    if (status != TRILLIM_OK) {
        return status;
    }
    Task down = {
        .input = scratch, .weight = inputs[3], .output = outputs[0],
        .tokens = p->tokens, .width = p->hidden, .outputs = p->width,
        .context = context
    };
    return run(&down, activation_workspace);
}

TRILLIM_KERNEL const char *trillim_scalar_packed_ffn_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(c):\n"
           "    x, up, gate, down = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16' or output.shape != x.shape:\n"
           "        raise ValueError('requires a matching BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if any(len(v.shape) != 2 or v.dtype != 'dubgrp_int6' or _storage_shape(v) != v.shape for v in (up, gate, down)):\n"
           "        raise ValueError('requires complete Q6 matrices')\n"
           "    hidden, width = up.shape\n"
           "    if up.shape != gate.shape or down.shape != (width, hidden) or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or hidden.value is None or not width.value or not hidden.value or width.value % 256 or hidden.value % 256:\n"
           "        raise ValueError('unaligned Q6 matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden)\n"
           "\n"
           "def pattern(x: Tensor, up: Tensor, down: Tensor, gate: Tensor, *, activation, up_bias=None, gate_bias=None, down_bias=None, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if activation is not Silu or dtype != 'bf16' or any(b is not None for b in (up_bias, gate_bias, down_bias)):\n"
           "        raise ValueError('requires a BF16 SwiGLU cast')\n"
           "    return (Cast(FFN(x, up=up, gate=gate, down=down), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_ffn_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 4, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_ffn_bf16_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
