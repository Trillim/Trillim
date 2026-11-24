#include "common.h"
#include <fenv.h>

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
    float values[32];
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
    float step = fabsf(peak) / 127.0f;
    if (step > 65504.0f || (*scale = trillim_fp16(trillim_to_fp16(step))) == 0.0f) {
        return false;
    }
    float inverse = 1.0f / step;
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
    size_t group = 32;
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
    size_t group = 32;
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

static inline bool packed_activation_ready(const PackedActivation *a, size_t row) {
    return a->scales != NULL && !isnan(a->scales[row * a->groups]);
}

static inline TrillimStatus packed_activation_dot(
    const PackedActivation *a, size_t first_row, size_t rows, const uint8_t *weight,
    uint32_t dtype, size_t outputs, size_t column,
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
    size_t low_size = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    const uint8_t *low = weight + block * low_size;
    const uint8_t *scales = weight + total * low_size + block * 2;
    const int8_t *codes = a->codes + first_row * a->width;
    const float *activation_scales = a->scales + first_row * a->groups;
    float output[PACKED_ACTIVATION_TILE] = {0};
    for (size_t g = 0; g < a->groups; ++g) {
        if (g % 8 == 0 && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float scale = trillim_bf16(bits);
        if (!isfinite(scale) || fabsf(scale) > 0x1p40f) {
            return TRILLIM_OK;
        }
        const uint8_t *l = low + g * low_size;
        int32_t dots[PACKED_ACTIVATION_TILE] = {0};
        int8_t decoded[32];
        const int8_t *w = (const int8_t *)l;
        if (dtype != TRILLIM_GRP_INT8) {
            for (size_t i = 0; i < 16; ++i) {
                decoded[i] = (int8_t)((l[i] & 15) - 8);
                decoded[i + 16] = (int8_t)((l[i] >> 4) - 8);
            }
            w = decoded;
        }
        for (size_t row = 0; row < rows; ++row) {
            const int8_t *x = codes + row * a->width + g * 32;
            for (size_t i = 0; i < 32; ++i) {
                dots[row] += w[i] * x[i];
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
    size_t tokens, width, outputs;
    uint32_t weight_dtype;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 2 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->width % 32 != 0 ||
        (p->weight_dtype != TRILLIM_GRP_INT4 && p->weight_dtype != TRILLIM_GRP_INT8) ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, inputs + 1) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_BF16, outputs)) {
        return TRILLIM_ERROR;
    }
    *scratch = packed_activation_space(p->tokens, p->width, p->outputs);
    return TRILLIM_OK;
}

static inline float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static void decode_block(
    const uint8_t *values, size_t total, size_t block, uint32_t dtype, size_t count, float *decoded
) {
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : 16;
    size_t scale_start = total * value_width;
    float scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)values + block * 32;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        int32_t code = (values[block * 16 + i % 16] >> ((i / 16) * 4)) & 15;
        decoded[i] = (float)(code - 8) * scale;
    }
}

static TrillimStatus accumulate(
    const void *input, const uint8_t *weight, size_t width, size_t outputs,
    size_t row, size_t rows, size_t column, uint32_t dtype,
    const TrillimKernelContext *context, const PackedActivation *activation, float sums[4]
) {
    if (activation->scales != NULL) {
        bool used;
        TrillimStatus status = packed_activation_dot(activation, row, rows, weight, dtype,
            outputs, column, sums, &used);
        if (status != TRILLIM_OK || used) {
            return status;
        }
    }
    size_t group = 32;
    size_t groups = width / group + (width % group != 0);
    size_t total = outputs * groups;
    for (size_t k = 0; k < width; k += group) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = width - k < group ? width - k : group;
        float decoded[32];
        decode_block(weight, total, column * groups + k / group, dtype, count, decoded);
        for (size_t r = 0; r < rows; ++r) {
            float sum = sums[r];
            for (size_t i = 0; i < count; ++i) {
                float x = read_bf16((const uint8_t *)input + ((row + r) * width + k + i) * sizeof(uint16_t));
                sum += x * decoded[i];
            }
            sums[r] = sum;
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

typedef struct {
    const void *input;
    const void *weight;
    void *output;
    size_t tokens, width, outputs, columns;
    uint32_t dtype;
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
            task->outputs, row, rows, column, task->dtype, task->context, &task->activation, sums);
        if (status != TRILLIM_OK) {
            return status;
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
        TRILLIM_BF16, task->tokens, task->width, task->outputs, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    return rows * tiles == 1 ? project(task, 0, 0)
        : pool->run(pool->context, rows * tiles, project, task);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_packed_grouped_linear_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    Task task = {
        .input = inputs[0], .weight = inputs[1], .output = outputs[0],
        .tokens = p->tokens, .width = p->width, .outputs = p->outputs,
        .dtype = p->weight_dtype,
        .context = context
    };
    return run(&task, scratch);
}

TRILLIM_KERNEL const char *trillim_scalar_packed_grouped_linear_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "\n"
           "physical_inputs = LINEAR_INPUTS = (\"x\", \"weight\")\n"
           "LINEAR_FORMATS = {\"grp_int4\", \"grp_int8\"}\n"
           "\n"
           "def contiguous(*bindings) -> None:\n"
           "    for binding in bindings:\n"
           "        stride = _contiguous_strides(binding.shape)\n"
           "        if (binding.strides or stride) != stride:\n"
           "            raise ValueError(\"requires contiguous operands\")\n"
           "\n"
           "def bind(context):\n"
           "    x, weight = (context.inputs[name] for name in LINEAR_INPUTS)\n"
           "    output = context.outputs[0]\n"
           "    if not x.shape or x.dtype != \"bf16\" or output.dtype != \"bf16\":\n"
           "        raise ValueError(\"requires BF16 input and output\")\n"
           "    if len(weight.shape) != 2 or weight.dtype not in LINEAR_FORMATS or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError(\"requires a complete Q4 or Q8 matrix\")\n"
           "    contiguous(x, weight, output)\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError(\"incompatible dimensions\")\n"
           "    if width.value is None or not width.value or width.value % 32:\n"
           "        raise ValueError(\"requires an aligned quantized input width\")\n"
           "    return {\"tokens\": _product(x.shape[:-1]), \"width\": width, \"outputs\": outputs,\n"
           "            \"weight_dtype\": _DTYPE_IDS[weight.dtype]}\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = \"bf16\") -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype != \"bf16\":\n"
           "        raise ValueError(\"requires unbiased BF16 projection\")\n"
           "    return (Cast(Linear(x, weight), \"bf16\"),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_packed_grouped_linear_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_packed_grouped_linear_bf16_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
