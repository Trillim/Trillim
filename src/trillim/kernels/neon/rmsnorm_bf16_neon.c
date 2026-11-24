#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

typedef struct {
    size_t rows, width;
    float eps, weight_offset;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const void *weight;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / sizeof(uint16_t) / p->width) {
        return TRILLIM_ERROR;
    }
    inputs[0] = outputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->width * sizeof(uint16_t);
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    return output == 0 && input == 0;
}

static inline float32x4_t load4(const void *data, size_t index) {
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, float32x4_t value) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    size_t start = row * p->width;
    float sum = 0.0f;
    size_t column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float lanes[4];
        vst1q_f32(lanes, load4(task->input, start + column));
        for (size_t lane = 0; lane < 4; ++lane) {
            sum += lanes[lane] * lanes[lane];
        }
    }
    for (; column < p->width; ++column) {
        float value = trillim_load(task->input, start + column, TRILLIM_BF16).f;
        sum += value * value;
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    float32x4_t divisor = vdupq_n_f32(denominator);
    float32x4_t offset = vdupq_n_f32(p->weight_offset);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vdivq_f32(load4(task->input, start + column), divisor);
        float32x4_t weight = load4(task->weight, column);
        value = vmulq_f32(value, vaddq_f32(weight, offset));
        store4(task->output, start + column, value);
    }
    for (; column < p->width; ++column) {
        float weight = trillim_load(task->weight, column, TRILLIM_BF16).f;
        float input = trillim_load(task->input, start + column, TRILLIM_BF16).f;
        TrillimValue value = {.f = (input / denominator) * (weight + p->weight_offset)};
        trillim_store(task->output, start + column, TRILLIM_BF16, value);
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize_pair(void *opaque, size_t pair, size_t worker) {
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    size_t row = pair * 2;
    if (p->rows - row == 1) {
        return normalize(opaque, row, worker);
    }
    const uint16_t *input0 = (const uint16_t *)task->input + row * p->width;
    const uint16_t *input1 = input0 + p->width;
    uint16_t *output0 = (uint16_t *)task->output + row * p->width;
    uint16_t *output1 = output0 + p->width;
    float32x2_t sum = vdup_n_f32(0.0f);
    for (size_t column = 0; column < p->width; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x4x2_t columns = vzip_u16(vld1_u16(input0 + column), vld1_u16(input1 + column));
        float32x4_t first = vreinterpretq_f32_u32(vshll_n_u16(columns.val[0], 16));
        float32x4_t second = vreinterpretq_f32_u32(vshll_n_u16(columns.val[1], 16));
        float32x2_t a = vget_low_f32(first);
        float32x2_t b = vget_high_f32(first);
        float32x2_t c = vget_low_f32(second);
        float32x2_t d = vget_high_f32(second);
        // Each lane adds its row's squares in column order, as in normalize.
        sum += a * a;
        sum += b * b;
        sum += c * c;
        sum += d * d;
    }
    float denominator0 = sqrtf(vget_lane_f32(sum, 0) / (float)p->width + p->eps);
    float denominator1 = sqrtf(vget_lane_f32(sum, 1) / (float)p->width + p->eps);
    if (!isfinite(denominator0) || !isfinite(denominator1) || denominator0 == 0.0f || denominator1 == 0.0f) {
        TrillimStatus status = normalize(opaque, row, worker);
        return status == TRILLIM_OK ? normalize(opaque, row + 1, worker) : status;
    }
    float32x4_t divisor0 = vdupq_n_f32(denominator0);
    float32x4_t divisor1 = vdupq_n_f32(denominator1);
    float32x4_t offset = vdupq_n_f32(p->weight_offset);
    for (size_t column = 0; column < p->width; column += 4) {
        if ((column & 511) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t weight = vaddq_f32(load4(task->weight, column), offset);
        float32x4_t value0 = vdivq_f32(load4(input0, column), divisor0);
        float32x4_t value1 = vdivq_f32(load4(input1, column), divisor1);
        store4(output0, column, vmulq_f32(value0, weight));
        store4(output1, column, vmulq_f32(value1, weight));
    }
    return TRILLIM_OK;
}

typedef struct {
    Task *task;
    size_t grain;
} PairBatch;

static TrillimStatus normalize_pair_batch(void *opaque, size_t tile, size_t worker) {
    const PairBatch *batch = opaque;
    size_t rows = batch->task->parameters->rows;
    size_t pairs = rows / 2 + (rows % 2 != 0);
    size_t first = tile * batch->grain;
    size_t count = pairs - first < batch->grain ? pairs - first : batch->grain;
    for (size_t pair = first; pair < first + count; ++pair) {
        TrillimStatus status = normalize_pair(batch->task, pair, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_bf16_rmsnorm(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *opaque, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = opaque;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    TrillimTask worker = normalize;
    size_t tasks = p->rows;
    if (p->width % 4 == 0 && p->rows >= 2 && isfinite(p->weight_offset) &&
        p->rows / 2 >= context->thread_pool->worker_count) {
        worker = normalize_pair;
        tasks = p->rows / 2 + (p->rows % 2 != 0);
    }
    if (p->rows == 1 || p->rows <= 2048 / p->width) {
        for (size_t tile = 0; tile < tasks; ++tile) {
            TrillimStatus status = worker(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (worker == normalize_pair && workers != 0) {
        size_t grain = tasks / workers / 2;
        if (grain > 1) {
            PairBatch batch = {&task, grain};
            size_t tiles = tasks / grain + (tasks % grain != 0);
            return context->thread_pool->run(context->thread_pool->context, tiles, normalize_pair_batch, &batch);
        }
    }
    return context->thread_pool->run(context->thread_pool->context, tasks, worker, &task);
}

TRILLIM_KERNEL const char *trillim_neon_bf16_rmsnorm_tdl(void) {
    return "from trillim.tdl import DType, RMSNorm, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or output.shape != x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('BF16 RMSNorm requires a matching channel vector')\n"
           "    for value in (x, weight, output):\n"
           "        if value.dtype != 'bf16' or (value.strides and value.strides != _contiguous_strides(value.shape)):\n"
           "            raise ValueError('BF16 RMSNorm requires contiguous BF16 tensors')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    return dict(rows=_product(x.shape[:-1]), width=x.shape[-1], eps=float(eps), weight_offset=float(offset))\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_bf16_rmsnorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_bf16_rmsnorm_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
