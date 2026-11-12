#include "common.h"
#include <arm_neon.h>

typedef struct {
    size_t outputs;
    size_t output_rank;
    size_t reduce_rank;
    size_t reduce_count;
    const size_t *output_shape;
    const size_t *output_strides;
    const size_t *reduce_shape;
    const size_t *reduce_strides;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimReductionParameters;


typedef struct {
    void *output;
    const void *input;
    const TrillimReductionParameters *parameters;
    const TrillimKernelContext *context;
} TrillimReductionTask;

static bool shape_product(size_t rank, const size_t *shape, size_t *product) {
    if (rank && !shape) {
        return false;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!shape[axis]) {
            *product = 0;
            return true;
        }
    }
    size_t result = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (result > SIZE_MAX / shape[axis]) {
            return false;
        }
        result *= shape[axis];
    }
    *product = result;
    return true;
}

static inline TrillimStatus trillim_reduction_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimReductionParameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t output_count, reduce_count;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        (p->output_rank && !p->output_strides) || (p->reduce_rank && !p->reduce_strides) ||
        !shape_product(p->output_rank, p->output_shape, &output_count) ||
        !shape_product(p->reduce_rank, p->reduce_shape, &reduce_count) ||
        output_count != p->outputs || reduce_count != p->reduce_count ||
        p->outputs > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->outputs * bytes;
    if (p->outputs && p->reduce_count) {
        size_t outer_span, inner_span;
        if ((p->storage_columns && p->storage_rows > SIZE_MAX / p->storage_columns) ||
            !trillim_span(p->output_rank, p->output_shape, p->output_strides, &outer_span) ||
            !trillim_span(p->reduce_rank, p->reduce_shape, p->reduce_strides, &inner_span)) {
            return TRILLIM_ERROR;
        }
        size_t capacity = p->storage_rows * p->storage_columns;
        if (!outer_span || !inner_span || outer_span - 1 > SIZE_MAX - inner_span ||
            outer_span - 1 + inner_span > capacity ||
            outer_span - 1 + inner_span > SIZE_MAX / trillim_itemsize(p->input_dtype)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = (outer_span - 1 + inner_span) * trillim_itemsize(p->input_dtype);
    }
    return TRILLIM_OK;
}

static float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
    } else if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index,
                 vreinterpret_u16_f16(vcvt_f16_f32(value)));
    } else {
        uint32x4_t bits = vreinterpretq_u32_f32(value);
        uint32x4_t upper = vshrq_n_u32(bits, 16);
        uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff),
                                   vandq_u32(upper, vdupq_n_u32(1)));
        uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
        uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)),
                                  vdupq_n_u32(0x7f800000));
        rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
        vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
    }
}


static TrillimStatus sum_float(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t base = trillim_offset(index, p->output_rank, p->output_shape, p->output_strides);
    float sum = 0.0f;
    size_t k = 0;
    if (p->reduce_rank == 1 && p->reduce_strides[0] == 1) {
        for (; p->reduce_count - k >= 4; k += 4) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t values = load4(task->input, base + k, p->input_dtype);
            sum += vgetq_lane_f32(values, 0);
            sum += vgetq_lane_f32(values, 1);
            sum += vgetq_lane_f32(values, 2);
            sum += vgetq_lane_f32(values, 3);
        }
    }
    for (; k < p->reduce_count; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
        sum += trillim_load(task->input, base + offset, p->input_dtype).f;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue value = {.f = sum};
    trillim_store(task->output, index, p->output_dtype, value);
    return TRILLIM_OK;
}

static TrillimStatus sum_columns(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t columns = p->output_shape[p->output_rank - 1];
    size_t tiles = columns / 64 + (columns % 64 != 0);
    size_t first = tile % tiles * 64;
    size_t count = columns - first < 64 ? columns - first : 64;
    size_t output = tile / tiles * columns + first;
    size_t base = trillim_offset(output, p->output_rank, p->output_shape, p->output_strides);
    size_t vectors = count / 4;
    float32x4_t sums[16];
    float tail[3] = {0};
    for (size_t lane = 0; lane < vectors; ++lane) {
        sums[lane] = vdupq_n_f32(0.0f);
    }
    for (size_t k = 0; k < p->reduce_count; ++k) {
        if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
        for (size_t lane = 0; lane < vectors; ++lane) {
            sums[lane] = vaddq_f32(sums[lane], load4(task->input, base + offset + lane * 4, p->input_dtype));
        }
        for (size_t lane = vectors * 4; lane < count; ++lane) {
            tail[lane - vectors * 4] += trillim_load(task->input, base + offset + lane, p->input_dtype).f;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t lane = 0; lane < vectors; ++lane) {
        store4(task->output, output + lane * 4, p->output_dtype, sums[lane]);
    }
    for (size_t lane = vectors * 4; lane < count; ++lane) {
        TrillimValue value = {.f = tail[lane - vectors * 4]};
        trillim_store(task->output, output + lane, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_sum(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimReductionParameters *p = parameters;
    if (!p->outputs) {
        return TRILLIM_OK;
    }
    TrillimReductionTask task = {outputs[0], inputs[0], p, context};
    if (p->output_rank && p->output_strides[p->output_rank - 1] == 1) {
        size_t columns = p->output_shape[p->output_rank - 1];
        size_t tiles = columns / 64 + (columns % 64 != 0);
        size_t tasks = p->outputs / columns * tiles;
        return tasks == 1 ? sum_columns(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tasks, sum_columns, &task);
    }
    return p->outputs == 1 ? sum_float(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->outputs, sum_float, &task);
}

TRILLIM_KERNEL const char *trillim_neon_sum_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bind_reduction as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> tuple[Tensor, ...]:\n"
           "    return (Sum(x, axes=axes, keepdims=keepdims),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_sum_parameters(void) {
    static const TrillimParameter fields[] = {
        {"outputs", "size_t", offsetof(TrillimReductionParameters, outputs), NULL},
        {"output_rank", "size_t", offsetof(TrillimReductionParameters, output_rank), NULL},
        {"reduce_rank", "size_t", offsetof(TrillimReductionParameters, reduce_rank), NULL},
        {"reduce_count", "size_t", offsetof(TrillimReductionParameters, reduce_count), NULL},
        {"output_shape", "size_t", offsetof(TrillimReductionParameters, output_shape), "output_rank"},
        {"output_strides", "size_t", offsetof(TrillimReductionParameters, output_strides), "output_rank"},
        {"reduce_shape", "size_t", offsetof(TrillimReductionParameters, reduce_shape), "reduce_rank"},
        {"reduce_strides", "size_t", offsetof(TrillimReductionParameters, reduce_strides), "reduce_rank"},
        {"storage_rows", "size_t", offsetof(TrillimReductionParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimReductionParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimReductionParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimReductionParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimReductionParameters), TRILLIM_ALIGNOF(TrillimReductionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_reduction_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_sum_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        /* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
        0
#else
        TRILLIM_CAP_NEON
#endif
    };
    return &capabilities;
}
