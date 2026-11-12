#include "common.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


static inline float32x4_t load4(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return vld1q_f32((const float *)data + index);
    }
    uint16x4_t bits = vld1_u16((const uint16_t *)data + index);
    if (dtype == TRILLIM_FP16) {
        return vcvt_f32_f16(vreinterpret_f16_u16(bits));
    }
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline void store4(void *data, size_t index, uint32_t dtype, float32x4_t value) {
    if (dtype == TRILLIM_FP32) {
        vst1q_f32((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        vst1_u16((uint16_t *)data + index, vreinterpret_u16_f16(vcvt_f16_f32(value)));
        return;
    }
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded);
    vst1_u16((uint16_t *)data + index, vmovn_u32(rounded));
}

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t itemsize = trillim_itemsize(dtype);
    if (columns != 0 && rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (itemsize != 0) {
        if (count > SIZE_MAX / itemsize) {
            return false;
        }
        *bytes = count * itemsize;
        return true;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        if (rows > (SIZE_MAX - count) / 2) {
            return false;
        }
        *bytes = count + rows * 2;
        return true;
    }
    size_t group;
    size_t record;
    switch (dtype) {
    case TRILLIM_MAT_TERNARY: group = 128; record = 32; break;
    case TRILLIM_GRP_INT8: group = 32; record = 34; break;
    case TRILLIM_GRP_INT4: group = 32; record = 18; break;
    case TRILLIM_GRP_INT5: group = 32; record = 22; break;
    case TRILLIM_DUBGRP_INT6: group = 256; record = 210; break;
    case TRILLIM_GRP_BINARY: group = 128; record = 18; break;
    case TRILLIM_GRP_TERNARY: group = 128; record = 34; break;
    case TRILLIM_MXFP4: group = 32; record = 17; break;
    default: return false;
    }
    size_t groups = columns / group + (columns % group != 0);
    if (rows != 0 && groups > SIZE_MAX / record / rows) {
        return false;
    }
    *bytes = rows * groups * record;
    if (dtype == TRILLIM_MAT_TERNARY) {
        if (*bytes > SIZE_MAX - sizeof(float)) {
            return false;
        }
        *bytes += sizeof(float);
    }
    return true;
}

typedef struct {
    size_t rows;
    size_t width;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    float eps;
    float weight_offset;
} TrillimRMSNormParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimRMSNormParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRMSNormTask;

static inline TrillimStatus trillim_rms_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                     size_t *scratch) {
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    inputs[0] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    if ((p->input_dtype != TRILLIM_FP16 && p->input_dtype != TRILLIM_BF16 && p->input_dtype != TRILLIM_FP32) ||
        (p->output_dtype != TRILLIM_FP16 && p->output_dtype != TRILLIM_BF16 && p->output_dtype != TRILLIM_FP32)) {
        return TRILLIM_ERROR;
    }
    if (p->rows == 0 || p->width == 0) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->width || p->rows * p->width > SIZE_MAX / bytes || p->storage_columns == 0 ||
        p->storage_rows > SIZE_MAX / p->storage_columns || p->rows * p->width > p->storage_rows * p->storage_columns ||
        !trillim_payload_size(p->storage_rows, p->storage_columns, p->input_dtype, &inputs[0])) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->rows * p->width * bytes;
    return TRILLIM_OK;
}

static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = (const TrillimRMSNormTask *)opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    size_t start = row * p->width;
    float sum = 0.0f;
    size_t column = 0;
    if (p->output_dtype != TRILLIM_FP32) {
        for (; p->width - column >= 4; column += 4) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            float lanes[4];
            vst1q_f32(lanes, load4(task->input, start + column, p->input_dtype));
            for (size_t lane = 0; lane < 4; ++lane) sum += lanes[lane] * lanes[lane];
        }
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column, p->input_dtype).f;
            sum += value * value;
        }
    } else {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        column = 0;
        for (; p->width - column >= 16; column += 16) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float32x4_t a = load4(task->input, start + column, p->input_dtype);
            float32x4_t b = load4(task->input, start + column + 4, p->input_dtype);
            float32x4_t c = load4(task->input, start + column + 8, p->input_dtype);
            float32x4_t d = load4(task->input, start + column + 12, p->input_dtype);
            sum0 = vaddq_f32(sum0, vmulq_f32(a, a));
            sum1 = vaddq_f32(sum1, vmulq_f32(b, b));
            sum2 = vaddq_f32(sum2, vmulq_f32(c, c));
            sum3 = vaddq_f32(sum3, vmulq_f32(d, d));
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
        for (; column < p->width; ++column) {
            float value = trillim_load(task->input, start + column, p->input_dtype).f;
            sum += value * value;
        }
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    float32x4_t divisor = vdupq_n_f32(denominator);
    column = 0;
    for (; p->width - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vdivq_f32(load4(task->input, start + column, p->input_dtype), divisor);
        store4(task->output, start + column, p->output_dtype, value);
    }
    for (; column < p->width; ++column) {
        TrillimValue value = {.f = trillim_load(task->input, start + column, p->input_dtype).f / denominator};
        trillim_store(task->output, start + column, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRMSNormParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_rmsnorm_plain(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], p, context};
    return p->rows == 1 ? normalize(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_neon_rmsnorm_plain_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _rmsnorm as bind\n"
           "def pattern(x: Tensor, *, eps: float = 1e-6, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, eps=eps, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_rmsnorm_plain_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(TrillimRMSNormParameters, rows), NULL},
        {"width", "size_t", offsetof(TrillimRMSNormParameters, width), NULL},
        {"storage_rows", "size_t", offsetof(TrillimRMSNormParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimRMSNormParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRMSNormParameters, output_dtype), NULL},
        {"eps", "float", offsetof(TrillimRMSNormParameters, eps), NULL},
        {"weight_offset", "float", offsetof(TrillimRMSNormParameters, weight_offset), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimRMSNormParameters), TRILLIM_ALIGNOF(TrillimRMSNormParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1, trillim_rms_requirements,
        can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_rmsnorm_plain_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
