#include "common.h"
#include <arm_neon.h>
#include <math.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif


static inline float trillim_read_float(const void *data, size_t index, uint32_t dtype) {
    TrillimValue value = trillim_load(data, index, dtype);
    return trillim_float(dtype) ? value.f : (trillim_unsigned(dtype) ? (float)value.u : (float)value.i);
}

typedef struct {
    size_t heads;
    size_t tokens;
    size_t dim;
    size_t rotary_dim;
    size_t rank;
    const size_t *strides;
    size_t position_stride;
    uint32_t position_dtype;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t interleaved;
    float frequency_step;
    float amplitude;
} TrillimRotaryParameters;

typedef struct {
    void *output;
    const void *input;
    const void *positions;
    float *angles;
    const TrillimRotaryParameters *parameters;
    const TrillimKernelContext *context;
} TrillimRotaryTask;

static inline TrillimStatus trillim_rotary_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                        size_t *scratch) {
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t position_bytes = trillim_itemsize(p->position_dtype);
    if (p->rank != 3 || p->rotary_dim == 0 || p->rotary_dim % 2 != 0 || p->rotary_dim > p->dim || position_bytes == 0 ||
        !trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) || p->interleaved > 1 || p->strides == NULL || p->strides[2] != 1) {
        return TRILLIM_ERROR;
    }
    if (p->heads == 0 || p->tokens == 0) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->heads, p->tokens, p->dim};
    size_t span;
    if (!trillim_span(3, shape, p->strides, &span) || span > SIZE_MAX / trillim_itemsize(p->input_dtype) ||
        p->heads > SIZE_MAX / p->tokens ||
        p->heads * p->tokens > SIZE_MAX / trillim_itemsize(p->output_dtype) / p->dim ||
        (p->position_stride != 0 && p->tokens - 1 > (SIZE_MAX - 1) / p->position_stride)) {
        return TRILLIM_ERROR;
    }
    size_t positions = (p->tokens - 1) * p->position_stride + 1;
    if (positions > SIZE_MAX / position_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * trillim_itemsize(p->input_dtype);
    inputs[1] = positions * position_bytes;
    outputs[0] = p->heads * p->tokens * p->dim * trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / sizeof(float) / p->rotary_dim) return TRILLIM_ERROR;
    *scratch = p->tokens * p->rotary_dim * sizeof(float);
    return TRILLIM_OK;
}

static void store(const TrillimRotaryTask *task, size_t index, float result) {
    TrillimValue value;
    value.f = result;
    trillim_store(task->output, index, task->parameters->output_dtype, value);
}

static inline void trillim_rotate_pair(const TrillimRotaryTask *task, size_t base, size_t output, const float *angles,
                                       size_t pair) {
    const TrillimRotaryParameters *p = task->parameters;
    size_t a = p->interleaved ? pair * 2 : pair;
    size_t b = p->interleaved ? a + 1 : pair + p->rotary_dim / 2;
    float cosine = angles[pair], sine = angles[pair + p->rotary_dim / 2];
    float first = trillim_read_float(task->input, base + a * p->strides[2], p->input_dtype);
    float second = trillim_read_float(task->input, base + b * p->strides[2], p->input_dtype);
    store(task, output + a, first * cosine - second * sine);
    store(task, output + b, second * cosine + first * sine);
}

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

static TrillimStatus trillim_rotary_task_neon(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    size_t rows = p->heads * p->tokens;
    size_t block = rows >= 128 ? 8 : 1;
    size_t first = tile * block;
    size_t last = rows - first < block ? rows : first + block;
    for (size_t index = first; index < last; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        size_t head = index % p->heads;
        size_t token = index / p->heads;
        size_t base = head * p->strides[0] + token * p->strides[1];
        size_t output = (token * p->heads + head) * p->dim;
        size_t column = p->rotary_dim;
        for (; p->dim - column >= 4; column += 4) {
            if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            store4(task->output, output + column, p->output_dtype,
                load4(task->input, base + column, p->input_dtype));
        }
        for (; column < p->dim; ++column) {
            store(task, output + column, trillim_read_float(task->input, base + column, p->input_dtype));
        }
        const float *angles = task->angles + token * p->rotary_dim;
        size_t pairs = p->rotary_dim / 2;
        size_t pair = 0;
        for (; pairs - pair >= 4; pair += 4) {
            if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            float32x4_t a, b;
            if (p->interleaved) {
                float32x4_t low = load4(task->input, base + pair * 2, p->input_dtype);
                float32x4_t high = load4(task->input, base + pair * 2 + 4, p->input_dtype);
                a = vuzp1q_f32(low, high);
                b = vuzp2q_f32(low, high);
            } else {
                a = load4(task->input, base + pair, p->input_dtype);
                b = load4(task->input, base + pair + pairs, p->input_dtype);
            }
            float32x4_t c = vld1q_f32(angles + pair);
            float32x4_t s = vld1q_f32(angles + pairs + pair);
            float32x4_t first = vfmaq_f32(vnegq_f32(vmulq_f32(b, s)), a, c);
            float32x4_t second = vfmaq_f32(vmulq_f32(a, s), b, c);
            if (p->interleaved) {
                store4(task->output, output + pair * 2, p->output_dtype, vzip1q_f32(first, second));
                store4(task->output, output + pair * 2 + 4, p->output_dtype, vzip2q_f32(first, second));
            } else {
                store4(task->output, output + pair, p->output_dtype, first);
                store4(task->output, output + pair + pairs, p->output_dtype, second);
            }
        }
        for (; pair < pairs; ++pair) {
            trillim_rotate_pair(task, base, output, angles, pair);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus prepare_angles(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const TrillimRotaryTask *task = opaque;
    const TrillimRotaryParameters *p = task->parameters;
    float position = trillim_read_float(task->positions, token * p->position_stride, p->position_dtype);
    float *angles = task->angles + token * p->rotary_dim;
    size_t pairs = p->rotary_dim / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context))
            return TRILLIM_CANCELLED;
        float angle = position * expf((float)pair * p->frequency_step);
        angles[pair] = cosf(angle) * p->amplitude;
        angles[pairs + pair] = sinf(angle) * p->amplitude;
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const TrillimRotaryParameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype && p->strides[2] == 1 &&
           (p->heads <= 1 || p->strides[0] == p->dim) && (p->tokens <= 1 || p->strides[1] == p->heads * p->dim);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_rotary(void *const *outputs, const void *const *inputs, void *scratch,
                                                     const void *parameters, const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRotaryParameters *p = (const TrillimRotaryParameters *)parameters;
    if (!p->heads || !p->tokens) return TRILLIM_OK;
    TrillimRotaryTask task = {outputs[0], inputs[0], inputs[1], scratch, p, context};
    TrillimStatus status =
        p->tokens == 1 ? prepare_angles(&task, 0, 0)
                       : context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_angles, &task);
    if (status != TRILLIM_OK) return status;
    size_t rows = p->heads * p->tokens, block = rows >= 128 ? 8 : 1;
    size_t tiles = rows / block + (rows % block != 0);
    if (p->tokens == 1) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus result = trillim_rotary_task_neon(&task, tile, 0);
            if (result != TRILLIM_OK) return result;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, trillim_rotary_task_neon, &task);
}

TRILLIM_KERNEL const char *trillim_neon_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rotary as bind\n"
           "def pattern(x: Tensor, positions: Tensor, *, dim: int, theta: float = 10000.0, interleaved: bool = False, "
           "amplitude: float = 1.0, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Rotary(x, positions, dim=dim, theta=theta, interleaved=interleaved, amplitude=amplitude, "
           "dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_rotary_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(TrillimRotaryParameters, heads), NULL},
        {"tokens", "size_t", offsetof(TrillimRotaryParameters, tokens), NULL},
        {"dim", "size_t", offsetof(TrillimRotaryParameters, dim), NULL},
        {"rotary_dim", "size_t", offsetof(TrillimRotaryParameters, rotary_dim), NULL},
        {"rank", "size_t", offsetof(TrillimRotaryParameters, rank), NULL},
        {"strides", "size_t", offsetof(TrillimRotaryParameters, strides), "rank"},
        {"position_stride", "size_t", offsetof(TrillimRotaryParameters, position_stride), NULL},
        {"position_dtype", "uint32_t", offsetof(TrillimRotaryParameters, position_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimRotaryParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimRotaryParameters, output_dtype), NULL},
        {"interleaved", "uint32_t", offsetof(TrillimRotaryParameters, interleaved), NULL},
        {"frequency_step", "float", offsetof(TrillimRotaryParameters, frequency_step), NULL},
        {"amplitude", "float", offsetof(TrillimRotaryParameters, amplitude), NULL}};
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(TrillimRotaryParameters),
                                                  TRILLIM_ALIGNOF(TrillimRotaryParameters),
                                                  sizeof(fields) / sizeof(fields[0]),
                                                  fields,
                                                  2,
                                                  1,
                                                  trillim_rotary_requirements,
                                                  can_alias,
                                                  inputs,
                                                  outputs,
                                                  1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM};
    return &capabilities;
}
