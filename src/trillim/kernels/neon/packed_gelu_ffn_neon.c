#include "common.h"
#include <arm_neon.h>
#include <stdatomic.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(__apple_build_version__) && __clang_major__ <= 17
__asm__(
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
    ".set trillim_fhm_v\\reg,\\reg\n"
    ".endr\n"
);
#define FHM
static inline float32x4_t fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x4e20ec00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
static inline float32x4_t fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    __asm__ volatile(".inst 0x6e20cc00 | trillim_fhm_%0 | (trillim_fhm_%1 << 5) | (trillim_fhm_%2 << 16)"
        : "+w"(sum) : "w"(a), "w"(b));
    return sum;
}
#else
__asm__(".arch armv8.2-a+fp16fml");
#if defined(__clang__)
#define FHM __attribute__((target("fp16fml")))
#else
#define FHM __attribute__((target("+fp16fml")))
#endif
FHM static inline float32x4_t fhm_low(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_low_f16(sum, a, b);
}
FHM static inline float32x4_t fhm_high(float32x4_t sum, float16x8_t a, float16x8_t b) {
    return vfmlalq_high_f16(sum, a, b);
}
#endif

static bool available_fhm(void) {
    static atomic_int cached;
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value) {
        return value == 2;
    }
    bool available = false;
#if defined(__APPLE__)
    int flag = 0;
    size_t size = sizeof(flag);
    available = sysctlbyname("hw.optional.arm.FEAT_FHM", &flag, &size, NULL, 0) == 0 && flag != 0;
#elif defined(__linux__) && defined(HWCAP_ASIMDFHM)
    available = (getauxval(AT_HWCAP) & HWCAP_ASIMDFHM) != 0;
#endif
    atomic_store_explicit(&cached, available ? 2 : 1, memory_order_relaxed);
    return available;
}

static bool fhm_default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

enum { MAX_FHM_WIDTH = 8192 };


#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

#if defined(__APPLE__) || defined(__linux__)
extern TrillimStatus trillim_neon_try_packed_ffn_int8(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8")
#endif
;

extern float32x4_t trillim_neon_gelu_tanh4_prefill(float32x4_t)
#if defined(__linux__)
    __asm__(".trillim_neon_gelu_tanh4_prefill")
#endif
;
extern TrillimStatus trillim_neon_try_packed_ffn_int8_prefill(
    void *, const void *, const void *, const void *, const void *, float (*)(float),
    float32x4_t (*)(float32x4_t), const TrillimKernelContext *, bool *
)
#if defined(__linux__)
    __asm__(".trillim_neon_try_packed_ffn_int8_prefill")
#endif
;

static float gelu_tanh(float x) {
    const float cube = (x * x) * x;
    const float correction = cube * 0.044715f;
    const float argument = (x + correction) * 0.7978845608028654f;
    const float scaled = argument * 2.0f;
    const float e = expf(-fabsf(scaled));
    const float denominator = e + 1.0f;
    const float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    const float doubled = probability * 2.0f;
    const float cdf = doubled - 1.0f;
    return (x * 0.5f) * (cdf + 1.0f);
}
#endif

extern COMPONENT_API TrillimStatus trillim_neon_packed_linear(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern COMPONENT_API TrillimStatus trillim_neon_gelu_tanh(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);
extern COMPONENT_API TrillimStatus trillim_neon_gelu_tanh_mul(
    void *const *, const void *const *, void *, const void *, const TrillimKernelContext *
);

enum { MAX_TILE = 64, MAX_FACTORED_WIDTH = 1 << 16 };

typedef struct {
    size_t tokens, width, hidden, outputs;
} Parameters;

typedef struct {
    size_t tokens, width, outputs;
    uint32_t input_dtype, weight_dtype, output_dtype;
} LinearParameters;

typedef struct {
    size_t count;
    uint32_t input_dtype, output_dtype;
} GeluParameters;

typedef struct {
    float *hidden;
    const uint8_t *input;
    const uint8_t *up, *gate;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t columns;
    const uint16_t *half;
} Task;

typedef struct {
    const uint8_t *codes, *scales;
} WeightRow;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 4 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width < 1024 || p->width > MAX_FACTORED_WIDTH || p->width % 32 != 0 ||
        p->hidden < 16 || p->hidden % 4 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->hidden, p->width, TRILLIM_GRP_INT4, inputs + 1) ||
        !trillim_packed_payload_size(p->outputs, p->hidden, TRILLIM_GRP_INT4, inputs + 3) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_FP32, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->hidden, TRILLIM_FP32, scratch) ||
        *scratch > SIZE_MAX / 2) {
        return TRILLIM_ERROR;
    }
    inputs[2] = inputs[1];
    *scratch *= 2;
    return TRILLIM_OK;
}

static TrillimStatus check_bounds(
    const uint8_t *values, size_t count, const TrillimKernelContext *context, bool *bounded
) {
    /* These exponent bounds and a width of at most 65536 leave room for the grouped dot. */
    const uint16x8_t minimum = vdupq_n_u16(87 << 7);
    const uint16x8_t span = vdupq_n_u16((81 << 7) - 1);
    uint16x8_t rejected = vdupq_n_u16(0);
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t raw = vreinterpretq_u16_u8(vld1q_u8(values + i * 2));
        uint16x8_t magnitude = vandq_u16(raw, vdupq_n_u16(0x7fff));
        uint16x8_t outside = vcgtq_u16(vsubq_u16(magnitude, minimum), span);
        rejected = vorrq_u16(rejected, vbicq_u16(outside, vceqzq_u16(magnitude)));
    }
    *bounded = vmaxvq_u16(rejected) == 0;
    for (; i < count && *bounded; ++i) {
        uint16_t raw;
        memcpy(&raw, values + i * 2, sizeof(raw));
        unsigned magnitude = raw & 0x7fff;
        *bounded = magnitude == 0 || (magnitude - (87u << 7)) <= (81u << 7) - 1;
    }
    return TRILLIM_OK;
}

static inline float16x8x2_t decode_half_codes(uint8x16_t codes) {
    /* Each centered Q4 code has a zero low byte in FP16. */
    static const uint8_t fp16_high[16] = {
        200, 199, 198, 197, 196, 194, 192, 188, 0, 60, 64, 66, 68, 69, 70, 71
    };
    uint8x16_t bits = vqtbl1q_u8(vld1q_u8(fp16_high), codes);
    float16x8x2_t halves = {{
        vreinterpretq_f16_u16(vshll_n_u8(vget_low_u8(bits), 8)),
        vreinterpretq_f16_u16(vshll_high_n_u8(bits, 8))
    }};
    return halves;
}

static inline void decode_codes(uint8x16_t codes, float32x4_t result[4]) {
    float16x8x2_t halves = decode_half_codes(codes);
    result[0] = vcvt_f32_f16(vget_low_f16(halves.val[0]));
    result[1] = vcvt_high_f32_f16(halves.val[0]);
    result[2] = vcvt_f32_f16(vget_low_f16(halves.val[1]));
    result[3] = vcvt_high_f32_f16(halves.val[1]);
}

static inline __attribute__((always_inline)) TrillimStatus project_pair(
    const Task *task, const WeightRow weights[4], bool factored, float result[4]
) {
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < task->p->width; k += 32) {
        if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t input[8];
        for (size_t i = 0; i < 8; ++i) {
            uint16x4_t bits = vreinterpret_u16_u8(vld1_u8(task->input + k * 2 + i * 8));
            input[i] = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
        }
        for (size_t c = 0; c < 4; ++c) {
            uint8x16_t packed = vld1q_u8(weights[c].codes + k / 2);
            float32x4_t codes[8];
            decode_codes(vandq_u8(packed, vdupq_n_u8(15)), codes);
            decode_codes(vshrq_n_u8(packed, 4), codes + 4);
            uint16_t bits;
            memcpy(&bits, weights[c].scales + k / 16, sizeof(bits));
            float scale = trillim_bf16(bits);
            if (factored) {
                float32x4_t lo = vdupq_n_f32(0), hi = lo;
                for (size_t i = 0; i < 4; ++i) {
                    lo = vfmaq_f32(lo, input[i], codes[i]);
                    hi = vfmaq_f32(hi, input[i + 4], codes[i + 4]);
                }
                sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), scale);
            } else {
                for (size_t i = 0; i < 8; ++i) {
                    sums[c] = vfmaq_f32(sums[c], input[i], vmulq_n_f32(codes[i], scale));
                }
            }
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

FHM static TrillimStatus project_fhm_pair(const Task *task, const WeightRow weights[4], float result[4]) {
    float32x4_t sums[4] = {0};
    for (size_t k = 0; k < task->p->width; k += 32) {
        if (k != 0 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8x4_t raw = vld1q_u16_x4(task->half + k);
        float16x8_t x0 = vreinterpretq_f16_u16(raw.val[0]), x1 = vreinterpretq_f16_u16(raw.val[1]);
        float16x8_t x2 = vreinterpretq_f16_u16(raw.val[2]), x3 = vreinterpretq_f16_u16(raw.val[3]);
        for (size_t c = 0; c < 4; ++c) {
            uint8x16_t packed = vld1q_u8(weights[c].codes + k / 2);
            float16x8x2_t low = decode_half_codes(vandq_u8(packed, vdupq_n_u8(15)));
            float16x8x2_t high = decode_half_codes(vshrq_n_u8(packed, 4));
            float32x4_t lo = vdupq_n_f32(0), hi = lo;
            lo = fhm_low(lo, x0, low.val[0]);
            hi = fhm_low(hi, x2, high.val[0]);
            lo = fhm_high(lo, x0, low.val[0]);
            hi = fhm_high(hi, x2, high.val[0]);
            lo = fhm_low(lo, x1, low.val[1]);
            hi = fhm_low(hi, x3, high.val[1]);
            lo = fhm_high(lo, x1, low.val[1]);
            hi = fhm_high(hi, x3, high.val[1]);
            uint16_t bits;
            memcpy(&bits, weights[c].scales + k / 16, sizeof(bits));
            sums[c] = vfmaq_n_f32(sums[c], vaddq_f32(lo, hi), trillim_bf16(bits));
        }
    }
    for (size_t c = 0; c < 4; ++c) {
        result[c] = vaddvq_f32(sums[c]);
    }
    return TRILLIM_OK;
}

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t groups = p->width / 32;
    size_t first = tile * task->columns;
    size_t count = p->hidden - first < task->columns ? p->hidden - first : task->columns;
    const uint8_t *up_scales = task->up + (p->hidden * 16 + first * 2) * groups;
    const uint8_t *gate_scales = task->gate + (p->hidden * 16 + first * 2) * groups;
    bool up_bounded, gate_bounded;
    TrillimStatus status = check_bounds(up_scales, count * groups, task->context, &up_bounded);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = check_bounds(gate_scales, count * groups, task->context, &gate_bounded);
    if (status != TRILLIM_OK) {
        return status;
    }
    bool factored = up_bounded && gate_bounded;
    bool fhm = factored && task->half != NULL && fhm_default_fpcr();
    float up[MAX_TILE], gate[MAX_TILE];
    for (size_t column = 0; column < count; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = column * groups;
        const uint8_t *up_codes = task->up + (first + column) * groups * 16;
        const uint8_t *gate_codes = task->gate + (first + column) * groups * 16;
        WeightRow weights[] = {
            {up_codes, up_scales + offset * 2},
            {gate_codes, gate_scales + offset * 2},
            {up_codes + groups * 16, up_scales + (offset + groups) * 2},
            {gate_codes + groups * 16, gate_scales + (offset + groups) * 2},
        };
        float values[4];
        if (fhm) {
            status = project_fhm_pair(task, weights, values);
        } else if (factored) {
            status = project_pair(task, weights, true, values);
        } else {
            status = project_pair(task, weights, false, values);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
        up[column] = values[0];
        gate[column] = values[1];
        up[column + 1] = values[2];
        gate[column + 1] = values[3];
    }
    void *outputs[] = {gate};
    const void *inputs[] = {gate};
    GeluParameters parameters = {count, TRILLIM_FP32, TRILLIM_FP32};
    status = trillim_neon_gelu_tanh(outputs, inputs, NULL, &parameters, task->context);
    if (status != TRILLIM_OK) {
        return status;
    }
    for (size_t i = 0; i < count; i += 4) {
        vst1q_f32(task->hidden + first + i, vmulq_f32(vld1q_f32(up + i), vld1q_f32(gate + i)));
    }
    return TRILLIM_OK;
}

static TrillimStatus run_hidden(Task *task, size_t tiles) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    return tiles == 1 ? project_hidden(task, 0, 0) : pool->run(pool->context, tiles, project_hidden, task);
}

static TrillimStatus run_hidden_fhm(Task *task, size_t tiles) {
    uint16_t input[MAX_FHM_WIDTH];
    uint32x4_t error = vdupq_n_u32(0);
    for (size_t k = 0; k < task->p->width; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16x8_t raw = vreinterpretq_u16_u8(vld1q_u8(task->input + k * 2));
        uint32x4_t lo = vshll_n_u16(vget_low_u16(raw), 16);
        uint32x4_t hi = vshll_high_n_u16(raw, 16);
        float16x8_t half = vcvt_high_f16_f32(vcvt_f16_f32(vreinterpretq_f32_u32(lo)), vreinterpretq_f32_u32(hi));
        error = vorrq_u32(error, veorq_u32(lo, vreinterpretq_u32_f32(vcvt_f32_f16(vget_low_f16(half)))));
        error = vorrq_u32(error, veorq_u32(hi, vreinterpretq_u32_f32(vcvt_high_f32_f16(half))));
        vst1q_u16(input + k, vreinterpretq_u16_f16(half));
    }
    if (vmaxvq_u32(error) == 0) {
        task->half = input;
    }
    return run_hidden(task, tiles);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_gelu_ffn(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    float *hidden = scratch;
    LinearParameters projection = {p->tokens, p->width, p->hidden, TRILLIM_BF16, TRILLIM_GRP_INT4, TRILLIM_FP32};
    bool hidden_handled = false;
#if defined(__APPLE__) || defined(__linux__)
    if (p->tokens == 1 || p->tokens >= 4) {
        TrillimStatus status = p->tokens == 1
            ? trillim_neon_try_packed_ffn_int8(
                hidden, inputs[0], inputs[1], inputs[2], &projection, gelu_tanh, context, &hidden_handled)
            : trillim_neon_try_packed_ffn_int8_prefill(
                hidden, inputs[0], inputs[1], inputs[2], &projection, gelu_tanh,
                trillim_neon_gelu_tanh4_prefill, context, &hidden_handled);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
#endif
    bool bounded = false;
    if (p->tokens == 1 && !hidden_handled) {
        TrillimStatus status = check_bounds(inputs[0], p->width, context, &bounded);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    if (bounded) {
        size_t workers = context->thread_pool->worker_count;
        if (workers == 0) {
            return TRILLIM_ERROR;
        }
        size_t columns = p->hidden / workers / 4;
        columns = columns < 16 ? 16 : columns > MAX_TILE ? MAX_TILE : columns / 4 * 4;
        size_t tiles = p->hidden / columns + (p->hidden % columns != 0);
        Task task = {hidden, inputs[0], inputs[1], inputs[2], p, context, columns, NULL};
        TrillimStatus status = p->width <= MAX_FHM_WIDTH && p->hidden >= 1152 && available_fhm() && fhm_default_fpcr()
            ? run_hidden_fhm(&task, tiles) : run_hidden(&task, tiles);
        if (status != TRILLIM_OK) {
            return status;
        }
    } else if (!hidden_handled) {
        size_t count = p->tokens * p->hidden;
        float *gate = hidden + count;
        void *up_outputs[] = {hidden}, *gate_outputs[] = {gate};
        const void *up_inputs[] = {inputs[0], inputs[1]}, *gate_inputs[] = {inputs[0], inputs[2]};
        TrillimStatus status = trillim_neon_packed_linear(up_outputs, up_inputs, NULL, &projection, context);
        if (status == TRILLIM_OK) {
            status = trillim_neon_packed_linear(gate_outputs, gate_inputs, NULL, &projection, context);
        }
        if (status == TRILLIM_OK) {
            const void *glu_inputs[] = {hidden, gate};
            status = trillim_neon_gelu_tanh_mul(up_outputs, glu_inputs, NULL, &count, context);
        }
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    const void *down_inputs[] = {hidden, inputs[3]};
    LinearParameters down = {p->tokens, p->hidden, p->outputs, TRILLIM_FP32, TRILLIM_GRP_INT4, TRILLIM_FP32};
    return trillim_neon_packed_linear(outputs, down_inputs, NULL, &down, context);
}

TRILLIM_KERNEL const char *trillim_neon_packed_gelu_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'up', 'gate', 'down')\n"
           "def bind(c):\n"
           "    x, up, gate, down = (c.inputs[name] for name in physical_inputs)\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or x.dtype != 'bf16' or output.dtype != 'fp32':\n"
           "        raise ValueError('packed GELU FFN requires BF16 input and FP32 output')\n"
           "    for tensor in (x, up, gate, down, output):\n"
           "        stride = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or stride) != stride:\n"
           "            raise ValueError('packed GELU FFN requires contiguous operands')\n"
           "    for weight in (up, gate, down):\n"
           "        if weight.dtype != 'grp_int4' or len(weight.shape) != 2 or _storage_shape(weight) != weight.shape:\n"
           "            raise ValueError('packed GELU FFN requires complete grouped INT4 matrices')\n"
           "    hidden, width = up.shape\n"
           "    if gate.shape != up.shape or down.shape[1] != hidden or x.shape[-1] != width or output.shape != (*x.shape[:-1], down.shape[0]):\n"
           "        raise ValueError('incompatible packed GELU FFN shapes')\n"
           "    if width.value is None or width.value < 1024 or width.value > 65536 or width.value % 32 or hidden.value is None or hidden.value < 16 or hidden.value % 4:\n"
           "        raise ValueError('packed GELU FFN requires aligned fixed-width projections')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, hidden=hidden, outputs=down.shape[0])\n"
           "def pattern(x: Tensor, up: Tensor, gate: Tensor, down: Tensor, *, up_bias=None, gate_bias=None, down_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if up_bias is not None or gate_bias is not None or down_bias is not None or activation is Silu:\n"
           "        raise ValueError('packed GELU FFN requires an unbiased GELU gate')\n"
           "    return (FFN(x, up=up, gate=gate, down=down, activation=lambda value: Gelu(value, approximate='tanh')),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_gelu_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 4, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_gelu_ffn_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
