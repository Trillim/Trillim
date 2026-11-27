#include "common.h"
#include <math.h>
#include <arm_neon.h>
#include <stdatomic.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t count;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const float *table;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static float round_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static float gelu_tanh(float x, uint32_t dtype) {
    float square = round_dtype(x * x, dtype);
    float cube = round_dtype(square * x, dtype);
    float correction = round_dtype(cube * round_dtype(0.044715f, dtype), dtype);
    float sum = round_dtype(x + correction, dtype);
    float argument = round_dtype(sum * round_dtype(0.7978845608028654f, dtype), dtype);
    float scaled = round_dtype(argument * 2.0f, dtype);
    float e = round_dtype(expf(-fabsf(scaled)), dtype);
    float denominator = round_dtype(e + 1.0f, dtype);
    float probability = scaled >= 0.0f ? 1.0f / denominator : round_dtype(e / denominator, dtype);
    float cdf = probability * 2.0f - 1.0f;
    float half = round_dtype(x * 0.5f, dtype);
    return half * (cdf + 1.0f);
}

static float bf16_gelu[UINT16_MAX + 1];
static atomic_uint table_state;

static TrillimStatus prepare_table(const TrillimKernelContext *context) {
    unsigned empty = 0;
    if (!atomic_compare_exchange_strong_explicit(&table_state, &empty, 1,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        return TRILLIM_OK;
    }
    for (size_t i = 0; i <= UINT16_MAX; ++i) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            atomic_store_explicit(&table_state, 0, memory_order_release);
            return TRILLIM_CANCELLED;
        }
        bf16_gelu[i] = gelu_tanh(trillim_bf16((uint16_t)i), TRILLIM_BF16);
    }
    atomic_store_explicit(&table_state, 2, memory_order_release);
    return TRILLIM_OK;
}

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        p->count > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->count * trillim_itemsize(p->input_dtype);
    outputs[0] = p->count * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; macOS catalog hardware flags are unavailable. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

static float32x4_t exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Cody-Waite reduction leaves |r| <= log(2)/2 for the degree-seven series. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

static float32x4_t gelu_tanh4(float32x4_t x) {
    float32x4_t cube = vmulq_f32(vmulq_f32(x, x), x);
    float32x4_t sum = vaddq_f32(x, vmulq_n_f32(cube, 0.044715f));
    float32x4_t argument = vmulq_n_f32(sum, 0.7978845608028654f);
    float32x4_t scaled = vmulq_n_f32(argument, 2.0f);
    float32x4_t e = exp4(vnegq_f32(vabsq_f32(scaled)));
    float32x4_t denominator = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t positive = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t negative = vdivq_f32(e, denominator);
    float32x4_t probability = vbslq_f32(vcgeq_f32(scaled, vdupq_n_f32(0.0f)), positive, negative);
    float32x4_t cdf = vsubq_f32(vmulq_n_f32(probability, 2.0f), vdupq_n_f32(1.0f));
    return vmulq_f32(vmulq_n_f32(x, 0.5f), vaddq_f32(cdf, vdupq_n_f32(1.0f)));
}

static TrillimStatus activate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t begin = tile * 1024;
    size_t end = p->count - begin < 1024 ? p->count : begin + 1024;
    if (task->table) {
        for (size_t i = begin; i < end; ++i) {
            uint16_t bits;
            memcpy(&bits, (const uint8_t *)task->input + i * sizeof(bits), sizeof(bits));
            memcpy((uint8_t *)task->output + i * sizeof(float), &task->table[bits], sizeof(float));
        }
        return TRILLIM_OK;
    }
    size_t i = begin;
    if (p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_FP32) {
        for (; end - i >= 4; i += 4) {
            float32x4_t input = vld1q_f32((const float *)task->input + i);
            vst1q_f32((float *)task->output + i, gelu_tanh4(input));
        }
    }
    for (; i < end; ++i) {
        float input = trillim_load(task->input, i, p->input_dtype).f;
        TrillimValue value = {.f = gelu_tanh(input, p->input_dtype)};
        trillim_store(task->output, i, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const Parameters *p = parameters;
    return output == 0 && input == 0 && p->input_dtype == p->output_dtype;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gelu_tanh(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    const float *table = NULL;
    if (p->input_dtype == TRILLIM_BF16 && p->output_dtype == TRILLIM_FP32) {
        unsigned state = atomic_load_explicit(&table_state, memory_order_acquire);
        if (state == 0 && p->count >= 4 * (UINT16_MAX + 1)) {
            /* Amortize the exact lookup table across a large activation. */
            TrillimStatus status = prepare_table(context);
            if (status != TRILLIM_OK) {
                return status;
            }
            state = atomic_load_explicit(&table_state, memory_order_acquire);
        }
        if (state == 2) {
            table = bf16_gelu;
        }
    }
    Task task = {outputs[0], inputs[0], table, p, context};
    size_t tiles = p->count / 1024 + (p->count % 1024 != 0);
    if (tiles == 1) {
        return activate(&task, 0, 0);
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, activate, &task);
}

TRILLIM_KERNEL const char *trillim_neon_gelu_tanh_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _native_dtype\n"
           "def bind(c):\n"
           "    x, output = c.inputs['x'], c.outputs[0]\n"
           "    if c.attrs.get('approximate', 'tanh') != 'tanh' or x.shape != output.shape:\n"
           "        raise ValueError('tanh GELU requires matching tensor shapes')\n"
           "    for value in (x, output):\n"
           "        if value.dtype not in {'fp16', 'bf16', 'fp32'} or (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('tanh GELU requires contiguous native floating tensors')\n"
           "    return dict(count=_product(x.shape), input_dtype=_native_dtype(x.dtype), output_dtype=_native_dtype(output.dtype))\n"
           "def pattern(x: Tensor, *, approximate: str = 'tanh', dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    y = Gelu(x, approximate=approximate)\n"
           "    return (y if dtype is None else Cast(y, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gelu_tanh_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 1, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gelu_tanh_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM | TRILLIM_NEON_REQUIRED
    };
    return &capabilities;
}
