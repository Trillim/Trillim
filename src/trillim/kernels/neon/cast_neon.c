#include "common.h"
#include <arm_neon.h>

#if defined(__APPLE__) && defined(__aarch64__)
/* AArch64 NEON is baseline; the macOS catalog does not read CPU feature flags. */
#define TRILLIM_NEON_REQUIRED UINT64_C(0)
#else
#define TRILLIM_NEON_REQUIRED TRILLIM_CAP_NEON
#endif

enum { TILE_ELEMENTS = 32768, CHUNK_ELEMENTS = 4096 };

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} Parameters;

typedef struct {
    uint8_t *output;
    const uint8_t *input;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t input_bytes, output_bytes;
    bool contiguous;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        p->input_dtype == p->output_dtype || (p->rank && (!p->shape || !p->strides))) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] == 0) {
            return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
        }
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    size_t span;
    size_t input_bytes = trillim_itemsize(p->input_dtype);
    size_t output_bytes = trillim_itemsize(p->output_dtype);
    if (count != p->count || count > SIZE_MAX / output_bytes ||
        !trillim_span(p->rank, p->shape, p->strides, &span) || span > SIZE_MAX / input_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * input_bytes;
    outputs[0] = count * output_bytes;
    return TRILLIM_OK;
}

static inline uint16x4_t narrow_bf16(uint32x4_t bits) {
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
    uint32x4_t nan = vcgtq_u32(magnitude, vdupq_n_u32(0x7f800000));
    uint16x4_t rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
    uint16x4_t quiet = vorr_u16(vmovn_u32(upper), vdup_n_u16(0x40));
    return vbsl_u16(vmovn_u32(nan), quiet, rounded);
}

static inline uint32x4_t widen_fp16(uint16x4_t bits) {
    uint32x4_t widened = vreinterpretq_u32_f32(vcvt_f32_f16(vreinterpret_f16_u16(bits)));
    uint16x4_t magnitude = vand_u16(bits, vdup_n_u16(0x7fff));
    uint16x4_t signaling = vand_u16(vcgt_u16(magnitude, vdup_n_u16(0x7c00)),
                                    vclt_u16(magnitude, vdup_n_u16(0x7e00)));
    /* FCVTL quiets signaling NaNs. The scalar FP16 decoder preserves that bit. */
    uint32x4_t quiet_bit = vshll_n_u16(vand_u16(signaling, vdup_n_u16(0x40)), 16);
    return vbicq_u32(widened, quiet_bit);
}

static inline void convert_eight(
    uint8_t *output, const uint8_t *input, uint32_t input_dtype, uint32_t output_dtype
) {
    uint32x4_t lo, hi;
    if (input_dtype == TRILLIM_FP32) {
        memcpy(&lo, input, sizeof(lo));
        memcpy(&hi, input + sizeof(lo), sizeof(hi));
    } else {
        uint16x8_t bits;
        memcpy(&bits, input, sizeof(bits));
        if (input_dtype == TRILLIM_BF16) {
            lo = vshll_n_u16(vget_low_u16(bits), 16);
            hi = vshll_high_n_u16(bits, 16);
        } else {
            lo = widen_fp16(vget_low_u16(bits));
            hi = widen_fp16(vget_high_u16(bits));
        }
    }
    if (output_dtype == TRILLIM_FP32) {
        memcpy(output, &lo, sizeof(lo));
        memcpy(output + sizeof(lo), &hi, sizeof(hi));
        return;
    }
    uint16x8_t bits;
    if (output_dtype == TRILLIM_BF16) {
        bits = vcombine_u16(narrow_bf16(lo), narrow_bf16(hi));
    } else {
        float16x4_t low_half = vcvt_f16_f32(vreinterpretq_f32_u32(lo));
        bits = vreinterpretq_u16_f16(vcvt_high_f16_f32(low_half, vreinterpretq_f32_u32(hi)));
    }
    memcpy(output, &bits, sizeof(bits));
}

static inline __attribute__((always_inline)) size_t convert_vectors(
    uint8_t *output, const uint8_t *input, size_t count, uint32_t input_dtype, uint32_t output_dtype
) {
    size_t input_bytes = trillim_itemsize(input_dtype);
    size_t output_bytes = trillim_itemsize(output_dtype);
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        convert_eight(output + i * output_bytes, input + i * input_bytes, input_dtype, output_dtype);
    }
    return i;
}

static void convert_chunk(
    const Task *task, size_t output_index, size_t input_index, size_t count, bool vectorize
) {
    const Parameters *p = task->p;
    uint8_t *output = task->output + output_index * task->output_bytes;
    const uint8_t *input = task->input + input_index * task->input_bytes;
    size_t i = 0;
    if (p->input_dtype == TRILLIM_BF16 && p->output_dtype == TRILLIM_FP32) {
        i = convert_vectors(output, input, count, TRILLIM_BF16, TRILLIM_FP32);
    } else if (p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_BF16) {
        i = convert_vectors(output, input, count, TRILLIM_FP32, TRILLIM_BF16);
    } else if (vectorize) {
        i = convert_vectors(output, input, count, p->input_dtype, p->output_dtype);
    }
    for (; i < count; ++i) {
        trillim_store(output, i, p->output_dtype, trillim_load(input, i, p->input_dtype));
    }
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    bool vectorize = true;
    if (p->input_dtype == TRILLIM_FP16 || p->output_dtype == TRILLIM_FP16) {
        /* FPCR affects hardware conversions. Scalar conversions operate on bits. */
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        vectorize = fpcr == 0;
    }
    size_t index = tile * TILE_ELEMENTS;
    size_t limit = p->count - index < TILE_ELEMENTS ? p->count : index + TILE_ELEMENTS;
    while (index < limit) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t length = limit - index < CHUNK_ELEMENTS ? limit - index : CHUNK_ELEMENTS;
        size_t source = index;
        size_t stride = 1;
        if (!task->contiguous) {
            size_t width = p->shape[p->rank - 1];
            size_t remaining = width - index % width;
            if (remaining < length) {
                length = remaining;
            }
            source = trillim_offset(index, p->rank, p->shape, p->strides);
            stride = p->strides[p->rank - 1];
        }
        if (stride == 1) {
            convert_chunk(task, index, source, length, vectorize);
        } else {
            for (size_t i = 0; i < length; ++i) {
                TrillimValue value = trillim_load(task->input, source + i * stride, p->input_dtype);
                trillim_store(task->output, index + i, p->output_dtype, value);
            }
        }
        index += length;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_cast(
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
    Task task = {
        outputs[0], inputs[0], p, context,
        trillim_itemsize(p->input_dtype), trillim_itemsize(p->output_dtype),
        trillim_contiguous(p->rank, p->shape, p->strides)
    };
    size_t tiles = p->count / TILE_ELEMENTS + (p->count % TILE_ELEMENTS != 0);
    if (tiles > 1 && context->thread_pool->worker_count > 1) {
        return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
    }
    for (size_t tile = 0; tile < tiles; ++tile) {
        TrillimStatus status = apply(&task, tile, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_neon_cast_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _cast\n"
           "def bind(context):\n"
           "    source, output = context.inputs['x'], context.outputs[0]\n"
           "    floats = ('fp16', 'bf16', 'fp32')\n"
           "    if source.dtype not in floats or output.dtype not in floats or source.dtype == output.dtype:\n"
           "        raise ValueError('NEON cast requires different floating dtypes')\n"
           "    p = _cast(context)\n"
           "    return {name: p[name] for name in ('count', 'rank', 'shape', 'strides', 'input_dtype', 'output_dtype')}\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_cast_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_cast_capabilities(void) {
    /* Keep the scalar Cast capability for default kernel selection on macOS. */
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_NEON_REQUIRED | TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
