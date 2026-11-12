#include "common.h"
#include <math.h>

#define USE_NEON 0
#if USE_NEON
#include <arm_neon.h>
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    size_t tokens;
    size_t heads;
    size_t dim;
    size_t axes;
    size_t rotary_dim;
    size_t pairs;
    size_t rank;
    const size_t *strides;
    const size_t *sections;
    const float *frequencies;
    size_t position_token_stride;
    size_t position_axis_stride;
    uint32_t position_dtype;
    uint32_t interleaved;
} AxialParameters;

typedef struct {
    float *output;
    const float *input;
    const void *positions;
    float *angles;
    const AxialParameters *parameters;
    const TrillimKernelContext *context;
} AxialTask;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const AxialParameters *p = parameters;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    size_t position_bytes = trillim_itemsize(p->position_dtype);
    if (p->rank != 3 || !p->axes || p->axes > 32 || !position_bytes || p->interleaved > 1 ||
        !p->rotary_dim || p->rotary_dim > 4096 || p->rotary_dim > p->dim || p->pairs != p->rotary_dim / 2) {
        return TRILLIM_ERROR;
    }
    size_t dimensions = 0;
    for (size_t axis = 0; axis < p->axes; ++axis) {
        size_t section = p->sections[axis];
        if (!section || section % 2 || section > p->rotary_dim - dimensions) {
            return TRILLIM_ERROR;
        }
        dimensions += section;
    }
    if (dimensions != p->rotary_dim) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->tokens) {
        return TRILLIM_OK;
    }
    size_t shape[] = {p->tokens, p->heads, p->dim};
    size_t position_shape[] = {p->tokens, p->axes};
    size_t position_strides[] = {p->position_token_stride, p->position_axis_stride};
    size_t span, positions;
    if (!trillim_span(3, shape, p->strides, &span) || span > SIZE_MAX / sizeof(float) ||
        !trillim_span(2, position_shape, position_strides, &positions) || positions > SIZE_MAX / position_bytes ||
        p->heads > SIZE_MAX / p->tokens || p->heads * p->tokens > SIZE_MAX / sizeof(float) / p->dim ||
        p->tokens > SIZE_MAX / sizeof(float) / p->rotary_dim) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * sizeof(float);
    inputs[1] = positions * position_bytes;
    outputs[0] = p->tokens * p->heads * p->dim * sizeof(float);
    *scratch = p->tokens * p->rotary_dim * sizeof(float);
    return TRILLIM_OK;
}

static TrillimStatus prepare(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const AxialTask *task = opaque;
    const AxialParameters *p = task->parameters;
    float *cosines = task->angles + token * p->rotary_dim;
    float *sines = cosines + p->pairs;
    size_t offset = 0;
    for (size_t axis = 0; axis < p->axes; ++axis) {
        TrillimValue raw = trillim_load(task->positions,
            token * p->position_token_stride + axis * p->position_axis_stride, p->position_dtype);
        float position = trillim_float(p->position_dtype) ? raw.f :
            (trillim_unsigned(p->position_dtype) ? (float)raw.u : (float)raw.i);
        size_t pairs = p->sections[axis] / 2;
        for (size_t pair = 0; pair < pairs; ++pair) {
            if ((pair & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float angle = position * p->frequencies[offset + pair];
            if (angle == 0) {
                cosines[offset + pair] = 1;
                sines[offset + pair] = angle;
            } else {
                cosines[offset + pair] = cosf(angle);
                sines[offset + pair] = sinf(angle);
            }
        }
        offset += pairs;
    }
    return TRILLIM_OK;
}

static TrillimStatus rotate(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const AxialTask *task = opaque;
    const AxialParameters *p = task->parameters;
    size_t rows = p->tokens * p->heads;
    size_t block = rows >= 2048 ? 64 : 8;
    size_t first = tile * block;
    size_t last = rows - first < block ? rows : first + block;
    for (size_t row = first; row < last; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t token = row / p->heads, head = row % p->heads;
        size_t base = token * p->strides[0] + head * p->strides[1];
        float *output = task->output + row * p->dim;
        const float *cosines = task->angles + token * p->rotary_dim;
        const float *sines = cosines + p->pairs;
        size_t offset = 0;
        for (size_t axis = 0; axis < p->axes; ++axis) {
            size_t pairs = p->sections[axis] / 2;
            size_t pair = 0;
#if USE_NEON
            if (p->strides[2] == 1) {
                for (; pairs - pair >= 4; pair += 4) {
                    if ((pair & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    float32x4_t a, b;
                    if (p->interleaved) {
                        float32x4x2_t values = vld2q_f32(task->input + base + offset + pair * 2);
                        a = values.val[0];
                        b = values.val[1];
                    } else {
                        a = vld1q_f32(task->input + base + offset + pair);
                        b = vld1q_f32(task->input + base + offset + pairs + pair);
                    }
                    float32x4_t c = vld1q_f32(cosines + offset / 2 + pair);
                    float32x4_t s = vld1q_f32(sines + offset / 2 + pair);
                    float32x4_t ac = vmulq_f32(a, c), bs = vmulq_f32(b, s);
                    float32x4_t bc = vmulq_f32(b, c), as = vmulq_f32(a, s);
                    float32x4_t first_value = vsubq_f32(ac, bs), second_value = vaddq_f32(bc, as);
                    if (p->interleaved) {
                        float32x4x2_t values = {{first_value, second_value}};
                        vst2q_f32(output + offset + pair * 2, values);
                    } else {
                        vst1q_f32(output + offset + pair, first_value);
                        vst1q_f32(output + offset + pairs + pair, second_value);
                    }
                }
            }
#endif
            for (; pair < pairs; ++pair) {
                if ((pair & 63) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t a = offset + (p->interleaved ? pair * 2 : pair);
                size_t b = p->interleaved ? a + 1 : a + pairs;
                float av = task->input[base + a * p->strides[2]];
                float bv = task->input[base + b * p->strides[2]];
                float c = cosines[offset / 2 + pair], s = sines[offset / 2 + pair];
                float ac = av * c, bs = bv * s;
                float bc = bv * c, as = av * s;
                output[a] = ac - bs;
                output[b] = bc + as;
            }
            offset += p->sections[axis];
        }
        for (size_t channel = offset; channel < p->dim; ++channel) {
            if ((channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            output[channel] = task->input[base + channel * p->strides[2]];
        }
    }
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (parameters == NULL) {
        return output == 0 && input == 0;
    }
    const AxialParameters *p = parameters;
    return output == 0 && input == 0 && p->strides[2] == 1 &&
        (p->heads <= 1 || p->strides[1] == p->dim) &&
        (p->tokens <= 1 || p->strides[0] == p->heads * p->dim);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_axial_rotary(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const AxialParameters *p = parameters;
    if (!p->tokens || !p->heads) {
        return TRILLIM_OK;
    }
    AxialTask task = {outputs[0], inputs[0], inputs[1], scratch, p, context};
    TrillimStatus status = context->thread_pool->run(context->thread_pool->context, p->tokens, prepare, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t rows = p->tokens * p->heads;
    size_t block = rows >= 2048 ? 64 : 8;
    size_t tiles = rows / block + (rows % block != 0);
    return context->thread_pool->run(context->thread_pool->context, tiles, rotate, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_axial_rotary_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _broadcast_strides, _native_dtype\n"
           "import math\n"
           "import numpy as np\n"
           "physical_inputs = ('x', 'positions')\n"
           "def pattern(x: Tensor, positions: Tensor, *, sections: tuple[int, ...], theta: float = 10000.0, interleaved: bool = True, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (AxialRotary(x, positions, sections=sections, theta=theta, interleaved=interleaved, dtype=dtype),)\n"
           "def bind(context):\n"
           "    x, positions = context.inputs['x'], context.inputs['positions']\n"
           "    if x.dtype != 'fp32' or context.outputs[0].dtype != 'fp32' or len(x.shape) != 3 or len(positions.shape) != 2:\n"
           "        raise ValueError('AxialRotary requires FP32 activations and rank-two positions')\n"
           "    sections = context.attrs['sections']\n"
           "    theta = context.attrs.get('theta', 10000.0)\n"
           "    interleaved = context.attrs.get('interleaved', True)\n"
           "    if not isinstance(sections, tuple) or not 1 <= len(sections) <= 32 or any(not isinstance(d, int) or d <= 0 or d % 2 for d in sections) or sum(sections) > 4096:\n"
           "        raise ValueError('AxialRotary requires positive even sections totaling at most 4096')\n"
           "    if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0 or not isinstance(interleaved, bool):\n"
           "        raise ValueError('invalid AxialRotary frequency attributes')\n"
           "    if positions.shape[1].op != 'const' or positions.shape[1].value < len(sections):\n"
           "        raise ValueError('AxialRotary positions have too few axes')\n"
           "    ps = positions.strides or _contiguous_strides(positions.shape)\n"
           "    frequencies = tuple(float(value) for d in sections for value in np.exp(np.arange(d // 2, dtype=np.float32) * np.float32(-2.0 * math.log(theta) / d)))\n"
           "    return dict(tokens=x.shape[0], heads=x.shape[1], dim=x.shape[2], axes=len(sections), rotary_dim=sum(sections), pairs=sum(sections) // 2, rank=3, strides=x.strides or _contiguous_strides(x.shape), sections=sections, frequencies=frequencies, position_token_stride=_broadcast_strides((positions.shape[0],), (x.shape[0],), (ps[0],))[0], position_axis_stride=ps[1], position_dtype=_native_dtype(positions.dtype), interleaved=int(interleaved))\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_axial_rotary_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(AxialParameters, tokens), NULL},
        {"heads", "size_t", offsetof(AxialParameters, heads), NULL},
        {"dim", "size_t", offsetof(AxialParameters, dim), NULL},
        {"axes", "size_t", offsetof(AxialParameters, axes), NULL},
        {"rotary_dim", "size_t", offsetof(AxialParameters, rotary_dim), NULL},
        {"pairs", "size_t", offsetof(AxialParameters, pairs), NULL},
        {"rank", "size_t", offsetof(AxialParameters, rank), NULL},
        {"strides", "size_t", offsetof(AxialParameters, strides), "rank"},
        {"sections", "size_t", offsetof(AxialParameters, sections), "axes"},
        {"frequencies", "float", offsetof(AxialParameters, frequencies), "pairs"},
        {"position_token_stride", "size_t", offsetof(AxialParameters, position_token_stride), NULL},
        {"position_axis_stride", "size_t", offsetof(AxialParameters, position_axis_stride), NULL},
        {"position_dtype", "uint32_t", offsetof(AxialParameters, position_dtype), NULL},
        {"interleaved", "uint32_t", offsetof(AxialParameters, interleaved), NULL},
    };
    static const size_t inputs[] = {4, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(AxialParameters), TRILLIM_ALIGNOF(AxialParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, can_alias, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_axial_rotary_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
