#include "common.h"
#include <arm_neon.h>
#include <math.h>

/* Keep the multiplication and addition rounding of the unfused loop. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { TILE = 32, DECODE_TILE = 64, LANES = 4, PACKED_KEYS = 128 };

typedef struct {
    size_t rows, heads, keys, values;
    uint32_t decay_dtype;
} Parameters;

typedef struct {
    const float *q, *k, *v, *beta;
    const void *log_decay;
    const float *initial;
    float *state, *output;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (!trillim_float(p->decay_dtype) || !p->heads || !p->keys || !p->values || p->values % TILE ||
        p->heads > SIZE_MAX / sizeof(float) / p->keys / p->values) {
        return TRILLIM_ERROR;
    }
    size_t head_bytes = p->heads * sizeof(float);
    if (p->rows > SIZE_MAX / head_bytes / p->keys ||
        p->rows > SIZE_MAX / head_bytes / p->values) {
        return TRILLIM_ERROR;
    }
    size_t scalar_bytes = p->rows * head_bytes;
    inputs[0] = inputs[1] = scalar_bytes * p->keys;
    inputs[2] = outputs[2] = scalar_bytes * p->values;
    inputs[3] = scalar_bytes;
    inputs[4] = scalar_bytes / sizeof(float) * trillim_itemsize(p->decay_dtype);
    inputs[5] = outputs[1] = head_bytes * p->keys * p->values;
    outputs[0] = sizeof(int64_t);
    *scratch = 0;
    return TRILLIM_OK;
}

static float decay_value(const Task *task, size_t element) {
    uint32_t dtype = task->parameters->decay_dtype;
    float value = expf(trillim_load(task->log_decay, element, dtype).f);
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    return value;
}

static inline __attribute__((always_inline)) TrillimStatus recurrence(
    const Task *task, size_t index, size_t columns, float *packed, bool single_row
) {
    const Parameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t tiles = p->values / columns;
    size_t head = index / tiles;
    size_t column = index % tiles * columns;
    size_t state_offset = head * p->keys * p->values + column;
    float *state = task->state + state_offset;
    const float *source = task->initial + state_offset;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->rows) {
        for (size_t i = 0; i < p->keys; ++i) {
            if (i && (i & 127) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            memcpy(state + i * p->values, source + i * p->values, columns * sizeof(float));
        }
        return TRILLIM_OK;
    }

    float *working = packed ? packed : state;
    size_t working_stride = packed ? columns : p->values;
    size_t source_stride = p->values;
    for (size_t row = 0; row < p->rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        bool final_row = row + 1 == p->rows;
        float *target = final_row ? state : working;
        size_t target_stride = final_row ? p->values : working_stride;
        size_t element = row * p->heads + head;
        const float *query = task->q + element * p->keys;
        const float *key = task->k + element * p->keys;
        const float *value = task->v + element * p->values + column;
        float32x4_t decay = vdupq_n_f32(decay_value(task, element));
        float32x4_t predicted[columns / LANES], update[columns / LANES];
        for (size_t lane = 0; lane < columns / LANES; ++lane) {
            predicted[lane] = vdupq_n_f32(0.0f);
        }
        for (size_t i = 0; i < p->keys; ++i) {
            if (i && (i & 127) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const float *input = source + i * source_stride;
            float *destination = working + i * working_stride;
            float32x4_t k = vdupq_n_f32(key[i]);
            for (size_t lane = 0; lane < columns / LANES; ++lane) {
                size_t offset = lane * LANES;
                float32x4_t memory = vmulq_f32(vld1q_f32(input + offset), decay);
                if (!single_row) {
                    vst1q_f32(destination + offset, memory);
                }
                predicted[lane] = vaddq_f32(predicted[lane], vmulq_f32(memory, k));
            }
        }
        float32x4_t result[columns / LANES];
        for (size_t lane = 0; lane < columns / LANES; ++lane) {
            float32x4_t residual = vsubq_f32(vld1q_f32(value + lane * LANES),
                                            predicted[lane]);
            update[lane] = vmulq_n_f32(residual, task->beta[element]);
            result[lane] = vdupq_n_f32(0.0f);
        }
        for (size_t i = 0; i < p->keys; ++i) {
            if (i && (i & 127) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const float *input = single_row ? source + i * source_stride : working + i * working_stride;
            float *destination = target + i * target_stride;
            float32x4_t k = vdupq_n_f32(key[i]), q = vdupq_n_f32(query[i]);
            for (size_t lane = 0; lane < columns / LANES; ++lane) {
                size_t offset = lane * LANES;
                float32x4_t memory = vld1q_f32(input + offset);
                if (single_row) {
                    memory = vmulq_f32(memory, decay);
                }
                memory = vaddq_f32(memory, vmulq_f32(k, update[lane]));
                vst1q_f32(destination + offset, memory);
                result[lane] = vaddq_f32(result[lane], vmulq_f32(memory, q));
            }
        }
        float *output = task->output + element * p->values + column;
        for (size_t lane = 0; lane < columns / LANES; ++lane) {
            vst1q_f32(output + lane * LANES, result[lane]);
        }
        source = working;
        source_stride = working_stride;
    }
    return TRILLIM_OK;
}

static TrillimStatus recurrence_strided(void *opaque, size_t index, size_t worker) {
    (void)worker;
    return recurrence(opaque, index, TILE, NULL, false);
}

static TrillimStatus recurrence_packed(void *opaque, size_t index, size_t worker) {
    (void)worker;
    /* Keep both per-token passes in a dense, cache-resident tile. */
    _Alignas(128) float packed[PACKED_KEYS * TILE];
    return recurrence(opaque, index, TILE, packed, false);
}

static TrillimStatus recurrence_decode(void *opaque, size_t index, size_t worker) {
    (void)worker;
    /* Recompute the decay on the second pass to avoid an intermediate state write. */
    return recurrence(opaque, index, DECODE_TILE, NULL, true);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_gated_delta(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
        outputs[1], outputs[2], p, context
    };
    size_t columns = TILE;
    TrillimTask compute = recurrence_strided;
    if (p->rows == 1 && p->values % DECODE_TILE == 0 &&
        p->heads * (p->values / DECODE_TILE) >= context->thread_pool->worker_count) {
        columns = DECODE_TILE;
        compute = recurrence_decode;
    } else if (p->rows > 1 && p->keys <= PACKED_KEYS && p->values > TILE) {
        compute = recurrence_packed;
    }
    size_t tiles = p->heads * (p->values / columns);
    TrillimStatus status = context->thread_pool->run(
        context->thread_pool->context, tiles, compute, &task);
    if (status == TRILLIM_OK) {
        *(int64_t *)outputs[0] = (int64_t)p->rows;
    }
    return status;
}

TRILLIM_KERNEL const char *trillim_neon_gated_delta_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "\n"
           "physical_inputs = ('q', 'k', 'v', 'beta', 'log_decay', 'initial.1')\n"
           "\n"
           "def bind(context):\n"
           "    q, k, v, beta, log_decay, memory = (context.inputs[name] for name in physical_inputs)\n"
           "    counter, state, output = context.outputs\n"
           "    for tensor in (q, k, v, beta, memory, state, output):\n"
           "        if tensor.dtype != 'fp32' or tensor.strides != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('GatedDelta requires contiguous FP32 tensors')\n"
           "    if log_decay.dtype not in {'fp16', 'bf16', 'fp32'} or log_decay.strides != _contiguous_strides(log_decay.shape):\n"
           "        raise ValueError('GatedDelta requires contiguous floating decay values')\n"
           "    rows, heads, keys = q.shape\n"
           "    values = v.shape[2]\n"
           "    # The loop replaces every row in the initial output buffer.\n"
           "    if (k.shape != q.shape or v.shape != (rows, heads, values) or\n"
           "        context.inputs['initial.2'].shape != v.shape or\n"
           "        beta.shape != (rows, heads) or log_decay.shape != beta.shape or\n"
           "        memory.shape != (heads, keys, values) or state.shape != memory.shape or\n"
           "        output.shape[1:] != v.shape[1:] or counter.shape or counter.dtype != 'int64'):\n"
           "        raise ValueError('GatedDelta requires matching sequence and state dimensions')\n"
           "    return {'rows': rows, 'heads': heads, 'keys': keys, 'values': values, 'decay_dtype': {'fp16': 9, 'bf16': 10, 'fp32': 11}[log_decay.dtype]}\n"
           "\n"
           "# The matcher seeds the three loop initials before its captures.\n"
           "def pattern(initial: tuple, _index_seed: Tensor, _state_seed: Tensor,\n"
           "            _output_seed: Tensor, v: Tensor, q: Tensor, k: Tensor,\n"
           "            beta: Tensor, log_decay: Tensor) -> tuple[Tensor, ...]:\n"
           "    if len(initial) != 3 or any(x.dtype != 'fp32' for x in (initial[1], q, k, v, beta)):\n"
           "        raise ValueError('GatedDelta requires an FP32 recurrence')\n"
           "    if log_decay.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('GatedDelta requires floating decay values')\n"
           "    rows, heads, values = v.shape.dimensions\n"
           "    if heads is None or values is None:\n"
           "        raise ValueError('GatedDelta requires fixed head and value widths')\n"
           "    if rows is None:\n"
           "        # The template carries an int64 counter. The target has its own bound.\n"
           "        v._producer.args = v._producer.args.model_copy(update={'maximum_shape': (2**63 - 1, heads, values)})\n"
           "\n"
           "    def step(carry):\n"
           "        index, memory, result = carry\n"
           "        qt = Take(q, index, rank=3, index_rank=0)\n"
           "        kt = Take(k, index, rank=3, index_rank=0)\n"
           "        vt = Take(v, index, rank=3, index_rank=0)\n"
           "        bt = Take(beta, index, rank=2, index_rank=0)\n"
           "        gt = Take(log_decay, index, rank=2, index_rank=0)\n"
           "        memory = Mul(memory, Reshape(Exp(gt), (-1, 1, 1)))\n"
           "        key = Reshape(kt, (Dim(kt, 0), Dim(kt, 1), 1))\n"
           "        residual = Sub(vt, Sum(Mul(memory, key), axes=(-2,)))\n"
           "        update = Mul(residual, Reshape(bt, (-1, 1)))\n"
           "        memory = Add(memory, Mul(key, Reshape(update, (Dim(update, 0), 1, Dim(update, 1)))))\n"
           "        token = Sum(Mul(memory, Reshape(qt, (Dim(qt, 0), Dim(qt, 1), 1))), axes=(-2,))\n"
           "        result = Scatter(result, indices=Reshape(index, (1,)), values=token)\n"
           "        return Add(index, 1), memory, result\n"
           "\n"
           "    return While((Constant(0), initial[1], initial[2]),\n"
           "                 condition=lambda s: Compare(s[0], Dim(v, 0), predicate='lt'), body=step)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_gated_delta_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"keys", "size_t", offsetof(Parameters, keys), NULL},
        {"values", "size_t", offsetof(Parameters, values), NULL},
        {"decay_dtype", "uint32_t", offsetof(Parameters, decay_dtype), NULL}
    };
    static const size_t inputs[] = {4, 4, 4, 4, 2, 4};
    static const size_t outputs[] = {8, 4, 4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 6, 3,
        requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_gated_delta_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION,
#if defined(__APPLE__) && defined(__aarch64__)
        TRILLIM_CAP_LIBM
#else
        TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM
#endif
    };
    return &capabilities;
}
