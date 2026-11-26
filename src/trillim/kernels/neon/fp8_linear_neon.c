#include "common.h"
#include <arm_neon.h>
#if defined(__APPLE__)
#include <dlfcn.h>
#include <stdatomic.h>
#endif

#define SCALE_BLOCK 128
#define DIRECT_COLUMNS 32
#define BLAS_COLUMNS 64
#define BLAS_DEPTH 512
#define BLAS_ROWS 256
#define BLAS_LANES 32
#define BLAS_WORKSPACE (BLAS_ROWS * BLAS_DEPTH + BLAS_COLUMNS * BLAS_DEPTH + BLAS_ROWS * BLAS_COLUMNS)

typedef struct {
    size_t tokens;
    size_t width;
    size_t outputs;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    const uint8_t *scales;
    void *output;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t scale_columns;
    float *scratch;
    size_t lanes;
} Task;

static size_t ceil_div(size_t count, size_t block) {
    return count / block + (count % block != 0);
}

#if defined(__APPLE__)
typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
static Sgemm sgemm;
static GetThreading get_threading;
static SetThreading set_threading;
static atomic_uint blas_state;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
        void *blas = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LOCAL | RTLD_LAZY);
        if (blas != NULL) {
            void *address = dlsym(blas, "cblas_sgemm$NEWLAPACK");
            if (address == NULL) {
                address = dlsym(blas, "cblas_sgemm");
            }
            void *getter = dlsym(blas, "BLASGetThreading");
            void *setter = dlsym(blas, "BLASSetThreading");
            if (address != NULL && getter != NULL && setter != NULL) {
                _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                memcpy(&sgemm, &address, sizeof(sgemm));
                memcpy(&get_threading, &getter, sizeof(get_threading));
                memcpy(&set_threading, &setter, sizeof(set_threading));
            } else {
                dlclose(blas);
            }
        }
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

static bool blas_shape(const Parameters *p) {
    return p->tokens >= 16 && p->width >= 128 && p->width <= INT32_MAX &&
        p->outputs >= BLAS_COLUMNS && p->outputs <= INT32_MAX;
}

static size_t blas_tiles(const Parameters *p) {
    return ceil_div(p->tokens, BLAS_ROWS) * ceil_div(p->outputs, BLAS_COLUMNS);
}
#endif

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    size_t input_size = trillim_itemsize(p->input_dtype);
    size_t output_size = trillim_itemsize(p->output_dtype);
    if ((p->width != 0 && p->tokens > SIZE_MAX / input_size / p->width) ||
        p->tokens > SIZE_MAX / output_size / p->outputs ||
        !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_FP8_E4M3, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width * input_size;
    outputs[0] = p->tokens * p->outputs * output_size;
#if defined(__APPLE__)
    if (blas_shape(p)) {
        size_t tiles = blas_tiles(p);
        size_t lanes = tiles < BLAS_LANES ? tiles : BLAS_LANES;
        *scratch = lanes * BLAS_WORKSPACE * sizeof(float);
    }
#endif
    return TRILLIM_OK;
}

typedef struct {
    float32x4_t v0, v1, v2, v3;
} Values16;

static inline Values16 load_activation(const void *input, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        const float *values = (const float *)input + index;
        return (Values16){vld1q_f32(values), vld1q_f32(values + 4),
                          vld1q_f32(values + 8), vld1q_f32(values + 12)};
    }
    const uint16_t *values = (const uint16_t *)input + index;
    uint16x8_t low = vld1q_u16(values);
    uint16x8_t high = vld1q_u16(values + 8);
    if (dtype == TRILLIM_BF16) {
        return (Values16){vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(low), 16)),
                          vreinterpretq_f32_u32(vshll_high_n_u16(low, 16)),
                          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(high), 16)),
                          vreinterpretq_f32_u32(vshll_high_n_u16(high, 16))};
    }
    return (Values16){vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(low))),
                      vcvt_high_f32_f16(vreinterpretq_f16_u16(low)),
                      vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(high))),
                      vcvt_high_f32_f16(vreinterpretq_f16_u16(high))};
}

static inline Values16 load_weight(const uint8_t *weight, float32x4_t scale) {
    uint8x16_t code = vld1q_u8(weight);
    uint8x16_t magnitude = vandq_u8(code, vdupq_n_u8(127));
    uint8x16_t high = vaddq_u8(vsriq_n_u8(code, magnitude, 1), vdupq_n_u8(0x20));
    uint8x16_t low = vandq_u8(vshlq_n_u8(code, 7), vcgeq_u8(magnitude, vdupq_n_u8(8)));
    /* Normal FP16 bits need only a bias adjustment. The table normalizes FP8 subnormals. */
    const uint8x16_t subnormal_xor = {0x20, 0x38, 0x3d, 0x3f, 2, 3, 1, 0};
    high = veorq_u8(high, vqtbl1q_u8(subnormal_xor, magnitude));
    high = vbslq_u8(vceqq_u8(magnitude, vdupq_n_u8(127)), vdupq_n_u8(0x7e), high);
    float16x8_t first = vreinterpretq_f16_u8(vzip1q_u8(low, high));
    float16x8_t last = vreinterpretq_f16_u8(vzip2q_u8(low, high));
    return (Values16){vmulq_f32(vcvt_f32_f16(vget_low_f16(first)), scale),
                      vmulq_f32(vcvt_high_f32_f16(first), scale),
                      vmulq_f32(vcvt_f32_f16(vget_low_f16(last)), scale),
                      vmulq_f32(vcvt_high_f32_f16(last), scale)};
}

static inline float weight_scale(const Task *task, size_t output, size_t depth) {
    size_t index = output / SCALE_BLOCK * task->scale_columns + depth / SCALE_BLOCK;
    float scale;
    memcpy(&scale, task->scales + index * sizeof(scale), sizeof(scale));
    return scale;
}

static inline bool cancelled(const Task *task) {
    return task->context->is_cancelled(task->context->cancellation_context);
}

static inline void store_output(void *output, size_t index, uint32_t dtype, float value) {
    if (dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)output + index * sizeof(value), &value, sizeof(value));
    } else {
        uint16_t bits = dtype == TRILLIM_BF16 ? trillim_to_bf16(value) : trillim_to_fp16(value);
        memcpy((uint8_t *)output + index * sizeof(bits), &bits, sizeof(bits));
    }
}

static inline __attribute__((always_inline)) TrillimStatus project_direct(
    const Task *task, size_t tile, uint32_t dtype
) {
    const Parameters *p = task->p;
    size_t columns = ceil_div(p->outputs, DIRECT_COLUMNS);
    size_t row = tile / columns;
    size_t first = tile % columns * DIRECT_COLUMNS;
    size_t last = p->outputs - first < DIRECT_COLUMNS ? p->outputs : first + DIRECT_COLUMNS;
    size_t input = row * p->width;
    for (size_t column = first; column < last; ++column) {
        if (cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        const uint8_t *weights = task->weight + column * p->width;
        Values16 sum = {0};
        size_t k = 0;
        for (size_t block = 0; block < task->scale_columns; ++block) {
            if ((block & 15) == 15 && cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            float scale = weight_scale(task, column, k);
            float32x4_t vector_scale = vdupq_n_f32(scale);
            size_t end = p->width - k < SCALE_BLOCK ? p->width : k + SCALE_BLOCK;
            for (; end - k >= 16; k += 16) {
                Values16 x = load_activation(task->input, input + k, dtype);
                Values16 w = load_weight(weights + k, vector_scale);
                sum.v0 = vfmaq_f32(sum.v0, w.v0, x.v0);
                sum.v1 = vfmaq_f32(sum.v1, w.v1, x.v1);
                sum.v2 = vfmaq_f32(sum.v2, w.v2, x.v2);
                sum.v3 = vfmaq_f32(sum.v3, w.v3, x.v3);
            }
            float tail = 0.0f;
            for (; k < end; ++k) {
                float w = trillim_fp8_e4m3(weights[k]) * scale;
                tail = fmaf(w, trillim_load(task->input, input + k, dtype).f, tail);
            }
            sum.v0 = vsetq_lane_f32(vgetq_lane_f32(sum.v0, 0) + tail, sum.v0, 0);
        }
        float result = vaddvq_f32(vaddq_f32(vaddq_f32(sum.v0, sum.v1), vaddq_f32(sum.v2, sum.v3)));
        store_output(task->output, row * p->outputs + column, p->output_dtype, result);
    }
    return TRILLIM_OK;
}

static TrillimStatus direct(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    switch (task->p->input_dtype) {
    case TRILLIM_BF16: return project_direct(task, tile, TRILLIM_BF16);
    case TRILLIM_FP16: return project_direct(task, tile, TRILLIM_FP16);
    default: return project_direct(task, tile, TRILLIM_FP32);
    }
}

#if defined(__APPLE__)
static void pack_weights(const Task *task, size_t first_column, size_t columns,
                         size_t first_depth, size_t depth, float *target) {
    const Parameters *p = task->p;
    for (size_t column = 0; column < columns; ++column) {
        size_t output = first_column + column;
        const uint8_t *weights = task->weight + output * p->width + first_depth;
        float *row = target + column * depth;
        size_t k = 0;
        while (k < depth) {
            float scale = weight_scale(task, output, first_depth + k);
            float32x4_t vector_scale = vdupq_n_f32(scale);
            size_t end = depth - k < SCALE_BLOCK ? depth : k + SCALE_BLOCK;
            for (; end - k >= 16; k += 16) {
                Values16 w = load_weight(weights + k, vector_scale);
                vst1q_f32(row + k, w.v0);
                vst1q_f32(row + k + 4, w.v1);
                vst1q_f32(row + k + 8, w.v2);
                vst1q_f32(row + k + 12, w.v3);
            }
            for (; k < end; ++k) {
                row[k] = trillim_fp8_e4m3(weights[k]) * scale;
            }
        }
    }
}

static void pack_activations(const Task *task, size_t first_row, size_t rows,
                             size_t first_depth, size_t depth, float *target) {
    const Parameters *p = task->p;
    for (size_t row = 0; row < rows; ++row) {
        size_t input = (first_row + row) * p->width + first_depth;
        float *values = target + row * depth;
        size_t k = 0;
        for (; depth - k >= 16; k += 16) {
            Values16 x = load_activation(task->input, input + k, p->input_dtype);
            vst1q_f32(values + k, x.v0);
            vst1q_f32(values + k + 4, x.v1);
            vst1q_f32(values + k + 8, x.v2);
            vst1q_f32(values + k + 12, x.v3);
        }
        for (; k < depth; ++k) {
            values[k] = trillim_load(task->input, input + k, p->input_dtype).f;
        }
    }
}

static TrillimStatus project_blas(const Task *task, size_t lane) {
    const Parameters *p = task->p;
    size_t tiles = blas_tiles(p);
    size_t column_tiles = ceil_div(p->outputs, BLAS_COLUMNS);
    float *a_pack = task->scratch + lane * BLAS_WORKSPACE;
    float *b_pack = a_pack + BLAS_ROWS * BLAS_DEPTH;
    float *c_pack = b_pack + BLAS_COLUMNS * BLAS_DEPTH;
    for (size_t tile = lane; tile < tiles; tile += task->lanes) {
        size_t row = tile / column_tiles * BLAS_ROWS;
        size_t column = tile % column_tiles * BLAS_COLUMNS;
        size_t rows = p->tokens - row < BLAS_ROWS ? p->tokens - row : BLAS_ROWS;
        size_t columns = p->outputs - column < BLAS_COLUMNS ? p->outputs - column : BLAS_COLUMNS;
        size_t output = row * p->outputs + column;
        bool direct_output = p->output_dtype == TRILLIM_FP32;
        float *c = direct_output ? (float *)task->output + output : c_pack;
        int ldc = direct_output ? (int)p->outputs : (int)columns;
        for (size_t k = 0; k < p->width; k += BLAS_DEPTH) {
            if (cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            size_t depth = p->width - k < BLAS_DEPTH ? p->width - k : BLAS_DEPTH;
            const float *a = a_pack;
            int lda = (int)depth;
            if (p->input_dtype == TRILLIM_FP32) {
                a = (const float *)task->input + row * p->width + k;
                lda = (int)p->width;
            } else {
                pack_activations(task, row, rows, k, depth, a_pack);
            }
            pack_weights(task, column, columns, k, depth, b_pack);
            sgemm(101, 111, 112, (int)rows, (int)columns, (int)depth, 1.0f,
                  a, lda, b_pack, (int)depth, k == 0 ? 0.0f : 1.0f, c, ldc);
        }
        if (!direct_output) {
            for (size_t r = 0; r < rows; ++r) {
                for (size_t col = 0; col < columns; ++col) {
                    store_output(task->output, output + r * p->outputs + col, p->output_dtype,
                                 c_pack[r * columns + col]);
                }
            }
        }
    }
    return cancelled(task) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus blas(void *opaque, size_t lane, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    unsigned previous = get_threading();
    if (previous != 1 && set_threading(1) != 0) {
        return TRILLIM_ERROR;
    }
    TrillimStatus status = project_blas(task, lane);
    if (previous != 1) {
        (void)set_threading(previous);
    }
    return status;
}
#endif

TRILLIM_KERNEL TrillimStatus trillim_neon_fp8_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    if (!p->width) {
        memset(outputs[0], 0, p->tokens * p->outputs * trillim_itemsize(p->output_dtype));
        return TRILLIM_OK;
    }
    const uint8_t *weight = inputs[1];
    Task task = {inputs[0], weight, weight + p->outputs * p->width, outputs[0], p,
                 context, ceil_div(p->width, SCALE_BLOCK), scratch, 0};
    const TrillimThreadPool *pool = context->thread_pool;
#if defined(__APPLE__)
    if (blas_shape(p) && initialize_blas()) {
        size_t tiles = blas_tiles(p);
        task.lanes = tiles < BLAS_LANES ? tiles : BLAS_LANES;
        return pool->run(pool->context, task.lanes, blas, &task);
    }
#endif
    size_t count = p->tokens * ceil_div(p->outputs, DIRECT_COLUMNS);
    return count == 1 ? direct(&task, 0, 0) : pool->run(pool->context, count, direct, &task);
}

TRILLIM_KERNEL const char *trillim_neon_fp8_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "physical_inputs = ('x', 'weight')\n"
           "def bind(c):\n"
           "    x, weight = (c.inputs[name] for name in physical_inputs)\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires floating activations and output')\n"
           "    if weight.dtype != 'fp8_e4m3' or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete FP8 E4M3 matrix')\n"
           "    for tensor in (x, weight):\n"
           "        if tensor.strides and tensor.strides != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    values = _linear(c)\n"
           "    outputs, width = weight.shape\n"
           "    return dict(tokens=values['m'], width=width, outputs=outputs, input_dtype=values['a_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased FP8 linear projection')\n"
           "    output = Linear(x, weight)\n"
           "    return (output if dtype is None else Cast(output, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_fp8_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 2, 1, requirements, NULL, inputs, outputs, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_fp8_linear_capabilities(void) {
#if defined(__APPLE__)
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &capabilities;
}
