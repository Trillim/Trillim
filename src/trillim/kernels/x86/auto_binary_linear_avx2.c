#include "common.h"
#include <stdatomic.h>
#include <immintrin.h>

enum { WEIGHT_GROUP = 128, ACTIVATION_GROUP = 32, ROWS = 4, COLUMNS = 16 };

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype, input_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    void *output;
    int8_t *codes;
    float *scales;
    size_t activation_groups;
    atomic_bool ordinary;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static size_t ceil_div(size_t n, size_t d) {
    return n / d + (n % d != 0);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->weight_dtype != TRILLIM_GRP_BINARY || !trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t in_bytes = trillim_itemsize(p->input_dtype), out_bytes = trillim_itemsize(p->output_dtype);
    if ((p->width && p->tokens > SIZE_MAX / p->width / in_bytes) ||
        p->tokens > SIZE_MAX / p->outputs / out_bytes ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    size_t count = p->tokens * p->width;
    size_t groups = ceil_div(p->width, ACTIVATION_GROUP);
    if (count > SIZE_MAX - 3 ||
        (groups && p->tokens > (SIZE_MAX - ((count + 3) & ~(size_t)3)) / sizeof(float) / groups)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count * in_bytes;
    outputs[0] = p->tokens * p->outputs * out_bytes;
    *scratch = ((count + 3) & ~(size_t)3) + p->tokens * groups * sizeof(float);
    return TRILLIM_OK;
}

#define AVX2 __attribute__((target("avx2")))

AVX2 static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)data + index);
    }
    if (dtype == TRILLIM_BF16) {
        __m128i bits = _mm_loadu_si128((const __m128i *)((const uint8_t *)data + index * 2));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    float lanes[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        lanes[lane] = trillim_load(data, index + lane, dtype).f;
    }
    return _mm256_loadu_ps(lanes);
}

AVX2 static inline void quantize8(int8_t *out, __m256 input, __m256 scale) {
    __m256 values = _mm256_mul_ps(input, scale);
    __m256 rounded = _mm256_round_ps(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i integers = _mm256_cvttps_epi32(rounded);
    __m128i words = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
    __m128i bytes = _mm_packs_epi16(words, words);
    _mm_storel_epi64((__m128i *)out, bytes);
}

AVX2 static TrillimStatus prepare_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    Task *task = opaque;
    const Parameters *p = task->p;
    for (size_t group = 0; group < task->activation_groups; ++group) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t first = group * ACTIVATION_GROUP;
        size_t end = p->width - first < ACTIVATION_GROUP ? p->width : first + ACTIVATION_GROUP;
        float maximum = 0.0f;
        for (size_t k = first; k < end; ++k) {
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f;
            if (!isfinite(x)) {
                atomic_store_explicit(&task->ordinary, false, memory_order_relaxed);
                return TRILLIM_OK;
            }
            maximum = fmaxf(maximum, fabsf(x));
        }
        /* Q8_0 rounds codes with the FP32 step and accumulates with its FP16 value. */
        float step = maximum / 127.0f;
        uint16_t bits = trillim_to_fp16(step);
        if (bits == 0x7c00 || (maximum != 0.0f && bits == 0)) {
            atomic_store_explicit(&task->ordinary, false, memory_order_relaxed);
            return TRILLIM_OK;
        }
        float inverse = step != 0.0f ? 1.0f / step : 0.0f;
        task->scales[row * task->activation_groups + group] = trillim_fp16(bits);
        size_t k = first;
        for (; end - k >= 8; k += 8) {
            size_t position = row * p->width + k;
            quantize8(task->codes + position, load8(task->input, position, p->input_dtype), _mm256_set1_ps(inverse));
        }
        for (; k < end; ++k) {
            float x = trillim_load(task->input, row * p->width + k, p->input_dtype).f * inverse;
            float magnitude = fabsf(x);
            int32_t code = (int32_t)magnitude;
            float fraction = magnitude - (float)code;
            if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
                ++code;
            }
            task->codes[row * p->width + k] = (int8_t)(x < 0.0f ? -code : code);
        }
    }
    return TRILLIM_OK;
}

AVX2 static inline int32_t sum_products(__m256i products) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static inline __m256i binary_signs(const uint8_t *packed) {
    uint32_t bytes;
    memcpy(&bytes, packed, sizeof(bytes));
    __m256i broadcast = _mm256_set1_epi32((int32_t)bytes);
    __m256i indices = _mm256_setr_epi64x(0, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
    __m256i mask = _mm256_set1_epi64x(0x0102040810204080);
    __m256i expanded = _mm256_shuffle_epi8(broadcast, indices);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(expanded, mask), _mm256_setzero_si256()),
                           _mm256_set1_epi8(1));
}

AVX2 static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = ceil_div(p->outputs, COLUMNS);
    size_t row = index / tiles * ROWS, first = index % tiles * COLUMNS;
    size_t rows = p->tokens - row < ROWS ? p->tokens - row : ROWS;
    size_t end = p->outputs - first < COLUMNS ? p->outputs : first + COLUMNS;
    size_t groups = ceil_div(p->width, WEIGHT_GROUP);
    bool ordinary = atomic_load_explicit(&task->ordinary, memory_order_relaxed);
    for (size_t column = first; column < end; ++column) {
        float totals[ROWS] = {0};
        if (ordinary) {
            for (size_t g = 0; g < groups; ++g) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float partial[ROWS] = {0};
                size_t first_group = g * (WEIGHT_GROUP / ACTIVATION_GROUP);
                size_t last_group = first_group + WEIGHT_GROUP / ACTIVATION_GROUP;
                if (last_group > task->activation_groups) {
                    last_group = task->activation_groups;
                }
                for (size_t a = first_group; a < last_group; ++a) {
                    size_t begin = a * ACTIVATION_GROUP;
                    size_t last = p->width - begin < ACTIVATION_GROUP ? p->width : begin + ACTIVATION_GROUP;
                    int32_t sums[ROWS] = {0};
                    const uint8_t *bits = task->weight + (column * groups + g) * 16 + (begin % WEIGHT_GROUP) / 8;
                    if (last - begin == ACTIVATION_GROUP) {
                        __m256i signs = binary_signs(bits);
                        for (size_t r = 0; r < rows; ++r) {
                            const int8_t *source = task->codes + (row + r) * p->width + begin;
                            __m256i x = _mm256_loadu_si256((const __m256i *)source);
                            __m256i signed_x = _mm256_sign_epi8(x, signs);
                            __m256i pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), signed_x);
                            sums[r] = sum_products(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
                        }
                    } else {
                        for (size_t k = 0; k < last - begin; ++k) {
                            int32_t sign = (bits[k / 8] >> (7 - k % 8)) & 1 ? 1 : -1;
                            for (size_t r = 0; r < rows; ++r) {
                                sums[r] += sign * task->codes[(row + r) * p->width + begin + k];
                            }
                        }
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        partial[r] = fmaf((float)sums[r],
                            task->scales[(row + r) * task->activation_groups + a], partial[r]);
                    }
                }
                uint16_t bits;
                size_t block = column * groups + g;
                memcpy(&bits, task->weight + p->outputs * groups * 16 + block * 2, sizeof(bits));
                float scale = trillim_bf16(bits);
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = fmaf(partial[r], scale, totals[r]);
                }
            }
        } else {
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float w = trillim_packed_read_float(task->weight, column * p->width + k,
                    p->weight_dtype, p->outputs, p->width);
                for (size_t r = 0; r < rows; ++r) {
                    float x = trillim_load(task->input, (row + r) * p->width + k, p->input_dtype).f;
                    totals[r] = fmaf(x, w, totals[r]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            trillim_store(task->output, (row + r) * p->outputs + column,
                p->output_dtype, (TrillimValue){.f = totals[r]});
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_auto_binary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || !p->tokens || !p->outputs) {
        return status;
    }
    size_t count = p->tokens * p->width;
    Task task = {inputs[0], inputs[1], outputs[0], scratch,
        scratch ? (float *)((uint8_t *)scratch + ((count + 3) & ~(size_t)3)) : NULL,
        ceil_div(p->width, ACTIVATION_GROUP), ATOMIC_VAR_INIT(true), p, context};
    const TrillimThreadPool *pool = context->thread_pool;
    status = p->tokens == 1 ? prepare_row(&task, 0, 0) :
        pool->run(pool->context, p->tokens, prepare_row, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    return pool->run(pool->context, ceil_div(p->tokens, ROWS) * ceil_div(p->outputs, COLUMNS), project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_auto_binary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if weight.dtype != 'grp_binary' or x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires grouped binary weights and floating activations')\n"
           "    for tensor in (x, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete packed matrix')\n"
           "    p = _linear(context)\n"
           "    return dict(tokens=p['m'], width=p['k'], outputs=p['n'], weight_dtype=p['b_dtype'], output_dtype=p['out_dtype'], input_dtype=p['a_dtype'])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased floating linear projection')\n"
           "    result = Linear(x, weight)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_auto_binary_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_auto_binary_linear_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM | TRILLIM_CAP_X86_AVX2};
    return &caps;
}
