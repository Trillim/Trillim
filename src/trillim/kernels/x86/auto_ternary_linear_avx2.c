#include "common.h"
#include <immintrin.h>
#include <stdatomic.h>

#define LOWBIT_WIDTH 128
#define LOWBIT_TYPE TRILLIM_GRP_TERNARY
#define LOWBIT_BYTES 32

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
    atomic_bool ordinary;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static inline __attribute__((always_inline)) float load_activation(const void *input, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)input + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)input + index * sizeof(bits), sizeof(bits));
    return dtype == TRILLIM_BF16 ? trillim_bf16(bits) : trillim_fp16(bits);
}

static inline __attribute__((always_inline)) void store_projection(void *output, size_t index, uint32_t dtype, float value) {
    if (dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)output + index * sizeof(value), &value, sizeof(value));
    } else {
        uint16_t bits = dtype == TRILLIM_BF16 ? trillim_to_bf16(value) : trillim_to_fp16(value);
        memcpy((uint8_t *)output + index * sizeof(bits), &bits, sizeof(bits));
    }
}


static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->weight_dtype != LOWBIT_TYPE || !trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t in_bytes = trillim_itemsize(p->input_dtype), out_bytes = trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / p->width ||
        (p->outputs && p->tokens > SIZE_MAX / out_bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    size_t count = p->tokens * p->width;
    if (count > SIZE_MAX / in_bytes || count > SIZE_MAX - 3 ||
        p->tokens > (SIZE_MAX - ((count + 3) & ~(size_t)3)) / sizeof(float) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count * in_bytes;
    outputs[0] = p->tokens * p->outputs * out_bytes;
    *scratch = ((count + 3) & ~(size_t)3) + p->tokens * sizeof(float);
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
    float maximum = 1e-5f;
    for (size_t k = 0; k < p->width; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = load_activation(task->input, row * p->width + k, p->input_dtype);
        if (!isfinite(x)) {
            atomic_store_explicit(&task->ordinary, false, memory_order_relaxed);
            return TRILLIM_OK;
        }
        maximum = fmaxf(maximum, fabsf(x));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    size_t k = 0;
    for (; p->width - k >= 8; k += 8) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t position = row * p->width + k;
        quantize8(task->codes + position, load8(task->input, position, p->input_dtype), _mm256_set1_ps(scale));
    }
    for (; k < p->width; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = load_activation(task->input, row * p->width + k, p->input_dtype) * scale;
        float magnitude = fminf(fabsf(x), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[row * p->width + k] = (int8_t)(x < 0.0f ? -code : code);
    }
    return TRILLIM_OK;
}

AVX2 static inline int32_t sum_products(__m256i products) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

AVX2 static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = index / tiles * 4, first = index % tiles * 16;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    bool ordinary = atomic_load_explicit(&task->ordinary, memory_order_relaxed);
    for (size_t column = first; column < end; ++column) {
        float totals[4] = {0};
        if (ordinary) {
            for (size_t g = 0; g < groups; ++g) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t block = column * groups + g;
                const uint8_t *packed = task->weight + block * LOWBIT_BYTES;
                size_t count = p->width - g * 128 < 128 ? p->width - g * 128 : 128;
                int32_t sums[4] = {0};
                bool invalid = false;
                size_t k = 0;
                if (count >= 32) {
                    __m256i packed_codes = _mm256_loadu_si256((const __m256i *)packed);
                    __m256i mask = _mm256_set1_epi8(3);
                    __m256i one = _mm256_set1_epi8(1);
                    __m256i weights[] = {
                        _mm256_srli_epi16(packed_codes, 6), _mm256_srli_epi16(packed_codes, 4),
                        _mm256_srli_epi16(packed_codes, 2), packed_codes
                    };
                    __m256i products[4] = {0};
                    for (; count - k >= 32; k += 32) {
                        __m256i code = _mm256_and_si256(weights[k / 32], mask);
                        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(code, mask))) {
                            invalid = true;
                            break;
                        }
                        __m256i sign = _mm256_sub_epi8(code, one);
                        for (size_t r = 0; r < rows; ++r) {
                            const int8_t *source = task->codes + (row + r) * p->width + g * 128 + k;
                            __m256i x = _mm256_loadu_si256((const __m256i *)source);
                            __m256i pairs = _mm256_maddubs_epi16(one, _mm256_sign_epi8(x, sign));
                            products[r] = _mm256_add_epi32(products[r], _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
                        }
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        sums[r] = sum_products(products[r]);
                    }
                }
                for (; !invalid && k < count; ++k) {
                    int32_t code = (packed[k % 32] >> (6 - 2 * (k / 32))) & 3;
                    if (code == 3) {
                        invalid = true;
                        break;
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        sums[r] += (code - 1) * task->codes[(row + r) * p->width + g * 128 + k];
                    }
                }
                uint16_t bits;
                memcpy(&bits, task->weight + p->outputs * groups * LOWBIT_BYTES + block * 2, sizeof(bits));
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = invalid ? NAN : fmaf((float)sums[r], trillim_bf16(bits), totals[r]);
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                totals[r] /= task->scales[row + r];
            }
        } else {
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float w = trillim_packed_read_float(task->weight, column * p->width + k,
                    p->weight_dtype, p->outputs, p->width);
                for (size_t r = 0; r < rows; ++r) {
                    float x = load_activation(task->input, (row + r) * p->width + k, p->input_dtype);
                    totals[r] = fmaf(x, w, totals[r]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            store_projection(task->output, (row + r) * p->outputs + column, p->output_dtype, totals[r]);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_auto_ternary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t count = p->tokens * p->width;
    Task task = {inputs[0], inputs[1], outputs[0], scratch,
        (float *)((uint8_t *)scratch + ((count + 3) & ~(size_t)3)), ATOMIC_VAR_INIT(true), p, context};
    status = p->tokens == 1 ? prepare_row(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_row, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
    return context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_auto_ternary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if weight.dtype != 'grp_ternary' or x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires grouped ternary weights and floating activations')\n"
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_auto_ternary_linear_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_auto_ternary_linear_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM | TRILLIM_CAP_X86_AVX2};
    return &caps;
}
