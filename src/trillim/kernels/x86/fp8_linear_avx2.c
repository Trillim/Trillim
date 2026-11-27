#include "common.h"
#include <immintrin.h>

enum { BLOCK_SIZE = 128, OUTPUT_TILE = 8, ROW_TILE = 4 };
_Static_assert(BLOCK_SIZE % OUTPUT_TILE == 0, "an output tile must fit within one scale block");

typedef struct {
    size_t tokens, width, outputs;
    size_t input_row_stride, input_column_stride;
    uint32_t input_dtype, output_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    void *output;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t tiles, blocks;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    size_t output_itemsize = trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / output_itemsize / p->outputs) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0) {
        size_t shape[] = {p->tokens, p->width};
        size_t strides[] = {p->input_row_stride, p->input_column_stride};
        size_t elements;
        size_t itemsize = trillim_itemsize(p->input_dtype);
        if (!trillim_span(2, shape, strides, &elements) || elements > SIZE_MAX / itemsize ||
            !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_FP8_E4M3, &inputs[1])) {
            return TRILLIM_ERROR;
        }
        inputs[0] = elements * itemsize;
    }
    outputs[0] = p->tokens * p->outputs * output_itemsize;
    return TRILLIM_OK;
}

#define VECTOR __attribute__((target("avx2,fma")))

VECTOR static inline __m256 decode8(const uint8_t *source) {
    __m256i code = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)source));
    __m256i magnitude = _mm256_and_si256(code, _mm256_set1_epi32(127));
    __m256i exponent = _mm256_srli_epi32(magnitude, 3);
    __m256i fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32(7));
    __m256i normal = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(120)), 23),
                                    _mm256_slli_epi32(fraction, 20));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-9f));
    __m256i bits = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                    _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(code, _mm256_set1_epi32(128)), 24);
    bits = _mm256_or_si256(bits, sign);
    return _mm256_castsi256_ps(_mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7fc00000),
                               _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(127))));
}

VECTOR static inline float horizontal_sum(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

VECTOR static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = index / task->tiles * ROW_TILE;
    size_t rows = p->tokens - row < ROW_TILE ? p->tokens - row : ROW_TILE;
    size_t first = index % task->tiles * OUTPUT_TILE;
    size_t columns = p->outputs - first < OUTPUT_TILE ? p->outputs - first : OUTPUT_TILE;
    float sums[OUTPUT_TILE][ROW_TILE] = {0};
    for (size_t start = 0; start < p->width;) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->width - start < BLOCK_SIZE ? p->width - start : BLOCK_SIZE;
        float panels[ROW_TILE][BLOCK_SIZE];
        const float *activations[ROW_TILE];
        for (size_t r = 0; r < rows; ++r) {
            size_t input = (row + r) * p->input_row_stride + start * p->input_column_stride;
            if (p->input_dtype == TRILLIM_FP32 && p->input_column_stride == 1) {
                activations[r] = (const float *)task->input + input;
            } else {
                size_t k = 0;
                if (p->input_dtype == TRILLIM_BF16 && p->input_column_stride == 1) {
                    const uint8_t *source = (const uint8_t *)task->input + input * 2;
                    for (; count - k >= 8; k += 8) {
                        __m128i bits = _mm_loadu_si128((const __m128i *)(source + k * 2));
                        __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16);
                        _mm256_storeu_ps(panels[r] + k, _mm256_castsi256_ps(wide));
                    }
                }
                for (; k < count; ++k) {
                    panels[r][k] = trillim_load(task->input, input + k * p->input_column_stride, p->input_dtype).f;
                }
                activations[r] = panels[r];
            }
        }
        size_t block = first / BLOCK_SIZE * task->blocks + start / BLOCK_SIZE;
        float scale;
        memcpy(&scale, task->weight + p->outputs * p->width + block * sizeof(scale), sizeof(scale));
        __m256 vector_scale = _mm256_set1_ps(scale);
        for (size_t column = 0; column < columns; ++column) {
            const uint8_t *weight = task->weight + (first + column) * p->width + start;
            __m256 totals[ROW_TILE] = {0};
            size_t k = 0;
            for (; count - k >= 8; k += 8) {
                __m256 w = _mm256_mul_ps(decode8(weight + k), vector_scale);
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = _mm256_fmadd_ps(_mm256_loadu_ps(activations[r] + k), w, totals[r]);
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                sums[column][r] += horizontal_sum(totals[r]);
            }
            for (; k < count; ++k) {
                float w = trillim_fp8_e4m3(weight[k]) * scale;
                for (size_t r = 0; r < rows; ++r) {
                    float x;
                    memcpy(&x, activations[r] + k, sizeof(x));
                    sums[column][r] = fmaf(x, w, sums[column][r]);
                }
            }
        }
        start += count;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t column = 0; column < columns; ++column) {
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue value = {.f = sums[column][r]};
            trillim_store(task->output, (row + r) * p->outputs + first + column, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_fp8_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    size_t tiles = p->outputs / OUTPUT_TILE + (p->outputs % OUTPUT_TILE != 0);
    size_t blocks = p->width / BLOCK_SIZE + (p->width % BLOCK_SIZE != 0);
    Task task = {inputs[0], inputs[1], outputs[0], p, context, tiles, blocks};
    size_t jobs = (p->tokens / ROW_TILE + (p->tokens % ROW_TILE != 0)) * tiles;
    return jobs == 1 ? project(&task, 0, 0)
        : context->thread_pool->run(context->thread_pool->context, jobs, project, &task);
}

TRILLIM_KERNEL const char *trillim_x86_fp8_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or weight.dtype != 'fp8_e4m3' or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('FP8 linear requires floating activations and output')\n"
           "    if len(weight.shape) != 2 or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('FP8 linear requires a complete weight matrix')\n"
           "    if (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "        raise ValueError('FP8 linear requires contiguous weights')\n"
           "    values = _linear(context)\n"
           "    outputs, width = weight.shape\n"
           "    row_stride, column_stride = values['a_row_stride'], values['a_col_stride']\n"
           "    if row_stride == values['k'] and column_stride.value == 1:\n"
           "        row_stride = width\n"
           "    return dict(tokens=values['m'], width=width, outputs=outputs, input_row_stride=row_stride, input_column_stride=column_stride, input_dtype=values['a_dtype'], output_dtype=values['out_dtype'])\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased FP8 linear projection')\n"
           "    output = Linear(x, weight)\n"
           "    return (output if dtype is None else Cast(output, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_fp8_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"input_row_stride", "size_t", offsetof(Parameters, input_row_stride), NULL},
        {"input_column_stride", "size_t", offsetof(Parameters, input_column_stride), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        2, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_fp8_linear_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA};
    return &capabilities;
}
