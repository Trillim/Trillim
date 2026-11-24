#include "common.h"
#include <fenv.h>
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2")
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { PACKED_ACTIVATION_TILE = 4, PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024 };

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    uint32_t dtype;
} PackedActivation;

static inline bool packed_activation_quantize(
    const void *input, uint32_t dtype, size_t start, size_t group, float *scale, int8_t *codes
) {
    float values[256];
    float peak = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        float value = trillim_load(input, start + i, dtype).f;
        if (!isfinite(value)) {
            return false;
        }
        if (fabsf(value) > fabsf(peak)) {
            peak = value;
        }
        values[i] = value;
    }
    if (peak == 0.0f) {
        *scale = 0.0f;
        memset(codes, 0, group);
        return true;
    }
    float inverse = -127.0f / peak;
    if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
        return false;
    }
    *scale = 1.0f / inverse;
    for (size_t i = 0; i < group; ++i) {
        /* Bias the significand to round to nearest, ties to even. */
        float rounded = values[i] * inverse + 12582912.0f;
        uint32_t bits;
        memcpy(&bits, &rounded, sizeof(bits));
        codes[i] = (int8_t)((int32_t)(bits & UINT32_C(0x007fffff)) - INT32_C(0x00400000));
    }
    return true;
}

static inline TrillimStatus packed_activation_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const PackedActivation *a = opaque;
    size_t group = a->width / a->groups;
    float *scales = a->scales + row * a->groups;
    int8_t *codes = a->codes + row * a->width;
    if (fegetround() != FE_TONEAREST) {
        scales[0] = NAN;
        return TRILLIM_OK;
    }
    for (size_t g = 0; g < a->groups; ++g) {
        if (g % 4 == 0 && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t start = g * group;
        if (!packed_activation_quantize(a->input, a->dtype, row * a->width + start,
                group, scales + g, codes + start)) {
            scales[0] = NAN;
            return TRILLIM_OK;
        }
    }
    return TRILLIM_OK;
}

static size_t packed_activation_space(size_t rows, size_t width, size_t columns) {
    size_t group = 256;
    if (width < group || columns < 16) {
        return 0;
    }
    if (rows > 1 && rows < 4) {
        rows = 1;
    }
    /* The compiler reserves scratch at the maximum shape. */
    size_t bounded_width = width < 65536 ? width : 65536;
    size_t groups = bounded_width / group;
    size_t row_bytes = groups * (group + sizeof(float));
    return rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes
        ? PACKED_ACTIVATION_MAX_BYTES : rows * row_bytes;
}

static inline TrillimStatus packed_activation_prepare(
    PackedActivation *a, void *workspace, const void *input, uint32_t input_dtype,
    size_t rows, size_t width, size_t groups, const TrillimKernelContext *context
) {
    float *scales = workspace;
    int8_t *codes = (int8_t *)(scales + rows * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, input_dtype};
    const TrillimThreadPool *pool = context->thread_pool;
    return rows == 1 ? packed_activation_row(a, 0, 0)
                     : pool->run(pool->context, rows, packed_activation_row, a);
}

static inline __m256i packed_pair_products(__m256i weight, __m256i input) {
    // The activation codes fit in [-127, 127], so neither negation nor pair sums saturate.
    return _mm256_maddubs_epi16(_mm256_abs_epi8(weight), _mm256_sign_epi8(input, weight));
}

static inline float packed_sum_f32(__m256 sums) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static inline __m256i packed_q6_weights(const uint8_t *low, const uint8_t *high, size_t lane) {
    __m256i lower = _mm256_loadu_si256((const __m256i *)low);
    __m256i upper = _mm256_loadu_si256((const __m256i *)high);
    lower = _mm256_srl_epi16(lower, _mm_cvtsi32_si128((int)(lane % 2 * 4)));
    upper = _mm256_srl_epi16(upper, _mm_cvtsi32_si128((int)(lane * 2)));
    lower = _mm256_and_si256(lower, _mm256_set1_epi8(15));
    upper = _mm256_and_si256(upper, _mm256_set1_epi8(3));
    return _mm256_sub_epi8(_mm256_or_si256(lower, _mm256_slli_epi16(upper, 4)), _mm256_set1_epi8(32));
}

static inline bool packed_activation_ready(const PackedActivation *a, size_t row) {
    return a->scales != NULL && !isnan(a->scales[row * a->groups]);
}

static inline TrillimStatus packed_activation_dot(
    const PackedActivation *a, size_t first_row, size_t rows, const uint8_t *weight,
    size_t outputs, size_t column,
    float sums[PACKED_ACTIVATION_TILE], bool *used
) {
    *used = false;
    for (size_t row = 0; row < rows; ++row) {
        if (!packed_activation_ready(a, first_row + row)) {
            return TRILLIM_OK;
        }
    }
    size_t total = outputs * a->groups;
    size_t block = column * a->groups;
    size_t low_size = 128;
    const uint8_t *low = weight + block * low_size;
    const uint8_t *scales = weight + total * low_size + block * 2;
    const uint8_t *high = weight + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (total * 64 + block * 16));
    size_t high_size = 64;
    high += block * high_size;
    const int8_t *codes = a->codes + first_row * a->width;
    const float *activation_scales = a->scales + first_row * a->groups;
    __m256 output[PACKED_ACTIVATION_TILE];
    for (size_t row = 0; row < rows; ++row) {
        output[row] = _mm256_setzero_ps();
    }
    for (size_t g = 0; g < a->groups; ++g) {
        if (a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float scale = trillim_bf16(bits);
        if (!isfinite(scale) || fabsf(scale) > 0x1p40f) {
            return TRILLIM_OK;
        }
        const uint8_t *l = low + g * low_size;
        const uint8_t *h = high + g * high_size;
        __m256i products[PACKED_ACTIVATION_TILE];
        for (size_t row = 0; row < rows; ++row) {
            products[row] = _mm256_setzero_si256();
        }
        for (size_t region = 0; region < 8; ++region) {
            size_t half = region / 4, lane = region % 4;
            __m256i w = packed_q6_weights(l + half * 64 + lane / 2 * 32, h + half * 32, lane);
            const int8_t *s = subscales + g * 16 + region * 2;
            __m256i scale_pair = _mm256_set_m128i(_mm_set1_epi16(s[1]), _mm_set1_epi16(s[0]));
            for (size_t row = 0; row < rows; ++row) {
                const int8_t *x = codes + row * a->width + g * 256 + region * 32;
                __m256i pairs = packed_pair_products(w, _mm256_loadu_si256((const __m256i *)x));
                products[row] = _mm256_add_epi32(products[row], _mm256_madd_epi16(pairs, scale_pair));
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float combined_scale = scale * activation_scales[row * a->groups + g];
            __m256 partial = _mm256_mul_ps(_mm256_cvtepi32_ps(products[row]), _mm256_set1_ps(combined_scale));
            output[row] = _mm256_add_ps(output[row], partial);
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = packed_sum_f32(output[row]);
    }
    *used = true;
    return TRILLIM_OK;
}

typedef struct {
    size_t tokens, width, outputs;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 2 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (p->width == 0 || p->width % 256 != 0 ||
        !trillim_packed_payload_size(p->tokens, p->width, TRILLIM_BF16, inputs) ||
        !trillim_packed_payload_size(p->outputs, p->width, TRILLIM_DUBGRP_INT6, inputs + 1) ||
        !trillim_packed_payload_size(p->tokens, p->outputs, TRILLIM_BF16, outputs)) {
        return TRILLIM_ERROR;
    }
    *scratch = packed_activation_space(p->tokens, p->width, p->outputs);
    return TRILLIM_OK;
}

static inline float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static inline __m256 decode8(
    const uint8_t *values, size_t total, size_t block, size_t offset
) {
    __m256i mask = _mm256_set1_epi32(15);
    __m256i codes;
    size_t scale_start = total * 128;
    float scale = read_bf16(values + scale_start + block * 2);
    size_t half = offset / 128, lane = offset % 128 / 32;
    size_t low = block * 128 + half * 64 + lane / 2 * 32 + offset % 32;
    size_t high = scale_start + total * 2 + block * 64 + half * 32 + offset % 32;
    codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + low)));
    codes = _mm256_and_si256(_mm256_srlv_epi32(codes, _mm256_set1_epi32((int)(lane % 2 * 4))), mask);
    __m256i upper = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(values + high)));
    upper = _mm256_and_si256(_mm256_srlv_epi32(upper, _mm256_set1_epi32((int)(lane * 2))), _mm256_set1_epi32(3));
    codes = _mm256_sub_epi32(_mm256_or_si256(codes, _mm256_slli_epi32(upper, 4)), _mm256_set1_epi32(32));
    int8_t subscale;
    memcpy(&subscale, values + scale_start + total * 66 + block * 16 + offset / 16, 1);
    codes = _mm256_mullo_epi32(codes, _mm256_set1_epi32(subscale));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
}

static inline __m256 load8(const void *input, size_t index) {
    __m128i bits = _mm_loadu_si128((const __m128i *)((const uint8_t *)input + index * sizeof(uint16_t)));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static TrillimStatus accumulate(
    const void *input, const uint8_t *weight, size_t width, size_t outputs,
    size_t row, size_t rows, size_t column,
    const TrillimKernelContext *context, const PackedActivation *activation, float sums[4]
) {
    if (activation->scales != NULL) {
        bool used;
        TrillimStatus status = packed_activation_dot(activation, row, rows, weight,
            outputs, column, sums, &used);
        if (status != TRILLIM_OK || used) {
            return status;
        }
    }
    size_t group = 256;
    size_t groups = width / group + (width % group != 0);
    size_t total = outputs * groups;
    size_t k = 0;
    for (size_t block = column * groups; k < width; ++block) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const size_t start = k;
        const size_t count = width - k < group ? width - k : group;
        const size_t end = k + count - count % 8;
        __m256 acc[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
        };
        for (; k < end; k += 8) {
            __m256 w = decode8(weight, total, block, k - start);
            for (size_t r = 0; r < rows; ++r) {
                __m256 x = load8(input, (row + r) * width + k);
                acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(x, w));
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            sums[r] += sum8(acc[r]);
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

typedef struct {
    const void *input;
    const void *weight;
    void *output;
    size_t tokens, width, outputs, columns;
    const TrillimKernelContext *context;
    PackedActivation activation;
} Task;

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    size_t tiles = task->outputs / task->columns + (task->outputs % task->columns != 0);
    size_t row = tile / tiles * PACKED_ACTIVATION_TILE;
    size_t rows = task->tokens - row < PACKED_ACTIVATION_TILE ? task->tokens - row : PACKED_ACTIVATION_TILE;
    size_t first = tile % tiles * task->columns;
    size_t end = task->outputs - first < task->columns ? task->outputs : first + task->columns;
    for (size_t column = first; column < end; ++column) {
        float sums[4] = {0};
        TrillimStatus status = accumulate(task->input, task->weight, task->width,
            task->outputs, row, rows, column, task->context, &task->activation, sums);
        if (status != TRILLIM_OK) {
            return status;
        }
        for (size_t r = 0; r < rows; ++r) {
            size_t index = (row + r) * task->outputs + column;
            trillim_store(task->output, index, TRILLIM_BF16, (TrillimValue){.f = sums[r]});
        }
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus run(const Task *task, void *workspace) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    if (pool->worker_count == 0) {
        return TRILLIM_ERROR;
    }
    size_t panel_rows = task->tokens;
    size_t groups = 0;
    size_t group = 256;
    if (workspace != NULL && (task->tokens == 1 || task->tokens >= PACKED_ACTIVATION_TILE) &&
        task->width >= group && task->width <= 65536 && task->width % group == 0 &&
        task->outputs >= 16 && fegetround() == FE_TONEAREST) {
        size_t row_groups = task->width / group;
        size_t row_bytes = task->width + row_groups * sizeof(float);
        size_t space = packed_activation_space(task->tokens, task->width, task->outputs);
        size_t capacity = space / row_bytes;
        if (capacity < task->tokens) {
            capacity = capacity / PACKED_ACTIVATION_TILE * PACKED_ACTIVATION_TILE;
        }
        if (capacity != 0) {
            panel_rows = capacity;
            groups = row_groups;
        }
    }

    Task panel = *task;
    size_t columns = task->outputs / pool->worker_count / 4;
    columns = columns / 16 * 16;
    panel.columns = columns < 16 ? 16 : columns;
    size_t tiles = task->outputs / panel.columns + (task->outputs % panel.columns != 0);
    uint32_t input_dtype = TRILLIM_BF16;
    size_t input_stride = task->width * trillim_itemsize(input_dtype);
    size_t output_stride = task->outputs * trillim_itemsize(TRILLIM_BF16);
    for (size_t first = 0; first < task->tokens; first += panel.tokens) {
        size_t remaining = task->tokens - first;
        panel.tokens = remaining < panel_rows ? remaining : panel_rows;
        panel.input = (const uint8_t *)task->input + first * input_stride;
        panel.output = (uint8_t *)task->output + first * output_stride;
        if (groups != 0) {
            TrillimStatus status = packed_activation_prepare(&panel.activation, workspace, panel.input,
                input_dtype, panel.tokens, panel.width, groups, panel.context);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t rows = panel.tokens / PACKED_ACTIVATION_TILE + (panel.tokens % PACKED_ACTIVATION_TILE != 0);
        size_t jobs = rows * tiles;
        TrillimStatus status = jobs == 1 ? project(&panel, 0, 0) : pool->run(pool->context, jobs, project, &panel);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_linear_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    Task task = {
        .input = inputs[0], .weight = inputs[1], .output = outputs[0],
        .tokens = p->tokens, .width = p->width, .outputs = p->outputs,
        .context = context
    };
    return run(&task, scratch);
}

TRILLIM_KERNEL const char *trillim_x86_packed_linear_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'weight')\n"
           "def bind(c):\n"
           "    x, weight = (c.inputs[name] for name in ('x', 'weight'))\n"
           "    output = c.outputs[0]\n"
           "    if len(x.shape) != 2 or x.dtype != 'bf16' or output.dtype != 'bf16':\n"
           "        raise ValueError('requires BF16 input and output')\n"
           "    if any((v.strides or _contiguous_strides(v.shape)) != _contiguous_strides(v.shape) for v in (*c.inputs.values(), output)):\n"
           "        raise ValueError('requires contiguous operands')\n"
           "    if len(weight.shape) != 2 or weight.dtype != 'dubgrp_int6' or _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete Q6 matrix')\n"
           "    outputs, width = weight.shape\n"
           "    if x.shape[-1] != width or output.shape != (*x.shape[:-1], outputs):\n"
           "        raise ValueError('incompatible dimensions')\n"
           "    if width.value is None or not width.value or width.value % 256:\n"
           "        raise ValueError('unaligned Q6 matrix')\n"
           "    return dict(tokens=_product(x.shape[:-1]), width=width, outputs=outputs)\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype != 'bf16':\n"
           "        raise ValueError('requires a BF16 projection cast')\n"
           "    return (Cast(Linear(x, weight), 'bf16'),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_linear_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_linear_bf16_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
