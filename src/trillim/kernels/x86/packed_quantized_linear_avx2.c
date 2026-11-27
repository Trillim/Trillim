#include "common.h"
#include <immintrin.h>
#include <math.h>

#define PACKED_TARGET __attribute__((target("avx2")))

enum {
    PACKED_ROWS = 4,
    PACKED_PREFILL_ROWS = 8,
    PACKED_COLUMNS = 16,
    PACKED_SUM_MAX_BYTES = 2 * 1024 * 1024,
};

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype;
} Parameters;

typedef struct {
    size_t width, values, scales, high, subscales;
} Layout;

typedef struct {
    const uint8_t *values, *high;
    const int8_t *subscales;
    uint32_t dtype;
} Block;

typedef struct {
    void *output;
    const int8_t *input;
    const uint8_t *values, *scales, *high;
    const int8_t *subscales;
    const float *activation_scales;
    int16_t *activation_sums;
    const Parameters *p;
    const Layout *layout;
    size_t groups, columns, column_tiles, rows_per_tile;
    const TrillimKernelContext *context;
} Task;

static bool centered_codes(uint32_t dtype) {
    return dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5;
}

static size_t activation_sum_space(size_t tokens, size_t groups, uint32_t dtype) {
    if (!centered_codes(dtype)) {
        return 0;
    }
    if (groups > PACKED_SUM_MAX_BYTES / sizeof(int16_t)) {
        return tokens == 0 ? 0 : PACKED_SUM_MAX_BYTES;
    }
    size_t row_bytes = groups * sizeof(int16_t);
    return tokens > PACKED_SUM_MAX_BYTES / row_bytes ? PACKED_SUM_MAX_BYTES : tokens * row_bytes;
}

static const Layout *packed_layout(uint32_t dtype) {
    static const Layout layouts[] = {
        [TRILLIM_GRP_INT4 - TRILLIM_GRP_INT4] = {32, 16, 2, 0, 0},
        [TRILLIM_GRP_INT5 - TRILLIM_GRP_INT4] = {32, 16, 2, 4, 0},
        [TRILLIM_DUBGRP_INT6 - TRILLIM_GRP_INT4] = {256, 128, 2, 64, 16},
        [TRILLIM_GRP_BINARY - TRILLIM_GRP_INT4] = {128, 16, 2, 0, 0},
        [TRILLIM_GRP_TERNARY - TRILLIM_GRP_INT4] = {128, 32, 2, 0, 0},
        [TRILLIM_MXFP4 - TRILLIM_GRP_INT4] = {32, 16, 1, 0, 0},
    };
    return dtype >= TRILLIM_GRP_INT4 && dtype <= TRILLIM_MXFP4 ? &layouts[dtype - TRILLIM_GRP_INT4] : NULL;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t record = layout->values + layout->scales + layout->high + layout->subscales;
    if (groups > SIZE_MAX / record || p->outputs > SIZE_MAX / (groups * record) ||
        p->tokens > SIZE_MAX / p->width || p->tokens > SIZE_MAX / sizeof(float) ||
        (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * record;
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    *scratch = activation_sum_space(p->tokens, groups, p->weight_dtype);
    return TRILLIM_OK;
}

static float weight_scale(const Task *task, size_t block) {
    if (task->p->weight_dtype == TRILLIM_MXFP4) {
        uint8_t exponent = task->scales[block];
        if (exponent == 255) {
            return NAN;
        }
        // Integer E2M1 codes represent twice the value, so halve the E8M0 scale.
        uint32_t bits = exponent <= 1 ? UINT32_C(0x00200000) << exponent : (uint32_t)(exponent - 1) << 23;
        float value;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, task->scales + block * 2, sizeof(bits));
    return trillim_bf16(bits);
}

static PACKED_TARGET __m256i expand_bits(const uint8_t *data, __m128i mask) {
    int32_t word;
    memcpy(&word, data, sizeof(word));
    const __m256i shuffle = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
    );
    __m256i spread = _mm256_shuffle_epi8(_mm256_set1_epi32(word), shuffle);
    __m256i bits = _mm256_broadcastsi128_si256(mask);
    return _mm256_cmpeq_epi8(_mm256_and_si256(spread, bits), bits);
}

static PACKED_TARGET int32_t sum_lanes(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

static PACKED_TARGET TrillimStatus sum_activation(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const int8_t *input = task->input + row * task->p->width;
    int16_t *sums = task->activation_sums + row * task->groups;
    const __m256i bytes = _mm256_set1_epi8(1);
    const __m256i words = _mm256_set1_epi16(1);
    for (size_t g = 0; g < task->groups; ++g) {
        if ((g & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256i x = _mm256_loadu_si256((const __m256i *)(input + g * 32));
        __m256i pairs = _mm256_maddubs_epi16(bytes, x);
        sums[g] = (int16_t)sum_lanes(_mm256_madd_epi16(pairs, words));
    }
    return TRILLIM_OK;
}

static PACKED_TARGET __m256i unpack_q4_q5(const Block *block) {
    const __m128i mask = _mm_set1_epi8(15);
    __m128i packed = _mm_loadu_si128((const __m128i *)block->values);
    __m256i codes = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(packed, 4), mask), _mm_and_si128(packed, mask));
    if (block->dtype == TRILLIM_GRP_INT5) {
        __m128i bits = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
        __m256i high = _mm256_and_si256(expand_bits(block->high, bits), _mm256_set1_epi8(16));
        codes = _mm256_or_si256(codes, high);
    }
    return codes;
}

static PACKED_TARGET void packed_centered_dot(
    const Block *block, const int8_t *input, size_t stride, const int16_t *activation_sums,
    size_t groups, size_t rows, int32_t *sums
) {
    int center = block->dtype == TRILLIM_GRP_INT4 ? 8 : 16;
    __m256i weight = unpack_q4_q5(block);
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t row = 0; row < rows; ++row) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(input + row * stride));
        __m256i pairs = _mm256_maddubs_epi16(weight, x);
        sums[row] = sum_lanes(_mm256_madd_epi16(pairs, ones)) - center * activation_sums[row * groups];
    }
}

static PACKED_TARGET bool packed_dot(const Block *block, const int8_t *input, size_t stride, size_t rows,
                                     size_t count, const int8_t *tail, int32_t *sums) {
    const uint8_t *values = block->values;
    __m256i acc[PACKED_ROWS] = {0};
    const __m256i nibbles = _mm256_set1_epi8(15);
    const __m256i two_bits = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);
    for (size_t k = 0; k < count; k += 32) {
        size_t valid = count - k;
        __m256i weight;
        __m256i subscale = _mm256_set1_epi16(1);
        switch (block->dtype) {
        case TRILLIM_GRP_BINARY: {
            __m128i mask = _mm_setr_epi8(-128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);
            weight = _mm256_sub_epi8(_mm256_setzero_si256(), _mm256_or_si256(expand_bits(values + k / 8, mask), one));
            break;
        }
        case TRILLIM_GRP_TERNARY: {
            __m256i packed = _mm256_loadu_si256((const __m256i *)values);
            __m256i codes = _mm256_and_si256(_mm256_srli_epi16(packed, 6 - (int)(k / 32) * 2), two_bits);
            uint32_t invalid = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(codes, two_bits));
            if (valid < 32) {
                invalid &= (UINT32_C(1) << valid) - 1;
            }
            if (invalid) {
                return false;
            }
            weight = _mm256_sub_epi8(codes, one);
            break;
        }
        case TRILLIM_GRP_INT4:
        case TRILLIM_GRP_INT5: {
            int center = block->dtype == TRILLIM_GRP_INT4 ? 8 : 16;
            weight = _mm256_sub_epi8(unpack_q4_q5(block), _mm256_set1_epi8((char)center));
            break;
        }
        case TRILLIM_MXFP4: {
            const __m128i mask = _mm_set1_epi8(15);
            __m128i packed = _mm_loadu_si128((const __m128i *)values);
            __m128i lo = _mm_and_si128(packed, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            __m128i palette = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
            __m256i codes = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            weight = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(palette), codes);
            break;
        }
        case TRILLIM_DUBGRP_INT6: {
            size_t half = k / 128;
            size_t lane = k % 128 / 32;
            __m256i packed = _mm256_loadu_si256((const __m256i *)(values + half * 64 + lane / 2 * 32));
            __m256i high = _mm256_loadu_si256((const __m256i *)(block->high + half * 32));
            __m256i low = _mm256_and_si256(_mm256_srli_epi16(packed, (int)(lane % 2) * 4), nibbles);
            high = _mm256_and_si256(_mm256_srli_epi16(high, (int)lane * 2), two_bits);
            weight = _mm256_sub_epi8(_mm256_or_si256(low, _mm256_slli_epi16(high, 4)), _mm256_set1_epi8(32));
            subscale = _mm256_set_m128i(_mm_set1_epi16(block->subscales[k / 16 + 1]),
                                      _mm_set1_epi16(block->subscales[k / 16]));
            break;
        }
        default:
            return false;
        }
        const int8_t *codes = valid < 32 ? tail : input + k;
        size_t step = valid < 32 ? 32 : stride;
        for (size_t row = 0; row < rows; ++row) {
            __m256i x = _mm256_loadu_si256((const __m256i *)(codes + row * step));
            // Magnitudes are unsigned, so -128 stays exact. Weight products fit in 16-bit pairs.
            __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(weight, x));
            acc[row] = _mm256_add_epi32(acc[row], _mm256_madd_epi16(pairs, subscale));
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = sum_lanes(acc[row]);
    }
    return true;
}

static PACKED_TARGET TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    const Layout *layout = task->layout;
    size_t row = tile / task->column_tiles * task->rows_per_tile;
    size_t rows = p->tokens - row < task->rows_per_tile ? p->tokens - row : task->rows_per_tile;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    int8_t tail[PACKED_ROWS * 32];
    size_t remainder = p->width % 32;
    if (remainder != 0) {
        memset(tail, 0, sizeof(tail));
        for (size_t r = 0; r < rows; ++r) {
            memcpy(tail + r * 32, task->input + (row + r + 1) * p->width - remainder, remainder);
        }
    }
    for (size_t column = first; column < end; ++column) {
        float totals[PACKED_PREFILL_ROWS] = {0};
        for (size_t g = 0; g < task->groups; ++g) {
            size_t k = g * layout->width;
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = p->width - k < layout->width ? p->width - k : layout->width;
            size_t index = column * task->groups + g;
            Block block = {
                task->values + index * layout->values,
                task->high + index * layout->high,
                task->subscales + index * layout->subscales,
                p->weight_dtype,
            };
            int32_t sums[PACKED_PREFILL_ROWS];
            const int8_t *input = task->input + row * p->width + k;
            if (task->activation_sums != NULL) {
                packed_centered_dot(&block, input, p->width, task->activation_sums + row * task->groups + g,
                    task->groups, rows, sums);
            } else if (!packed_dot(&block, input, p->width, rows, count, tail, sums)) {
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = NAN;
                }
                break;
            }
            float scale = weight_scale(task, index);
            for (size_t r = 0; r < rows; ++r) {
                totals[r] += (float)sums[r] * scale;
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            TrillimValue result = {.f = totals[r] / task->activation_scales[row + r]};
            trillim_store(task->output, (row + r) * p->outputs + column, p->output_dtype, result);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_quantized_linear_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Layout *layout = packed_layout(p->weight_dtype);
    if (!layout || !p->width) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t groups = p->width / layout->width + (p->width % layout->width != 0);
    size_t blocks = p->outputs * groups;
    const uint8_t *values = inputs[1];
    const uint8_t *scales = values + blocks * layout->values;
    const uint8_t *high = scales + blocks * layout->scales;
    const int8_t *subscales = (const int8_t *)(high + blocks * layout->high);
    size_t columns = p->outputs / workers / (4 * PACKED_COLUMNS) * PACKED_COLUMNS;
    if (columns < PACKED_COLUMNS) {
        columns = PACKED_COLUMNS;
    }
    size_t tiles = p->outputs / columns + (p->outputs % columns != 0);
    size_t capacity = p->tokens;
    size_t rows_per_tile = PACKED_ROWS;
    size_t sum_space = activation_sum_space(p->tokens, groups, p->weight_dtype);
    if (scratch != NULL && p->width % 32 == 0 && groups <= sum_space / sizeof(int16_t) && p->outputs >= PACKED_COLUMNS) {
        capacity = sum_space / sizeof(int16_t) / groups;
        if (p->weight_dtype == TRILLIM_GRP_INT5 && p->tokens >= PACKED_PREFILL_ROWS && tiles >= workers) {
            rows_per_tile = PACKED_PREFILL_ROWS;
        }
    } else {
        scratch = NULL;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    const size_t itemsize = trillim_itemsize(p->output_dtype);
    for (size_t first = 0; first < p->tokens;) {
        Parameters panel = *p;
        panel.tokens = p->tokens - first < capacity ? p->tokens - first : capacity;
        Task task = {
            (uint8_t *)outputs[0] + first * p->outputs * itemsize,
            (const int8_t *)inputs[0] + first * p->width,
            values, scales, high, subscales, (const float *)inputs[2] + first,
            scratch, &panel, layout, groups, columns, tiles, rows_per_tile, context,
        };
        if (scratch != NULL) {
            TrillimStatus status = panel.tokens == 1 ? sum_activation(&task, 0, 0)
                : pool->run(pool->context, panel.tokens, sum_activation, &task);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t jobs = (panel.tokens / rows_per_tile + (panel.tokens % rows_per_tile != 0)) * tiles;
        TrillimStatus status = jobs == 1 ? project(&task, 0, 0) : pool->run(pool->context, jobs, project, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        first += panel.tokens;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_packed_quantized_linear_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _DTYPE_IDS, _quantized_linear, _storage_shape\n"
           "def bind(context):\n"
           "    weight = context.inputs['weight']\n"
           "    if weight.dtype not in {'grp_binary', 'grp_ternary', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'mxfp4'}:\n"
           "        raise ValueError('projection requires grouped low-bit or MXFP4 weights')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('projection requires a complete packed weight matrix')\n"
           "    return dict(_quantized_linear(context, weight_dtype=weight.dtype), weight_dtype=_DTYPE_IDS[weight.dtype])\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_quantized_linear_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 3, 1, requirements, NULL, inputs, outputs, 2,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_quantized_linear_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
