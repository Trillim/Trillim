#include "common.h"
#include <immintrin.h>
#include <math.h>

#define PACKED_TARGET __attribute__((target("avx2")))

enum { PACKED_ROWS = 4, PACKED_COLUMNS = 16 };

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
    const Parameters *p;
    const Layout *layout;
    size_t groups, columns, column_tiles;
    const TrillimKernelContext *context;
} Task;

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

static inline bool scalar_tail(const Block *block, const int8_t *input, size_t stride, size_t rows,
                               size_t begin, size_t end, int32_t *sums) {
    const uint8_t *values = block->values;
    for (size_t i = begin; i < end; ++i) {
        int32_t code;
        switch (block->dtype) {
        case TRILLIM_GRP_BINARY:
            code = ((values[i / 8] >> (7 - i % 8)) & 1) * 2 - 1;
            break;
        case TRILLIM_GRP_TERNARY:
            code = (values[i % 32] >> (6 - (i / 32) * 2)) & 3;
            if (code == 3) {
                return false;
            }
            code -= 1;
            break;
        case TRILLIM_GRP_INT4:
            code = ((values[i % 16] >> ((i / 16) * 4)) & 15) - 8;
            break;
        case TRILLIM_GRP_INT5:
            code = (values[i % 16] >> ((i / 16) * 4)) & 15;
            code |= ((block->high[i / 8] >> (i % 8)) & 1) << 4;
            code -= 16;
            break;
        case TRILLIM_DUBGRP_INT6: {
            size_t half = i / 128;
            size_t lane = i % 128 / 32;
            size_t low = half * 64 + lane / 2 * 32 + i % 32;
            size_t high = half * 32 + i % 32;
            code = (values[low] >> ((lane % 2) * 4)) & 15;
            code |= ((block->high[high] >> (lane * 2)) & 3) << 4;
            code = (code - 32) * block->subscales[i / 16];
            break;
        }
        case TRILLIM_MXFP4: {
            static const int8_t e2m1[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
            code = e2m1[(values[i / 2] >> ((i % 2) * 4)) & 15];
            break;
        }
        default:
            return false;
        }
        for (size_t row = 0; row < rows; ++row) {
            sums[row] += input[row * stride + i] * code;
        }
    }
    return true;
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

static PACKED_TARGET bool packed_dot(const Block *block, const int8_t *input, size_t stride, size_t rows,
                                     size_t count, int32_t *sums) {
    const uint8_t *values = block->values;
    __m256i acc[PACKED_ROWS] = {0};
    const __m256i nibbles = _mm256_set1_epi8(15);
    const __m256i two_bits = _mm256_set1_epi8(3);
    const __m256i one = _mm256_set1_epi8(1);
    size_t full = count / 32 * 32;
    for (size_t k = 0; k < full; k += 32) {
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
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(codes, two_bits))) {
                return false;
            }
            weight = _mm256_sub_epi8(codes, one);
            break;
        }
        case TRILLIM_GRP_INT4:
        case TRILLIM_GRP_INT5:
        case TRILLIM_MXFP4: {
            const __m128i mask = _mm_set1_epi8(15);
            __m128i packed = _mm_loadu_si128((const __m128i *)values);
            __m128i lo = _mm_and_si128(packed, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            if (block->dtype == TRILLIM_MXFP4) {
                __m128i palette = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
                __m256i codes = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
                weight = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(palette), codes);
            } else {
                __m256i codes = _mm256_set_m128i(hi, lo);
                if (block->dtype == TRILLIM_GRP_INT5) {
                    __m128i bits = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
                    __m256i high = _mm256_and_si256(expand_bits(block->high, bits), _mm256_set1_epi8(16));
                    weight = _mm256_sub_epi8(_mm256_or_si256(codes, high), _mm256_set1_epi8(16));
                } else {
                    weight = _mm256_sub_epi8(codes, _mm256_set1_epi8(8));
                }
            }
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
        for (size_t row = 0; row < rows; ++row) {
            __m256i x = _mm256_loadu_si256((const __m256i *)(input + row * stride + k));
            // Magnitudes are unsigned, so -128 stays exact. Weight products fit in 16-bit pairs.
            __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(weight, x));
            acc[row] = _mm256_add_epi32(acc[row], _mm256_madd_epi16(pairs, subscale));
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        sums[row] = sum_lanes(acc[row]);
    }
    return full == count || scalar_tail(block, input, stride, rows, full, count, sums);
}

static PACKED_TARGET TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    const Layout *layout = task->layout;
    size_t row = tile / task->column_tiles * PACKED_ROWS;
    size_t rows = p->tokens - row < PACKED_ROWS ? p->tokens - row : PACKED_ROWS;
    size_t first = tile % task->column_tiles * task->columns;
    size_t end = p->outputs - first < task->columns ? p->outputs : first + task->columns;
    for (size_t column = first; column < end; ++column) {
        float totals[PACKED_ROWS] = {0};
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
            int32_t sums[PACKED_ROWS] = {0};
            if (!packed_dot(&block, task->input + row * p->width + k, p->width, rows, count, sums)) {
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
    (void)scratch;
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
    size_t rows = p->tokens / PACKED_ROWS + (p->tokens % PACKED_ROWS != 0);
    Task task = {
        outputs[0], inputs[0], values, scales, high, subscales, inputs[2],
        p, layout, groups, columns, tiles, context,
    };
    return rows * tiles == 1 ? project(&task, 0, 0)
                            : context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
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
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 3, 1, requirements, NULL, inputs, outputs, 1,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_quantized_linear_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &capabilities;
}
