#include "common.h"
#include <fenv.h>
#include <immintrin.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct {
    size_t tokens;
    size_t routes;
    size_t experts;
    size_t n;
    size_t k;
    size_t a_token_stride;
    size_t a_route_stride;
    size_t index_token_stride;
    size_t index_route_stride;
    uint32_t a_dtype;
} RoutedParameters;

enum { Q8_GROUP = 32, Q8_LOCAL_GROUPS = 512, Q8_MAX_WIDTH = 8192, Q8_MAX_BYTES = 8 * 1024 * 1024 };

typedef struct {
    int8_t codes[Q8_GROUP];
    float scale;
} RoutedQ8;

typedef struct {
    void *output;
    const void *activation;
    const uint8_t *weight;
    const uint8_t *scales;
    const void *indices;
    const RoutedParameters *parameters;
    const TrillimKernelContext *context;
    size_t columns;
    size_t tiles;
    const RoutedQ8 *quantized;
} RoutedTask;

static size_t quantized_groups(const RoutedParameters *p) {
    if (p->n < 16 || p->k < Q8_GROUP || p->k > Q8_MAX_WIDTH || p->k % Q8_GROUP) {
        return 0;
    }
    size_t groups = p->k / Q8_GROUP;
    size_t token_rows = p->a_token_stride == 0 ? 1 : p->tokens;
    size_t route_rows = p->a_route_stride == 0 ? 1 : p->routes;
    size_t max_groups = Q8_MAX_BYTES / sizeof(RoutedQ8);
    if (route_rows == 0 || token_rows > max_groups / route_rows / groups) {
        return 0;
    }
    return token_rows * route_rows * groups;
}

static inline float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static inline bool mxfp4_palette(const uint8_t *codes, uint8_t exponent, float palette[16]) {
    static const float magnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    /* The packed gather rejects NaN scales and codes that decode to infinity. */
    if (exponent >= 253) {
        if (exponent == 255) {
            return false;
        }
        uint8_t limit = exponent == 253 ? 6 : 4;
        for (size_t i = 0; i < 16; ++i) {
            if ((codes[i] & 7) >= limit || ((codes[i] >> 4) & 7) >= limit) {
                return false;
            }
        }
    }
    float scale = trillim_mxfp4_scale(exponent);
    for (size_t i = 0; i < 8; ++i) {
        palette[i] = magnitudes[i] * scale;
        palette[i + 8] = -palette[i];
    }
    return true;
}

static inline bool load_activation(const void *data, size_t offset, uint32_t dtype, float row[32]) {
    switch (dtype) {
    case TRILLIM_FP32:
        memcpy(row, (const uint8_t *)data + offset * sizeof(float), 32 * sizeof(float));
        return true;
    case TRILLIM_FP16:
        for (size_t i = 0; i < 32; ++i) {
            row[i] = trillim_load(data, offset + i, TRILLIM_FP16).f;
        }
        return true;
    case TRILLIM_BF16:
        for (size_t i = 0; i < 32; ++i) {
            row[i] = trillim_load(data, offset + i, TRILLIM_BF16).f;
        }
        return true;
    default:
        return false;
    }
}

__attribute__((target("avx2"))) static inline __m256i decode_mxfp4(const uint8_t *codes) {
    const __m128i table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15);
    __m128i packed = _mm_loadu_si128((const __m128i *)codes);
    __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
    __m128i high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    return _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high));
}

__attribute__((target("avx2"))) static inline int32_t dot_mxfp4_q8(__m256i weights, __m256i magnitudes,
                                                                  const int8_t *x) {
    __m256i activation = _mm256_loadu_si256((const __m256i *)x);
    __m256i products = _mm256_maddubs_epi16(magnitudes, _mm256_sign_epi8(activation, weights));
    __m256i sums = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}

static bool default_floating_point(void) {
    return (_mm_getcsr() & 0xe040) == 0 && fegetround() == FE_TONEAREST;
}

static bool quantize_q8_group(const void *data, size_t offset, uint32_t dtype, RoutedQ8 *output) {
    float values[Q8_GROUP];
    if (!load_activation(data, offset, dtype, values)) {
        return false;
    }
    float peak = 0.0f;
    for (size_t i = 0; i < Q8_GROUP; ++i) {
        float magnitude = fabsf(values[i]);
        if (!isfinite(magnitude)) {
            return false;
        }
        if (magnitude > peak) {
            peak = magnitude;
        }
    }
    float scale = peak / 127.0f;
    float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    if (!isfinite(inverse) || scale > 65504.0f) {
        return false;
    }
    uint16_t bits = trillim_to_fp16(scale);
    if (bits == 0 && peak != 0.0f) {
        return false;
    }
    output->scale = trillim_fp16(bits);
    for (size_t i = 0; i < Q8_GROUP; ++i) {
        /* Adding 1.5 * 2^23 rounds to integer bits in nearest-even mode. */
        float rounded = values[i] * inverse + 12582912.0f;
        uint32_t integer;
        memcpy(&integer, &rounded, sizeof(integer));
        output->codes[i] = (int8_t)((int32_t)(integer & UINT32_C(0x007fffff)) - INT32_C(0x00400000));
    }
    return true;
}

static inline int64_t expert_id(const void *indices, size_t token, size_t route, const RoutedParameters *p) {
    size_t offset = token * p->index_token_stride + route * p->index_route_stride;
    return trillim_load(indices, offset, TRILLIM_INT64).i;
}

enum { ROUTED_ROWS = 4, ROUTED_MAX_ITEMS = 512, ROUTED_MAX_EXPERTS = 128 };

__attribute__((target("avx2")))
static TrillimStatus project_int8(
    const RoutedTask *task, const size_t *items, size_t rows, size_t expert, size_t first, size_t end
) {
    const RoutedParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->k / Q8_GROUP;
    size_t rows_per_token = p->a_route_stride == 0 ? 1 : p->routes;
    const RoutedQ8 *activation[ROUTED_ROWS];
    for (size_t row = 0; row < rows; ++row) {
        size_t token = p->a_token_stride == 0 ? 0 : items[row] / p->routes;
        size_t route = p->a_route_stride == 0 ? 0 : items[row] % p->routes;
        activation[row] = task->quantized + (token * rows_per_token + route) * groups;
    }
    for (size_t column = first; column < end; ++column) {
        size_t block = (expert * p->n + column) * groups;
        float sums[ROUTED_ROWS] = {0}, partials[ROUTED_ROWS] = {0};
        for (size_t group = 0; group < groups; ++group) {
            if ((group & 31) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256i weights = decode_mxfp4(task->weight + (block + group) * 16);
            __m256i magnitudes = _mm256_abs_epi8(weights);
            float weight_scale = trillim_mxfp4_scale(task->scales[block + group]) * 0.5f;
            for (size_t row = 0; row < rows; ++row) {
                const RoutedQ8 *x = activation[row] + group;
                int32_t dot = dot_mxfp4_q8(weights, magnitudes, x->codes);
                float scale = weight_scale * x->scale;
                if ((group & 1) == 0) {
                    partials[row] = scale * (float)dot;
                } else {
                    sums[row] += multiply_add(scale, (float)dot, partials[row]);
                }
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float result = groups % 2 == 0 ? sums[row] : sums[row] + partials[row];
            memcpy((uint8_t *)task->output + (items[row] * p->n + column) * sizeof(result), &result, sizeof(result));
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus check_int8_scales(
    const uint8_t *scales, size_t count, const TrillimKernelContext *context, bool *bounded
) {
    *bounded = false;
    for (size_t i = 0; i < count; ++i) {
        if ((i & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        /* Keep the grouped scale and full-width sum within the FP32 range. */
        if (scales[i] < 32 || scales[i] > 212) {
            return TRILLIM_OK;
        }
    }
    *bounded = true;
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const RoutedTask *task = opaque;
    const RoutedParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t item = tile / task->tiles;
    size_t token = item / p->routes;
    size_t route = item % p->routes;
    size_t expert = (size_t)expert_id(task->indices, token, route, p);
    size_t a_offset = token * p->a_token_stride + route * p->a_route_stride;
    size_t groups = p->k / 32;
    size_t first = tile % task->tiles * task->columns;
    size_t end = p->n - first < task->columns ? p->n : first + task->columns;
    if (task->quantized != NULL && default_floating_point()) {
        const uint8_t *scales = task->scales + (expert * p->n + first) * groups;
        bool bounded;
        TrillimStatus status = check_int8_scales(scales, (end - first) * groups, context, &bounded);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (bounded) {
            return project_int8(task, &item, 1, expert, first, end);
        }
    }
    for (size_t column = first; column < end; column += 4) {
        size_t count = end - column < 4 ? end - column : 4;
        size_t block = (expert * p->n + column) * groups;
        float sums[4] = {0};
        for (size_t group = 0; group < groups; ++group) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float palette[4][16];
            const uint8_t *codes[4];
            for (size_t j = 0; j < count; ++j) {
                size_t offset = block + j * groups + group;
                codes[j] = task->weight + offset * 16;
                if (!mxfp4_palette(codes[j], task->scales[offset], palette[j])) {
                    return TRILLIM_ERROR;
                }
            }
            float a[32];
            if (!load_activation(task->activation, a_offset + group * 32, p->a_dtype, a)) {
                return TRILLIM_ERROR;
            }
            for (size_t i = 0; i < 16; ++i) {
                for (size_t j = 0; j < count; ++j) {
                    uint8_t code = codes[j][i];
                    sums[j] = multiply_add(a[i * 2], palette[j][code & 15], sums[j]);
                    sums[j] = multiply_add(a[i * 2 + 1], palette[j][code >> 4], sums[j]);
                }
            }
        }
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        memcpy((uint8_t *)task->output + (item * p->n + column) * sizeof(float), sums, count * sizeof(float));
    }
    return TRILLIM_OK;
}

static TrillimStatus run_tiles(RoutedTask *task, size_t tasks) {
    const TrillimThreadPool *pool = task->context->thread_pool;
    return tasks == 1 ? project(task, 0, 0) : pool->run(pool->context, tasks, project, task);
}

typedef struct {
    size_t expert, first, count;
} ExpertTile;

typedef struct {
    RoutedTask *projection;
    const ExpertTile *batches;
    const size_t *items;
} GroupedTask;

static TrillimStatus project_expert(void *opaque, size_t tile, size_t worker) {
    const GroupedTask *grouped = opaque;
    RoutedTask *task = grouped->projection;
    const RoutedParameters *p = task->parameters;
    const ExpertTile *batch = grouped->batches + tile / task->tiles;
    const size_t *items = grouped->items + batch->first;
    size_t column_tile = tile % task->tiles;
    if (default_floating_point()) {
        size_t first = column_tile * task->columns;
        size_t end = p->n - first < task->columns ? p->n : first + task->columns;
        size_t groups = p->k / Q8_GROUP;
        const uint8_t *scales = task->scales + (batch->expert * p->n + first) * groups;
        bool bounded;
        TrillimStatus status = check_int8_scales(scales, (end - first) * groups, task->context, &bounded);
        if (status != TRILLIM_OK) {
            return status;
        }
        if (bounded) {
            return project_int8(task, items, batch->count, batch->expert, first, end);
        }
    }
    for (size_t row = 0; row < batch->count; ++row) {
        TrillimStatus status = project(task, items[row] * task->tiles + column_tile, worker);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus run_quantized_tiles(RoutedTask *task, size_t tasks) {
    const RoutedParameters *p = task->parameters;
    size_t count = p->tokens * p->routes;
    if (count < 8 || count > ROUTED_MAX_ITEMS || p->experts > ROUTED_MAX_EXPERTS) {
        return run_tiles(task, tasks);
    }
    size_t totals[ROUTED_MAX_EXPERTS] = {0}, offsets[ROUTED_MAX_EXPERTS];
    size_t experts[ROUTED_MAX_ITEMS], items[ROUTED_MAX_ITEMS];
    for (size_t item = 0; item < count; ++item) {
        size_t expert = (size_t)expert_id(task->indices, item / p->routes, item % p->routes, p);
        experts[item] = expert;
        ++totals[expert];
    }
    ExpertTile batches[ROUTED_MAX_ITEMS];
    size_t first = 0, batch_count = 0;
    for (size_t expert = 0; expert < p->experts; ++expert) {
        offsets[expert] = first;
        for (size_t row = 0; row < totals[expert]; row += ROUTED_ROWS) {
            size_t rows = totals[expert] - row < ROUTED_ROWS ? totals[expert] - row : ROUTED_ROWS;
            batches[batch_count++] = (ExpertTile){expert, first + row, rows};
        }
        first += totals[expert];
    }
    if (batch_count == count) {
        return run_tiles(task, tasks);
    }
    for (size_t item = 0; item < count; ++item) {
        items[offsets[experts[item]]++] = item;
    }
    GroupedTask grouped = {task, batches, items};
    const TrillimThreadPool *pool = task->context->thread_pool;
    return pool->run(pool->context, batch_count * task->tiles, project_expert, &grouped);
}

static TrillimStatus try_routed_int8(RoutedTask *task, size_t tasks, RoutedQ8 *scratch, bool *handled) {
    const RoutedParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t total = quantized_groups(p);
    if (total == 0) {
        return TRILLIM_OK;
    }
    size_t groups = p->k / Q8_GROUP;
    size_t token_rows = p->a_token_stride == 0 ? 1 : p->tokens;
    size_t route_rows = p->a_route_stride == 0 ? 1 : p->routes;
    RoutedQ8 local[Q8_LOCAL_GROUPS];
    RoutedQ8 *quantized = total <= Q8_LOCAL_GROUPS ? local : scratch;
    if (quantized == NULL) {
        return TRILLIM_ERROR;
    }
    TrillimStatus status = TRILLIM_OK;
    bool valid = true;
    for (size_t t = 0; t < token_rows && valid; ++t) {
        for (size_t r = 0; r < route_rows && valid; ++r) {
            size_t start = t * p->a_token_stride + r * p->a_route_stride;
            RoutedQ8 *row = quantized + (t * route_rows + r) * groups;
            for (size_t g = 0; g < groups; ++g) {
                if ((g & 31) == 0 && context->is_cancelled(context->cancellation_context)) {
                    status = TRILLIM_CANCELLED;
                    valid = false;
                    break;
                }
                if (!quantize_q8_group(task->activation, start + g * Q8_GROUP, p->a_dtype, row + g)) {
                    valid = false;
                    break;
                }
            }
        }
    }
    if (valid) {
        task->quantized = quantized;
        *handled = true;
        status = run_quantized_tiles(task, tasks);
        task->quantized = NULL;
    }
    return status;
}

static inline TrillimStatus routed_matmul(
    void *output, const void *activation, const uint8_t *weight, const void *indices,
    const RoutedParameters *p, RoutedQ8 *scratch, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    if (!pool->worker_count) {
        return TRILLIM_ERROR;
    }
    size_t items = p->tokens * p->routes;
    for (size_t item = 0; item < items; ++item) {
        if ((item & 1023) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int64_t expert = expert_id(indices, item / p->routes, item % p->routes, p);
        if (expert < 0 || (uint64_t)expert >= p->experts) {
            return TRILLIM_ERROR;
        }
    }
    size_t columns = p->n / pool->worker_count / 4 / 16 * 16;
    if (columns < 16) {
        columns = 16;
    }
    size_t tiles = p->n / columns + (p->n % columns != 0);
    const uint8_t *scales = weight + p->experts * p->n * (p->k / 32) * 16;
    RoutedTask task = {output, activation, weight, scales, indices, p, context, columns, tiles, NULL};
    size_t tasks = items * tiles;
    if (default_floating_point()) {
        bool handled = false;
        TrillimStatus status = try_routed_int8(&task, tasks, scratch, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
    return run_tiles(&task, tasks);
}



static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const RoutedParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !p->experts || !p->n || !p->k || p->k % 32) {
        return TRILLIM_ERROR;
    }
    if (!p->tokens || !p->routes) {
        return TRILLIM_OK;
    }
    size_t a_shape[] = {p->tokens, p->routes, p->k};
    size_t a_strides[] = {p->a_token_stride, p->a_route_stride, 1};
    size_t index_strides[] = {p->index_token_stride, p->index_route_stride};
    size_t a_span, index_span;
    size_t a_bytes = trillim_itemsize(p->a_dtype);
    if (!trillim_span(3, a_shape, a_strides, &a_span) || a_span > SIZE_MAX / a_bytes ||
        !trillim_span(2, a_shape, index_strides, &index_span) || index_span > SIZE_MAX / sizeof(int64_t) ||
        p->n > SIZE_MAX / p->k ||
        !trillim_packed_payload_size(p->experts, p->n * p->k, TRILLIM_MXFP4, &inputs[1]) ||
        p->tokens > SIZE_MAX / p->routes ||
        p->tokens * p->routes > SIZE_MAX / p->n / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = a_span * a_bytes;
    inputs[2] = index_span * sizeof(int64_t);
    outputs[0] = p->tokens * p->routes * p->n * sizeof(float);
    /* Runtime rows can be smaller than the arena's bound and still qualify for Q8. */
    if (p->n >= 16) {
        size_t groups_per_row = (p->k < Q8_MAX_WIDTH ? p->k : Q8_MAX_WIDTH) / Q8_GROUP;
        size_t max_groups = Q8_MAX_BYTES / sizeof(RoutedQ8);
        size_t rows = p->tokens * p->routes;
        size_t groups = rows > max_groups / groups_per_row ? max_groups : rows * groups_per_row;
        if (groups > Q8_LOCAL_GROUPS) {
            *scratch = groups * sizeof(RoutedQ8);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_routed_matmul_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return routed_matmul(outputs[0], inputs[0], inputs[1], inputs[2], parameters, scratch, context);
}

TRILLIM_KERNEL const char *trillim_x86_routed_matmul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _native_dtype, _product, _storage_shape\n"
           "physical_inputs = ('a', 'weight', 'indices')\n"
           "def bind(c):\n"
           "    a, weight, indices = (c.inputs[name] for name in physical_inputs)\n"
           "    out = c.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    if a.dtype not in floating or out.dtype != 'fp32' or weight.dtype != 'mxfp4' or indices.dtype != 'int64':\n"
           "        raise ValueError('routed matmul requires floating activations, MXFP4 weights, int64 indices and an FP32 output')\n"
           "    if len(a.shape) != 4 or len(weight.shape) != 3 or len(indices.shape) != 3 or len(out.shape) != 4:\n"
           "        raise ValueError('routed matmul requires a rank-three expert bank and rank-four matvecs')\n"
           "    experts, n, k = weight.shape\n"
           "    if any(d.value is None or d.value <= 0 for d in weight.shape) or k.value % 32:\n"
           "        raise ValueError('routed matmul requires fixed expert shapes and group-aligned weight rows')\n"
           "    if a.shape[2].value != 1 or indices.shape[2].value != 1 or a.shape[3] != k or out.shape[2].value != 1 or out.shape[3] != n:\n"
           "        raise ValueError('routed matmul requires one activation row and one expert index per route')\n"
           "    ws = _contiguous_strides(weight.shape)\n"
           "    if (weight.strides or ws) != ws or _storage_shape(weight) != (experts, _product(weight.shape[1:])):\n"
           "        raise ValueError('routed matmul requires contiguous expert-major weights')\n"
           "    ast = a.strides or _contiguous_strides(a.shape)\n"
           "    ist = _contiguous_strides(indices.shape)\n"
           "    if ast[3].value != 1 or (indices.strides or ist) != ist:\n"
           "        raise ValueError('routed matmul requires contiguous activation rows and indices')\n"
           "    at, ar = _broadcast_strides(a.shape[:2], out.shape[:2], ast[:2])\n"
           "    it, ir = _broadcast_strides(indices.shape[:2], out.shape[:2], ist[:2])\n"
           "    return dict(tokens=out.shape[0], routes=out.shape[1], experts=experts, n=n, k=k,\n"
           "                a_token_stride=at, a_route_stride=ar, index_token_stride=it, index_route_stride=ir,\n"
           "                a_dtype=_native_dtype(a.dtype))\n"
           "def pattern(a: Tensor, weight: Tensor, indices: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, Permute(Gather(weight, indices=indices), (0, 1, 3, 2))),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_routed_matmul_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(RoutedParameters, tokens), NULL},
        {"routes", "size_t", offsetof(RoutedParameters, routes), NULL},
        {"experts", "size_t", offsetof(RoutedParameters, experts), NULL},
        {"n", "size_t", offsetof(RoutedParameters, n), NULL},
        {"k", "size_t", offsetof(RoutedParameters, k), NULL},
        {"a_token_stride", "size_t", offsetof(RoutedParameters, a_token_stride), NULL},
        {"a_route_stride", "size_t", offsetof(RoutedParameters, a_route_stride), NULL},
        {"index_token_stride", "size_t", offsetof(RoutedParameters, index_token_stride), NULL},
        {"index_route_stride", "size_t", offsetof(RoutedParameters, index_route_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(RoutedParameters, a_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(RoutedParameters), TRILLIM_ALIGNOF(RoutedParameters), sizeof(fields) / sizeof(fields[0]),
        fields, 3, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(RoutedQ8)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_routed_matmul_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
