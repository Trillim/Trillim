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



enum { ROUTES = 4, PANEL_ROWS = 128 };

typedef struct {
    size_t rows, width, hidden, experts;
} Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    memset(inputs, 0, 7 * sizeof(*inputs));
    outputs[0] = *scratch = 0;
    if (p->rows > 65536 || p->experts != 32 || p->width == 0 || p->hidden == 0 ||
        p->width > 8192 || p->hidden > 8192 || p->width % 32 || p->hidden % 32) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->experts * p->width * sizeof(uint16_t);
    inputs[2] = p->experts * sizeof(uint16_t);
    inputs[3] = p->experts * p->hidden * 2 * (p->width / 32) * 17;
    inputs[4] = p->experts * p->width * (p->hidden / 32) * 17;
    inputs[5] = p->experts * p->hidden * 2 * sizeof(uint16_t);
    inputs[6] = p->experts * p->width * sizeof(uint16_t);
    outputs[0] = p->rows * p->width * sizeof(float);
    if (p->rows != 0) {
        size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
        size_t slots = rows * ROUTES;
        size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
        *scratch = slots * ((projection_width + p->hidden) * sizeof(float) +
                            sizeof(int64_t) + sizeof(uint16_t));
        size_t groups_up = rows * (p->width / Q8_GROUP);
        size_t groups_down = slots * (p->hidden / Q8_GROUP);
        size_t groups = groups_up > groups_down ? groups_up : groups_down;
        if (groups > Q8_LOCAL_GROUPS) {
            *scratch += groups * sizeof(RoutedQ8);
        }
    }
    return TRILLIM_OK;
}

/* Match the staged FP32 arithmetic in the decomposed operations. */
static float activate(float gate, float up) {
    gate = gate <= 7.0f ? gate : 7.0f;
    up = up >= -7.0f ? up : -7.0f;
    up = up <= 7.0f ? up : 7.0f;
    float scaled = gate * 1.702f;
    float e = expf(-fabsf(scaled));
    float denominator = e + 1.0f;
    float probability = scaled >= 0.0f ? 1.0f / denominator : e / denominator;
    return (gate * probability) * (up + 1.0f);
}

#define SIMD __attribute__((target("avx2,fma")))

SIMD static __m256 exp8(__m256 x) {
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(0.00021219444005469058f), r);
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 720.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 120.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 24.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f / 6.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(0.5f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    value = _mm256_fmadd_ps(value, r, _mm256_set1_ps(1.0f));
    __m256i exponent = _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_mul_ps(value, _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23)));
}

SIMD static __m256 activate8(__m256 gate, __m256 up) {
    gate = _mm256_min_ps(gate, _mm256_set1_ps(7.0f));
    up = _mm256_min_ps(_mm256_max_ps(up, _mm256_set1_ps(-7.0f)), _mm256_set1_ps(7.0f));
    __m256 scaled = _mm256_mul_ps(gate, _mm256_set1_ps(1.702f));
    if (_mm256_movemask_ps(_mm256_cmp_ps(scaled, _mm256_set1_ps(-80.0f), _CMP_LT_OQ))) {
        float gates[8], ups[8];
        _mm256_storeu_ps(gates, gate);
        _mm256_storeu_ps(ups, up);
        for (size_t lane = 0; lane < 8; ++lane) {
            gates[lane] = activate(gates[lane], ups[lane]);
        }
        return _mm256_loadu_ps(gates);
    }
    __m256 e = exp8(_mm256_or_ps(scaled, _mm256_set1_ps(-0.0f)));
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denominator = _mm256_add_ps(e, one);
    __m256 numerator = _mm256_blendv_ps(e, one, _mm256_cmp_ps(scaled, _mm256_setzero_ps(), _CMP_GE_OQ));
    __m256 probability = _mm256_div_ps(numerator, denominator);
    return _mm256_mul_ps(_mm256_mul_ps(gate, probability), _mm256_add_ps(up, one));
}

SIMD static __m256 activate_interleaved(__m256 first, __m256 second) {
    const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    __m256 gate = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(first, second, 0x88), order);
    __m256 up = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(first, second, 0xdd), order);
    return activate8(gate, up);
}

SIMD static inline __m256 bias8(const uint16_t *bias) {
    __m128i bits = _mm_loadu_si128((const __m128i *)bias);
    __m128i magnitude = _mm_and_si128(bits, _mm_set1_epi16(0x7fff));
    __m128i nan = _mm_cmpgt_epi16(magnitude, _mm_set1_epi16(0x7f80));
    bits = _mm_or_si128(bits, _mm_and_si128(nan, _mm_set1_epi16(0x0040)));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

typedef struct {
    float *projected, *hidden;
    int64_t *indices;
    uint16_t *scores;
    RoutedQ8 *quantized;
} Workspace;

typedef struct {
    const Parameters *p;
    const uint16_t *activation, *router, *router_bias;
    int64_t *indices;
    uint16_t *scores;
    const TrillimKernelContext *context;
} RouterTask;

typedef struct {
    const Parameters *p;
    const Workspace *workspace;
    const uint16_t *bias;
    float *output;
    const TrillimKernelContext *context;
} Epilogue;

static inline float round_bf16(float value) {
    return trillim_bf16(trillim_to_bf16(value));
}

SIMD static inline __m128 widen4(const uint16_t *source) {
    __m128i bits = _mm_loadl_epi64((const __m128i *)source);
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(bits), 16));
}

SIMD static TrillimStatus route_token(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const RouterTask *task = opaque;
    const Parameters *p = task->p;
    const uint16_t *source = task->activation + token * p->width;
    float logits[32];
    for (size_t expert = 0; expert < p->experts; expert += 4) {
        const uint16_t *weights = task->router + expert * p->width;
        __m128 dots = _mm_setzero_ps();
        for (size_t channel = 0; channel < p->width; channel += 4) {
            if ((channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m128 w0 = widen4(weights + channel);
            __m128 w1 = widen4(weights + p->width + channel);
            __m128 w2 = widen4(weights + 2 * p->width + channel);
            __m128 w3 = widen4(weights + 3 * p->width + channel);
            _MM_TRANSPOSE4_PS(w0, w1, w2, w3);
            dots = _mm_add_ps(dots, _mm_mul_ps(w0, _mm_set1_ps(trillim_bf16(source[channel]))));
            dots = _mm_add_ps(dots, _mm_mul_ps(w1, _mm_set1_ps(trillim_bf16(source[channel + 1]))));
            dots = _mm_add_ps(dots, _mm_mul_ps(w2, _mm_set1_ps(trillim_bf16(source[channel + 2]))));
            dots = _mm_add_ps(dots, _mm_mul_ps(w3, _mm_set1_ps(trillim_bf16(source[channel + 3]))));
        }
        _mm_storeu_ps(logits + expert, dots);
        for (size_t lane = 0; lane < 4; ++lane) {
            size_t index = expert + lane;
            logits[index] = round_bf16(round_bf16(logits[index]) + trillim_bf16(task->router_bias[index]));
        }
    }

    uint32_t available = UINT32_MAX;
    float scores[ROUTES];
    float maximum = -INFINITY;
    for (size_t route = 0; route < ROUTES; ++route) {
        size_t selected = p->experts;
        float best = 0.0f;
        for (size_t expert = 0; expert < p->experts; ++expert) {
            if ((available & (UINT32_C(1) << expert)) && (selected == p->experts || !(best >= logits[expert]))) {
                selected = expert;
                best = logits[expert];
            }
        }
        available &= ~(UINT32_C(1) << selected);
        task->indices[token * ROUTES + route] = (int64_t)selected;
        scores[route] = best;
        maximum = maximum >= best ? maximum : best;
    }
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float sum = 0.0f;
    for (size_t route = 0; route < ROUTES; ++route) {
        scores[route] = round_bf16(expf(round_bf16(scores[route] - maximum)));
        sum += scores[route];
    }
    sum = round_bf16(sum);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t route = 0; route < ROUTES; ++route) {
        task->scores[token * ROUTES + route] = trillim_to_bf16(scores[route] / denominator);
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus activate_slot(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const Epilogue *task = opaque;
    size_t hidden = task->p->hidden;
    const float *projected = task->workspace->projected + slot * hidden * 2;
    const uint16_t *bias = task->bias + (size_t)task->workspace->indices[slot] * hidden * 2;
    float *output = task->workspace->hidden + slot * hidden;
    for (size_t channel = 0; channel < hidden; channel += 8) {
        if ((channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = 2 * channel;
        __m256 first = _mm256_add_ps(_mm256_loadu_ps(projected + offset), bias8(bias + offset));
        __m256 second = _mm256_add_ps(_mm256_loadu_ps(projected + offset + 8), bias8(bias + offset + 8));
        _mm256_storeu_ps(output + channel, activate_interleaved(first, second));
    }
    return TRILLIM_OK;
}

SIMD static TrillimStatus finish_output(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const Epilogue *task = opaque;
    size_t width = task->p->width, slot = token * ROUTES;
    float *destination = task->output + token * width;
    const float *projected[ROUTES];
    const uint16_t *bias[ROUTES];
    float scores[ROUTES];
    for (size_t route = 0; route < ROUTES; ++route) {
        projected[route] = task->workspace->projected + (slot + route) * width;
        bias[route] = task->bias + (size_t)task->workspace->indices[slot + route] * width;
        scores[route] = trillim_bf16(task->workspace->scores[slot + route]);
    }
    for (size_t channel = 0; channel < width; channel += 8) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 sum = _mm256_setzero_ps();
        for (size_t route = 0; route < ROUTES; ++route) {
            __m256 value = _mm256_add_ps(_mm256_loadu_ps(projected[route] + channel), bias8(bias[route] + channel));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(value, _mm256_set1_ps(scores[route])));
        }
        _mm256_storeu_ps(destination + channel, sum);
    }
    return TRILLIM_OK;
}

static TrillimStatus panel(
    float *output, const uint16_t *activation, const void *const *inputs,
    const Parameters *p, size_t rows, const Workspace *workspace, const TrillimKernelContext *context
) {
    const TrillimThreadPool *pool = context->thread_pool;
    RouterTask routing = {p, activation, inputs[1], inputs[2], workspace->indices, workspace->scores, context};
    TrillimStatus status = rows == 1 ? route_token(&routing, 0, 0) :
                                       pool->run(pool->context, rows, route_token, &routing);
    if (status != TRILLIM_OK) return status;

    RoutedParameters up = {
        .tokens = rows, .routes = ROUTES, .experts = p->experts, .n = p->hidden * 2, .k = p->width,
        .a_token_stride = p->width, .a_route_stride = 0,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_BF16,
    };
    status = routed_matmul(workspace->projected, activation, inputs[3], workspace->indices, &up, workspace->quantized, context);
    if (status != TRILLIM_OK) return status;

    Epilogue gate = {p, workspace, inputs[5], NULL, context};
    size_t slots = rows * ROUTES;
    if (slots > 4 && slots > 16384 / p->hidden) {
        status = pool->run(pool->context, slots, activate_slot, &gate);
    } else {
        for (size_t slot = 0; slot < slots && status == TRILLIM_OK; ++slot) {
            status = activate_slot(&gate, slot, 0);
        }
    }
    if (status != TRILLIM_OK) return status;

    RoutedParameters down = {
        .tokens = rows, .routes = ROUTES, .experts = p->experts, .n = p->width, .k = p->hidden,
        .a_token_stride = ROUTES * p->hidden, .a_route_stride = p->hidden,
        .index_token_stride = ROUTES, .index_route_stride = 1, .a_dtype = TRILLIM_FP32,
    };
    status = routed_matmul(workspace->projected, workspace->hidden, inputs[4], workspace->indices, &down, workspace->quantized, context);
    if (status != TRILLIM_OK) return status;

    Epilogue result = {p, workspace, inputs[6], output, context};
    return rows == 1 ? finish_output(&result, 0, 0) : pool->run(pool->context, rows, finish_output, &result);
}
TRILLIM_KERNEL TrillimStatus trillim_x86_gpt_oss_moe_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->rows == 0) {
        return TRILLIM_OK;
    }
    size_t spans[7], output_span, required;
    if (!scratch || requirements(p, spans, &output_span, &required) != TRILLIM_OK) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->rows < PANEL_ROWS ? p->rows : PANEL_ROWS;
    size_t slots = rows * ROUTES;
    size_t projection_width = p->width > p->hidden * 2 ? p->width : p->hidden * 2;
    Workspace workspace;
    workspace.projected = scratch;
    workspace.hidden = workspace.projected + slots * projection_width;
    workspace.indices = (int64_t *)(workspace.hidden + slots * p->hidden);
    workspace.scores = (uint16_t *)(workspace.indices + slots);
    workspace.quantized = (RoutedQ8 *)(workspace.scores + slots);

    const uint16_t *activation = inputs[0];
    float *output = outputs[0];
    for (size_t first = 0; first < p->rows; first += rows) {
        size_t length = p->rows - first < rows ? p->rows - first : rows;
        TrillimStatus status = panel(output + first * p->width, activation + first * p->width,
                                     inputs, p, length, &workspace, context);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_gpt_oss_moe_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _storage_shape\n"
           "physical_inputs = ('x', 'router', 'router_bias', 'gate_up', 'down', 'gate_up_bias', 'down_bias')\n"
           "def pattern(x: Tensor, *, router: Tensor, router_bias: Tensor, gate_up: Tensor, down: Tensor,\n"
           "            gate_up_bias: Tensor, down_bias: Tensor, top_k: int = 4, limit: float = 7.0) -> tuple[Tensor, ...]:\n"
           "    if top_k != 4 or limit != 7.0 or x.dtype != 'bf16' or router.dtype != 'bf16' or gate_up.dtype != 'mxfp4' or down.dtype != 'mxfp4':\n"
           "        raise ValueError('expected four routes, BF16 logits, and MXFP4 experts')\n"
           "    return (GptOssMoE(x, router=router, router_bias=router_bias, gate_up=gate_up, down=down,\n"
           "                     gate_up_bias=gate_up_bias, down_bias=down_bias, top_k=top_k, limit=limit),)\n"
           "def bind(c):\n"
           "    x, router, rb, gate_up, down, gb, db = (c.inputs[k] for k in physical_inputs)\n"
           "    if tuple(len(v.shape) for v in (x, router, rb, gate_up, down, gb, db)) != (2, 2, 1, 3, 3, 2, 2):\n"
           "        raise ValueError('expected contiguous activation and expert banks')\n"
           "    if any(v.strides and v.strides != _contiguous_strides(v.shape) for v in (x, router, rb, gate_up, down, gb, db)):\n"
           "        raise ValueError('noncontiguous MoE input')\n"
           "    if rb.dtype != 'bf16' or gb.dtype != 'bf16' or db.dtype != 'bf16' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('expected BF16 biases and FP32 output')\n"
           "    experts, doubled, width = gate_up.shape\n"
           "    if any(_storage_shape(w) != (experts, _product(w.shape[1:])) for w in (gate_up, down)):\n"
           "        raise ValueError('expected canonical MXFP4 expert banks')\n"
           "    if router.shape != (experts, width) or rb.shape != (experts,) or gb.shape != (experts, doubled) or down.shape[:2] != (experts, width) or db.shape != (experts, width):\n"
           "        raise ValueError('incompatible expert bank dimensions')\n"
           "    hidden = down.shape[-1]\n"
           "    if hidden.value is None or doubled.value != 2 * hidden.value or x.shape[-1] != width:\n"
           "        raise ValueError('incompatible expert projection widths')\n"
           "    out = c.outputs[0]\n"
           "    if len(out.shape) != 2 or out.shape[-1] != width or (out.strides and out.strides != _contiguous_strides(out.shape)):\n"
           "        raise ValueError('expected contiguous FP32 output rows')\n"
           "    return dict(rows=x.shape[0], width=width, hidden=hidden, experts=experts)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_gpt_oss_moe_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"hidden", "size_t", offsetof(Parameters, hidden), NULL},
        {"experts", "size_t", offsetof(Parameters, experts), NULL},
    };
    static const size_t inputs[] = {2, 2, 2, 1, 1, 2, 2}, outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields,
        7, 1, requirements, NULL, inputs, outputs, 64,
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_gpt_oss_moe_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &capabilities;
}
