#include "common.h"
#include <fenv.h>

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

static bool default_floating_point(void) {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
#else
    return fegetround() == FE_TONEAREST;
#endif
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

static TrillimStatus project_int8(
    const RoutedTask *task, size_t item, size_t token, size_t route, size_t expert, size_t first, size_t end
) {
    static const int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const RoutedParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->k / Q8_GROUP;
    size_t rows_per_token = p->a_route_stride == 0 ? 1 : p->routes;
    size_t row_token = p->a_token_stride == 0 ? 0 : token;
    size_t row_route = p->a_route_stride == 0 ? 0 : route;
    const RoutedQ8 *activation = task->quantized + (row_token * rows_per_token + row_route) * groups;
    for (size_t column = first; column < end; ++column) {
        size_t block = (expert * p->n + column) * groups;
        float sum = 0.0f, partial = 0.0f;
        for (size_t group = 0; group < groups; ++group) {
            if ((group & 31) == 0 && context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const uint8_t *codes = task->weight + (block + group) * 16;
            const int8_t *x = activation[group].codes;
            int32_t dot = 0;
            for (size_t i = 0; i < 16; ++i) {
                dot += values[codes[i] & 15] * x[i * 2] + values[codes[i] >> 4] * x[i * 2 + 1];
            }
            float scale = (trillim_mxfp4_scale(task->scales[block + group]) * 0.5f) * activation[group].scale;
            if ((group & 1) == 0) {
                partial = scale * (float)dot;
            } else {
                sum += multiply_add(scale, (float)dot, partial);
            }
        }
        float result = groups % 2 == 0 ? sum : sum + partial;
        memcpy((uint8_t *)task->output + (item * p->n + column) * sizeof(result), &result, sizeof(result));
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
            return project_int8(task, item, token, route, expert, first, end);
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
        status = run_tiles(task, tasks);
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

static inline float load_bias_bf16(uint16_t bits) {
    if ((bits & 0x7fff) > 0x7f80) {
        bits |= 0x0040;
    }
    return trillim_bf16(bits);
}

static TrillimStatus route_token(void *opaque, size_t token, size_t worker) {
    (void)worker;
    const RouterTask *task = opaque;
    const Parameters *p = task->p;
    const uint16_t *source = task->activation + token * p->width;
    float logits[32];
    for (size_t expert = 0; expert < p->experts; ++expert) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint16_t *weights = task->router + expert * p->width;
        float dot = 0.0f;
        for (size_t channel = 0; channel < p->width; ++channel) {
            dot += trillim_bf16(source[channel]) * trillim_bf16(weights[channel]);
        }
        logits[expert] = round_bf16(round_bf16(dot) + trillim_bf16(task->router_bias[expert]));
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

static TrillimStatus activate_slot(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const Epilogue *task = opaque;
    size_t hidden = task->p->hidden;
    const float *projected = task->workspace->projected + slot * hidden * 2;
    const uint16_t *bias = task->bias + (size_t)task->workspace->indices[slot] * hidden * 2;
    float *output = task->workspace->hidden + slot * hidden;
    for (size_t channel = 0; channel < hidden; ++channel) {
        if ((channel & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float gate = projected[2 * channel] + load_bias_bf16(bias[2 * channel]);
        float up = projected[2 * channel + 1] + load_bias_bf16(bias[2 * channel + 1]);
        output[channel] = activate(gate, up);
    }
    return TRILLIM_OK;
}

static TrillimStatus finish_output(void *opaque, size_t token, size_t worker) {
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
    for (size_t channel = 0; channel < width; ++channel) {
        if ((channel & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float sum = 0.0f;
        for (size_t route = 0; route < ROUTES; ++route) {
            sum += (projected[route][channel] + load_bias_bf16(bias[route][channel])) * scores[route];
        }
        destination[channel] = sum;
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
TRILLIM_KERNEL TrillimStatus trillim_scalar_gpt_oss_moe(
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

TRILLIM_KERNEL const char *trillim_scalar_gpt_oss_moe_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_gpt_oss_moe_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_gpt_oss_moe_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
