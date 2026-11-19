#include "common.h"
#include <arm_neon.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

enum {
    ROWS_PER_TASK = 4,
    MAX_TASKS = 64,
    PACK_ROWS = 256,
    BLAS_COLUMNS = 64,
    BLAS_DEPTH = 1024
};

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);

static Sgemm sgemm;
static atomic_uint blas_state;
#ifdef __APPLE__
static unsigned (*get_threading)(void);
static int (*set_threading)(unsigned);
#endif

typedef struct {
    size_t heads, kv_heads, queries, length, key_dim, value_dim, rank;
    const size_t *q_strides, *k_strides, *v_strides;
    size_t q_rows, q_columns, k_rows, k_columns, v_rows, v_columns;
    size_t window;
    uint32_t output_dtype, q_dtype, k_dtype, v_dtype, causal, windowed, default_scale;
    float scale;
    uint32_t sink_dtype;
} Parameters;

typedef struct {
    uint16_t *output;
    const uint16_t *q, *k, *v;
    const void *sinks;
    float *scratch;
    float *packed_values;
    size_t tiles, tasks;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static size_t query_tiles(const Parameters *p) {
    return p->queries / ROWS_PER_TASK + (p->queries % ROWS_PER_TASK != 0);
}

static bool blas_shape(const Parameters *p) {
    return p->length >= 256 && p->value_dim > 32767 / p->length;
}

static void load_blas(void) {
#ifdef _WIN32
    static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        HMODULE library = LoadLibraryA(names[i]);
        if (library == NULL) {
            continue;
        }
        FARPROC symbol = GetProcAddress(library, "cblas_sgemm");
        if (symbol != NULL) {
            _Static_assert(sizeof(sgemm) == sizeof(symbol), "function pointer size");
            memcpy(&sgemm, &symbol, sizeof(sgemm));
            return;
        }
        FreeLibrary(library);
    }
#else
#ifdef __APPLE__
    static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
    static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
#endif
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        void *library = dlopen(names[i], RTLD_LOCAL | RTLD_LAZY);
        if (library == NULL) {
            continue;
        }
        void *symbol = NULL;
#ifdef __APPLE__
        symbol = dlsym(library, "cblas_sgemm$NEWLAPACK");
#endif
        if (symbol == NULL) {
            symbol = dlsym(library, "cblas_sgemm");
        }
        if (symbol != NULL) {
            _Static_assert(sizeof(sgemm) == sizeof(symbol), "function pointer size");
            memcpy(&sgemm, &symbol, sizeof(sgemm));
#ifdef __APPLE__
            void *getter = dlsym(library, "BLASGetThreading");
            void *setter = dlsym(library, "BLASSetThreading");
            if (getter != NULL && setter != NULL) {
                memcpy(&get_threading, &getter, sizeof(get_threading));
                memcpy(&set_threading, &setter, sizeof(set_threading));
            }
#endif
            return;
        }
        dlclose(library);
    }
#endif
}

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
        unsigned expected = 0;
        if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                    memory_order_acq_rel, memory_order_acquire)) {
            load_blas();
            atomic_store_explicit(&blas_state, 2, memory_order_release);
        } else {
            while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
                atomic_signal_fence(memory_order_acquire);
            }
        }
    }
    return sgemm != NULL;
}

static bool input_span(const size_t *shape, const size_t *strides,
                       size_t rows, size_t columns, size_t *bytes) {
    size_t span;
    if ((columns && rows > SIZE_MAX / sizeof(uint16_t) / columns) ||
        !trillim_span(3, shape, strides, &span) || span > rows * columns ||
        strides[2] != 1) {
        return false;
    }
    *bytes = rows * columns * sizeof(uint16_t);
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = inputs[3] = outputs[0] = *scratch = 0;
    if (p->rank != 3 || p->q_dtype != TRILLIM_BF16 || p->k_dtype != TRILLIM_BF16 ||
        p->v_dtype != TRILLIM_BF16 || p->output_dtype != TRILLIM_BF16 ||
        !trillim_float(p->sink_dtype) || p->causal > 1 || p->windowed > 1 ||
        p->default_scale > 1 || (p->key_dim != 64 && p->key_dim != 128) ||
        (p->value_dim != 64 && p->value_dim != 128) || !p->kv_heads ||
        p->heads % p->kv_heads || p->queries > INT64_MAX || p->length > INT64_MAX) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    size_t q_shape[] = {p->heads, p->queries, p->key_dim};
    size_t k_shape[] = {p->kv_heads, p->length, p->key_dim};
    size_t v_shape[] = {p->kv_heads, p->length, p->value_dim};
    if (p->heads > SIZE_MAX / sizeof(uint16_t) / p->value_dim / p->queries ||
        p->heads > SIZE_MAX / trillim_itemsize(p->sink_dtype) ||
        !input_span(q_shape, p->q_strides, p->q_rows, p->q_columns, &inputs[0]) ||
        !input_span(k_shape, p->k_strides, p->k_rows, p->k_columns, &inputs[1]) ||
        !input_span(v_shape, p->v_strides, p->v_rows, p->v_columns, &inputs[2])) {
        return TRILLIM_ERROR;
    }
    size_t work = p->heads * query_tiles(p);
    size_t tasks = work < MAX_TASKS ? work : MAX_TASKS;
    if (p->length > SIZE_MAX / sizeof(float) / tasks) {
        return TRILLIM_ERROR;
    }
    inputs[3] = p->heads * trillim_itemsize(p->sink_dtype);
    outputs[0] = p->heads * p->queries * p->value_dim * sizeof(uint16_t);
    *scratch = tasks * p->length * sizeof(float);
    if (blas_shape(p)) {
        size_t available = SIZE_MAX - *scratch;
        if (p->length > available / sizeof(float) / p->value_dim / p->kv_heads) {
            return TRILLIM_ERROR;
        }
        *scratch += p->kv_heads * p->length * p->value_dim * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline float32x4_t widen(uint16x4_t bits) {
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

static inline uint16x4_t narrow(float32x4_t value) {
    uint32x4_t bits = vreinterpretq_u32_f32(value);
    uint32x4_t upper = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(upper, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    return vmovn_u32(vbslq_u32(nan, vorrq_u32(upper, vdupq_n_u32(0x40)), rounded));
}

static inline float32x4_t scores4(const uint16_t *q, const uint16_t *k, size_t stride,
                                 size_t width) {
    float32x4_t sums = vdupq_n_f32(0.0f);
    for (size_t channel = 0; channel < width; channel += 4) {
        uint16x4_t a = vld1_u16(k + channel);
        uint16x4_t b = vld1_u16(k + stride + channel);
        uint16x4_t c = vld1_u16(k + 2 * stride + channel);
        uint16x4_t d = vld1_u16(k + 3 * stride + channel);
        uint32x2_t ab_even = vreinterpret_u32_u16(vtrn1_u16(a, b));
        uint32x2_t ab_odd = vreinterpret_u32_u16(vtrn2_u16(a, b));
        uint32x2_t cd_even = vreinterpret_u32_u16(vtrn1_u16(c, d));
        uint32x2_t cd_odd = vreinterpret_u32_u16(vtrn2_u16(c, d));
        float32x4_t x = widen(vld1_u16(q + channel));
        /* Each lane accumulates one QK dot product in channel order. */
        sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn1_u32(ab_even, cd_even))), x, 0);
        sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn1_u32(ab_odd, cd_odd))), x, 1);
        sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn2_u32(ab_even, cd_even))), x, 2);
        sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn2_u32(ab_odd, cd_odd))), x, 3);
    }
    return sums;
}

static TrillimStatus compute_scores(const Task *task, const uint16_t *q, const uint16_t *k,
                                     size_t begin, size_t end, float scale, float *scores) {
    const Parameters *p = task->p;
    size_t key = 0;
    for (; key < begin; ++key) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    for (; end - key >= 4; key += 4) {
        if (((key - begin) & 31) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t sums = scores4(q, k + key * p->k_strides[1], p->k_strides[1], p->key_dim);
        vst1q_f32(scores + key, vmulq_n_f32(widen(narrow(sums)), scale));
    }
    for (; key < end; ++key) {
        const uint16_t *row = k + key * p->k_strides[1];
        float sum = 0.0f;
        for (size_t channel = 0; channel < p->key_dim; ++channel) {
            sum = fmaf(trillim_bf16(q[channel]), trillim_bf16(row[channel]), sum);
        }
        scores[key] = trillim_bf16(trillim_to_bf16(sum)) * scale;
    }
    for (; key < p->length; ++key) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    return TRILLIM_OK;
}

static float32x4_t exp4(float32x4_t x) {
    if (vminvq_u32(vandq_u32(vcgeq_f32(x, vdupq_n_f32(-87.0f)),
                              vcleq_f32(x, vdupq_n_f32(0.0f)))) != UINT32_MAX) {
        if (vminvq_u32(vceqq_f32(x, vdupq_n_f32(-INFINITY))) == UINT32_MAX) {
            return vdupq_n_f32(0.0f);
        }
        float lanes[4];
        vst1q_f32(lanes, x);
        for (size_t lane = 0; lane < 4; ++lane) {
            lanes[lane] = lanes[lane] == -INFINITY ? 0.0f : expf(lanes[lane]);
        }
        return vld1q_f32(lanes);
    }
    /* Match the FP32 softmax kernel's range reduction and polynomial. */
    int32x4_t power = vcvtnq_s32_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t integer = vcvtq_f32_s32(power);
    float32x4_t r = vfmaq_n_f32(x, integer, -0.693145751953125f);
    r = vfmaq_n_f32(r, integer, -1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t exponent = vshlq_n_s32(vaddq_s32(power, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

static TrillimStatus normalize(const Task *task, float *scores, float sink) {
    size_t length = task->p->length;
    float maximum = -INFINITY;
    float32x4_t maximum4 = vdupq_n_f32(-INFINITY);
    bool precise = false;
    size_t key = 0;
    for (; length - key >= 4; key += 4) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t value = vld1q_f32(scores + key);
        if (vminvq_u32(vceqq_f32(value, value)) != UINT32_MAX) {
            precise = true;
            break;
        }
        maximum4 = vmaxnmq_f32(maximum4, value);
    }
    if (!precise) {
        maximum = vmaxnmvq_f32(maximum4);
        for (; key < length; ++key) {
            precise |= isnan(scores[key]);
            maximum = maximum >= scores[key] ? maximum : scores[key];
        }
    }
    if (precise) {
        maximum = -INFINITY;
        for (key = 0; key < length; ++key) {
            if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            maximum = maximum >= scores[key] ? maximum : scores[key];
        }
    }
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    key = 0;
    for (; !precise && length - key >= 4; key += 4) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t numerator = exp4(vsubq_f32(vld1q_f32(scores + key), vdupq_n_f32(maximum)));
        vst1q_f32(scores + key, numerator);
        sum4 = vaddq_f32(sum4, numerator);
    }
    float sink_numerator = expf(sink - maximum);
    float sum = precise ? 0.0f : vaddvq_f32(sum4) + sink_numerator;
    for (; key < length; ++key) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = expf(scores[key] - maximum);
        sum += scores[key];
    }
    if (precise) {
        sum += sink_numerator;
    }
    float denominator = sum > 0.0f ? sum : 1.0f;
    key = 0;
    for (; length - key >= 4; key += 4) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        vst1q_f32(scores + key, vdivq_f32(vld1q_f32(scores + key), vdupq_n_f32(denominator)));
    }
    for (; key < length; ++key) {
        scores[key] /= denominator;
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_values(const Task *task, const uint16_t *v, const float *probabilities,
                                   uint16_t *output) {
    const Parameters *p = task->p;
    for (size_t channel = 0; channel < p->value_dim; channel += 16) {
        float32x4_t a = vdupq_n_f32(0.0f), b = a, c = a, d = a;
        for (size_t key = 0; key < p->length; ++key) {
            if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            const uint16_t *value = v + key * p->v_strides[1] + channel;
            float weight = probabilities[key];
            a = vfmaq_n_f32(a, widen(vld1_u16(value)), weight);
            b = vfmaq_n_f32(b, widen(vld1_u16(value + 4)), weight);
            c = vfmaq_n_f32(c, widen(vld1_u16(value + 8)), weight);
            d = vfmaq_n_f32(d, widen(vld1_u16(value + 12)), weight);
        }
        vst1_u16(output + channel, narrow(a));
        vst1_u16(output + channel + 4, narrow(b));
        vst1_u16(output + channel + 8, narrow(c));
        vst1_u16(output + channel + 12, narrow(d));
    }
    return TRILLIM_OK;
}

static TrillimStatus pack_values(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->length / PACK_ROWS + (p->length % PACK_ROWS != 0);
    size_t bands = p->value_dim / BLAS_COLUMNS;
    size_t band = index / tiles;
    size_t first = index % tiles * PACK_ROWS;
    size_t last = p->length - first < PACK_ROWS ? p->length : first + PACK_ROWS;
    const uint16_t *v = task->v + (band / bands) * p->v_strides[0] + (band % bands) * BLAS_COLUMNS;
    float *packed = task->packed_values + band * p->length * BLAS_COLUMNS;
    for (size_t key = first; key < last; ++key) {
        if (((key - first) & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint16_t *value = v + key * p->v_strides[1];
        for (size_t channel = 0; channel < BLAS_COLUMNS; channel += 4) {
            vst1q_f32(packed + key * BLAS_COLUMNS + channel, widen(vld1_u16(value + channel)));
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_blas_values(const Task *task, size_t kv_head, const float *probabilities,
                                       uint16_t *output) {
    const Parameters *p = task->p;
    const float *packed = task->packed_values + kv_head * p->length * p->value_dim;
    for (size_t channel = 0; channel < p->value_dim; channel += BLAS_COLUMNS) {
        _Alignas(64) float values[BLAS_COLUMNS];
        /* Use the decomposed PV Matmul's depth blocks and accumulation order. */
        for (size_t first = 0; first < p->length; first += BLAS_DEPTH) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int depth = (int)(p->length - first < BLAS_DEPTH ? p->length - first : BLAS_DEPTH);
            sgemm(101, 111, 111, 1, BLAS_COLUMNS, depth, 1.0f,
                  probabilities + first, depth, packed + first * BLAS_COLUMNS,
                  BLAS_COLUMNS, first == 0 ? 0.0f : 1.0f, values, BLAS_COLUMNS);
        }
        for (size_t lane = 0; lane < BLAS_COLUMNS; lane += 4) {
            vst1_u16(output + channel + lane, narrow(vld1q_f32(values + lane)));
        }
        packed += p->length * BLAS_COLUMNS;
    }
    return TRILLIM_OK;
}

static TrillimStatus attention_rows(const Task *task, size_t index) {
    const Parameters *p = task->p;
    float *scores = p->length ? task->scratch + index * p->length : NULL;
    size_t group = p->heads / p->kv_heads;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t tile = index; tile < p->heads * task->tiles; tile += task->tasks) {
        size_t head = tile / task->tiles;
        size_t kv_head = head / group;
        size_t first = tile % task->tiles * ROWS_PER_TASK;
        size_t last = p->queries - first < ROWS_PER_TASK ? p->queries : first + ROWS_PER_TASK;
        float sink = trillim_load(task->sinks, head, p->sink_dtype).f;
        for (size_t row = first; row < last; ++row) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            uint16_t *output = task->output + (row * p->heads + head) * p->value_dim;
            if (p->length == 0) {
                memset(output, 0, p->value_dim * sizeof(*output));
                continue;
            }
            int64_t position = (int64_t)p->length - (int64_t)(p->queries - row);
            size_t end = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
            size_t begin = 0;
            if (p->windowed && position >= 0 && (size_t)position + 1 > p->window) {
                begin = (size_t)position + 1 - p->window;
            }
            const uint16_t *q = task->q + head * p->q_strides[0] + row * p->q_strides[1];
            const uint16_t *k = task->k + kv_head * p->k_strides[0];
            const uint16_t *v = task->v + kv_head * p->v_strides[0];
            TrillimStatus status = compute_scores(task, q, k, begin, end, scale, scores);
            if (status != TRILLIM_OK) {
                return status;
            }
            status = normalize(task, scores, sink);
            if (status != TRILLIM_OK) {
                return status;
            }
            status = task->packed_values ? apply_blas_values(task, kv_head, scores, output) :
                                          apply_values(task, v, scores, output);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus attention(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
#ifdef __APPLE__
    if (task->packed_values && get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = attention_rows(task, index);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return attention_rows(task, index);
}

TRILLIM_KERNEL TrillimStatus trillim_neon_attention_sinks_bf16(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    size_t tiles = query_tiles(p);
    size_t work = p->heads * tiles;
    size_t tasks = work < MAX_TASKS ? work : MAX_TASKS;
    Task task = {
        outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], scratch, NULL, tiles, tasks, p, context
    };
    const TrillimThreadPool *pool = context->thread_pool;
    if (blas_shape(p) && initialize_blas()) {
        task.packed_values = task.scratch + tasks * p->length;
        size_t pack_tiles = p->length / PACK_ROWS + (p->length % PACK_ROWS != 0);
        size_t packs = p->kv_heads * (p->value_dim / BLAS_COLUMNS) * pack_tiles;
        TrillimStatus status = pool->run(pool->context, packs, pack_values, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return pool->run(pool->context, tasks, attention, &task);
}

TRILLIM_KERNEL const char *trillim_neon_attention_sinks_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bf16_attention, _contiguous_strides, _product, _DTYPE_IDS\n"
           "physical_inputs = ('q', 'k', 'v', 'sinks')\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, sinks: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype, sinks=sinks),)\n"
           "def bind(context):\n"
           "    if context.outputs[0].dtype != 'bf16' or context.attrs.get('softcap') is not None:\n"
           "        raise ValueError('this attention kernel requires BF16 output and uncapped scores')\n"
           "    if context.inputs['v'].shape[1] != context.inputs['k'].shape[1]:\n"
           "        raise ValueError('fused attention requires matching key and value geometry')\n"
           "    result = _bf16_attention(context)\n"
           "    sinks = context.inputs['sinks']\n"
           "    if sinks.dtype not in {'fp16', 'bf16', 'fp32'} or _product(sinks.shape) != context.inputs['q'].shape[1]:\n"
           "        raise ValueError('attention sinks require one floating scalar per query head')\n"
           "    if sinks.strides is not None and sinks.strides != _contiguous_strides(sinks.shape):\n"
           "        raise ValueError('attention sinks require contiguous storage')\n"
           "    result['sink_dtype'] = _DTYPE_IDS[sinks.dtype]\n"
           "    return result\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_attention_sinks_bf16_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(Parameters, heads), NULL},
        {"kv_heads", "size_t", offsetof(Parameters, kv_heads), NULL},
        {"queries", "size_t", offsetof(Parameters, queries), NULL},
        {"length", "size_t", offsetof(Parameters, length), NULL},
        {"key_dim", "size_t", offsetof(Parameters, key_dim), NULL},
        {"value_dim", "size_t", offsetof(Parameters, value_dim), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"q_strides", "size_t", offsetof(Parameters, q_strides), "rank"},
        {"k_strides", "size_t", offsetof(Parameters, k_strides), "rank"},
        {"v_strides", "size_t", offsetof(Parameters, v_strides), "rank"},
        {"q_rows", "size_t", offsetof(Parameters, q_rows), NULL},
        {"q_columns", "size_t", offsetof(Parameters, q_columns), NULL},
        {"k_rows", "size_t", offsetof(Parameters, k_rows), NULL},
        {"k_columns", "size_t", offsetof(Parameters, k_columns), NULL},
        {"v_rows", "size_t", offsetof(Parameters, v_rows), NULL},
        {"v_columns", "size_t", offsetof(Parameters, v_columns), NULL},
        {"window", "size_t", offsetof(Parameters, window), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"q_dtype", "uint32_t", offsetof(Parameters, q_dtype), NULL},
        {"k_dtype", "uint32_t", offsetof(Parameters, k_dtype), NULL},
        {"v_dtype", "uint32_t", offsetof(Parameters, v_dtype), NULL},
        {"causal", "uint32_t", offsetof(Parameters, causal), NULL},
        {"windowed", "uint32_t", offsetof(Parameters, windowed), NULL},
        {"default_scale", "uint32_t", offsetof(Parameters, default_scale), NULL},
        {"scale", "float", offsetof(Parameters, scale), NULL},
        {"sink_dtype", "uint32_t", offsetof(Parameters, sink_dtype), NULL},
    };
    static const size_t inputs[] = {2, 2, 2, 1}, outputs[] = {2};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters),
        sizeof(fields) / sizeof(fields[0]), fields, 4, 1,
        requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_attention_sinks_bf16_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &capabilities;
}
