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
    GROUPED_KEYS = 512,
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
    atomic_bool *finite_values;
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
    static const char *const names[] = {"libopenblas.dll", "openblas.dll"};
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
    static const char *const names[] = {"libopenblas.so.0", "libopenblas.so"};
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

static inline __attribute__((always_inline)) float32x4_t score_channel4(
    float32x4_t sums, float32x4_t q, const uint16_t *k, size_t stride
) {
    uint16x4_t a = vld1_u16(k), b = vld1_u16(k + stride);
    uint16x4_t c = vld1_u16(k + 2 * stride), d = vld1_u16(k + 3 * stride);
    uint32x2_t ab_even = vreinterpret_u32_u16(vtrn1_u16(a, b));
    uint32x2_t ab_odd = vreinterpret_u32_u16(vtrn2_u16(a, b));
    uint32x2_t cd_even = vreinterpret_u32_u16(vtrn1_u16(c, d));
    uint32x2_t cd_odd = vreinterpret_u32_u16(vtrn2_u16(c, d));
    sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn1_u32(ab_even, cd_even))), q, 0);
    sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn1_u32(ab_odd, cd_odd))), q, 1);
    sums = vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn2_u32(ab_even, cd_even))), q, 2);
    return vfmaq_laneq_f32(sums, widen(vreinterpret_u16_u32(vtrn2_u32(ab_odd, cd_odd))), q, 3);
}

static inline float32x4_t scores4(const uint16_t *q, const uint16_t *k, size_t stride, size_t width) {
    float32x4_t sums = vdupq_n_f32(0);
    for (size_t channel = 0; channel < width; channel += 4) {
        sums = score_channel4(sums, widen(vld1_u16(q + channel)), k + channel, stride);
    }
    return sums;
}

static inline void scores16(const uint16_t *q, const uint16_t *k, size_t stride,
                            size_t width, float scale, float *output) {
    const uint16_t *k1 = k + 4 * stride, *k2 = k + 8 * stride, *k3 = k + 12 * stride;
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    for (size_t channel = 0; channel < width; channel += 4) {
        float32x4_t query = widen(vld1_u16(q + channel));
        a = score_channel4(a, query, k + channel, stride);
        b = score_channel4(b, query, k1 + channel, stride);
        c = score_channel4(c, query, k2 + channel, stride);
        d = score_channel4(d, query, k3 + channel, stride);
    }
    vst1q_f32(output, vmulq_n_f32(widen(narrow(a)), scale));
    vst1q_f32(output + 4, vmulq_n_f32(widen(narrow(b)), scale));
    vst1q_f32(output + 8, vmulq_n_f32(widen(narrow(c)), scale));
    vst1q_f32(output + 12, vmulq_n_f32(widen(narrow(d)), scale));
}

static TrillimStatus compute_scores(const Task *task, const uint16_t *q, const uint16_t *k,
                                     size_t begin, size_t end, float scale, float *scores) {
    const Parameters *p = task->p;
    size_t key = begin & ~(size_t)3;
    for (; key < begin; ++key) {
        scores[key] = -INFINITY;
    }
    for (; end > key && end - key >= 16; key += 16) {
        if (((key - begin) & 31) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores16(q, k + key * p->k_strides[1], p->k_strides[1], p->key_dim, scale, scores + key);
    }
    for (; end > key && end - key >= 4; key += 4) {
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
    size_t boundary = (end + 3) & ~(size_t)3;
    boundary = boundary < p->length ? boundary : p->length;
    for (; key < boundary; ++key) {
        scores[key] = -INFINITY;
    }
    size_t tail = p->length & ~(size_t)3;
    for (key = key > tail ? key : tail; key < p->length; ++key) {
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

static TrillimStatus fill_mask(const Task *task, float *scores, size_t begin, size_t end) {
    for (size_t key = 0; key < begin; ++key) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    for (size_t key = end; key < task->p->length; ++key) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    return TRILLIM_OK;
}

static TrillimStatus clear_scores(const Task *task, float *scores, size_t count) {
    for (size_t first = 0; first < count; first += 256) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t width = count - first < 256 ? count - first : 256;
        memset(scores + first, 0, width * sizeof(float));
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize(const Task *task, float *scores, float sink, size_t begin, size_t end) {
    size_t length = task->p->length;
    bool full_scan = isnan(sink);
    if (full_scan) {
        TrillimStatus status = fill_mask(task, scores, begin, end);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    /* Keep the original four-lane sum order across the mask boundaries. */
    size_t full_vectors = length & ~(size_t)3;
    size_t first_vector = full_scan ? 0 : begin & ~(size_t)3;
    size_t last_vector = full_scan ? full_vectors : (end + 3) & ~(size_t)3;
    last_vector = last_vector < full_vectors ? last_vector : full_vectors;
    float maximum = -INFINITY;
    float32x4_t maximum4 = vdupq_n_f32(-INFINITY);
    bool precise = false;
    size_t key = first_vector;
    for (; key < last_vector; key += 4) {
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
        for (key = full_vectors; key < length; ++key) {
            precise |= isnan(scores[key]);
            maximum = maximum >= scores[key] ? maximum : scores[key];
        }
    }
    if (precise) {
        if (!full_scan) {
            TrillimStatus status = fill_mask(task, scores, begin, end);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
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
    key = first_vector;
    for (; !precise && key < last_vector; key += 4) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t numerator = exp4(vsubq_f32(vld1q_f32(scores + key), vdupq_n_f32(maximum)));
        vst1q_f32(scores + key, numerator);
        sum4 = vaddq_f32(sum4, numerator);
    }
    float sink_numerator = expf(sink - maximum);
    float sum = precise ? 0.0f : vaddvq_f32(sum4) + sink_numerator;
    key = precise ? 0 : full_vectors;
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
    if (precise) {
        first_vector = 0;
        last_vector = full_vectors;
    }
    size_t first = first_vector < last_vector ? first_vector : last_vector;
    TrillimStatus status = clear_scores(task, scores, first);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = clear_scores(task, scores + last_vector, full_vectors - last_vector);
    if (status != TRILLIM_OK) {
        return status;
    }
    key = first_vector;
    for (; key < last_vector; key += 4) {
        if ((key & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        vst1q_f32(scores + key, vdivq_f32(vld1q_f32(scores + key), vdupq_n_f32(denominator)));
    }
    for (key = full_vectors; key < length; ++key) {
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
    uint16x4_t maximum = vdup_n_u16(0);
    for (size_t key = first; key < last; ++key) {
        if (((key - first) & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint16_t *value = v + key * p->v_strides[1];
        for (size_t channel = 0; channel < BLAS_COLUMNS; channel += 4) {
            uint16x4_t bits = vld1_u16(value + channel);
            maximum = vmax_u16(maximum, vand_u16(bits, vdup_n_u16(0x7fff)));
            vst1q_f32(packed + key * BLAS_COLUMNS + channel, widen(bits));
        }
    }
    if (vmaxv_u16(maximum) >= 0x7f80) {
        atomic_store_explicit(task->finite_values, false, memory_order_relaxed);
    }
    return TRILLIM_OK;
}

static bool default_fpcr(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
}

static TrillimStatus apply_blas_values(const Task *task, size_t kv_head, size_t begin, size_t end,
                                      const float *probabilities, uint16_t *output) {
    const Parameters *p = task->p;
    bool skip = p->length > BLAS_DEPTH && default_fpcr() && atomic_load_explicit(task->finite_values, memory_order_relaxed) &&
        (!begin || probabilities[0] == 0.0f) && (end == p->length || probabilities[p->length - 1] == 0.0f);
    if (skip && begin >= end) {
        memset(output, 0, p->value_dim * sizeof(*output));
        return TRILLIM_OK;
    }
    size_t first_block = skip ? begin / BLAS_DEPTH * BLAS_DEPTH : 0;
    size_t last_key = skip ? end : p->length;
    bool skipped_suffix = skip && (end / BLAS_DEPTH + (end % BLAS_DEPTH != 0)) <
        (p->length / BLAS_DEPTH + (p->length % BLAS_DEPTH != 0));
    const float *packed = task->packed_values + kv_head * p->length * p->value_dim;
    for (size_t channel = 0; channel < p->value_dim; channel += BLAS_COLUMNS) {
        /* Accelerate can change the accumulation order at a different C alignment. */
        _Alignas(256) float values[BLAS_COLUMNS];
        if (first_block) {
            memset(values, 0, sizeof(values));
        }
        /* Keep the same BLAS depth blocks for the visible mask. */
        for (size_t first = first_block; first < last_key; first += BLAS_DEPTH) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int depth = (int)(p->length - first < BLAS_DEPTH ? p->length - first : BLAS_DEPTH);
            sgemm(101, 111, 111, 1, BLAS_COLUMNS, depth, 1.0f,
                  probabilities + first, depth, packed + first * BLAS_COLUMNS,
                  BLAS_COLUMNS, first == 0 ? 0.0f : 1.0f, values, BLAS_COLUMNS);
        }
        for (size_t lane = 0; lane < BLAS_COLUMNS; lane += 4) {
            float32x4_t value = vld1q_f32(values + lane);
            if (skipped_suffix) {
                /* The omitted all-zero term also clears a negative zero. */
                value = vaddq_f32(value, vdupq_n_f32(0));
            }
            vst1_u16(output + channel + lane, narrow(value));
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
            status = normalize(task, scores, sink, begin, end);
            if (status != TRILLIM_OK) {
                return status;
            }
            status = task->packed_values ? apply_blas_values(task, kv_head, begin, end, scores, output) :
                                          apply_values(task, v, scores, output);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus grouped_scores(const Task *task, const uint16_t *query, const uint16_t *keys,
                                    size_t begin, size_t end, float scale, float **scores) {
    const Parameters *p = task->p;
    const uint16_t *q1 = query + p->q_strides[0];
    const uint16_t *q2 = q1 + p->q_strides[0], *q3 = q2 + p->q_strides[0];
    float packed[4 * 128];
    for (size_t channel = 0; channel < p->key_dim; channel += 4) {
        float32x4x4_t data = {{
            widen(vld1_u16(query + channel)), widen(vld1_u16(q1 + channel)),
            widen(vld1_u16(q2 + channel)), widen(vld1_u16(q3 + channel))
        }};
        vst4q_f32(packed + 4 * channel, data);
    }
    uint32x4_t invalid = vdupq_n_u32(0);
    size_t key = begin;
    for (; end - key >= 4; key += 4) {
        if (((key - begin) & 31) == 0 &&
            task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const uint16_t *k0 = keys + key * p->k_strides[1];
        const uint16_t *k1 = k0 + p->k_strides[1], *k2 = k1 + p->k_strides[1], *k3 = k2 + p->k_strides[1];
        float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
        for (size_t channel = 0; channel < p->key_dim; channel += 4) {
            float32x4_t q0 = vld1q_f32(packed + channel * 4);
            float32x4_t q1 = vld1q_f32(packed + channel * 4 + 4);
            float32x4_t q2 = vld1q_f32(packed + channel * 4 + 8);
            float32x4_t q3 = vld1q_f32(packed + channel * 4 + 12);
#define UPDATE_SUM(SUM, KEY)             SUM = vfmaq_laneq_f32(SUM, q0, KEY, 0);             SUM = vfmaq_laneq_f32(SUM, q1, KEY, 1);             SUM = vfmaq_laneq_f32(SUM, q2, KEY, 2);             SUM = vfmaq_laneq_f32(SUM, q3, KEY, 3)
            UPDATE_SUM(a, widen(vld1_u16(k0 + channel)));
            UPDATE_SUM(b, widen(vld1_u16(k1 + channel)));
            UPDATE_SUM(c, widen(vld1_u16(k2 + channel)));
            UPDATE_SUM(d, widen(vld1_u16(k3 + channel)));
#undef UPDATE_SUM
        }
        uint32x4_t valid = vandq_u32(vandq_u32(vceqq_f32(a, a), vceqq_f32(b, b)),
                                            vandq_u32(vceqq_f32(c, c), vceqq_f32(d, d)));
        invalid = vorrq_u32(invalid, vmvnq_u32(valid));
        float32x4_t ab0 = vtrn1q_f32(a, b), ab1 = vtrn2q_f32(a, b);
        float32x4_t cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
        vst1q_f32(scores[0] + key, vmulq_n_f32(widen(narrow(vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)))), scale));
        vst1q_f32(scores[1] + key, vmulq_n_f32(widen(narrow(vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)))), scale));
        vst1q_f32(scores[2] + key, vmulq_n_f32(widen(narrow(vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)))), scale));
        vst1q_f32(scores[3] + key, vmulq_n_f32(widen(narrow(vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)))), scale));
    }
    if (vmaxvq_u32(invalid)) {
        return TRILLIM_ERROR;
    }
    for (size_t head = 0; head < 4; ++head) {
        TrillimStatus status = fill_mask(task, scores[head], begin, end);
        if (status != TRILLIM_OK) {
            return status;
        }
        const uint16_t *q = query + head * p->q_strides[0];
        size_t tail = key;
        for (; tail < end; ++tail) {
            float sum = 0;
            for (size_t channel = 0; channel < p->key_dim; ++channel) {
                sum = fmaf(trillim_bf16(q[channel]), trillim_bf16(keys[tail * p->k_strides[1] + channel]), sum);
            }
            if (isnan(sum)) {
                return TRILLIM_ERROR;
            }
            scores[head][tail] = trillim_bf16(trillim_to_bf16(sum)) * scale;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus grouped_values(const Task *task, const uint16_t *values, float *const scores[4],
                                    uint16_t *output) {
    const Parameters *p = task->p;
    /* NaN payload choice can depend on how the compiler orders FMA operands. */
    uint32x4_t magnitude = vdupq_n_u32(0);
    const uint32x4_t absolute = vdupq_n_u32(0x7fffffff);
    bool ordinary = default_fpcr();
    for (size_t head = 0; head < 4; ++head) {
        size_t key = 0;
        for (; p->length - key >= 4; key += 4) {
            uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(scores[head] + key));
            magnitude = vmaxq_u32(magnitude, vandq_u32(bits, absolute));
        }
        for (; key < p->length; ++key) {
            ordinary &= isfinite(scores[head][key]);
        }
    }
    if (!ordinary || vmaxvq_u32(magnitude) >= UINT32_C(0x7f800000)) {
        for (size_t head = 0; head < 4; ++head) {
            TrillimStatus status = apply_values(task, values, scores[head], output + head * p->value_dim);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    for (size_t channel = 0; channel < p->value_dim; channel += 8) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (size_t key = 0; key < p->length; ++key) {
            const uint16_t *row = values + key * p->v_strides[1] + channel;
            uint16x8_t bits = vld1q_u16(row);
            float32x4_t a = widen(vget_low_u16(bits));
            float32x4_t b = widen(vget_high_u16(bits));
            a0 = vfmaq_n_f32(a0, a, scores[0][key]);
            b0 = vfmaq_n_f32(b0, b, scores[0][key]);
            a1 = vfmaq_n_f32(a1, a, scores[1][key]);
            b1 = vfmaq_n_f32(b1, b, scores[1][key]);
            a2 = vfmaq_n_f32(a2, a, scores[2][key]);
            b2 = vfmaq_n_f32(b2, b, scores[2][key]);
            a3 = vfmaq_n_f32(a3, a, scores[3][key]);
            b3 = vfmaq_n_f32(b3, b, scores[3][key]);
        }
        uint16_t *row = output + channel;
        vst1_u16(row, narrow(a0));
        vst1_u16(row + 4, narrow(b0));
        row += p->value_dim;
        vst1_u16(row, narrow(a1));
        vst1_u16(row + 4, narrow(b1));
        row += p->value_dim;
        vst1_u16(row, narrow(a2));
        vst1_u16(row + 4, narrow(b2));
        row += p->value_dim;
        vst1_u16(row, narrow(a3));
        vst1_u16(row + 4, narrow(b3));
    }
    return TRILLIM_OK;
}

static TrillimStatus grouped_attention_rows(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    float storage[4][GROUPED_KEYS];
    float *scores[4] = {storage[0], storage[1], storage[2], storage[3]};
    bool ordinary = default_fpcr();
    size_t group = p->heads / p->kv_heads;
    size_t work = p->heads / 4 * task->tiles;
    size_t slots = work < MAX_TASKS ? work : MAX_TASKS;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t tile = index; tile < work; tile += slots) {
        size_t first_head = (tile / task->tiles) * 4;
        size_t kv_head = first_head / group;
        size_t first = tile % task->tiles * ROWS_PER_TASK;
        size_t last = p->queries - first < ROWS_PER_TASK ? p->queries : first + ROWS_PER_TASK;
        const uint16_t *k = task->k + kv_head * p->k_strides[0];
        const uint16_t *v = task->v + kv_head * p->v_strides[0];
        for (size_t row = first; row < last; ++row) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            int64_t position = (int64_t)p->length - (int64_t)(p->queries - row);
            size_t end = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
            size_t begin = p->windowed && position >= 0 && (size_t)position + 1 > p->window ?
                (size_t)position + 1 - p->window : 0;
            const uint16_t *q = task->q + first_head * p->q_strides[0] + row * p->q_strides[1];
            TrillimStatus grouped = ordinary ? grouped_scores(task, q, k, begin, end, scale, scores) : TRILLIM_ERROR;
            if (grouped == TRILLIM_CANCELLED) {
                return grouped;
            }
            for (size_t head = 0; head < 4; ++head) {
                TrillimStatus status;
                if (grouped == TRILLIM_ERROR) {
                    status = compute_scores(task, q + head * p->q_strides[0], k, begin, end, scale, scores[head]);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                float sink = trillim_load(task->sinks, first_head + head, p->sink_dtype).f;
                status = normalize(task, scores[head], sink, begin, end);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            uint16_t *out = task->output + (row * p->heads + first_head) * p->value_dim;
            TrillimStatus status = grouped_values(task, v, scores, out);
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
    atomic_bool finite_values = ATOMIC_VAR_INIT(true);
    Task task = {
        outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], scratch, NULL, &finite_values, tiles, tasks, p, context
    };
    const TrillimThreadPool *pool = context->thread_pool;
    size_t visible = p->windowed && p->window < p->length ? p->window : p->length;
    if (visible >= 32 && p->length <= GROUPED_KEYS && !blas_shape(p) &&
        (p->heads / p->kv_heads) % 4 == 0) {
        size_t groups = p->heads / 4 * tiles;
        size_t slots = groups < MAX_TASKS ? groups : MAX_TASKS;
        return pool->run(pool->context, slots, grouped_attention_rows, &task);
    }
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
