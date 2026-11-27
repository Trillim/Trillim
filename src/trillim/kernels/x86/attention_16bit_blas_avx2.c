#include "common.h"
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

enum { QUERY_TILE = 64, KEY_TILE = 128, MAX_TASKS = 32, MAX_CHANNELS = 256 };
#define SIMD __attribute__((target("avx2")))

typedef struct {
    size_t heads;
    size_t kv_heads;
    size_t queries;
    size_t length;
    size_t key_dim;
    size_t value_dim;
    size_t rank;
    const size_t *q_strides;
    const size_t *k_strides;
    const size_t *v_strides;
    size_t q_rows;
    size_t q_columns;
    size_t k_rows;
    size_t k_columns;
    size_t v_rows;
    size_t v_columns;
    size_t window;
    uint32_t output_dtype;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t causal;
    uint32_t windowed;
    uint32_t default_scale;
    float scale;
} TrillimAttentionParameters;

typedef struct {
    void *output;
    const void *query;
    const void *keys;
    const void *values;
    float *scratch;
    const TrillimAttentionParameters *parameters;
    const TrillimKernelContext *context;
    bool use_blas;
} TrillimAttentionTask;

static inline bool trillim_attention_span(const size_t *shape, const size_t *strides, size_t rows, size_t columns) {
    size_t span;
    return (columns == 0 || rows <= SIZE_MAX / columns) && trillim_span(3, shape, strides, &span) &&
           span <= rows * columns;
}

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);

static Sgemm sgemm;
static atomic_uint blas_state;
#ifdef __APPLE__
static unsigned (*get_threading)(void);
static int (*set_threading)(unsigned);
#endif

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
    static const char *const names[] = {"libopenblas.so.0", "libblas.so.3", "libblas.so"};
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

static bool initialize_blas(const TrillimKernelContext *context) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
        unsigned expected = 0;
        if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                    memory_order_acq_rel, memory_order_acquire)) {
            load_blas();
            atomic_store_explicit(&blas_state, 2, memory_order_release);
        } else {
            while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return false;
                }
                atomic_signal_fence(memory_order_acquire);
            }
        }
    }
    return sgemm != NULL;
}

static bool payload(size_t rows, size_t columns, size_t *bytes) {
    if (columns && rows > SIZE_MAX / 2 / columns) return false;
    *bytes = rows * columns * 2;
    return true;
}


static size_t query_tiles(const TrillimAttentionParameters *p) {
    return p->queries / QUERY_TILE + (p->queries % QUERY_TILE != 0);
}

static size_t workspace_stride(const TrillimAttentionParameters *p) {
    size_t channels = p->key_dim > p->value_dim ? p->key_dim : p->value_dim;
    return QUERY_TILE * (p->key_dim + p->value_dim + KEY_TILE) + KEY_TILE * channels;
}

static TrillimStatus trillim_attention_requirements(const void *parameters, size_t *inputs, size_t *outputs,
                                                    size_t *scratch) {
    const TrillimAttentionParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    if (p->rank != 3 || !p->kv_heads || p->heads % p->kv_heads ||
        !p->key_dim || p->key_dim > MAX_CHANNELS || !p->value_dim || p->value_dim > MAX_CHANNELS ||
        (p->q_dtype != TRILLIM_FP16 && p->q_dtype != TRILLIM_BF16) ||
        p->k_dtype != p->q_dtype || p->v_dtype != p->q_dtype || p->output_dtype != p->q_dtype ||
        p->q_strides[2] != 1 || p->k_strides[2] != 1 || p->v_strides[2] != 1 || p->causal > 1 ||
        p->windowed > 1 || p->default_scale > 1 || p->heads > SIZE_MAX / p->queries ||
        p->heads * p->queries > SIZE_MAX / sizeof(uint16_t) / p->value_dim) {
        return TRILLIM_ERROR;
    }
    size_t qshape[] = {p->heads, p->queries, p->key_dim};
    size_t kshape[] = {p->kv_heads, p->length, p->key_dim};
    size_t vshape[] = {p->kv_heads, p->length, p->value_dim};
    if (!trillim_attention_span(qshape, p->q_strides, p->q_rows, p->q_columns) ||
        !trillim_attention_span(kshape, p->k_strides, p->k_rows, p->k_columns) ||
        !trillim_attention_span(vshape, p->v_strides, p->v_rows, p->v_columns) ||
        !payload(p->q_rows, p->q_columns, &inputs[0]) || !payload(p->k_rows, p->k_columns, &inputs[1]) ||
        !payload(p->v_rows, p->v_columns, &inputs[2])) {
        return TRILLIM_ERROR;
    }
    size_t tasks = p->heads * query_tiles(p);
    tasks = tasks < MAX_TASKS ? tasks : MAX_TASKS;
    *scratch = tasks * workspace_stride(p) * sizeof(float);
    outputs[0] = p->heads * p->queries * p->value_dim * sizeof(uint16_t);
    return TRILLIM_OK;
}

SIMD static inline __attribute__((always_inline)) __m256 load8(const uint16_t *data, uint32_t dtype) {
    __m128i raw = _mm_loadu_si128((const __m128i *)data);
    __m256i h = _mm256_cvtepu16_epi32(raw);
    if (dtype == TRILLIM_BF16) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(h, 16));
    }
    __m256i fraction = _mm256_and_si256(h, _mm256_set1_epi32(1023));
    __m256i exponent = _mm256_and_si256(h, _mm256_set1_epi32(0x7c00));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
    __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x7fff)), 13),
                                      _mm256_set1_epi32(112 << 23));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
    normal = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000), _mm256_slli_epi32(fraction, 13));
    normal = _mm256_blendv_epi8(normal, special, _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00)));
    return _mm256_castsi256_ps(_mm256_or_si256(normal, sign));
}

SIMD static bool pack_rows(float *dst, const uint16_t *src, size_t rows, size_t width,
                            size_t stride, uint32_t dtype) {
    __m256 limit = _mm256_set1_ps(0x1p32f);
    __m256 sign = _mm256_set1_ps(-0.0f);
    bool ordinary = true;
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *source = src + row * stride;
        size_t channel = 0;
        for (; width - channel >= 8; channel += 8) {
            __m256 value = load8(source + channel, dtype);
            __m256 magnitude = _mm256_andnot_ps(sign, value);
            ordinary &= _mm256_movemask_ps(_mm256_cmp_ps(magnitude, limit, _CMP_LT_OQ)) == 0xff;
            _mm256_storeu_ps(dst + channel, value);
        }
        for (; channel < width; ++channel) {
            float value = trillim_load(source, channel, dtype).f;
            ordinary &= fabsf(value) < 0x1p32f;
            dst[channel] = value;
        }
        dst += width;
    }
    return ordinary;
}

SIMD static inline float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum,sum));
}
SIMD static inline __m256 exp_negative(__m256 original) {
    __m256 x = _mm256_max_ps(_mm256_set1_ps(-104.0f), _mm256_min_ps(original, _mm256_set1_ps(0.0f)));
    __m256 exponent = _mm256_floor_ps(_mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f))));
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(exponent, _mm256_set1_ps(0.693359375f)));
    r = _mm256_add_ps(r, _mm256_mul_ps(exponent, _mm256_set1_ps(0.00021219444005469058f)));
    /* Range reduction keeps |r| <= ln(2)/2 for the degree-seven polynomial. */
    __m256 value = _mm256_set1_ps(1.0f / 5040.0f);
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 720.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 120.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 24.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f / 6.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(value, r));
    value = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(value, r));
    __m256i n = _mm256_cvttps_epi32(exponent);
    __m256i low = _mm256_min_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(126)), _mm256_set1_epi32(0));
    __m256i high = _mm256_sub_epi32(n, low);
    __m256 main_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(high, _mm256_set1_epi32(127)), 23));
    __m256 tail_scale = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(low, _mm256_set1_epi32(127)), 23));
    value = _mm256_mul_ps(_mm256_mul_ps(value, main_scale), tail_scale);
    return _mm256_blendv_ps(original, value, _mm256_cmp_ps(original, original, _CMP_EQ_OQ));
}
SIMD static float normalize_block(float *scores, size_t count, size_t from, size_t to, float peak) {
    size_t key = 0;
    for (; key < from; ++key) {
        scores[key] = 0.0f;
    }
    __m256 weights = _mm256_setzero_ps();
    for (; to - key >= 8; key += 8) {
        __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(scores + key), _mm256_set1_ps(peak));
        __m256 value = exp_negative(shifted);
        _mm256_storeu_ps(scores + key, value);
        weights = _mm256_add_ps(weights, value);
    }
    float total = sum8(weights);
    for (; key < to; ++key) {
        scores[key] = expf(scores[key] - peak);
        total += scores[key];
    }
    for (; key < count; ++key) {
        scores[key] = 0.0f;
    }
    return total;
}

static TrillimStatus prefill_tile(const TrillimAttentionTask *task, size_t index, size_t slot) {
    const TrillimAttentionParameters *p = task->parameters;
    const TrillimKernelContext *context = task->context;
    size_t tiles = query_tiles(p);
    size_t head = index / tiles, kv_head = head / (p->heads / p->kv_heads);
    size_t first = index % tiles * QUERY_TILE;
    size_t rows = p->queries - first < QUERY_TILE ? p->queries - first : QUERY_TILE;
    float *query = task->scratch + slot * workspace_stride(p);
    float *kv = query + QUERY_TILE * p->key_dim;
    float *scores = kv + KEY_TILE * (p->key_dim > p->value_dim ? p->key_dim : p->value_dim);
    float *accum = scores + QUERY_TILE * KEY_TILE;
    float peak[QUERY_TILE], sum[QUERY_TILE];
    size_t begin[QUERY_TILE], end[QUERY_TILE], first_key = p->length, last_key = 0;
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t distance = p->queries - first - row;
        size_t position_end = distance > p->length ? 0 : p->length - distance + 1;
        begin[row] = p->windowed && position_end > p->window ? position_end - p->window : 0;
        end[row] = p->causal ? position_end : p->length;
        if (begin[row] < first_key) first_key = begin[row];
        if (end[row] > last_key) last_key = end[row];
        peak[row] = -INFINITY;
        sum[row] = 0.0f;
        memset(accum + row * p->value_dim, 0, p->value_dim * sizeof(float));
    }
    if (first_key < last_key) {
        const uint16_t *q = (const uint16_t *)task->query + head * p->q_strides[0] + first * p->q_strides[1];
        const uint16_t *k = (const uint16_t *)task->keys + kv_head * p->k_strides[0];
        const uint16_t *v = (const uint16_t *)task->values + kv_head * p->v_strides[0];
        float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
        bool ordinary_query = pack_rows(query, q, rows, p->key_dim, p->q_strides[1], p->q_dtype) &&
            isfinite(scale) && fabsf(scale) <= 1.0f;
        for (size_t block = first_key; block < last_key;) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = last_key - block < KEY_TILE ? last_key - block : KEY_TILE;
            bool ordinary_key = pack_rows(kv, k + block * p->k_strides[1], count, p->key_dim,
                                          p->k_strides[1], p->k_dtype);
            if (task->use_blas && ordinary_query && ordinary_key) {
                sgemm(101, 111, 112, (int)rows, (int)count, (int)p->key_dim, scale,
                      query, (int)p->key_dim, kv, (int)p->key_dim, 0.0f, scores, KEY_TILE);
            } else {
                for (size_t row = 0; row < rows; ++row) {
                    if (context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    for (size_t key = 0; key < count; ++key) {
                        float dot = 0.0f;
                        for (size_t c = 0; c < p->key_dim; ++c) {
                            dot += query[row * p->key_dim + c] * kv[key * p->key_dim + c];
                        }
                        scores[row * KEY_TILE + key] = dot * scale;
                    }
                }
            }
            for (size_t row = 0; row < rows; ++row) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float *probabilities = scores + row * KEY_TILE;
                float maximum = peak[row];
                size_t from = begin[row] > block ? begin[row] - block : 0;
                size_t to = end[row] > block ? end[row] - block : 0;
                if (to > count) to = count;
                if (from > to) from = to;
                for (size_t key = from; key < to; ++key) {
                    maximum = fmaxf(maximum, probabilities[key]);
                }
                float normalization_peak = maximum == -INFINITY ? 0.0f : maximum;
                float previous = peak[row] == -INFINITY ? 0.0f : expf(peak[row] - normalization_peak);
                peak[row] = maximum;
                float total = normalize_block(probabilities, count, from, to, normalization_peak);
                sum[row] = sum[row] * previous + total;
                for (size_t c = 0; c < p->value_dim; ++c) {
                    accum[row * p->value_dim + c] *= previous;
                }
            }
            bool ordinary_value = pack_rows(kv, v + block * p->v_strides[1], count, p->value_dim,
                                            p->v_strides[1], p->v_dtype);
            if (task->use_blas && ordinary_query && ordinary_key && ordinary_value) {
                if (context->is_cancelled(context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                sgemm(101, 111, 111, (int)rows, (int)p->value_dim, (int)count, 1.0f,
                      scores, KEY_TILE, kv, (int)p->value_dim, 1.0f, accum, (int)p->value_dim);
            } else {
                for (size_t row = 0; row < rows; ++row) {
                    if (context->is_cancelled(context->cancellation_context)) {
                        return TRILLIM_CANCELLED;
                    }
                    for (size_t key = 0; key < count; ++key) {
                        if (block + key >= begin[row] && block + key < end[row]) {
                            float weight = scores[row * KEY_TILE + key];
                            for (size_t c = 0; c < p->value_dim; ++c) {
                                accum[row * p->value_dim + c] += weight * kv[key * p->value_dim + c];
                            }
                        }
                    }
                }
            }
            block += count;
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float denominator = sum[row] > 0.0f ? sum[row] : 1.0f;
        size_t output = ((first + row) * p->heads + head) * p->value_dim;
        for (size_t c = 0; c < p->value_dim; ++c) {
            TrillimValue value = {.f = accum[row * p->value_dim + c] / denominator};
            trillim_store(task->output, output + c, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus prefill(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const TrillimAttentionTask *task = opaque;
    const TrillimAttentionParameters *p = task->parameters;
    size_t jobs = p->heads * query_tiles(p), slots = jobs < MAX_TASKS ? jobs : MAX_TASKS;
#ifdef __APPLE__
    unsigned previous = task->use_blas && get_threading != NULL ? get_threading() : 1;
    bool restore = set_threading != NULL && previous != 1 && set_threading(1) == 0;
#endif
    TrillimStatus status = TRILLIM_OK;
    for (size_t index = slot; index < jobs; index += slots) {
        status = prefill_tile(task, index, slot);
        if (status != TRILLIM_OK) break;
    }
#ifdef __APPLE__
    if (restore) (void)set_threading(previous);
#endif
    return status;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_attention_16bit_blas_avx2(
    void *const *outputs, const void *const *inputs, void *scratch, const void *parameters,
    const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const TrillimAttentionParameters *p = parameters;
    if (!p->heads || !p->queries) return TRILLIM_OK;
    bool use_blas = (_mm_getcsr() & 0xffc0) == 0x1f80 && initialize_blas(context);
    TrillimAttentionTask task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context, use_blas};
    size_t jobs = p->heads * query_tiles(p), slots = jobs < MAX_TASKS ? jobs : MAX_TASKS;
    return context->thread_pool->run(context->thread_pool->context, slots, prefill, &task);
}

TRILLIM_KERNEL const char *trillim_x86_attention_16bit_blas_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention, _contiguous_strides\n"
           "def bind(context):\n"
           "    q, k, v = (context.inputs[name] for name in ('q', 'k', 'v'))\n"
           "    dtype = context.outputs[0].dtype\n"
           "    if dtype not in {'bf16', 'fp16'} or any(t.dtype != dtype for t in (q, k, v)):\n"
           "        raise ValueError('16-bit attention requires one floating dtype for all tensors')\n"
           "    for t in (q, k, v):\n"
           "        if t.shape[-1].value not in range(64, 257) or (t.strides or _contiguous_strides(t.shape))[-1].value != 1:\n"
           "            raise ValueError('16-bit attention requires contiguous channels of width 64 to 256')\n"
           "    return _attention(context)\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: "
           "DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_attention_16bit_blas_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"heads", "size_t", offsetof(TrillimAttentionParameters, heads), NULL},
        {"kv_heads", "size_t", offsetof(TrillimAttentionParameters, kv_heads), NULL},
        {"queries", "size_t", offsetof(TrillimAttentionParameters, queries), NULL},
        {"length", "size_t", offsetof(TrillimAttentionParameters, length), NULL},
        {"key_dim", "size_t", offsetof(TrillimAttentionParameters, key_dim), NULL},
        {"value_dim", "size_t", offsetof(TrillimAttentionParameters, value_dim), NULL},
        {"rank", "size_t", offsetof(TrillimAttentionParameters, rank), NULL},
        {"q_strides", "size_t", offsetof(TrillimAttentionParameters, q_strides), "rank"},
        {"k_strides", "size_t", offsetof(TrillimAttentionParameters, k_strides), "rank"},
        {"v_strides", "size_t", offsetof(TrillimAttentionParameters, v_strides), "rank"},
        {"q_rows", "size_t", offsetof(TrillimAttentionParameters, q_rows), NULL},
        {"q_columns", "size_t", offsetof(TrillimAttentionParameters, q_columns), NULL},
        {"k_rows", "size_t", offsetof(TrillimAttentionParameters, k_rows), NULL},
        {"k_columns", "size_t", offsetof(TrillimAttentionParameters, k_columns), NULL},
        {"v_rows", "size_t", offsetof(TrillimAttentionParameters, v_rows), NULL},
        {"v_columns", "size_t", offsetof(TrillimAttentionParameters, v_columns), NULL},
        {"window", "size_t", offsetof(TrillimAttentionParameters, window), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimAttentionParameters, output_dtype), NULL},
        {"q_dtype", "uint32_t", offsetof(TrillimAttentionParameters, q_dtype), NULL},
        {"k_dtype", "uint32_t", offsetof(TrillimAttentionParameters, k_dtype), NULL},
        {"v_dtype", "uint32_t", offsetof(TrillimAttentionParameters, v_dtype), NULL},
        {"causal", "uint32_t", offsetof(TrillimAttentionParameters, causal), NULL},
        {"windowed", "uint32_t", offsetof(TrillimAttentionParameters, windowed), NULL},
        {"default_scale", "uint32_t", offsetof(TrillimAttentionParameters, default_scale), NULL},
        {"scale", "float", offsetof(TrillimAttentionParameters, scale), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(TrillimAttentionParameters),
                                                  TRILLIM_ALIGNOF(TrillimAttentionParameters),
                                                  sizeof(fields) / sizeof(fields[0]),
                                                  fields,
                                                  3,
                                                  1,
                                                  trillim_attention_requirements,
                                                  NULL,
                                                  inputs,
                                                  outputs,
                                                  4};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_attention_16bit_blas_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM | TRILLIM_CAP_BLAS};
    return &capabilities;
}
