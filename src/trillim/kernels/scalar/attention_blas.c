#include "common.h"
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define QUERIES 128
#define KEYS 256
#define CHANNELS 256
#define WORKSPACE (QUERIES * (KEYS + CHANNELS))

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
#ifdef __APPLE__
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
typedef void (*VectorExp)(float *, const float *, const int *);
static GetThreading get_threading;
static SetThreading set_threading;
static VectorExp vector_exp;
#endif
static atomic_uint blas_state;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1, memory_order_acq_rel, memory_order_acquire)) {
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#elif defined(_WIN32)
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
#else
        static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
#endif
        for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
#ifdef _WIN32
            HMODULE library = LoadLibraryA(names[index]);
            FARPROC address = library ? GetProcAddress(library, "cblas_sgemm") : NULL;
#else
            void *library = dlopen(names[index], RTLD_LOCAL | RTLD_LAZY);
            void *address = library ? dlsym(library, "cblas_sgemm") : NULL;
#endif
            if (!library) {
                continue;
            }
#ifdef __APPLE__
            void *modern = dlsym(library, "cblas_sgemm$NEWLAPACK");
            if (modern) {
                address = modern;
            }
#endif
            if (address) {
                _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                void *getter = dlsym(library, "BLASGetThreading");
                void *setter = dlsym(library, "BLASSetThreading");
                if (getter && setter) {
                    memcpy(&get_threading, &getter, sizeof(get_threading));
                    memcpy(&set_threading, &setter, sizeof(set_threading));
                    void *exponential = dlsym(library, "vvexpf");
                    if (exponential) {
                        memcpy(&vector_exp, &exponential, sizeof(vector_exp));
                    }
                } else {
                    sgemm = NULL;
                }
#endif
                if (sgemm) {
                    break;
                }
            }
#ifdef _WIN32
            FreeLibrary(library);
#else
            dlclose(library);
#endif
        }
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

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
} Parameters;

typedef struct {
    float *output;
    const float *query;
    const float *keys;
    const float *values;
    float *scratch;
    const Parameters *p;
    const TrillimKernelContext *context;
    atomic_bool *ordinary;
} Task;

static bool payload(size_t rows, size_t columns, const size_t *shape,
                     const size_t *strides, size_t *bytes) {
    if (columns && rows > SIZE_MAX / sizeof(float) / columns) {
        return false;
    }
    size_t count = rows * columns, span;
    if (!trillim_span(3, shape, strides, &span) || span > count) {
        return false;
    }
    *bytes = count * sizeof(float);
    return true;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (p->output_dtype != TRILLIM_FP32 || p->q_dtype != TRILLIM_FP32 ||
        p->k_dtype != TRILLIM_FP32 || p->v_dtype != TRILLIM_FP32 || p->rank != 3 ||
        !p->kv_heads || p->heads % p->kv_heads || p->causal > 1 || p->windowed > 1 || p->default_scale > 1) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    if (p->queries > INT64_MAX || p->length > INT64_MAX ||
        p->key_dim < 64 || p->key_dim > CHANNELS || p->value_dim < 64 || p->value_dim > CHANNELS ||
        p->q_strides[2] != 1 || p->k_strides[2] != 1 || p->v_strides[2] != 1 ||
        p->q_strides[1] < p->key_dim || p->k_strides[1] < p->key_dim || p->v_strides[1] < p->value_dim ||
        p->q_strides[1] > INT32_MAX || p->k_strides[1] > INT32_MAX || p->v_strides[1] > INT32_MAX ||
        p->heads > SIZE_MAX / p->queries ||
        p->heads * p->queries > SIZE_MAX / sizeof(float) / p->value_dim ||
        (p->length && p->kv_heads > (SIZE_MAX - p->heads * p->queries) / 2 / p->length)) {
        return TRILLIM_ERROR;
    }
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    if (!isfinite(scale) || fabsf(scale) > 1) {
        return TRILLIM_ERROR;
    }
    size_t qs[] = {p->heads, p->queries, p->key_dim};
    size_t ks[] = {p->kv_heads, p->length, p->key_dim};
    size_t vs[] = {p->kv_heads, p->length, p->value_dim};
    if (!payload(p->q_rows, p->q_columns, qs, p->q_strides, &inputs[0]) ||
        !payload(p->k_rows, p->k_columns, ks, p->k_strides, &inputs[1]) ||
        !payload(p->v_rows, p->v_columns, vs, p->v_strides, &inputs[2])) {
        return TRILLIM_ERROR;
    }
    size_t groups = p->queries / QUERIES + (p->queries % QUERIES != 0);
    size_t jobs = p->heads * groups;
    size_t slots = jobs < 128 ? jobs : 128;
    outputs[0] = p->heads * p->queries * p->value_dim * sizeof(float);
    *scratch = slots * WORKSPACE * sizeof(float);
    return TRILLIM_OK;
}

static size_t first_key(const Parameters *p) {
    if (p->windowed && p->length >= p->queries && p->length - p->queries + 1 > p->window) {
        return p->length - p->queries + 1 - p->window;
    }
    return 0;
}

static TrillimStatus ordinary_inputs(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t queries = p->heads * p->queries;
    size_t first = first_key(p);
    size_t keys = p->kv_heads * (p->length - first);
    size_t total = queries + 2 * keys;
    size_t begin = index * 128, end = total - begin < 128 ? total : begin + 128;
    for (size_t row = begin; row < end; ++row) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!atomic_load_explicit(task->ordinary, memory_order_relaxed)) {
            break;
        }
        const float *data;
        size_t base, channels;
        if (row < queries) {
            data = task->query;
            base = row / p->heads * p->q_strides[1] + row % p->heads * p->q_strides[0];
            channels = p->key_dim;
        } else {
            bool values = row - queries >= keys;
            size_t key = row - queries - (values ? keys : 0);
            const size_t *strides = values ? p->v_strides : p->k_strides;
            data = values ? task->values : task->keys;
            base = (key / p->kv_heads + first) * strides[1] + key % p->kv_heads * strides[0];
            channels = values ? p->value_dim : p->key_dim;
        }
        for (size_t channel = 0; channel < channels; ++channel) {
            uint32_t bits;
            memcpy(&bits, data + base + channel, sizeof(bits));
            bits &= UINT32_C(0x7fffffff);
            if (bits && (bits < UINT32_C(0x2f800000) || bits >= UINT32_C(0x4f800000))) {
                atomic_store_explicit(task->ordinary, false, memory_order_relaxed);
                break;
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus scalar_tile(const Task *task, size_t index, size_t slot) {
    const Parameters *p = task->p;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->queries / QUERIES + (p->queries % QUERIES != 0);
    size_t head = index / groups, kv_head = head / (p->heads / p->kv_heads);
    size_t first = index % groups * QUERIES;
    size_t last = p->queries - first < QUERIES ? p->queries : first + QUERIES;
    float *scores = task->scratch + slot * WORKSPACE;
    float *accumulator = scores + 64;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t query = first; query < last; ++query) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            accumulator[channel] = 0;
        }
        size_t qb = head * p->q_strides[0] + query * p->q_strides[1];
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)query;
        size_t end = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        size_t begin = p->windowed && position >= 0 && (uint64_t)position + 1 > p->window ?
            (size_t)position + 1 - p->window : 0;
        float maximum = -INFINITY, denominator = 0;
        for (size_t block = begin; block < end; block += 64) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t count = end - block < 64 ? end - block : 64;
            float peak = -INFINITY;
            for (size_t key = 0; key < count; ++key) {
                size_t kb = kv_head * p->k_strides[0] + (block + key) * p->k_strides[1];
                float score = 0;
                for (size_t channel = 0; channel < p->key_dim; ++channel) {
                    score += task->query[qb + channel] * task->keys[kb + channel];
                }
                scores[key] = score * scale;
                peak = fmaxf(peak, scores[key]);
            }
            peak = fmaxf(maximum, peak);
            float normalization_peak = peak == -INFINITY ? 0 : peak;
            float previous = maximum == -INFINITY ? 0 : expf(maximum - normalization_peak);
            float sum = 0;
            for (size_t key = 0; key < count; ++key) {
                scores[key] = expf(scores[key] - normalization_peak);
                sum += scores[key];
            }
            for (size_t channel = 0; channel < p->value_dim; ++channel) {
                accumulator[channel] *= previous;
            }
            denominator = denominator * previous + sum;
            maximum = peak;
            for (size_t key = 0; key < count; ++key) {
                size_t vb = kv_head * p->v_strides[0] + (block + key) * p->v_strides[1];
                for (size_t channel = 0; channel < p->value_dim; ++channel) {
                    accumulator[channel] += scores[key] * task->values[vb + channel];
                }
            }
        }
        float *output = task->output + (query * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            output[channel] = accumulator[channel] / (denominator > 0 ? denominator : 1);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus blas_tile(const Task *task, size_t index, size_t slot) {
    const Parameters *p = task->p;
    const TrillimKernelContext *context = task->context;
    size_t groups = p->queries / QUERIES + (p->queries % QUERIES != 0);
    size_t head = index / groups, kv_head = head / (p->heads / p->kv_heads);
    size_t first = index % groups * QUERIES;
    size_t rows = p->queries - first < QUERIES ? p->queries - first : QUERIES;
    float *scores = task->scratch + slot * WORKSPACE;
    float *accumulator = scores + QUERIES * KEYS;
    float maximum[QUERIES], denominator[QUERIES], normalized[KEYS];
    size_t begin[QUERIES], end[QUERIES];
    size_t first_key_index = p->length, last_key = 0;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        maximum[row] = -INFINITY;
        denominator[row] = 0;
        memset(accumulator + row * p->value_dim, 0, p->value_dim * sizeof(float));
        int64_t position = (int64_t)p->length - (int64_t)p->queries + (int64_t)(first + row);
        end[row] = p->causal ? (position < 0 ? 0 : (size_t)position + 1) : p->length;
        begin[row] = p->windowed && position >= 0 && (uint64_t)position + 1 > p->window ?
            (size_t)position + 1 - p->window : 0;
        if (begin[row] < first_key_index) {
            first_key_index = begin[row];
        }
        if (end[row] > last_key) {
            last_key = end[row];
        }
    }
    const float *query = task->query + head * p->q_strides[0] + first * p->q_strides[1];
    for (size_t block = first_key_index; block < last_key; block += KEYS) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = last_key - block < KEYS ? last_key - block : KEYS;
        const float *keys = task->keys + kv_head * p->k_strides[0] + block * p->k_strides[1];
        sgemm(101, 111, 112, (int)rows, (int)count, (int)p->key_dim, scale,
              query, (int)p->q_strides[1], keys, (int)p->k_strides[1], 0, scores, KEYS);
        for (size_t row = 0; row < rows; ++row) {
            if (context->is_cancelled(context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float *probabilities = scores + row * KEYS;
            float peak = maximum[row];
            for (size_t key = 0; key < count; ++key) {
                if (block + key < begin[row] || block + key >= end[row]) {
                    probabilities[key] = -INFINITY;
                }
                peak = fmaxf(peak, probabilities[key]);
            }
            float normalization_peak = peak == -INFINITY ? 0 : peak;
            float previous = maximum[row] == -INFINITY ? 0 : expf(maximum[row] - normalization_peak);
            maximum[row] = peak;
            for (size_t key = 0; key < count; ++key) {
                normalized[key] = probabilities[key] - normalization_peak;
            }
#ifdef __APPLE__
            if (vector_exp) {
                int elements = (int)count;
                vector_exp(probabilities, normalized, &elements);
            } else
#endif
            {
                for (size_t key = 0; key < count; ++key) {
                    probabilities[key] = expf(normalized[key]);
                }
            }
            float sum = 0;
            for (size_t key = 0; key < count; ++key) {
                sum += probabilities[key];
            }
            denominator[row] = denominator[row] * previous + sum;
            for (size_t channel = 0; channel < p->value_dim; ++channel) {
                accumulator[row * p->value_dim + channel] *= previous;
            }
        }
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const float *values = task->values + kv_head * p->v_strides[0] + block * p->v_strides[1];
        sgemm(101, 111, 111, (int)rows, (int)p->value_dim, (int)count, 1,
              scores, KEYS, values, (int)p->v_strides[1], 1, accumulator, (int)p->value_dim);
    }
    for (size_t row = 0; row < rows; ++row) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float reciprocal = 1.0f / (denominator[row] > 0 ? denominator[row] : 1);
        float *output = task->output + ((first + row) * p->heads + head) * p->value_dim;
        for (size_t channel = 0; channel < p->value_dim; ++channel) {
            output[channel] = accumulator[row * p->value_dim + channel] * reciprocal;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus execute_tiles(void *opaque, size_t slot, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t groups = p->queries / QUERIES + (p->queries % QUERIES != 0);
    size_t jobs = p->heads * groups;
    size_t slots = jobs < 128 ? jobs : 128;
    bool enabled = atomic_load_explicit(task->ordinary, memory_order_relaxed);
#ifdef __APPLE__
    unsigned previous = enabled ? get_threading() : 1;
    if (enabled && previous != 1) {
        enabled = set_threading(1) == 0;
    }
#endif
    TrillimStatus status = TRILLIM_OK;
    for (size_t index = slot; index < jobs;) {
        status = enabled ? blas_tile(task, index, slot) : scalar_tile(task, index, slot);
        if (status != TRILLIM_OK || jobs - index <= slots) {
            break;
        }
        index += slots;
    }
#ifdef __APPLE__
    if (enabled && previous != 1) {
        (void)set_threading(previous);
    }
#endif
    return status;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_accelerated_attention(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    size_t visible = p->windowed && p->window < p->length ? p->window : p->length;
    bool available = visible >= 64 && (p->queries == 1 || p->queries >= 32) && initialize_blas();
    atomic_bool ordinary = ATOMIC_VAR_INIT(available);
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], scratch, p, context, &ordinary};
    size_t rows = p->heads * p->queries + 2 * p->kv_heads * (p->length - first_key(p));
    size_t scans = rows / 128 + (rows % 128 != 0);
    if (available) {
        TrillimStatus status = context->thread_pool->run(context->thread_pool->context, scans, ordinary_inputs, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    size_t groups = p->queries / QUERIES + (p->queries % QUERIES != 0);
    size_t jobs = p->heads * groups;
    return context->thread_pool->run(context->thread_pool->context, jobs < 128 ? jobs : 128, execute_tiles, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_accelerated_attention_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype),)\n"
           "def bind(context):\n"
           "    tensors = (context.inputs[name] for name in ('q', 'k', 'v'))\n"
           "    if context.outputs[0].dtype != 'fp32' or any(t.dtype != 'fp32' or (len(t.shape) != 3 or any(t.shape[axis].op != 'const' for axis in (1, 2))) for t in tensors):\n"
           "        raise ValueError('accelerated attention requires FP32 and fixed head/channel dimensions')\n"
           "    return _attention(context)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_accelerated_attention_parameters(void) {
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
    };
    static const size_t inputs[] = {4, 4, 4};
    static const size_t outputs[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        3, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_accelerated_attention_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
