#include "common.h"
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

enum { MAX_TASKS = 64, VALUE_TILE = 32, BLAS_DEPTH = 1024, PACK_ROWS = 256, ROWS_PER_TILE = 2 };

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
    float scale;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static size_t row_tiles(const Parameters *p) {
    size_t rows = (p->heads / p->kv_heads) * p->queries;
    return rows / ROWS_PER_TILE + (rows % ROWS_PER_TILE != 0);
}

static bool blas_shape(const Parameters *p) {
    return p->length && p->value_dim > 4096 / p->length &&
        p->value_dim <= INT32_MAX;
}

static size_t blas_columns(const Parameters *p) {
    return p->value_dim <= 1024 ? 64 : 128;
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

static inline bool cancelled(const Task *task) {
    return task->context->is_cancelled(task->context->cancellation_context);
}

static inline bool input_span(const size_t *shape, const size_t *strides,
                              size_t rows, size_t columns, size_t *bytes) {
    size_t span;
    if ((columns && rows > SIZE_MAX / sizeof(uint16_t) / columns) ||
        !trillim_span(3, shape, strides, &span) || span > rows * columns) {
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
        p->default_scale > 1 || !p->key_dim || !p->value_dim ||
        !p->kv_heads || p->heads % p->kv_heads) {
        return TRILLIM_ERROR;
    }
    if (!p->heads || !p->queries) {
        return TRILLIM_OK;
    }
    if (p->heads > SIZE_MAX / sizeof(uint16_t) / p->value_dim / p->queries) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->heads * p->queries;
    outputs[0] = rows * p->value_dim * sizeof(uint16_t);
    if (!p->length) {
        return TRILLIM_OK;
    }
    size_t tiles = p->kv_heads * row_tiles(p);
    size_t tasks = tiles < MAX_TASKS ? tiles : MAX_TASKS;
    size_t q_shape[] = {p->heads, p->queries, p->key_dim};
    size_t k_shape[] = {p->kv_heads, p->length, p->key_dim};
    size_t v_shape[] = {p->kv_heads, p->length, p->value_dim};
    if (p->length > SIZE_MAX / sizeof(float) / ROWS_PER_TILE / tasks ||
        p->heads > SIZE_MAX / trillim_itemsize(p->sink_dtype) ||
        !input_span(q_shape, p->q_strides, p->q_rows, p->q_columns, &inputs[0]) ||
        !input_span(k_shape, p->k_strides, p->k_rows, p->k_columns, &inputs[1]) ||
        !input_span(v_shape, p->v_strides, p->v_rows, p->v_columns, &inputs[2])) {
        return TRILLIM_ERROR;
    }
    inputs[3] = p->heads * trillim_itemsize(p->sink_dtype);
    *scratch = tasks * ROWS_PER_TILE * p->length * sizeof(float);
    if (blas_shape(p)) {
        size_t available = SIZE_MAX - *scratch;
        if (p->length > available / sizeof(float) / p->kv_heads / p->value_dim) {
            return TRILLIM_ERROR;
        }
        *scratch += p->kv_heads * p->length * p->value_dim * sizeof(float);
    }
    return TRILLIM_OK;
}

static TrillimStatus compute_scores(const Task *task, const uint16_t *q, const uint16_t *k,
                                    size_t begin, size_t end, float *scores) {
    const Parameters *p = task->p;
    size_t key = 0;
    for (; key < begin; ++key) {
        if ((key & 1023) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    size_t stride = p->k_strides[1];
    size_t q_channel = p->q_strides[2], k_channel = p->k_strides[2];
    for (; end - key >= 4; key += 4) {
        const uint16_t *base = k + key * stride;
        float sums[4] = {0};
        for (size_t first = 0; first < p->key_dim;) {
            if (cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            size_t last = p->key_dim - first < 1024 ? p->key_dim : first + 1024;
            for (size_t channel = first; channel < last; ++channel) {
                float value = trillim_bf16(q[channel * q_channel]);
                size_t offset = channel * k_channel;
                sums[0] += value * trillim_bf16(base[offset]);
                sums[1] += value * trillim_bf16(base[stride + offset]);
                sums[2] += value * trillim_bf16(base[2 * stride + offset]);
                sums[3] += value * trillim_bf16(base[3 * stride + offset]);
            }
            first = last;
        }
        for (size_t lane = 0; lane < 4; ++lane) {
            scores[key + lane] = trillim_bf16(trillim_to_bf16(sums[lane])) * task->scale;
        }
    }
    for (; key < end; ++key) {
        const uint16_t *base = k + key * stride;
        float sum = 0.0f;
        for (size_t first = 0; first < p->key_dim;) {
            if (cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            size_t last = p->key_dim - first < 1024 ? p->key_dim : first + 1024;
            for (size_t channel = first; channel < last; ++channel) {
                sum += trillim_bf16(q[channel * q_channel]) * trillim_bf16(base[channel * k_channel]);
            }
            first = last;
        }
        scores[key] = trillim_bf16(trillim_to_bf16(sum)) * task->scale;
    }
    for (; key < p->length; ++key) {
        if ((key & 1023) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] = -INFINITY;
    }
    return TRILLIM_OK;
}

static TrillimStatus normalize(const Task *task, float *scores, float sink) {
    size_t length = task->p->length;
    float maximum = -INFINITY;
    for (size_t key = 0; key < length; ++key) {
        if ((key & 1023) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        maximum = maximum >= scores[key] ? maximum : scores[key];
    }
    maximum = maximum >= sink ? maximum : sink;
    if (maximum == -INFINITY) {
        maximum = 0.0f;
    }
    bool numeric_maximum = !isnan(maximum);
    float sum = 0.0f;
    for (size_t key = 0; key < length; ++key) {
        if ((key & 1023) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        float numerator = scores[key] == -INFINITY && numeric_maximum ?
            0.0f : expf(scores[key] - maximum);
        scores[key] = numerator;
        sum += numerator;
    }
    sum += expf(sink - maximum);
    float denominator = sum > 0.0f ? sum : 1.0f;
    for (size_t key = 0; key < length; ++key) {
        if ((key & 1023) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        scores[key] /= denominator;
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_values(const Task *task, const uint16_t *v,
                                  float *const *probabilities, uint16_t *const *outputs,
                                  size_t rows) {
    const Parameters *p = task->p;
    for (size_t first = 0; first < p->value_dim;) {
        size_t width = p->value_dim - first < VALUE_TILE ? p->value_dim - first : VALUE_TILE;
        float sums[ROWS_PER_TILE][VALUE_TILE] = {{0}};
        for (size_t key = 0; key < p->length; ++key) {
            if ((key & 255) == 0 && cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            const uint16_t *value = v + key * p->v_strides[1] + first * p->v_strides[2];
            float weight0 = probabilities[0][key];
            float weight1 = rows == 2 ? probabilities[1][key] : 0.0f;
            for (size_t channel = 0; channel < width; ++channel) {
                float decoded = trillim_bf16(value[channel * p->v_strides[2]]);
                sums[0][channel] += weight0 * decoded;
                if (rows == 2) {
                    sums[1][channel] += weight1 * decoded;
                }
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t channel = 0; channel < width; ++channel) {
                outputs[row][first + channel] = trillim_to_bf16(sums[row][channel]);
            }
        }
        first += width;
    }
    return TRILLIM_OK;
}

static TrillimStatus pack_values(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t columns = blas_columns(p);
    size_t bands = p->value_dim / columns + (p->value_dim % columns != 0);
    size_t tiles = p->length / PACK_ROWS + (p->length % PACK_ROWS != 0);
    size_t band = index / tiles;
    size_t kv_head = band / bands;
    size_t channel = band % bands * columns;
    size_t width = p->value_dim - channel < columns ? p->value_dim - channel : columns;
    size_t first = index % tiles * PACK_ROWS;
    size_t end = p->length - first < PACK_ROWS ? p->length : first + PACK_ROWS;
    const uint16_t *v = task->v + kv_head * p->v_strides[0] + channel * p->v_strides[2];
    float *packed = task->packed_values + (kv_head * p->value_dim + channel) * p->length;
    if (p->v_strides[1] == 1) {
        size_t block = first / BLAS_DEPTH * BLAS_DEPTH;
        size_t depth = p->length - block < BLAS_DEPTH ? p->length - block : BLAS_DEPTH;
        for (size_t lane = 0; lane < width; ++lane) {
            if ((lane & 7) == 0 && cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            const uint16_t *source = v + lane * p->v_strides[2];
            float *target = packed + block * width + lane * depth;
            for (size_t key = first; key < end; ++key) {
                target[key - block] = trillim_bf16(source[key]);
            }
        }
        return TRILLIM_OK;
    }
    for (size_t key = first; key < end; ++key) {
        if (((key - first) & 7) == 0 && cancelled(task)) {
            return TRILLIM_CANCELLED;
        }
        const uint16_t *source = v + key * p->v_strides[1];
        for (size_t lane = 0; lane < width; ++lane) {
            packed[key * width + lane] = trillim_bf16(source[lane * p->v_strides[2]]);
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus apply_blas_values(const Task *task, size_t kv_head,
                                       const float *probabilities, uint16_t *output) {
    const Parameters *p = task->p;
    size_t columns = blas_columns(p);
    const float *packed = task->packed_values + kv_head * p->value_dim * p->length;
    bool transpose = p->v_strides[1] == 1;
    for (size_t channel = 0; channel < p->value_dim;) {
        size_t width = p->value_dim - channel < columns ? p->value_dim - channel : columns;
        _Alignas(64) float sums[128];
        /* Match the scalar BLAS matmul for one output row. */
        for (size_t first = 0; first < p->length;) {
            if (cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            size_t depth = p->length - first < BLAS_DEPTH ? p->length - first : BLAS_DEPTH;
            sgemm(101, 111, transpose ? 112 : 111, 1, (int)width, (int)depth, 1.0f,
                  probabilities + first, (int)depth, packed + first * width,
                  transpose ? (int)depth : (int)width, first == 0 ? 0.0f : 1.0f,
                  sums, (int)p->value_dim);
            first += depth;
        }
        for (size_t lane = 0; lane < width; ++lane) {
            output[channel + lane] = trillim_to_bf16(sums[lane]);
        }
        packed += p->length * width;
        channel += width;
    }
    return TRILLIM_OK;
}

static TrillimStatus attention_rows(const Task *task, size_t index) {
    const Parameters *p = task->p;
    size_t group = p->heads / p->kv_heads;
    size_t rows = group * p->queries;
    float *scores = p->length ? task->scratch + index * ROWS_PER_TILE * p->length : NULL;
    for (size_t flat = index; flat < p->kv_heads * task->tiles; flat += task->tasks) {
        size_t kv_head = flat / task->tiles;
        size_t first = flat % task->tiles * ROWS_PER_TILE;
        size_t count = rows - first < ROWS_PER_TILE ? rows - first : ROWS_PER_TILE;
        uint16_t *outputs[ROWS_PER_TILE];
        float *probabilities[ROWS_PER_TILE];
        for (size_t lane = 0; lane < count; ++lane) {
            size_t head = kv_head * group + (first + lane) / p->queries;
            size_t row = (first + lane) % p->queries;
            uint16_t *output = task->output + (row * p->heads + head) * p->value_dim;
            if (cancelled(task)) {
                return TRILLIM_CANCELLED;
            }
            if (!p->length) {
                for (size_t offset = 0; offset < p->value_dim;) {
                    if (cancelled(task)) {
                        return TRILLIM_CANCELLED;
                    }
                    size_t width = p->value_dim - offset < 1024 ? p->value_dim - offset : 1024;
                    memset(output + offset, 0, width * sizeof(*output));
                    offset += width;
                }
                continue;
            }
            size_t distance = p->queries - row;
            size_t position_end = distance > p->length ? 0 : p->length - distance + 1;
            size_t end = p->causal ? position_end : p->length;
            size_t begin = p->windowed && position_end > p->window ? position_end - p->window : 0;
            const uint16_t *q = task->q + head * p->q_strides[0] + row * p->q_strides[1];
            const uint16_t *k = task->k + kv_head * p->k_strides[0];
            float sink = trillim_load(task->sinks, head, p->sink_dtype).f;
            probabilities[lane] = scores + lane * p->length;
            outputs[lane] = output;
            TrillimStatus status = compute_scores(task, q, k, begin, end, probabilities[lane]);
            if (status != TRILLIM_OK) {
                return status;
            }
            status = normalize(task, probabilities[lane], sink);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        if (!p->length) {
            continue;
        }
        if (task->packed_values) {
            for (size_t lane = 0; lane < count; ++lane) {
                TrillimStatus status = apply_blas_values(task, kv_head, probabilities[lane], outputs[lane]);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
        } else {
            const uint16_t *v = task->v + kv_head * p->v_strides[0];
            TrillimStatus status = apply_values(task, v, probabilities, outputs, count);
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

TRILLIM_KERNEL TrillimStatus trillim_scalar_attention_sinks_bf16(
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
    size_t tiles = row_tiles(p);
    size_t work = p->kv_heads * tiles;
    size_t tasks = work < MAX_TASKS ? work : MAX_TASKS;
    float scale = p->default_scale ? 1.0f / sqrtf((float)p->key_dim) : p->scale;
    Task task = {
        outputs[0], inputs[0], inputs[1], inputs[2], inputs[3],
        scratch, NULL, tiles, tasks, scale, p, context
    };
    const TrillimThreadPool *pool = context->thread_pool;
    if (blas_shape(p) && initialize_blas()) {
        task.packed_values = task.scratch + tasks * ROWS_PER_TILE * p->length;
        size_t columns = blas_columns(p);
        size_t bands = p->value_dim / columns + (p->value_dim % columns != 0);
        size_t tiles = p->length / PACK_ROWS + (p->length % PACK_ROWS != 0);
        TrillimStatus status = pool->run(pool->context, p->kv_heads * bands * tiles,
                                         pack_values, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return pool->run(pool->context, tasks, attention, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_attention_sinks_bf16_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _attention, _contiguous_strides, _product, _DTYPE_IDS\n"
           "physical_inputs = ('q', 'k', 'v', 'sinks')\n"
           "def pattern(q: Tensor, k: Tensor, v: Tensor, sinks: Tensor, *, causal: bool = True, scale = None, window = None, dtype: DType = 'bf16') -> tuple[Tensor, ...]:\n"
           "    return (Attention(q, k, v, causal=causal, scale=scale, window=window, dtype=dtype, sinks=sinks),)\n"
           "def bind(context):\n"
           "    if context.outputs[0].dtype != 'bf16' or context.attrs.get('softcap') is not None:\n"
           "        raise ValueError('this attention kernel requires BF16 output and uncapped scores')\n"
           "    if context.inputs['v'].shape[1] != context.inputs['k'].shape[1]:\n"
           "        raise ValueError('fused attention requires matching key and value geometry')\n"
           "    if any(context.inputs[name].dtype != 'bf16' for name in ('q', 'k', 'v')):\n"
           "        raise ValueError('this attention kernel requires BF16 query, key, and value storage')\n"
           "    result = _attention(context)\n"
           "    sinks = context.inputs['sinks']\n"
           "    if sinks.dtype not in {'fp16', 'bf16', 'fp32'} or _product(sinks.shape) != context.inputs['q'].shape[1]:\n"
           "        raise ValueError('attention sinks require one floating scalar per query head')\n"
           "    if sinks.strides is not None and sinks.strides != _contiguous_strides(sinks.shape):\n"
           "        raise ValueError('attention sinks require contiguous storage')\n"
           "    result['sink_dtype'] = _DTYPE_IDS[sinks.dtype]\n"
           "    return result\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_attention_sinks_bf16_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_attention_sinks_bf16_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
