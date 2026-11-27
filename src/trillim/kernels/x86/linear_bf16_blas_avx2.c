#include "common.h"
#include <immintrin.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

enum { BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 256, MAX_LANES = 16 };
enum { PACK_ROWS = 8 };
enum { WORKSPACE = BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K + BLOCK_M * BLOCK_N };
#define VECTOR __attribute__((target("avx2,fma")))

typedef struct {
    size_t rows, columns, width;
} Parameters;

typedef struct {
    const uint8_t *input, *weight;
    void *output;
    float *scratch;
    float *activations;
    size_t lanes, column_tiles;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);

static Sgemm sgemm;
static atomic_uint provider_state;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&provider_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&provider_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
                    break;
                }
                FreeLibrary(library);
            }
        }
#else
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
        static const char *const names[] = {"libopenblas.so.0", "libblas.so.3", "libblas.so"};
#endif
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            void *library = dlopen(names[i], RTLD_LOCAL | RTLD_LAZY);
            if (library != NULL) {
#ifdef __APPLE__
                void *address = dlsym(library, "cblas_sgemm$NEWLAPACK");
                if (address == NULL) {
                    address = dlsym(library, "cblas_sgemm");
                }
#else
                void *address = dlsym(library, "cblas_sgemm");
#endif
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&provider_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&provider_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

static size_t ceil_div(size_t n, size_t d) {
    return n / d + (n % d != 0);
}

static bool prefill_shape(const Parameters *p) {
    return p->rows >= 32 && p->columns >= 128 && p->width >= 256;
}

static size_t scratch_lanes(const Parameters *p) {
    size_t tiles = ceil_div(p->rows, BLOCK_M) * ceil_div(p->columns, BLOCK_N);
    return tiles < MAX_LANES ? tiles : MAX_LANES;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!p->rows || !p->columns) {
        return TRILLIM_OK;
    }
    if (p->rows > SIZE_MAX / p->columns / sizeof(uint16_t) ||
        (p->width && (p->rows > SIZE_MAX / p->width / sizeof(uint16_t) ||
                      p->columns > SIZE_MAX / p->width / sizeof(uint16_t)))) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->rows * p->width * sizeof(uint16_t);
    inputs[1] = p->columns * p->width * sizeof(uint16_t);
    outputs[0] = p->rows * p->columns * sizeof(uint16_t);
    if (prefill_shape(p)) {
        *scratch = scratch_lanes(p) * WORKSPACE * sizeof(float);
    }
    return TRILLIM_OK;
}

VECTOR static inline __m256 widen8(const uint8_t *source) {
    __m128i bits = _mm_loadu_si128((const __m128i *)source);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

VECTOR static TrillimStatus direct(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t row = tile / task->column_tiles;
    size_t first = tile % task->column_tiles * BLOCK_N;
    size_t end = p->columns - first < BLOCK_N ? p->columns : first + BLOCK_N;
    for (size_t column = first; column < end; ++column) {
        __m256 total = _mm256_setzero_ps();
        size_t a = row * p->width, b = column * p->width;
        size_t k = 0;
        for (; p->width - k >= 8; k += 8) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 x = widen8(task->input + (a + k) * 2);
            __m256 w = widen8(task->weight + (b + k) * 2);
            total = _mm256_fmadd_ps(x, w, total);
        }
        __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        sum = _mm_hadd_ps(sum, sum);
        float value = _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
        for (; k < p->width; ++k) {
            value = fmaf(trillim_load(task->input, a + k, TRILLIM_BF16).f,
                         trillim_load(task->weight, b + k, TRILLIM_BF16).f, value);
        }
        trillim_store(task->output, row * p->columns + column, TRILLIM_BF16, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

VECTOR static void pack(const uint8_t *source, size_t first, size_t depth, size_t width,
                        size_t rows, size_t columns, float *packed) {
    for (size_t r = 0; r < rows; ++r) {
        size_t index = (first + r) * width + depth;
        size_t k = 0;
        for (; columns - k >= 8; k += 8) {
            _mm256_storeu_ps(packed + r * columns + k, widen8(source + (index + k) * 2));
        }
        for (; k < columns; ++k) {
            packed[r * columns + k] = trillim_load(source, index + k, TRILLIM_BF16).f;
        }
    }
}

static TrillimStatus pack_activations(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t column_tiles = ceil_div(p->width, BLOCK_K);
    size_t first = tile / column_tiles * PACK_ROWS;
    size_t end = p->rows - first < PACK_ROWS ? p->rows : first + PACK_ROWS;
    size_t start = tile % column_tiles * BLOCK_K;
    size_t k = p->width - start < BLOCK_K ? p->width - start : BLOCK_K;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t row = first; row < end; ++row) {
        pack(task->input, row, start, p->width, 1, k, task->activations + row * p->width + start);
    }
    return TRILLIM_OK;
}

static TrillimStatus prefill(void *opaque, size_t lane, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = ceil_div(p->rows, BLOCK_M) * task->column_tiles;
    float *a = task->scratch + lane * WORKSPACE;
    float *b = a + BLOCK_M * BLOCK_K;
    float *c = b + BLOCK_N * BLOCK_K;
    for (size_t tile = lane; tile < tiles; tile += task->lanes) {
        size_t row = tile / task->column_tiles * BLOCK_M;
        size_t column = tile % task->column_tiles * BLOCK_N;
        size_t m = p->rows - row < BLOCK_M ? p->rows - row : BLOCK_M;
        size_t n = p->columns - column < BLOCK_N ? p->columns - column : BLOCK_N;
        for (size_t start = 0; start < p->width;) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t k = p->width - start < BLOCK_K ? p->width - start : BLOCK_K;
            const float *x = a;
            int input_stride = (int)k;
            if (task->activations) {
                x = task->activations + row * p->width + start;
                input_stride = (int)p->width;
            } else {
                pack(task->input, row, start, p->width, m, k, a);
            }
            pack(task->weight, column, start, p->width, n, k, b);
            sgemm(101, 111, 112, (int)m, (int)n, (int)k, 1.0f,
                  x, input_stride, b, (int)k, start ? 1.0f : 0.0f, c, (int)n);
            start += k;
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        for (size_t r = 0; r < m; ++r) {
            for (size_t i = 0; i < n; ++i) {
                TrillimValue value = {.f = c[r * n + i]};
                trillim_store(task->output, (row + r) * p->columns + column + i, TRILLIM_BF16, value);
            }
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_linear_bf16_blas_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(p, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || !output_size) {
        return status;
    }
    const TrillimThreadPool *pool = context->thread_pool;
    Task task = {
        .input = inputs[0], .weight = inputs[1], .output = outputs[0], .scratch = scratch,
        .column_tiles = ceil_div(p->columns, BLOCK_N), .p = p, .context = context
    };
    if (scratch_size && initialize_blas()) {
        size_t reserved = scratch_lanes(p);
        size_t workers = pool->worker_count;
        task.lanes = reserved;
        if (workers && workers < reserved && p->width <= INT32_MAX &&
            p->rows <= (reserved - workers) * WORKSPACE / p->width) {
            task.lanes = workers;
            task.activations = task.scratch + workers * WORKSPACE;
            size_t tiles = ceil_div(p->rows, PACK_ROWS) * ceil_div(p->width, BLOCK_K);
            status = pool->run(pool->context, tiles, pack_activations, &task);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return pool->run(pool->context, task.lanes, prefill, &task);
    }
    return pool->run(pool->context, p->rows * task.column_tiles, direct, &task);
}

TRILLIM_KERNEL const char *trillim_x86_linear_bf16_blas_avx2_tdl(void) {
    return "from trillim.tdl import Linear, Tensor\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n"
           "def bind(context):\n"
           "    x, w, y = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if any(t.dtype != 'bf16' for t in (x, w, y)):\n"
           "        raise ValueError('requires BF16 activations, weights and output')\n"
           "    for t in (x, w):\n"
           "        if (t.strides or _contiguous_strides(t.shape)) != _contiguous_strides(t.shape) or _storage_shape(t) != t.shape:\n"
           "            raise ValueError('requires contiguous complete operands')\n"
           "    p = _linear(context)\n"
           "    return dict(rows=p['m'], columns=p['n'], width=p['k'])\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_linear_bf16_blas_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"columns", "size_t", offsetof(Parameters, columns), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 2, 1,
        requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_linear_bf16_blas_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_BLAS
    };
    return &caps;
}
