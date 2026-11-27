#include "common.h"
#include <math.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__FLT16_MANT_DIG__)
__extension__ typedef _Float16 TrillimFloat16;
#endif

#define BLOCK_M 64
#define BLOCK_N 128
#define BLOCK_K 1024
#define LANES 32
#define WORKSPACE (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N + BLOCK_M * BLOCK_N)

static bool initialize_blas(void);

typedef struct {
    size_t batches;
    size_t m;
    size_t n;
    size_t k;
    size_t rank;
    const size_t *batch_shape;
    const size_t *a_batch_strides;
    const size_t *b_batch_strides;
    size_t a_row_stride;
    size_t a_col_stride;
    size_t b_row_stride;
    size_t b_col_stride;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t out_dtype;
    size_t a_storage_rows;
    size_t a_storage_columns;
    size_t b_storage_rows;
    size_t b_storage_columns;
} MatmulParameters;

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const MatmulParameters *p;
    const TrillimKernelContext *context;
} MatmulTask;

static bool matrix_span(size_t base, size_t rows, size_t columns,
                        size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (base >= capacity || (row_stride != 0 && rows - 1 > (capacity - 1 - base) / row_stride)) {
        return false;
    }
    base += (rows - 1) * row_stride;
    return col_stride == 0 || columns - 1 <= (capacity - 1 - base) / col_stride;
}

static size_t block_n(const MatmulParameters *p) {
    return p->n <= 1024 ? 64 : 128;
}

static size_t block_k(const MatmulParameters *p) {
    return p->m < 64 || p->n <= 1024 ? 1024 : 512;
}

static size_t block_m(const MatmulParameters *p) {
    bool direct_f32 = p->a_dtype == TRILLIM_FP32 && p->b_dtype == TRILLIM_FP32 &&
        p->out_dtype == TRILLIM_FP32 && p->m >= 1024 && p->n >= 512 && p->n <= 2048 &&
        ((p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) ||
         (p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX));
    if (direct_f32) {
        return 256;
    }
    /* Direct A/C access permits taller tiles without larger packing buffers. */
    bool direct = p->a_dtype == TRILLIM_FP32 && p->out_dtype == TRILLIM_FP32 &&
        p->b_dtype != TRILLIM_FP32 && p->n >= 2560 &&
        ((p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) ||
         (p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX));
    return direct && p->m >= 128 ? 256 : BLOCK_M;
}


static bool small(const MatmulParameters *p) {
    return p->k == 0 || (p->n != 0 && p->m <= 4096 / p->n &&
                         p->m * p->n <= 4096 / p->k);
}

static TrillimStatus blas_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const MatmulParameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        p->out_dtype != TRILLIM_FP32) {
        return TRILLIM_ERROR;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    size_t a_span, b_span;
    if (!trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span) ||
        p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / trillim_itemsize(p->out_dtype)) {
        return TRILLIM_ERROR;
    }
    size_t batches = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->batch_shape[axis] != 0 && batches > SIZE_MAX / p->batch_shape[axis]) {
            return TRILLIM_ERROR;
        }
        batches *= p->batch_shape[axis];
    }
    if (batches != p->batches) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if ((p->a_storage_columns != 0 && p->a_storage_rows > SIZE_MAX / trillim_itemsize(p->a_dtype) / p->a_storage_columns) ||
            (p->b_storage_columns != 0 && p->b_storage_rows > SIZE_MAX / trillim_itemsize(p->b_dtype) / p->b_storage_columns)) {
            return TRILLIM_ERROR;
        }
        size_t a_count = p->a_storage_rows * p->a_storage_columns;
        size_t b_count = p->b_storage_rows * p->b_storage_columns;
        if (!matrix_span(a_span ? a_span - 1 : 0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !matrix_span(b_span ? b_span - 1 : 0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = a_count * trillim_itemsize(p->a_dtype);
        inputs[1] = b_count * trillim_itemsize(p->b_dtype);
    }
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    if (small(p)) {
        return TRILLIM_OK;
    }
    if (p->n > INT32_MAX || !initialize_blas()) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t columns = p->n / 64 + (p->n % 64 != 0);
    size_t count = p->batches * rows * columns;
    *scratch = (count < LANES ? count : LANES) * WORKSPACE * sizeof(float);
    return TRILLIM_OK;
}

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
static Sgemm sgemm;
#ifdef __APPLE__
typedef unsigned (*GetThreading)(void);
typedef int (*SetThreading)(unsigned);
static GetThreading get_threading;
static SetThreading set_threading;
#endif
static atomic_uint blas_state = 0;

static bool initialize_blas(void) {
    if (atomic_load_explicit(&blas_state, memory_order_acquire) == 2) {
        return sgemm != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&blas_state, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
#ifdef _WIN32
        static const char *const names[] = {"libopenblas.dll", "openblas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                FreeLibrary(library);
            }
        }
#else
#ifdef __APPLE__
        static const char *const names[] = {"/System/Library/Frameworks/Accelerate.framework/Accelerate"};
#else
        static const char *const names[] = {"libopenblas.so.0", "libopenblas.so"};
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
#ifdef __APPLE__
                    void *getter = dlsym(library, "BLASGetThreading");
                    void *setter = dlsym(library, "BLASSetThreading");
                    if (getter != NULL && setter != NULL) {
                        memcpy(&get_threading, &getter, sizeof(get_threading));
                        memcpy(&set_threading, &setter, sizeof(set_threading));
                    }
#endif
                    break;
                }
                dlclose(library);
            }
        }
#endif
        atomic_store_explicit(&blas_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&blas_state, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

static inline float load(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)data + index * sizeof(bits), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(bits);
    }
#if defined(__FLT16_MANT_DIG__)
    TrillimFloat16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
#else
    return trillim_fp16(bits);
#endif
}

static TrillimStatus pack(const void *source, uint32_t dtype, size_t base,
                          size_t rows, size_t columns, size_t row_stride,
                          size_t col_stride, float *target,
                          const TrillimKernelContext *context) {
    for (size_t row = 0; row < rows; ++row) {
        if ((row & 7) == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t index = base + row * row_stride;
        if (dtype == TRILLIM_FP32 && col_stride == 1) {
            memcpy(target + row * columns, (const uint8_t *)source + index * sizeof(float), columns * sizeof(float));
        } else if (dtype == TRILLIM_BF16 && col_stride == 1) {
            for (size_t column = 0; column < columns; ++column) {
                uint16_t bits;
                memcpy(&bits, (const uint8_t *)source + (index + column) * sizeof(bits), sizeof(bits));
                target[row * columns + column] = trillim_bf16(bits);
            }
#if defined(__FLT16_MANT_DIG__)
        } else if (dtype == TRILLIM_FP16 && col_stride == 1) {
            for (size_t column = 0; column < columns; ++column) {
                TrillimFloat16 value;
                memcpy(&value, (const uint8_t *)source + (index + column) * sizeof(value), sizeof(value));
                target[row * columns + column] = (float)value;
            }
#endif
        } else {
            for (size_t column = 0; column < columns; ++column) {
                target[row * columns + column] = load(source, index + column * col_stride, dtype);
            }
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus blas_small_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const MatmulTask *t = opaque;
    const MatmulParameters *p = t->p;
    size_t columns = p->n / 32 + (p->n % 32 != 0);
    size_t batch = index / columns / p->m;
    size_t row = index / columns % p->m;
    size_t first = index % columns * 32;
    size_t last = p->n - first < 32 ? p->n : first + 32;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    float sums[32] = {0};
    for (size_t k = 0; k < p->k; ++k) {
        if ((k & 255) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float a = load(t->a, a_base + k * p->a_col_stride, p->a_dtype);
        for (size_t column = first; column < last; ++column) {
            float b = load(t->b, b_base + k * p->b_row_stride + column * p->b_col_stride, p->b_dtype);
            sums[column - first] += a * b;
        }
    }
    for (size_t column = first; column < last; ++column) {
        TrillimValue value = {.f = sums[column - first]};
        trillim_store(t->output, (batch * p->m + row) * p->n + column, p->out_dtype, value);
    }
    return TRILLIM_OK;
}

static size_t row_reuse(const MatmulParameters *p, const TrillimKernelContext *context) {
    size_t workers = context->thread_pool->worker_count;
    size_t minimum_rows = workers <= 8 || p->n >= 2048 ? 128 : 512;
    if (p->m < minimum_rows || p->out_dtype != TRILLIM_FP32 || p->b_dtype == TRILLIM_FP32 || p->k == 0) {
        return 1;
    }
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
    if (columns == 0) {
        return 1;
    }
    size_t target = workers == 0 ? 1 : (workers < LANES / 2 ? workers * 2 : LANES);
    size_t groups = target / columns + (target % columns != 0);
    size_t reuse = rows / groups;
    return reuse < 2 ? 1 : (reuse < 8 ? reuse : 8);
}

typedef struct {
    MatmulTask task;
    float *scratch;
    size_t count;
    size_t lanes;
    size_t row_reuse;
} BlasLinearTask;

static TrillimStatus ungrouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasLinearTask *job = opaque;
    const MatmulTask *t = &job->task;
    const MatmulParameters *p = t->p;
    size_t width = block_n(p);
    size_t depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / block_m(p) + (p->m % block_m(p) != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    float *c_pack = b_pack + BLOCK_K * BLOCK_N;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / row_tiles;
        size_t row = tile / columns % row_tiles * block_m(p);
        size_t column = tile % columns * width;
        size_t m = p->m - row < block_m(p) ? p->m - row : block_m(p);
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        size_t output = (batch * p->m + row) * p->n + column;
        float *c = p->out_dtype == TRILLIM_FP32 ? (float *)t->output + output : c_pack;
        int ldc = p->out_dtype == TRILLIM_FP32 ? (int)p->n : (int)n;
        if (p->k == 0) {
            for (size_t r = 0; r < m; ++r) {
                memset(c + r * (size_t)ldc, 0, n * sizeof(float));
            }
        }
        for (size_t first = 0; first < p->k; first += depth) {
            if (t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t k = p->k - first < depth ? p->k - first : depth;
            size_t a_index = a_base + first * p->a_col_stride;
            size_t b_index = b_base + first * p->b_row_stride;
            const float *a = a_pack;
            const float *b = b_pack;
            int a_trans = 111, b_trans = 111, lda = (int)k, ldb = (int)n;
            TrillimStatus status;
            if (p->a_dtype == TRILLIM_FP32 && p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                lda = (int)p->a_row_stride;
            } else if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX) {
                a = (const float *)t->a + a_index;
                a_trans = 112;
                lda = (int)p->a_col_stride;
            } else {
                status = pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            if (p->b_dtype == TRILLIM_FP32 && p->b_col_stride == 1 && p->b_row_stride >= p->n && p->b_row_stride <= INT32_MAX) {
                b = (const float *)t->b + b_index;
                ldb = (int)p->b_row_stride;
            } else if (p->b_dtype == TRILLIM_FP32 && p->b_row_stride == 1 && p->b_col_stride >= p->k && p->b_col_stride <= INT32_MAX) {
                b = (const float *)t->b + b_index;
                b_trans = 112;
                ldb = (int)p->b_col_stride;
            } else if (p->b_row_stride == 1) {
                status = pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                if (status != TRILLIM_OK) {
                    return status;
                }
            }
            sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                  a, lda, b, ldb, first == 0 ? 0.0f : 1.0f, c, ldc);
        }
        if (p->out_dtype != TRILLIM_FP32) {
            for (size_t r = 0; r < m; ++r) {
                for (size_t col = 0; col < n; ++col) {
                    TrillimValue value = {.f = c_pack[r * n + col]};
                    trillim_store(t->output, output + r * p->n + col, p->out_dtype, value);
                }
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus grouped_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasLinearTask *job = opaque;
    const MatmulTask *t = &job->task;
    const MatmulParameters *p = t->p;
    size_t width = block_n(p), depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t groups = rows / job->row_reuse + (rows % job->row_reuse != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / groups;
        size_t group = tile / columns % groups;
        size_t first_row = group * job->row_reuse;
        size_t last_row = rows - first_row < job->row_reuse ? rows : first_row + job->row_reuse;
        size_t column = tile % columns * width;
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides);
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        for (size_t first = 0; first < p->k; first += depth) {
            if (t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t k = p->k - first < depth ? p->k - first : depth;
            /* Partial C stays in output while packed B is shared across row tiles. */
            size_t b_index = b_base + first * p->b_row_stride;
            int b_trans = 111, ldb = (int)n;
            TrillimStatus status;
            if (p->b_row_stride == 1) {
                status = pack(t->b, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                b_trans = 112;
                ldb = (int)k;
            } else {
                status = pack(t->b, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
            }
            if (status != TRILLIM_OK) {
                return status;
            }
            for (size_t block = first_row; block < last_row; ++block) {
                if (t->context->is_cancelled(t->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t row = block * block_m(p);
                size_t m = p->m - row < block_m(p) ? p->m - row : block_m(p);
                size_t a_index = a_base + row * p->a_row_stride + first * p->a_col_stride;
                const float *a = a_pack;
                int a_trans = 111, lda = (int)k;
                if (p->a_dtype == TRILLIM_FP32 && p->a_col_stride == 1 && p->a_row_stride >= p->k && p->a_row_stride <= INT32_MAX) {
                    a = (const float *)t->a + a_index;
                    lda = (int)p->a_row_stride;
                } else if (p->a_dtype == TRILLIM_FP32 && p->a_row_stride == 1 && p->a_col_stride >= p->m && p->a_col_stride <= INT32_MAX) {
                    a = (const float *)t->a + a_index;
                    a_trans = 112;
                    lda = (int)p->a_col_stride;
                } else {
                    status = pack(t->a, p->a_dtype, a_index, m, k, p->a_row_stride, p->a_col_stride, a_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                size_t output = (batch * p->m + row) * p->n + column;
                float *c = (float *)t->output + output;
                sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                      a, lda, b_pack, ldb, first == 0 ? 0.0f : 1.0f, c, (int)p->n);
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus blas_linear_project_impl(void *opaque, size_t index, size_t worker) {
    const BlasLinearTask *job = opaque;
    return job->row_reuse > 1 ? grouped_project_impl(opaque, index, worker) :
        ungrouped_project_impl(opaque, index, worker);
}

static TrillimStatus blas_linear_project(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = blas_linear_project_impl(opaque, index, worker);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return blas_linear_project_impl(opaque, index, worker);
}

static TrillimStatus blas_linear(void *const *outputs, const void *const *inputs,
                                  void *scratch, const void *parameters,
                                  const TrillimKernelContext *context) {
    const MatmulParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (small(p)) {
        MatmulTask task = {outputs[0], inputs[0], inputs[1], p, context};
        size_t columns = p->n / 32 + (p->n % 32 != 0);
        size_t count = p->batches * p->m * columns;
        return count == 1 ? blas_small_project(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, count, blas_small_project, &task);
    }
    if (!initialize_blas()) {
        return TRILLIM_ERROR;
    }
    size_t rows = p->m / block_m(p) + (p->m % block_m(p) != 0);
    size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
    size_t reuse = row_reuse(p, context);
    size_t groups = rows / reuse + (rows % reuse != 0);
    size_t count = p->batches * groups * columns;
    size_t lanes = count < LANES ? count : LANES;
    BlasLinearTask task = {{outputs[0], inputs[0], inputs[1], p, context}, scratch, count, lanes, reuse};
    return context->thread_pool->run(context->thread_pool->context, lanes, blas_linear_project, &task);
}

static inline float trillim_load_fp32(const void *data, size_t index) {
    float value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(float), sizeof(value));
    return value;
}

static inline float trillim_load_bf16(const void *data, size_t index) {
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return trillim_bf16(value);
}

static inline float trillim_load_fp16(const void *data, size_t index) {
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return trillim_fp16(value);
}

#define TRILLIM_NATIVE_PROJECTION(NAME, LOAD_B) \
static TrillimStatus trillim_native_##NAME( \
    const MatmulTask *task, size_t a_base, \
    size_t output, size_t first, size_t last \
) { \
    const MatmulParameters *p = task->p; \
    if (p->b_col_stride == 1) { \
        float sums[32] = {0}; \
        for (size_t start = 0; start < p->k; start += 256) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t end = p->k - start < 256 ? p->k : start + 256; \
            for (size_t k = start; k < end; ++k) { \
                float value = trillim_load_fp32(task->a, a_base + k * p->a_col_stride); \
                size_t offset = k * p->b_row_stride + first; \
                for (size_t column = 0; column < last - first; ++column) { \
                    sums[column] += value * LOAD_B(task->b, offset + column); \
                } \
            } \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t column = first; column < last; ++column) { \
            TrillimValue result = {.f = sums[column - first]}; \
            trillim_store(task->output, output + column, p->out_dtype, result); \
        } \
        return TRILLIM_OK; \
    } \
    size_t column = first; \
    for (; last - column >= 4; column += 4) { \
        float sums[4] = {0}; \
        for (size_t start = 0; start < p->k; start += 1024) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t end = p->k - start < 1024 ? p->k : start + 1024; \
            for (size_t k = start; k < end; ++k) { \
                float value = trillim_load_fp32(task->a, a_base + k * p->a_col_stride); \
                size_t offset = k * p->b_row_stride + column * p->b_col_stride; \
                sums[0] += value * LOAD_B(task->b, offset); \
                sums[1] += value * LOAD_B(task->b, offset + p->b_col_stride); \
                sums[2] += value * LOAD_B(task->b, offset + 2 * p->b_col_stride); \
                sums[3] += value * LOAD_B(task->b, offset + 3 * p->b_col_stride); \
            } \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t i = 0; i < 4; ++i) { \
            TrillimValue result = {.f = sums[i]}; \
            trillim_store(task->output, output + column + i, p->out_dtype, result); \
        } \
    } \
    for (; column < last; ++column) { \
        float sum = 0.0f; \
        for (size_t start = 0; start < p->k; start += 1024) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t end = p->k - start < 1024 ? p->k : start + 1024; \
            for (size_t k = start; k < end; ++k) { \
                sum += trillim_load_fp32(task->a, a_base + k * p->a_col_stride) * \
                       LOAD_B(task->b, k * p->b_row_stride + column * p->b_col_stride); \
            } \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        TrillimValue result = {.f = sum}; \
        trillim_store(task->output, output + column, p->out_dtype, result); \
    } \
    return TRILLIM_OK; \
}

TRILLIM_NATIVE_PROJECTION(fp32_bf16, trillim_load_bf16)
TRILLIM_NATIVE_PROJECTION(fp32_fp16, trillim_load_fp16)

#undef TRILLIM_NATIVE_PROJECTION

static inline TrillimStatus scalar_linear_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const MatmulTask *task = opaque;
    const MatmulParameters *p = task->p;
    size_t tiles = p->n / 32 + (p->n % 32 != 0);
    size_t row = index / tiles;
    size_t first = index % tiles * 32;
    size_t last = p->n - first < 32 ? p->n : first + 32;
    size_t a_base = row * p->a_row_stride;
    if (p->b_dtype == TRILLIM_FP32 && p->b_col_stride == 1) {
        float sums[32] = {0};
        for (size_t start = 0; start < p->k; start += 256) {
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            size_t end = p->k - start < 256 ? p->k : start + 256;
            for (size_t k = start; k < end; ++k) {
                float value = trillim_load_fp32(task->a, a_base + k * p->a_col_stride);
                size_t offset = k * p->b_row_stride + first;
                for (size_t column = 0; column < last - first; ++column) {
                    sums[column] += value * trillim_load_fp32(task->b, offset + column);
                }
            }
        }
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        memcpy((uint8_t *)task->output + (row * p->n + first) * sizeof(float),
               sums, (last - first) * sizeof(float));
        return TRILLIM_OK;
    }
    if (p->b_dtype == TRILLIM_FP32) {
        for (size_t column = first; column < last; ++column) {
            float sum = 0.0f;
            for (size_t start = 0; start < p->k; start += 1024) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t end = p->k - start < 1024 ? p->k : start + 1024;
                for (size_t k = start; k < end; ++k) {
                    sum += trillim_load_fp32(task->a, a_base + k * p->a_col_stride) *
                           trillim_load_fp32(task->b, k * p->b_row_stride + column * p->b_col_stride);
                }
            }
            if (task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            TrillimValue result = {.f = sum};
            trillim_store(task->output, row * p->n + column, p->out_dtype, result);
        }
        return TRILLIM_OK;
    }
    if (p->b_dtype == TRILLIM_BF16) {
        return trillim_native_fp32_bf16(task, a_base, row * p->n, first, last);
    }
    return trillim_native_fp32_fp16(task, a_base, row * p->n, first, last);
}

static TrillimStatus scalar_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    const MatmulParameters *p = parameters;
    MatmulTask task = {outputs[0], inputs[0], inputs[1], p, context};
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t tiles = p->n / 32 + (p->n % 32 != 0);
    size_t tasks = p->m * tiles;
    return tasks == 1 ? scalar_linear_project(&task, 0, 0)
                      : context->thread_pool->run(context->thread_pool->context, tasks, scalar_linear_project, &task);
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    MatmulParameters projection;
    uint32_t bias_dtype;
} Parameters;

static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(Parameters, projection.batches), NULL},
        {"m", "size_t", offsetof(Parameters, projection.m), NULL},
        {"n", "size_t", offsetof(Parameters, projection.n), NULL},
        {"k", "size_t", offsetof(Parameters, projection.k), NULL},
        {"rank", "size_t", offsetof(Parameters, projection.rank), NULL},
        {"batch_shape", "size_t", offsetof(Parameters, projection.batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(Parameters, projection.a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(Parameters, projection.b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(Parameters, projection.a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(Parameters, projection.a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Parameters, projection.b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(Parameters, projection.b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, projection.a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, projection.b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(Parameters, projection.out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(Parameters, projection.a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Parameters, projection.a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Parameters, projection.b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Parameters, projection.b_storage_columns), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
};

enum { FIELD_COUNT = sizeof(fields) / sizeof(fields[0]) };
/* Availability may differ between compilation and execution. */
#define OPTIONAL_WORKSPACE (32u * 1024u * 1024u)

static TrillimStatus prepare(const Parameters *parameters, size_t *inputs, size_t *outputs,
                             size_t *scratch, TrillimKernel *projection) {
    const MatmulParameters *p = &parameters->projection;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (p->batches != 1 || p->rank != 0 || p->a_dtype != TRILLIM_FP32 ||
        !trillim_float(p->b_dtype) || !trillim_float(parameters->bias_dtype) ||
        p->out_dtype != TRILLIM_FP32) {
        return TRILLIM_ERROR;
    }
    if (p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / sizeof(float) / p->n) {
        return TRILLIM_ERROR;
    }
    size_t a_bytes = 0, b_bytes = 0;
    if (p->k != 0) {
        size_t b_itemsize = trillim_itemsize(p->b_dtype);
        if (p->a_storage_columns == 0 || p->b_storage_columns == 0 ||
            p->a_storage_rows > SIZE_MAX / sizeof(float) / p->a_storage_columns ||
            p->b_storage_rows > SIZE_MAX / b_itemsize / p->b_storage_columns) {
            return TRILLIM_ERROR;
        }
        size_t a_count = p->a_storage_rows * p->a_storage_columns;
        size_t b_count = p->b_storage_rows * p->b_storage_columns;
        if (!matrix_span(0, p->m, p->k, p->a_row_stride, p->a_col_stride, a_count) ||
            !matrix_span(0, p->k, p->n, p->b_row_stride, p->b_col_stride, b_count)) {
            return TRILLIM_ERROR;
        }
        a_bytes = a_count * sizeof(float);
        b_bytes = b_count * b_itemsize;
    }
    size_t workspace;
    *projection = blas_requirements(p, inputs, outputs, &workspace) == TRILLIM_OK && workspace <= OPTIONAL_WORKSPACE
        ? blas_linear : scalar_linear;
    inputs[0] = a_bytes;
    inputs[1] = b_bytes;
    inputs[2] = p->n * trillim_itemsize(parameters->bias_dtype);
    outputs[0] = p->m * p->n * sizeof(float);
    *scratch = OPTIONAL_WORKSPACE;
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    TrillimKernel projection;
    return prepare(parameters, inputs, outputs, scratch, &projection);
}

static float silu(float value) {
    float e = expf(-fabsf(value));
    float denominator = 1.0f + e;
    float sigmoid = value >= 0.0f ? 1.0f / denominator : e / denominator;
    return value * sigmoid;
}


typedef struct {
    float *output;
    const void *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus epilogue(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t total = task->p->projection.m * task->p->projection.n;
    size_t begin = tile * 1024;
    size_t end = total - begin < 1024 ? total : begin + 1024;
    while (begin < end) {
        size_t column = begin % task->p->projection.n;
        size_t count = task->p->projection.n - column;
        if (count > end - begin) {
            count = end - begin;
        }
        size_t lane = 0;

        for (; lane < count; ++lane) {
            float value = task->output[begin + lane] + trillim_load(task->bias, column + lane, task->p->bias_dtype).f;
            task->output[begin + lane] = silu(value);
        }
        begin += count;
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_linear_silu(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t input_spans[3], output_spans[1], scratch_span;
    TrillimKernel projection;
    TrillimStatus status = prepare(parameters, input_spans, output_spans, &scratch_span, &projection);
    if (status != TRILLIM_OK || p->projection.m == 0 || p->projection.n == 0) {
        return status;
    }
    status = projection(outputs, inputs, scratch, &p->projection, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    Task task = {outputs[0], inputs[2], p, context};
    size_t count = p->projection.m * p->projection.n;
    size_t tiles = count / 1024 + (count % 1024 != 0);
    if (count <= 4096) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            status = epilogue(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, epilogue, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_linear_silu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _DTYPE_IDS\n"
           "\n"
           "def bind(c):\n"
           "    x, weight, bias = (c.inputs[name] for name in ('x', 'weight', 'bias'))\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or len(weight.shape) != 2 or x.shape[-1] != weight.shape[1]:\n"
           "        raise ValueError('Linear-SiLU requires compatible channel and matrix dimensions')\n"
           "    if x.dtype != 'fp32' or weight.dtype not in {'fp16', 'bf16', 'fp32'} or bias.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype != 'fp32':\n"
           "        raise ValueError('Linear-SiLU requires FP32 activations with floating weights and bias')\n"
           "    if bias.shape != weight.shape[:1] or output.shape != (*x.shape[:-1], weight.shape[0]):\n"
           "        raise ValueError('Linear-SiLU requires a channel bias and the projected output shape')\n"
           "    for value in (bias, output):\n"
           "        if (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('Linear-SiLU requires contiguous bias and output')\n"
           "    return dict(_linear(c), bias_dtype=_DTYPE_IDS[bias.dtype])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Silu(Linear(x, weight, bias=bias)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_linear_silu_parameters(void) {
    static const size_t input_alignments[] = {4, 4, 4}, output_alignments[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), FIELD_COUNT, fields,
        3, 1, requirements, NULL, input_alignments, output_alignments, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_linear_silu_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
