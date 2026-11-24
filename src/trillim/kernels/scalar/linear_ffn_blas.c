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
typedef void (*VectorExp)(float *, const float *, const int *);
static VectorExp vector_exp;
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
        static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            HMODULE library = LoadLibraryA(names[i]);
            if (library != NULL) {
                FARPROC address = GetProcAddress(library, "cblas_sgemm");
                if (address != NULL) {
                    _Static_assert(sizeof(sgemm) == sizeof(address), "function pointer size");
                    memcpy(&sgemm, &address, sizeof(sgemm));
#ifdef __APPLE__
                    void *exponential = dlsym(library, "vvexpf");
                    if (exponential != NULL) {
                        memcpy(&vector_exp, &exponential, sizeof(vector_exp));
                    }
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
        static const char *const names[] = {"libblas.so.3", "libopenblas.so.0", "libblas.so"};
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
                    void *exponential = dlsym(library, "vvexpf");
                    if (exponential != NULL) {
                        memcpy(&vector_exp, &exponential, sizeof(vector_exp));
                    }
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

_Static_assert(BLOCK_K >= BLOCK_M && BLOCK_K >= BLOCK_N, "activation workspace must fit the row tile");

typedef struct {
    void *output;
    const void *a;
    const void *b;
    const void *up;
    const MatmulParameters *p;
    const TrillimKernelContext *context;
} GluTask;

static bool blas_shape(const MatmulParameters *p) {
    return p->n <= INT32_MAX && p->k != 0 && p->n != 0 &&
        (p->m > 4096 / p->n || p->m * p->n > 4096 / p->k);
}

static bool reserve_blas(const MatmulParameters *p) {
    return p->k != 0 && p->n != 0 &&
        (p->m > 4096 / p->n || p->m * p->n > 4096 / p->k);
}

static bool use_blas(const MatmulParameters *p) {
    return blas_shape(p) && initialize_blas();
}

static TrillimStatus glu_requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const MatmulParameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->b_dtype) ||
        p->out_dtype != TRILLIM_FP32 || p->rank != 0 || p->batches != 1 ||
        p->a_col_stride != 1 || p->b_row_stride != 1 || p->b_col_stride != p->k ||
        (p->a_dtype == p->b_dtype && p->a_dtype != TRILLIM_FP32)) {
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
    inputs[2] = inputs[1];
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    /* Reserve optional-library workspace independently of host availability. */
    if (reserve_blas(p)) {
        (void)initialize_blas();
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / 64 + (p->n % 64 != 0);
        size_t count = p->batches * rows * columns;
        *scratch = (count < LANES ? count : LANES) * WORKSPACE * sizeof(float);
    }
    return TRILLIM_OK;
}

static inline float activate(float gate, float up) {
    float e = expf(-fabsf(gate));
    float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return (gate * sigmoid) * up;
}

static inline void store(const GluTask *t, size_t index, float gate, float up) {
    ((float *)t->output)[index] = activate(gate, up);
}

static TrillimStatus glu_small_project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const GluTask *t = opaque;
    const MatmulParameters *p = t->p;
    size_t columns = p->n / 32 + (p->n % 32 != 0);
    size_t row = index / columns;
    size_t first = index % columns * 32;
    size_t last = p->n - first < 32 ? p->n : first + 32;
    for (size_t column = first; column < last; ++column) {
        float gate = 0.0f, up = 0.0f;
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            float x = load(t->a, row * p->a_row_stride + k, p->a_dtype);
            size_t weight = column * p->b_col_stride + k;
            gate = fmaf(x, load(t->b, weight, p->b_dtype), gate);
            up = fmaf(x, load(t->up, weight, p->b_dtype), up);
        }
        store(t, row * p->n + column, gate, up);
    }
    return TRILLIM_OK;
}

typedef struct {
    GluTask task;
    float *scratch;
    size_t count;
    size_t lanes;
} BlasGluTask;

static TrillimStatus glu_project_impl(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const BlasGluTask *job = opaque;
    const GluTask *t = &job->task;
    const MatmulParameters *p = t->p;
    size_t width = block_n(p);
    size_t depth = block_k(p);
    size_t columns = p->n / width + (p->n % width != 0);
    size_t row_tiles = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
    float *a_pack = job->scratch + index * WORKSPACE;
    float *b_pack = a_pack + BLOCK_M * BLOCK_K;
    float *c_pack = b_pack + BLOCK_K * BLOCK_N;
    for (size_t tile = index; tile < job->count; tile += job->lanes) {
        if (t->context->is_cancelled(t->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t batch = tile / columns / row_tiles;
        size_t row = tile / columns % row_tiles * BLOCK_M;
        size_t column = tile % columns * width;
        size_t m = p->m - row < BLOCK_M ? p->m - row : BLOCK_M;
        size_t n = p->n - column < width ? p->n - column : width;
        size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
        size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) + column * p->b_col_stride;
        size_t output = (batch * p->m + row) * p->n + column;
        float *c = (float *)t->output + output;
        int ldc = (int)p->n;
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
            int a_trans = 111, lda = (int)k;
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
            for (size_t projection = 0; projection < 2; ++projection) {
                const void *weight = projection == 0 ? t->b : t->up;
                const float *b = b_pack;
                int b_trans = 111, ldb = (int)n;
                if (p->b_dtype == TRILLIM_FP32 && p->b_col_stride == 1 && p->b_row_stride >= p->n && p->b_row_stride <= INT32_MAX) {
                    b = (const float *)weight + b_index;
                    ldb = (int)p->b_row_stride;
                } else if (p->b_dtype == TRILLIM_FP32 && p->b_row_stride == 1 && p->b_col_stride >= p->k && p->b_col_stride <= INT32_MAX) {
                    b = (const float *)weight + b_index;
                    b_trans = 112;
                    ldb = (int)p->b_col_stride;
                } else if (p->b_row_stride == 1) {
                    status = pack(weight, p->b_dtype, b_index, n, k, p->b_col_stride, 1, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                    b_trans = 112;
                    ldb = (int)k;
                } else {
                    status = pack(weight, p->b_dtype, b_index, k, n, p->b_row_stride, p->b_col_stride, b_pack, t->context);
                    if (status != TRILLIM_OK) {
                        return status;
                    }
                }
                float *destination = projection == 0 ? c : c_pack;
                int destination_stride = projection == 0 ? ldc : (int)n;
                sgemm(101, a_trans, b_trans, (int)m, (int)n, (int)k, 1.0f,
                      a, lda, b, ldb, first == 0 ? 0.0f : 1.0f, destination, destination_stride);
            }
        }
        if (vector_exp != NULL) {
            for (size_t r = 0; r < m; ++r) {
                for (size_t col = 0; col < n; ++col) {
                    b_pack[r * n + col] = -fabsf(c[r * (size_t)ldc + col]);
                }
            }
            int count = (int)(m * n);
            vector_exp(a_pack, b_pack, &count);
            for (size_t r = 0; r < m; ++r) {
                for (size_t col = 0; col < n; ++col) {
                    float gate = c[r * (size_t)ldc + col];
                    float up = c_pack[r * n + col];
                    float e = a_pack[r * n + col];
                    if (fabsf(gate) > 80 || fabsf(up) > 1e30f || !isfinite(gate) || !isfinite(up)) {
                        e = expf(-fabsf(gate));
                    }
                    float probability = gate >= 0 ? 1 / (1 + e) : e / (1 + e);
                    ((float *)t->output)[output + r * p->n + col] = (gate * probability) * up;
                }
            }
        } else {
            for (size_t r = 0; r < m; ++r) {
                for (size_t col = 0; col < n; ++col) {
                    store(t, output + r * p->n + col, c[r * (size_t)ldc + col], c_pack[r * n + col]);
                }
            }
        }
    }
    return t->context->is_cancelled(t->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus glu_project(void *opaque, size_t index, size_t worker) {
#ifdef __APPLE__
    if (get_threading != NULL && set_threading != NULL) {
        unsigned previous = get_threading();
        if (previous != 1 && set_threading(1) == 0) {
            TrillimStatus status = glu_project_impl(opaque, index, worker);
            (void)set_threading(previous);
            return status;
        }
    }
#endif
    return glu_project_impl(opaque, index, worker);
}

static TrillimStatus blas_glu(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    const MatmulParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    GluTask task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    if (use_blas(p)) {
        size_t rows = p->m / BLOCK_M + (p->m % BLOCK_M != 0);
        size_t columns = p->n / block_n(p) + (p->n % block_n(p) != 0);
        size_t count = rows * columns;
        size_t lanes = count < LANES ? count : LANES;
        BlasGluTask job = {task, scratch, count, lanes};
        return context->thread_pool->run(context->thread_pool->context, lanes, glu_project, &job);
    }
    size_t columns = p->n / 32 + (p->n % 32 != 0);
    size_t count = p->m * columns;
    return count == 1 ? glu_small_project(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, count, glu_small_project, &task);
}

typedef struct {
    size_t rows;
    size_t input_width;
    size_t hidden_width;
    size_t output_width;
    size_t x_row_stride;
    size_t x_storage_rows;
    size_t x_storage_columns;
    uint32_t x_dtype;
    uint32_t gate_dtype;
    uint32_t down_dtype;
} Parameters;

typedef struct {
    size_t inputs[4];
    size_t output;
    size_t hidden_span;
    size_t scratch;
} Layout;

static TrillimStatus prepare(const Parameters *p, MatmulParameters *glu, MatmulParameters *dense, Layout *layout) {
    if (p->rows == 0 || p->output_width == 0) {
        memset(layout, 0, sizeof(*layout));
        return TRILLIM_OK;
    }
    *glu = (MatmulParameters){
        .batches = 1, .m = p->rows, .n = p->hidden_width, .k = p->input_width,
        .a_row_stride = p->x_row_stride, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->input_width,
        .a_dtype = p->x_dtype, .b_dtype = p->gate_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->x_storage_rows, .a_storage_columns = p->x_storage_columns,
        .b_storage_rows = p->hidden_width, .b_storage_columns = p->input_width
    };
    *dense = (MatmulParameters){
        .batches = 1, .m = p->rows, .n = p->output_width, .k = p->hidden_width,
        .a_row_stride = p->hidden_width, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->hidden_width,
        .a_dtype = TRILLIM_FP32, .b_dtype = p->down_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->rows, .a_storage_columns = p->hidden_width,
        .b_storage_rows = p->output_width, .b_storage_columns = p->hidden_width
    };
    size_t glu_inputs[3], glu_outputs[1], glu_scratch;
    size_t dense_inputs[2], dense_outputs[1], dense_scratch;
    TrillimStatus status = glu_requirements(glu, glu_inputs, glu_outputs, &glu_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = blas_requirements(dense, dense_inputs, dense_outputs, &dense_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (glu_outputs[0] != dense_inputs[0] || glu_outputs[0] > SIZE_MAX - 63) {
        return TRILLIM_ERROR;
    }
    layout->hidden_span = (glu_outputs[0] + 63) / 64 * 64;
    size_t workspace = glu_scratch > dense_scratch ? glu_scratch : dense_scratch;
    if (workspace > SIZE_MAX - layout->hidden_span) {
        return TRILLIM_ERROR;
    }
    for (size_t i = 0; i < 3; ++i) {
        layout->inputs[i] = glu_inputs[i];
    }
    layout->inputs[3] = dense_inputs[1];
    layout->output = dense_outputs[0];
    layout->scratch = layout->hidden_span + workspace;
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    for (size_t i = 0; i < 4; ++i) {
        inputs[i] = 0;
    }
    outputs[0] = *scratch = 0;
    MatmulParameters glu, dense;
    Layout layout;
    TrillimStatus status = prepare(parameters, &glu, &dense, &layout);
    if (status != TRILLIM_OK) {
        return status;
    }
    memcpy(inputs, layout.inputs, sizeof(layout.inputs));
    outputs[0] = layout.output;
    *scratch = layout.scratch;
    return TRILLIM_OK;
}

static bool can_alias(const void *parameters, size_t output, size_t input) {
    (void)parameters;
    /* The projections finish reading x before the down projection writes output. */
    return output == 0 && input == 0;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_blas_ffn(void *const *outputs, const void *const *inputs, void *scratch,
                                    const void *parameters, const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->rows == 0 || p->output_width == 0) {
        return TRILLIM_OK;
    }
    MatmulParameters glu, dense;
    Layout layout;
    TrillimStatus status = prepare(p, &glu, &dense, &layout);
    if (status != TRILLIM_OK) {
        return status;
    }
    void *hidden = scratch;
    void *workspace = layout.hidden_span != 0 ? (uint8_t *)scratch + layout.hidden_span : scratch;
    void *glu_outputs[] = {hidden};
    const void *glu_inputs[] = {inputs[0], inputs[1], inputs[2]};
    status = blas_glu(glu_outputs, glu_inputs, workspace, &glu, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const void *dense_inputs[] = {hidden, inputs[3]};
    return blas_linear(outputs, dense_inputs, workspace, &dense, context);
}

TRILLIM_KERNEL const char *trillim_scalar_blas_ffn_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _constant_expression, _DTYPE_IDS\n"
           "from trillim.compiler.kernels.kernel_dataclasses import _KernelBindingContext\n"
           "from trillim.compiler.passes.infer import _promote\n"
           "def bind(c):\n"
           "    x, gate, up, down = (c.inputs[name] for name in ('x', 'gate', 'up', 'down'))\n"
           "    if x.dtype not in {'fp16', 'bf16', 'fp32'} or gate.dtype != up.dtype or gate.shape != up.shape or len(gate.shape) != 2 or len(down.shape) != 2 or down.shape[1] != gate.shape[0] or _promote(x.dtype, gate.dtype) != 'fp32' or c.outputs[0].dtype != 'fp32':\n"
           "        raise ValueError('fused FFN requires compatible floating projections to FP32')\n"
           "    for weight in (gate, up, down):\n"
           "        if weight.dtype not in {'fp16', 'bf16', 'fp32'} or (weight.strides or _contiguous_strides(weight.shape)) != _contiguous_strides(weight.shape):\n"
           "            raise ValueError('fused FFN requires canonical floating weights')\n"
           "    base = _linear(_KernelBindingContext({'x': x, 'weight': gate}, c.outputs, c.attrs))\n"
           "    if base['a_col_stride'] != _constant_expression(1):\n"
           "        raise ValueError('fused FFN requires contiguous input channels')\n"
           "    return {'rows': base['m'], 'input_width': base['k'], 'hidden_width': base['n'], 'output_width': down.shape[0], 'x_row_stride': base['a_row_stride'], 'x_storage_rows': base['a_storage_rows'], 'x_storage_columns': base['a_storage_columns'], 'x_dtype': base['a_dtype'], 'gate_dtype': base['b_dtype'], 'down_dtype': _DTYPE_IDS[down.dtype]}\n"
           "def pattern(x: Tensor, gate: Tensor, up: Tensor, down: Tensor, *, up_bias=None, down_bias=None, gate_bias=None, activation=Silu) -> tuple[Tensor, ...]:\n"
           "    if gate is None or up_bias is not None or down_bias is not None or gate_bias is not None or activation is not Silu:\n"
           "        raise ValueError('fused FFN requires an unbiased SiLU gate')\n"
           "    return (FFN(x, up=up, down=down, gate=gate, activation=activation),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_blas_ffn_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"input_width", "size_t", offsetof(Parameters, input_width), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"output_width", "size_t", offsetof(Parameters, output_width), NULL},
        {"x_row_stride", "size_t", offsetof(Parameters, x_row_stride), NULL},
        {"x_storage_rows", "size_t", offsetof(Parameters, x_storage_rows), NULL},
        {"x_storage_columns", "size_t", offsetof(Parameters, x_storage_columns), NULL},
        {"x_dtype", "uint32_t", offsetof(Parameters, x_dtype), NULL},
        {"gate_dtype", "uint32_t", offsetof(Parameters, gate_dtype), NULL},
        {"down_dtype", "uint32_t", offsetof(Parameters, down_dtype), NULL},
    };
    static const size_t input_alignments[] = {1, 1, 1, 1}, output_alignments[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, can_alias, input_alignments, output_alignments, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_blas_ffn_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM | TRILLIM_CAP_BLAS
    };
    return &capabilities;
}
