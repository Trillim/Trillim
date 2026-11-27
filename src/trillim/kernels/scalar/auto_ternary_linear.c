#include "common.h"
#include <stdatomic.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <fenv.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#endif

#define LOWBIT_WIDTH 128
#define LOWBIT_TYPE TRILLIM_GRP_TERNARY
#define LOWBIT_BYTES 32

typedef struct {
    size_t tokens, width, outputs;
    uint32_t weight_dtype, output_dtype, input_dtype;
} Parameters;

typedef struct {
    const void *input;
    const uint8_t *weight;
    void *output;
    int8_t *codes;
    float *scales;
    atomic_bool ordinary;
    bool prepared;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static inline __attribute__((always_inline)) float load_activation(const void *input, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) {
        float value;
        memcpy(&value, (const uint8_t *)input + index * sizeof(value), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, (const uint8_t *)input + index * sizeof(bits), sizeof(bits));
    return dtype == TRILLIM_BF16 ? trillim_bf16(bits) : trillim_fp16(bits);
}

static inline __attribute__((always_inline)) void store_projection(void *output, size_t index, uint32_t dtype, float value) {
    if (dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)output + index * sizeof(value), &value, sizeof(value));
    } else {
        uint16_t bits = dtype == TRILLIM_BF16 ? trillim_to_bf16(value) : trillim_to_fp16(value);
        memcpy((uint8_t *)output + index * sizeof(bits), &bits, sizeof(bits));
    }
}

#if defined(__APPLE__)
enum { BLAS_ROWS = 512, BLAS_DECODE_COLUMNS = 16 };
static size_t blas_rows(const Parameters *p) {
    if (p->tokens < 64 || p->outputs < 1024 || p->width < 512 ||
        p->width < (1 << 20) / p->outputs || p->width % LOWBIT_WIDTH != 0) {
        return 0;
    }
    size_t rows = (1 << 23) / p->width;
    if (rows < 32) {
        return 0;
    }
    if (rows > BLAS_ROWS) {
        rows = BLAS_ROWS;
    }
    return p->tokens < rows ? p->tokens : rows;
}

static size_t blas_workspace(const Parameters *p) {
    size_t rows = blas_rows(p);
    if (!rows || p->outputs > (1 << 27) / p->width) {
        return 0;
    }
    return (rows * p->width + p->outputs * (p->width + rows)) * sizeof(float);
}

static size_t blas_reservation(const Parameters *p) {
    if (p->tokens < 64 || p->outputs < 1024 || p->width < 512) {
        return 0;
    }
    /* Bounds also cover smaller aligned widths and shapes below the work limit. */
    size_t rows = p->tokens < BLAS_ROWS ? p->tokens : BLAS_ROWS;
    size_t weights = p->outputs > (1 << 27) / p->width ? (1 << 27) : p->outputs * p->width;
    size_t width = p->width < (1 << 18) ? p->width : (1 << 18);
    size_t columns = p->outputs < (1 << 18) ? p->outputs : (1 << 18);
    size_t activation = rows * width < (1 << 23) ? rows * width : (1 << 23);
    return (weights + activation + rows * columns) * sizeof(float);
}

#endif

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (p->weight_dtype != LOWBIT_TYPE || !trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) || !p->width) {
        return TRILLIM_ERROR;
    }
    size_t in_bytes = trillim_itemsize(p->input_dtype), out_bytes = trillim_itemsize(p->output_dtype);
    if (p->tokens > SIZE_MAX / p->width ||
        (p->outputs && p->tokens > SIZE_MAX / out_bytes / p->outputs)) {
        return TRILLIM_ERROR;
    }
    size_t count = p->tokens * p->width;
    if (count > SIZE_MAX / in_bytes || count > SIZE_MAX - 3 ||
        p->tokens > (SIZE_MAX - ((count + 3) & ~(size_t)3)) / sizeof(float) ||
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, &inputs[1])) {
        return TRILLIM_ERROR;
    }
    inputs[0] = count * in_bytes;
    outputs[0] = p->tokens * p->outputs * out_bytes;
    *scratch = ((count + 3) & ~(size_t)3) + p->tokens * sizeof(float);
#if defined(__APPLE__)
    size_t workspace = blas_reservation(p);
    if (*scratch > SIZE_MAX - workspace) {
        return TRILLIM_ERROR;
    }
    *scratch += workspace;
#endif
    return TRILLIM_OK;
}

/* Keep direct decode preparation inline while allowing parallel row callbacks. */
static inline __attribute__((always_inline)) TrillimStatus prepare_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    Task *task = opaque;
    const Parameters *p = task->p;
    float maximum = 1e-5f;
    for (size_t k = 0; k < p->width; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = load_activation(task->input, row * p->width + k, p->input_dtype);
        if (!isfinite(x)) {
            atomic_store_explicit(&task->ordinary, false, memory_order_relaxed);
            return TRILLIM_OK;
        }
        maximum = fmaxf(maximum, fabsf(x));
    }
    float scale = 127.0f / maximum;
    task->scales[row] = scale;
    for (size_t k = 0; k < p->width; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = load_activation(task->input, row * p->width + k, p->input_dtype) * scale;
        float magnitude = fminf(fabsf(x), 127.0f);
        int32_t code = (int32_t)magnitude;
        float fraction = magnitude - (float)code;
        if (fraction > 0.5f || (fraction == 0.5f && (code & 1))) {
            ++code;
        }
        task->codes[row * p->width + k] = (int8_t)(x < 0.0f ? -code : code);
    }
    return TRILLIM_OK;
}

static TrillimStatus project_single(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = index / tiles, first = index % tiles * 16;
    size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    for (size_t column = first; column < end; ++column) {
        float total = 0.0f;
        if (task->prepared) {
            for (size_t g = 0; g < groups; ++g) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t block = column * groups + g;
                const uint8_t *packed = task->weight + block * 32;
                size_t count = p->width - g * 128 < 128 ? p->width - g * 128 : 128;
                int32_t sum = 0;
                for (size_t k = 0; k < count; ++k) {
                    int32_t code = (packed[k % 32] >> (6 - 2 * (k / 32))) & 3;
                    if (code == 3) {
                        total = NAN;
                        break;
                    }
                    sum += (code - 1) * task->codes[row * p->width + g * 128 + k];
                }
                uint16_t bits;
                memcpy(&bits, task->weight + p->outputs * groups * 32 + block * 2, sizeof(bits));
                total = fmaf((float)sum, trillim_bf16(bits), total);
            }
            total /= task->scales[row];
        } else {
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float x = load_activation(task->input, row * p->width + k, p->input_dtype);
                float w = trillim_packed_read_float(task->weight, column * p->width + k,
                    p->weight_dtype, p->outputs, p->width);
                total = fmaf(x, w, total);
            }
        }
        store_projection(task->output, row * p->outputs + column, p->output_dtype, total);
    }
    return TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    size_t row = index / tiles * 4, first = index % tiles * 16;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t end = p->outputs - first < 16 ? p->outputs : first + 16;
    size_t groups = p->width / 128 + (p->width % 128 != 0);
    bool ordinary = atomic_load_explicit(&task->ordinary, memory_order_relaxed);
    for (size_t column = first; column < end; ++column) {
        float totals[4] = {0};
        if (ordinary) {
            for (size_t g = 0; g < groups; ++g) {
                if (task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                size_t block = column * groups + g;
                const uint8_t *packed = task->weight + block * LOWBIT_BYTES;
                size_t count = p->width - g * 128 < 128 ? p->width - g * 128 : 128;
                int32_t sums[4] = {0};
                bool invalid = false;
                for (size_t k = 0; k < count; ++k) {
                    int32_t code;
                    if (LOWBIT_TYPE == TRILLIM_GRP_TERNARY) {
                        code = (packed[k % 32] >> (6 - 2 * (k / 32))) & 3;
                        if (code == 3) {
                            invalid = true;
                            break;
                        }
                        --code;
                    } else {
                        code = ((packed[k / 8] >> (7 - k % 8)) & 1) ? 1 : -1;
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        sums[r] += code * task->codes[(row + r) * p->width + g * 128 + k];
                    }
                }
                uint16_t bits;
                memcpy(&bits, task->weight + p->outputs * groups * LOWBIT_BYTES + block * 2, sizeof(bits));
                for (size_t r = 0; r < rows; ++r) {
                    totals[r] = invalid ? NAN : fmaf((float)sums[r], trillim_bf16(bits), totals[r]);
                }
            }
            for (size_t r = 0; r < rows; ++r) {
                totals[r] /= task->scales[row + r];
            }
        } else {
            for (size_t k = 0; k < p->width; ++k) {
                if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                    return TRILLIM_CANCELLED;
                }
                float w = trillim_packed_read_float(task->weight, column * p->width + k,
                    p->weight_dtype, p->outputs, p->width);
                for (size_t r = 0; r < rows; ++r) {
                    float x = load_activation(task->input, (row + r) * p->width + k, p->input_dtype);
                    totals[r] = fmaf(x, w, totals[r]);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            store_projection(task->output, (row + r) * p->outputs + column, p->output_dtype, totals[r]);
        }
    }
    return TRILLIM_OK;
}

#if defined(__APPLE__)

typedef void (*Sgemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
typedef unsigned (*GetThreading)(void);

static Sgemm sgemm;
static GetThreading get_threading;
typedef ssize_t (*MatmulWorkspace)(bool, bool, float, const BNNSNDArrayDescriptor *,
    const BNNSNDArrayDescriptor *, const BNNSNDArrayDescriptor *, const BNNSFilterParameters *);
typedef int (*Matmul)(bool, bool, float, const BNNSNDArrayDescriptor *,
    const BNNSNDArrayDescriptor *, const BNNSNDArrayDescriptor *, void *, const BNNSFilterParameters *);
static MatmulWorkspace matmul_workspace;
static Matmul matmul;

static bool initialize_blas(void) {
    static atomic_uint state;
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&state, &expected, 1, memory_order_acq_rel, memory_order_acquire)) {
        void *library = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LOCAL | RTLD_LAZY);
        if (library != NULL) {
            void *gemm = dlsym(library, "cblas_sgemm$NEWLAPACK");
            void *getter = dlsym(library, "BLASGetThreading");
            if (gemm == NULL) {
                gemm = dlsym(library, "cblas_sgemm");
            }
            if (gemm != NULL && getter != NULL) {
                memcpy(&sgemm, &gemm, sizeof(sgemm));
                memcpy(&get_threading, &getter, sizeof(get_threading));
                void *size = dlsym(library, "BNNSMatMulWorkspaceSize");
                void *multiply = dlsym(library, "BNNSMatMul");
                if (size != NULL && multiply != NULL) {
                    memcpy(&matmul_workspace, &size, sizeof(matmul_workspace));
                    memcpy(&matmul, &multiply, sizeof(matmul));
                }
            } else {
                dlclose(library);
            }
        }
        atomic_store_explicit(&state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&state, memory_order_acquire) == 1) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return sgemm != NULL;
}

typedef struct {
    const Parameters *p;
    const TrillimKernelContext *context;
    const int8_t *codes;
    const float *scales;
    const uint8_t *weights, *weight_scales;
    void *output;
    float *activation, *expanded_weights, *accumulator;
    size_t row, rows;
    atomic_bool ordinary;
    bool compact;
    void *workspace;
    size_t workspace_size;
} BlasTask;

static bool default_fpcr(void) {
    return fegetround() == FE_TONEAREST;
}

static TrillimStatus widen_activation(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    size_t width = job->p->width;
    const int8_t *codes = job->codes + (job->row + row) * width;
    for (size_t k = 0; k < width; ++k) {
        if ((k & 1023) == 0 && job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (job->compact) {
            ((uint16_t *)job->activation)[row * width + k] = trillim_to_bf16((float)codes[k]);
        } else {
            job->activation[row * width + k] = (float)codes[k];
        }
    }
    return TRILLIM_OK;
}

static bool widen_weight_row(const BlasTask *job, size_t block, size_t groups, void *output) {
    for (size_t g = 0; g < groups; ++g) {
        uint16_t bits;
        memcpy(&bits, job->weight_scales + (block + g) * 2, 2);
        uint16_t magnitude = bits & 0x7fff;
        if (magnitude && (magnitude < 0x1000 || magnitude > 0x6b80)) {
            return false;
        }
        const uint8_t *packed = job->weights + (block + g) * LOWBIT_BYTES;
        for (size_t k = 0; k < 128; ++k) {
            uint16_t value;
            if (LOWBIT_TYPE == TRILLIM_GRP_TERNARY) {
                unsigned code = (packed[k % 32] >> (6 - 2 * (k / 32))) & 3;
                if (code == 3) {
                    return false;
                }
                value = code == 1 ? 0 : (uint16_t)(bits ^ (code == 0 ? 0x8000 : 0));
            } else {
                value = (uint16_t)(bits ^ (((packed[k / 8] >> (7 - k % 8)) & 1) ? 0 : 0x8000));
            }
            if (job->compact) {
                ((uint16_t *)output)[g * 128 + k] = value;
            } else {
                ((float *)output)[g * 128 + k] = trillim_bf16(value);
            }
        }
    }
    return true;
}

static TrillimStatus prepare_weights(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    BlasTask *job = opaque;
    size_t width = job->p->width, groups = width / LOWBIT_WIDTH;
    size_t begin = tile * BLAS_DECODE_COLUMNS;
    size_t end = job->p->outputs - begin < BLAS_DECODE_COLUMNS ? job->p->outputs : begin + BLAS_DECODE_COLUMNS;
    if (!default_fpcr()) {
        atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
        return TRILLIM_OK;
    }
    for (size_t c = begin; c < end && atomic_load_explicit(&job->ordinary, memory_order_relaxed); ++c) {
        if (job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (!widen_weight_row(job, c * groups, groups, (uint8_t *)job->expanded_weights + c * width * (job->compact ? sizeof(uint16_t) : sizeof(float)))) {
            atomic_store_explicit(&job->ordinary, false, memory_order_relaxed);
            break;
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus finish_output(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const BlasTask *job = opaque;
    const size_t columns = job->p->outputs;
    if (job->context->is_cancelled(job->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const float *values = job->accumulator + row * columns;
    size_t start = (job->row + row) * columns;
    float scale = job->scales[job->row + row];
    size_t c = 0;
    for (; c < columns; ++c) {
        if ((c & 1023) == 0 && job->context->is_cancelled(job->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        store_projection(job->output, start + c, job->p->output_dtype, values[c] / scale);
    }
    return TRILLIM_OK;
}

static bool blas_thread_limit(size_t workers) {
    const char *limit = getenv("VECLIB_MAXIMUM_THREADS");
    if (limit == NULL || workers < 4 || workers > 32) {
        return false;
    }
    size_t threads = 0;
    for (; *limit != '\0'; ++limit) {
        if (*limit < '0' || *limit > '9') {
            return false;
        }
        threads = threads * 10 + (size_t)(*limit - '0');
        if (threads > workers) {
            return false;
        }
    }
    return threads >= 4;
}

typedef struct {
    const BlasTask *job;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    uint64_t deadline;
    size_t workers, arrived;
    bool declined, done;
} SharedBlas;

static TrillimStatus multiply_blas(const BlasTask *job) {
    const Parameters *p = job->p;
    const TrillimKernelContext *context = job->context;
    // Bound the work between cancellation checks inside a large projection.
    size_t columns = UINT64_C(16000000000) / (job->rows * p->width);
    columns = columns / 128 * 128;
    if (columns < 128) {
        columns = 128;
    }
    for (size_t first = 0; first < p->outputs; first += columns) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->outputs - first < columns ? p->outputs - first : columns;
        if (job->compact) {
            BNNSNDArrayDescriptor a = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {p->width, job->rows}, .data = job->activation, .data_type = BNNSDataTypeBFloat16};
            BNNSNDArrayDescriptor b = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {p->width, count}, .data = (uint16_t *)job->expanded_weights + first * p->width,
                .data_type = BNNSDataTypeBFloat16};
            BNNSNDArrayDescriptor c = {.layout = BNNSDataLayoutRowMajorMatrix,
                .size = {count, job->rows}, .stride = {1, p->outputs},
                .data = job->accumulator + first, .data_type = BNNSDataTypeFloat32};
            BNNSFilterParameters options = {.n_threads = context->thread_pool->worker_count};
            ssize_t needed = matmul_workspace(false, true, 1.0f, &a, &b, &c, &options);
            if (needed < 0 || (size_t)needed > job->workspace_size ||
                matmul(false, true, 1.0f, &a, &b, &c, job->workspace, &options) != 0) {
                return TRILLIM_ERROR;
            }
        } else {
            sgemm(101, 111, 112, (int)job->rows, (int)count, (int)p->width, 1.0f,
                job->activation, (int)p->width, job->expanded_weights + first * p->width, (int)p->width,
                0.0f, job->accumulator + first, (int)p->outputs);
        }
    }
    return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus run_shared_blas(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    SharedBlas *shared = opaque;
    const TrillimKernelContext *context = shared->job->context;
    pthread_mutex_lock(&shared->mutex);
    uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (!now || now >= shared->deadline || (tile == 0 && (!default_fpcr() || (!shared->job->compact && get_threading() != 0)))) {
        shared->declined = true;
    }
    if (++shared->arrived == shared->workers || shared->declined) {
        pthread_cond_broadcast(&shared->changed);
    }
    // The pool may provide fewer concurrent workers than it advertises.
    while (!shared->declined && shared->arrived < shared->workers) {
        now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (!now || now >= shared->deadline || context->is_cancelled(context->cancellation_context)) {
            shared->declined = true;
        } else {
            const struct timespec remaining = {.tv_sec = 0, .tv_nsec = (long)(shared->deadline - now)};
            if (pthread_cond_timedwait_relative_np(&shared->changed, &shared->mutex, &remaining) != 0 &&
                shared->arrived < shared->workers) {
                shared->declined = true;
            }
        }
        if (shared->declined) {
            pthread_cond_broadcast(&shared->changed);
        }
    }
    if (shared->declined) {
        pthread_mutex_unlock(&shared->mutex);
        return context->is_cancelled(context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
    }
    if (tile != 0) {
        while (!shared->done) {
            pthread_cond_wait(&shared->changed, &shared->mutex);
        }
        pthread_mutex_unlock(&shared->mutex);
        return TRILLIM_OK;
    }
    pthread_mutex_unlock(&shared->mutex);
    const TrillimStatus status = multiply_blas(shared->job);
    pthread_mutex_lock(&shared->mutex);
    shared->done = true;
    pthread_cond_broadcast(&shared->changed);
    pthread_mutex_unlock(&shared->mutex);
    return status;
}

static TrillimStatus try_shared_blas(const BlasTask *job, bool *completed) {
    const TrillimThreadPool *pool = job->context->thread_pool;
    uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (!now) {
        *completed = false;
        return TRILLIM_OK;
    }
    SharedBlas shared = {
        .job = job, .mutex = PTHREAD_MUTEX_INITIALIZER, .changed = PTHREAD_COND_INITIALIZER,
        .deadline = now + UINT64_C(2000000), .workers = pool->worker_count,
    };
    const TrillimStatus status = pool->run(pool->context, pool->worker_count, run_shared_blas, &shared);
    pthread_cond_destroy(&shared.changed);
    pthread_mutex_destroy(&shared.mutex);
    *completed = status == TRILLIM_OK && shared.done;
    return status;
}

static TrillimStatus try_blas(
    const Parameters *p, const TrillimKernelContext *context, void *output, const void *weights,
    const int8_t *codes, const float *scales, float *scratch, bool compact, bool *handled
) {
    *handled = false;
    const size_t rows = blas_rows(p);
    const TrillimThreadPool *pool = context->thread_pool;
    if (!blas_workspace(p) || !initialize_blas() || !default_fpcr() ||
        (compact ? (matmul == NULL || pool->worker_count < 4 || pool->worker_count > 32) :
                   !blas_thread_limit(pool->worker_count))) {
        return TRILLIM_OK;
    }
    /* INT8 codes and BF16 weight scales are exact in the compact BF16 operands. */
    float *expanded = scratch + rows * p->width;
    size_t blocks = p->outputs * (p->width / LOWBIT_WIDTH);
    BlasTask job = {
        p, context, codes, scales, weights, (const uint8_t *)weights + blocks * LOWBIT_BYTES,
        output, scratch, expanded, expanded + p->outputs * p->width,
        0, 0, ATOMIC_VAR_INIT(true), compact, NULL, 0,
    };
    if (compact) {
        job.expanded_weights = (float *)((uint16_t *)scratch + rows * p->width);
        job.accumulator = (float *)((uint16_t *)job.expanded_weights + p->outputs * p->width);
        uintptr_t end = (uintptr_t)(job.accumulator + rows * p->outputs);
        uintptr_t aligned = (end + 63) & ~(uintptr_t)63;
        uintptr_t limit = (uintptr_t)scratch + blas_workspace(p);
        if (aligned >= limit) {
            return TRILLIM_OK;
        }
        job.workspace = (void *)aligned;
        job.workspace_size = limit - aligned;
    }
    size_t tiles = p->outputs / BLAS_DECODE_COLUMNS + (p->outputs % BLAS_DECODE_COLUMNS != 0);
    TrillimStatus status = pool->run(pool->context, tiles, prepare_weights, &job);
    if (status != TRILLIM_OK || !atomic_load_explicit(&job.ordinary, memory_order_relaxed)) {
        return status;
    }
    while (job.row < p->tokens) {
        job.rows = p->tokens - job.row < rows ? p->tokens - job.row : rows;
        status = pool->run(pool->context, job.rows, widen_activation, &job);
        if (status != TRILLIM_OK) {
            return status;
        }
        bool completed;
        status = try_shared_blas(&job, &completed);
        if (status != TRILLIM_OK || !completed) {
            return compact && status == TRILLIM_ERROR ? TRILLIM_OK : status;
        }
        status = pool->run(pool->context, job.rows, finish_output, &job);
        if (status != TRILLIM_OK) {
            return status;
        }
        job.row += job.rows;
    }
    *handled = true;
    return TRILLIM_OK;
}
#endif

TRILLIM_KERNEL TrillimStatus trillim_scalar_auto_ternary_linear(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) {
        return TRILLIM_OK;
    }
    size_t sizes[2], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t count = p->tokens * p->width;
    Task task = {inputs[0], inputs[1], outputs[0], scratch,
        (float *)((uint8_t *)scratch + ((count + 3) & ~(size_t)3)), ATOMIC_VAR_INIT(true), false, p, context};
    status = p->tokens == 1 ? prepare_row(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->tokens, prepare_row, &task);
    if (status != TRILLIM_OK) {
        return status;
    }
    size_t tiles = p->outputs / 16 + (p->outputs % 16 != 0);
    if (p->tokens == 1) {
        task.prepared = atomic_load_explicit(&task.ordinary, memory_order_relaxed);
        return context->thread_pool->run(context->thread_pool->context, tiles, project_single, &task);
    }
#if defined(__APPLE__)
    if (atomic_load_explicit(&task.ordinary, memory_order_relaxed)) {
        bool handled;
        status = try_blas(p, context, outputs[0], inputs[1], task.codes, task.scales,
            task.scales + p->tokens, true, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
        status = try_blas(p, context, outputs[0], inputs[1], task.codes, task.scales,
            task.scales + p->tokens, false, &handled);
        if (status != TRILLIM_OK || handled) {
            return status;
        }
    }
#endif
    size_t rows = p->tokens / 4 + (p->tokens % 4 != 0);
    return context->thread_pool->run(context->thread_pool->context, rows * tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_auto_ternary_linear_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides, _storage_shape\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, output = context.inputs['x'], context.inputs['weight'], context.outputs[0]\n"
           "    if weight.dtype != 'grp_ternary' or x.dtype not in {'fp16', 'bf16', 'fp32'} or output.dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires grouped ternary weights and floating activations')\n"
           "    for tensor in (x, weight):\n"
           "        if (tensor.strides or _contiguous_strides(tensor.shape)) != _contiguous_strides(tensor.shape):\n"
           "            raise ValueError('requires contiguous operands')\n"
           "    if _storage_shape(weight) != weight.shape:\n"
           "        raise ValueError('requires a complete packed matrix')\n"
           "    p = _linear(context)\n"
           "    return dict(tokens=p['m'], width=p['k'], outputs=p['n'], weight_dtype=p['b_dtype'], output_dtype=p['out_dtype'], input_dtype=p['a_dtype'])\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, *, bias, dtype=None) -> tuple[Tensor, ...]:\n"
           "    if bias is not None or dtype not in {None, 'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('requires an unbiased floating linear projection')\n"
           "    result = Linear(x, weight)\n"
           "    return (result if dtype is None else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_auto_ternary_linear_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_auto_ternary_linear_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &caps;
}
