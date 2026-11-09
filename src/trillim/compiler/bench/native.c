#define _POSIX_C_SOURCE 200809L
#include "native.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <time.h>
#endif

static volatile sig_atomic_t signal_cancelled;

static void cancel_signal(int signal_number) {
    (void)signal_number;
    signal_cancelled = 1;
}

static void *allocate(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void *address = NULL;
    if (posix_memalign(&address, alignment, size) != 0) {
        return NULL;
    }
    return address;
#endif
}

static void release(void *address) {
#ifdef _WIN32
    _aligned_free(address);
#else
    free(address);
#endif
}

static bool is_cancelled(void *context) {
    TrillimBenchContext *backend = context;
    return signal_cancelled != 0 || atomic_load_explicit(&backend->cancelled, memory_order_relaxed);
}

static void spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

static uint64_t monotonic_nanoseconds(void) {
#ifdef _WIN32
    LARGE_INTEGER now, frequency;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0 ||
        !QueryPerformanceCounter(&now)) {
        return 0;
    }
    uint64_t ticks = (uint64_t)now.QuadPart;
    uint64_t rate = (uint64_t)frequency.QuadPart;
    return (ticks / rate) * UINT64_C(1000000000) +
        (uint64_t)((long double)(ticks % rate) * 1000000000.0L / rate);
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    return (uint64_t)now.tv_sec * 1000000000 + (uint64_t)now.tv_nsec;
#endif
}

static TrillimStatus execute_tasks(
    TrillimBenchWorker *worker, TrillimTask task,
    void *task_context, size_t task_count, bool parallel
) {
    TrillimBenchContext *backend = worker->backend;
    size_t completed = 0;
    TrillimStatus status = TRILLIM_OK;
    while (!is_cancelled(backend)) {
        size_t index = parallel
            ? atomic_fetch_add_explicit(&backend->next_task, 1, memory_order_relaxed)
            : completed;
        if (index >= task_count) {
            break;
        }
        status = task(task_context, index, worker->index);
        ++completed;
        if (status != TRILLIM_OK) {
            atomic_store_explicit(&backend->cancelled, true, memory_order_relaxed);
            break;
        }
    }
    return status == TRILLIM_OK || status == TRILLIM_CANCELLED ? status : TRILLIM_ERROR;
}

static void *worker_main(void *context) {
    TrillimBenchWorker *worker = context;
    TrillimBenchContext *backend = worker->backend;
    size_t generation = 0;
    for (;;) {
        uint64_t start = monotonic_nanoseconds();
        size_t next = atomic_load_explicit(&backend->generation, memory_order_acquire);
        while (!atomic_load_explicit(&backend->shutdown, memory_order_relaxed)
               && generation == next) {
            for (size_t pause = 0; pause < 1024 && generation == next; ++pause) {
                spin_pause();
                next = atomic_load_explicit(&backend->generation, memory_order_acquire);
            }
            if (generation != next) {
                break;
            }
            uint64_t now = monotonic_nanoseconds();
            if (now == 0 || now - start >= 10000000) {
                /* Register and recheck under the dispatcher's wake-up mutex. */
                pthread_mutex_lock(&backend->mutex);
                ++backend->sleepers;
                while (!atomic_load_explicit(&backend->shutdown, memory_order_relaxed)
                       && generation == atomic_load_explicit(
                           &backend->generation, memory_order_acquire)) {
                    pthread_cond_wait(&backend->work_ready, &backend->mutex);
                }
                --backend->sleepers;
                next = atomic_load_explicit(&backend->generation, memory_order_acquire);
                pthread_mutex_unlock(&backend->mutex);
            }
        }
        if (atomic_load_explicit(&backend->shutdown, memory_order_relaxed)) {
            break;
        }
        generation = next;
        TrillimTask task = backend->task;
        void *task_context = backend->task_context;
        size_t task_count = backend->task_count;
        worker->status = execute_tasks(worker, task, task_context, task_count, true);
        /* Publish task status and output writes before acknowledging this dispatch. */
        atomic_store_explicit(&worker->generation, generation, memory_order_release);
    }
    return NULL;
}

static TrillimStatus run(
    void *context, size_t task_count, TrillimTask task, void *task_context
) {
    TrillimBenchContext *backend = context;
    if (task_count == 0) {
        return TRILLIM_OK;
    }
    pthread_mutex_lock(&backend->mutex);
    /* Every worker can claim one index past the end. */
    if (backend->task != NULL || task == NULL ||
        atomic_load_explicit(&backend->shutdown, memory_order_relaxed) ||
        task_count > SIZE_MAX - backend->pool.worker_count) {
        pthread_mutex_unlock(&backend->mutex);
        return TRILLIM_ERROR;
    }
    if (is_cancelled(backend)) {
        pthread_mutex_unlock(&backend->mutex);
        return TRILLIM_CANCELLED;
    }
    atomic_store_explicit(&backend->next_task, 0, memory_order_relaxed);
    backend->task_count = task_count;
    backend->task = task;
    backend->task_context = task_context;
    bool parallel = task_count > 1 && backend->pool.worker_count > 1;
    size_t generation = 0;
    if (parallel) {
        generation = atomic_load_explicit(&backend->generation, memory_order_relaxed) + 1;
        atomic_store_explicit(&backend->generation, generation, memory_order_release);
        if (backend->sleepers != 0) {
            pthread_cond_broadcast(&backend->work_ready);
        }
    }
    pthread_mutex_unlock(&backend->mutex);
    TrillimStatus status = execute_tasks(
        &backend->workers[0], task, task_context, task_count, parallel);
    if (parallel) {
        /* Cancellation still drains every worker before releasing borrowed task data. */
        for (size_t index = 1; index < backend->pool.worker_count; ++index) {
            TrillimBenchWorker *worker = &backend->workers[index];
            while (atomic_load_explicit(&worker->generation, memory_order_acquire) != generation) {
                spin_pause();
            }
            if (worker->status == TRILLIM_ERROR) {
                status = TRILLIM_ERROR;
            } else if (status == TRILLIM_OK && worker->status == TRILLIM_CANCELLED) {
                status = TRILLIM_CANCELLED;
            }
        }
    }
    if (status == TRILLIM_OK && is_cancelled(backend)) {
        status = TRILLIM_CANCELLED;
    }
    pthread_mutex_lock(&backend->mutex);
    backend->task = NULL;
    backend->task_context = NULL;
    pthread_mutex_unlock(&backend->mutex);
    return status;
}

TrillimBenchContext *trillim_bench_create(size_t worker_count) {
    if (worker_count == 0 || worker_count > SIZE_MAX / sizeof(TrillimBenchWorker)) {
        return NULL;
    }
    TrillimBenchContext *backend = allocate(sizeof(*backend), 64);
    if (backend == NULL) {
        return NULL;
    }
    memset(backend, 0, sizeof(*backend));
    atomic_init(&backend->cancelled, false);
    atomic_init(&backend->shutdown, false);
    atomic_init(&backend->generation, 0);
    atomic_init(&backend->next_task, 0);
    if (pthread_mutex_init(&backend->mutex, NULL) != 0) {
        release(backend);
        return NULL;
    }
    if (pthread_cond_init(&backend->work_ready, NULL) != 0) {
        pthread_mutex_destroy(&backend->mutex);
        release(backend);
        return NULL;
    }
    backend->pool = (TrillimThreadPool){backend, worker_count, run};
    backend->context = (TrillimKernelContext){&backend->pool, is_cancelled, backend};
    backend->workers = allocate(worker_count * sizeof(*backend->workers), 64);
    if (backend->workers == NULL) {
        backend->pool.worker_count = 0;
        trillim_bench_destroy(backend);
        return NULL;
    }
    memset(backend->workers, 0, worker_count * sizeof(*backend->workers));
    for (size_t index = 0; index < worker_count; ++index) {
        TrillimBenchWorker *worker = &backend->workers[index];
        worker->backend = backend;
        worker->index = index;
        atomic_init(&worker->generation, 0);
        if (index != 0 && pthread_create(&worker->thread, NULL, worker_main, worker) != 0) {
            backend->pool.worker_count = index;
            trillim_bench_destroy(backend);
            return NULL;
        }
    }
    return backend;
}

void trillim_bench_destroy(TrillimBenchContext *backend) {
    pthread_mutex_lock(&backend->mutex);
    atomic_store_explicit(&backend->shutdown, true, memory_order_relaxed);
    atomic_store(&backend->cancelled, true);
    pthread_cond_broadcast(&backend->work_ready);
    pthread_mutex_unlock(&backend->mutex);
    for (size_t index = 1; index < backend->pool.worker_count; ++index) {
        pthread_join(backend->workers[index].thread, NULL);
    }
    release(backend->workers);
    pthread_cond_destroy(&backend->work_ready);
    pthread_mutex_destroy(&backend->mutex);
    release(backend);
}

void trillim_bench_cancel(TrillimBenchContext *context) {
    atomic_store_explicit(&context->cancelled, true, memory_order_relaxed);
}

TrillimStatus trillim_bench_run(
    TrillimBenchContext *backend, TrillimKernel kernel,
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, size_t samples, double *seconds
) {
#ifdef _WIN32
    if (signal(SIGTERM, cancel_signal) == SIG_ERR) {
        return TRILLIM_ERROR;
    }
#else
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = cancel_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, NULL) != 0) {
        return TRILLIM_ERROR;
    }
#endif
    signal_cancelled = 0;
    for (size_t index = 0; index < samples; ++index) {
        uint64_t start = monotonic_nanoseconds();
        if (start == 0) {
            return TRILLIM_ERROR;
        }
        TrillimStatus status = kernel(outputs, inputs, scratch, parameters, &backend->context);
        if (status != TRILLIM_OK) {
            return status;
        }
        uint64_t end = monotonic_nanoseconds();
        if (end == 0 || end < start) {
            return TRILLIM_ERROR;
        }
        seconds[index] = (double)(end - start) * 1e-9;
    }
    return TRILLIM_OK;
}
