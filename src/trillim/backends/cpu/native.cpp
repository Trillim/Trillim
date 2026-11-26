#include "native.h"

#include <new>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#include <signal.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "cancellation requires lock-free atomics");
static std::atomic_bool signal_cancelled;
static std::atomic_flag executing = ATOMIC_FLAG_INIT;

uint64_t trillim_cpu_features(void) {
    uint64_t features = 0;
#if defined(__x86_64__) || defined(__i386__)
    unsigned int a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1U << 27)) || !(c & (1U << 28))) {
        return 0;
    }
    unsigned int xcr0, high;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0), "=d"(high) : "c"(0));
    (void)high;
    if ((xcr0 & 6) != 6) return 0;
    if (c & (1U << 12)) features |= TRILLIM_CAP_X86_FMA;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return features;
    if (b & (1U << 5)) features |= TRILLIM_CAP_X86_AVX2;
    if ((xcr0 & 0xe6) == 0xe6) {
        if (b & (1U << 16)) features |= TRILLIM_CAP_X86_AVX512F;
        if (b & (1U << 30)) features |= TRILLIM_CAP_X86_AVX512BW;
        if (b & (1U << 17)) features |= TRILLIM_CAP_X86_AVX512DQ;
        if (b & (1U << 31)) features |= TRILLIM_CAP_X86_AVX512VL;
        if (c & (1U << 11)) features |= TRILLIM_CAP_X86_AVX512_VNNI;
    }
    if (a >= 1 && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1U << 4))) {
        features |= TRILLIM_CAP_X86_AVX_VNNI;
    }
#endif
    return features;
}

static void cancel_signal(int signal_number) {
    (void)signal_number;
    std::atomic_store_explicit(&signal_cancelled, true, std::memory_order_relaxed);
}

static bool is_cancelled(void *context) {
    auto *backend = static_cast<TrillimCpuContext *>(context);
    return std::atomic_load_explicit(&signal_cancelled, std::memory_order_relaxed) ||
        std::atomic_load_explicit(&backend->cancelled, std::memory_order_relaxed);
}

static void spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
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
    TrillimCpuWorker *worker, TrillimTask task,
    void *task_context, size_t task_count, bool parallel
) {
    TrillimCpuContext *backend = worker->backend;
    size_t completed = 0;
    TrillimStatus status = TRILLIM_OK;
    while (!is_cancelled(backend)) {
        size_t index = parallel
            ? std::atomic_fetch_add_explicit(&backend->next_task, 1, std::memory_order_relaxed)
            : completed;
        if (index >= task_count) {
            break;
        }
        status = task(task_context, index, worker->index);
        ++completed;
        if (status != TRILLIM_OK) {
            std::atomic_store_explicit(&backend->cancelled, true, std::memory_order_relaxed);
            break;
        }
    }
    return status == TRILLIM_OK || status == TRILLIM_CANCELLED ? status : TRILLIM_ERROR;
}

static void worker_main(void *context) {
    auto *worker = static_cast<TrillimCpuWorker *>(context);
    TrillimCpuContext *backend = worker->backend;
    size_t generation = 0;
    for (;;) {
        uint64_t start = monotonic_nanoseconds();
        size_t next = std::atomic_load_explicit(&backend->generation, std::memory_order_acquire);
        while (!std::atomic_load_explicit(&backend->shutdown, std::memory_order_relaxed)
               && generation == next) {
            for (size_t pause = 0; pause < 1024 && generation == next; ++pause) {
                spin_pause();
                next = std::atomic_load_explicit(&backend->generation, std::memory_order_acquire);
            }
            if (generation != next) {
                break;
            }
            uint64_t now = monotonic_nanoseconds();
            if (now == 0 || now - start >= 10000000) {
                /* Register and recheck under the dispatcher's wake-up mutex. */
                std::unique_lock lock(backend->mutex);
                ++backend->sleepers;
                while (!std::atomic_load_explicit(&backend->shutdown, std::memory_order_relaxed)
                       && generation == std::atomic_load_explicit(
                           &backend->generation, std::memory_order_acquire)) {
                    backend->work_ready.wait(lock);
                }
                --backend->sleepers;
                next = std::atomic_load_explicit(&backend->generation, std::memory_order_acquire);
            }
        }
        if (std::atomic_load_explicit(&backend->shutdown, std::memory_order_relaxed)) {
            break;
        }
        generation = next;
        TrillimTask task = backend->task;
        void *task_context = backend->task_context;
        size_t task_count = backend->task_count;
        worker->status = execute_tasks(worker, task, task_context, task_count, true);
        /* Publish task status and output writes before acknowledging this dispatch. */
        std::atomic_store_explicit(&worker->generation, generation, std::memory_order_release);
    }
}

static TrillimStatus run(
    void *context, size_t task_count, TrillimTask task, void *task_context
) {
    auto *backend = static_cast<TrillimCpuContext *>(context);
    if (task_count == 0) {
        return TRILLIM_OK;
    }
    backend->mutex.lock();
    /* Every worker can claim one index past the end. */
    if (backend->task != nullptr || task == nullptr ||
        std::atomic_load_explicit(&backend->shutdown, std::memory_order_relaxed) ||
        task_count > SIZE_MAX - backend->pool.worker_count) {
        backend->mutex.unlock();
        return TRILLIM_ERROR;
    }
    if (is_cancelled(backend)) {
        backend->mutex.unlock();
        return TRILLIM_CANCELLED;
    }
    std::atomic_store_explicit(&backend->next_task, 0, std::memory_order_relaxed);
    backend->task_count = task_count;
    backend->task = task;
    backend->task_context = task_context;
    bool parallel = task_count > 1 && backend->pool.worker_count > 1;
    size_t generation = 0;
    if (parallel) {
        generation = std::atomic_load_explicit(&backend->generation, std::memory_order_relaxed) + 1;
        std::atomic_store_explicit(&backend->generation, generation, std::memory_order_release);
        if (backend->sleepers != 0) {
            backend->work_ready.notify_all();
        }
    }
    backend->mutex.unlock();
    TrillimStatus status = execute_tasks(
        &backend->workers[0], task, task_context, task_count, parallel);
    if (parallel) {
        /* Cancellation still drains every worker before releasing borrowed task data. */
        for (size_t index = 1; index < backend->pool.worker_count; ++index) {
            TrillimCpuWorker *worker = &backend->workers[index];
            while (std::atomic_load_explicit(&worker->generation, std::memory_order_acquire) != generation) {
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
    backend->mutex.lock();
    backend->task = nullptr;
    backend->task_context = nullptr;
    backend->mutex.unlock();
    return status;
}

TrillimCpuContext *trillim_cpu_create(size_t worker_count) {
    if (worker_count == 0 || worker_count > SIZE_MAX / sizeof(TrillimCpuWorker)) {
        return nullptr;
    }
    TrillimCpuContext *backend = nullptr;
    try {
        backend = new TrillimCpuContext{};
        backend->pool = {backend, 0, run};
        backend->context = {&backend->pool, is_cancelled, backend};
        backend->workers = new TrillimCpuWorker[worker_count]{};
        for (size_t index = 0; index < worker_count; ++index) {
            auto &worker = backend->workers[index];
            worker.backend = backend;
            worker.index = index;
            if (index != 0) {
                worker.thread = std::thread(worker_main, &worker);
            }
            ++backend->pool.worker_count;
        }
    } catch (...) {
        if (backend) {
            trillim_cpu_destroy(backend);
        }
        return nullptr;
    }
    return backend;
}

void trillim_cpu_destroy(TrillimCpuContext *backend) {
    backend->mutex.lock();
    backend->shutdown.store(true, std::memory_order_relaxed);
    backend->cancelled.store(true, std::memory_order_relaxed);
    backend->work_ready.notify_all();
    backend->mutex.unlock();
    for (size_t index = 1; index < backend->pool.worker_count; ++index) {
        backend->workers[index].thread.join();
    }
    delete[] backend->workers;
    delete backend;
}

void trillim_cpu_cancel(TrillimCpuContext *context) {
    std::atomic_store_explicit(&context->cancelled, true, std::memory_order_relaxed);
}

bool trillim_cpu_cancelled(TrillimCpuContext *context) {
    return is_cancelled(context);
}

TrillimStatus trillim_cpu_enter(TrillimCpuContext *context) {
    if (std::atomic_flag_test_and_set(&executing)) {
        return TRILLIM_ERROR;
    }
    std::atomic_store_explicit(&signal_cancelled, false, std::memory_order_relaxed);
#ifdef _WIN32
    context->previous_interrupt = signal(SIGINT, cancel_signal);
    if (context->previous_interrupt == SIG_ERR) {
        std::atomic_flag_clear(&executing);
        return TRILLIM_ERROR;
    }
    context->previous_terminate = signal(SIGTERM, cancel_signal);
    if (context->previous_terminate == SIG_ERR) {
        signal(SIGINT, context->previous_interrupt);
        std::atomic_flag_clear(&executing);
        return TRILLIM_ERROR;
    }
#else
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = cancel_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &context->previous_interrupt) != 0) {
        std::atomic_flag_clear(&executing);
        return TRILLIM_ERROR;
    }
    if (sigaction(SIGTERM, &action, &context->previous_terminate) != 0) {
        sigaction(SIGINT, &context->previous_interrupt, nullptr);
        std::atomic_flag_clear(&executing);
        return TRILLIM_ERROR;
    }
#endif
    return TRILLIM_OK;
}

void trillim_cpu_leave(TrillimCpuContext *context) {
#ifdef _WIN32
    signal(SIGINT, context->previous_interrupt);
    signal(SIGTERM, context->previous_terminate);
#else
    sigaction(SIGINT, &context->previous_interrupt, nullptr);
    sigaction(SIGTERM, &context->previous_terminate, nullptr);
#endif
    std::atomic_flag_clear(&executing);
}

TrillimStatus trillim_cpu_invoke(
    TrillimCpuContext *backend, TrillimKernel kernel,
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters
) {
    if (is_cancelled(backend)) {
        return TRILLIM_CANCELLED;
    }
    return kernel(outputs, inputs, scratch, parameters, &backend->context);
}

TrillimStatus trillim_cpu_run(
    TrillimCpuContext *backend, TrillimKernel kernel,
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
    if (sigaction(SIGTERM, &action, nullptr) != 0) {
        return TRILLIM_ERROR;
    }
#endif
    std::atomic_store_explicit(&signal_cancelled, false, std::memory_order_relaxed);
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
