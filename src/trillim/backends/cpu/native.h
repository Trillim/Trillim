#ifndef TRILLIM_CPU_NATIVE_H
#define TRILLIM_CPU_NATIVE_H

#include "common.h"
#include <pthread.h>
#include <stdatomic.h>
#include <signal.h>

typedef struct TrillimCpuContext TrillimCpuContext;
typedef struct {
    _Alignas(64) atomic_size_t generation;
    TrillimStatus status;
    TrillimCpuContext *backend;
    pthread_t thread;
    size_t index;
} TrillimCpuWorker;

struct TrillimCpuContext {
    TrillimThreadPool pool;
    TrillimKernelContext context;
    TrillimCpuWorker *workers;
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    atomic_bool cancelled;
    atomic_bool shutdown;
    size_t sleepers;
    _Alignas(64) atomic_size_t generation;
    _Alignas(64) atomic_size_t next_task;
    size_t task_count;
    TrillimTask task;
    void *task_context;
#ifdef _WIN32
    void (*previous_interrupt)(int);
    void (*previous_terminate)(int);
#else
    struct sigaction previous_interrupt;
    struct sigaction previous_terminate;
#endif
};

TRILLIM_KERNEL TrillimCpuContext *trillim_cpu_create(size_t workers);
TRILLIM_KERNEL uint64_t trillim_cpu_features(void);
TRILLIM_KERNEL void trillim_cpu_destroy(TrillimCpuContext *context);
TRILLIM_KERNEL void trillim_cpu_cancel(TrillimCpuContext *context);
TRILLIM_KERNEL bool trillim_cpu_cancelled(TrillimCpuContext *context);
TRILLIM_KERNEL TrillimStatus trillim_cpu_invoke(
    TrillimCpuContext *context, TrillimKernel kernel,
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters
);
TRILLIM_KERNEL TrillimStatus trillim_cpu_enter(TrillimCpuContext *context);
TRILLIM_KERNEL void trillim_cpu_leave(TrillimCpuContext *context);
TRILLIM_KERNEL TrillimStatus trillim_cpu_run(
    TrillimCpuContext *context, TrillimKernel kernel,
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, size_t samples, double *seconds
);

#endif
