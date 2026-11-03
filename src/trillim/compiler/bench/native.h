#ifndef TRILLIM_COMPILER_BENCH_NATIVE_H
#define TRILLIM_COMPILER_BENCH_NATIVE_H

#include "common.h"
#include <pthread.h>
#include <stdatomic.h>

typedef struct TrillimBenchContext TrillimBenchContext;
typedef struct {
    _Alignas(64) atomic_size_t generation;
    TrillimStatus status;
    atomic_size_t completed;
    TrillimBenchContext *backend;
    pthread_t thread;
    size_t index;
} TrillimBenchWorker;

struct TrillimBenchContext {
    TrillimThreadPool pool;
    TrillimKernelContext context;
    TrillimBenchWorker *workers;
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
};

TrillimBenchContext *trillim_bench_create(size_t workers);
void trillim_bench_destroy(TrillimBenchContext *context);
void trillim_bench_cancel(TrillimBenchContext *context);
TrillimStatus trillim_bench_run(
    TrillimBenchContext *context, TrillimKernel kernel,
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, size_t samples, double *seconds
);

#endif
