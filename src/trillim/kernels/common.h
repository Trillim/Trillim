#ifndef TRILLIM_KERNEL_COMMON_H
#define TRILLIM_KERNEL_COMMON_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define TRILLIM_KERNEL_ABI_VERSION 1
#ifdef __cplusplus
#define TRILLIM_KERNEL extern "C"
#define TRILLIM_ALIGNOF(type) alignof(type)
extern "C" {
#else
#define TRILLIM_KERNEL
#define TRILLIM_ALIGNOF(type) _Alignof(type)
#endif
    typedef int32_t TrillimStatus;
    enum {TRILLIM_OK = 0, TRILLIM_CANCELLED = 1, TRILLIM_ERROR = 2}
    ;
    typedef struct {
        const char *name;
        const char *type;
        size_t offset;
        const char *length;
    } TrillimParameter;
    typedef struct {
        size_t size;
        size_t alignment;
        size_t field_count;
        const TrillimParameter *fields;
        size_t input_count;
        size_t output_count;
        TrillimStatus (*requirements)(const void *, size_t *, size_t *, size_t *);
        bool (*can_alias)(const void *, size_t, size_t);
        const size_t *input_alignments;
        const size_t *output_alignments;
        size_t scratch_alignment;
    } TrillimParameterSchema;
    typedef struct {
        uint32_t abi_version;
        const char *backend;
        const char *hardware;
        const char *software;
    } TrillimCapabilities;
    typedef TrillimStatus (*TrillimTask)(void *, size_t, size_t);
    typedef struct {
        void *context;
        size_t worker_count;
        TrillimStatus (*run)(void *, size_t, TrillimTask, void *);
    } TrillimThreadPool;
    typedef struct {
        const TrillimThreadPool *thread_pool;
        bool (*is_cancelled)(void *);
        void *cancellation_context;
    } TrillimKernelContext;
    typedef TrillimStatus (*TrillimKernel)(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
#ifdef __cplusplus
}
#endif
#endif
