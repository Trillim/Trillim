#include "common.h"
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_neon_dense_ffn(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern COMPONENT_API const char *trillim_neon_dense_ffn_tdl(void);
extern COMPONENT_API const TrillimParameterSchema *trillim_neon_dense_ffn_parameters(void);
extern COMPONENT_API const TrillimCapabilities *trillim_neon_dense_ffn_capabilities(void);

static atomic_uint initialization = 0;
static const TrillimParameterSchema *component;
static TrillimParameterSchema schema;
static TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, UINT64_MAX};
static bool available;

static TrillimStatus requirements(const void *, size_t *, size_t *, size_t *);

static bool provider_available(void) {
#ifdef _WIN32
    static const char *const names[] = {"libopenblas.dll", "openblas.dll", "blas.dll"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        HMODULE library = LoadLibraryA(names[i]);
        if (library != NULL) {
            if (GetProcAddress(library, "cblas_sgemm") != NULL) {
                return true;
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
            if (dlsym(library, "cblas_sgemm$NEWLAPACK") != NULL ||
                dlsym(library, "cblas_sgemm") != NULL) {
#else
            if (dlsym(library, "cblas_sgemm") != NULL) {
#endif
                return true;
            }
            dlclose(library);
        }
    }
#endif
    return false;
}

static bool initialize(void) {
    if (atomic_load_explicit(&initialization, memory_order_acquire) == 2) {
        return available;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&initialization, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
        component = trillim_neon_dense_ffn_parameters();
        const TrillimCapabilities *base = trillim_neon_dense_ffn_capabilities();
        if (component != NULL && component->requirements != NULL && base != NULL &&
            base->abi_version == TRILLIM_KERNEL_ABI_VERSION) {
            schema = *component;
            schema.requirements = requirements;
            capabilities.required = base->required | TRILLIM_CAP_BLAS;
            available = provider_available();
        }
        atomic_store_explicit(&initialization, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&initialization, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return available;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    if (!initialize()) {
        if (component != NULL) {
            memset(inputs, 0, component->input_count * sizeof(*inputs));
            memset(outputs, 0, component->output_count * sizeof(*outputs));
        }
        *scratch = 0;
        return TRILLIM_ERROR;
    }
    return component->requirements(parameters, inputs, outputs, scratch);
}

/* Keep the optional-provider implementation selectable alongside portable BLAS FFN. */
TRILLIM_KERNEL TrillimStatus trillim_neon_dense_ffn_blas(void *const *outputs, const void *const *inputs,
    void *scratch, const void *parameters, const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!initialize()) {
        return TRILLIM_ERROR;
    }
    return trillim_neon_dense_ffn(outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_dense_ffn_blas_tdl(void) {
    return trillim_neon_dense_ffn_tdl();
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_ffn_blas_parameters(void) {
    (void)initialize();
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_ffn_blas_capabilities(void) {
    (void)initialize();
    return &capabilities;
}
