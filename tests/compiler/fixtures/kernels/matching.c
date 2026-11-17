#include "common.h"

typedef struct { size_t count, input_count; const size_t *input_counts; } Parameters;

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                 size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    for (size_t i = 0; i < INPUT_COUNT; ++i) inputs[i] = p->input_counts[i] * sizeof(float);
    for (size_t i = 0; i < OUTPUT_COUNT; ++i) outputs[i] = p->count * sizeof(float);
    *scratch = 0;
    return TRILLIM_OK;
}

/* Matching fixtures deliberately have no executable numerical implementation. */
TRILLIM_KERNEL TrillimStatus matching(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    (void)outputs; (void)inputs; (void)scratch; (void)parameters; (void)context;
    return TRILLIM_ERROR;
}

TRILLIM_KERNEL const char *matching_tdl(void) { return TDL_SOURCE; }

TRILLIM_KERNEL const TrillimParameterSchema *matching_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"input_count", "size_t", offsetof(Parameters, input_count), NULL},
        {"input_counts", "size_t", offsetof(Parameters, input_counts), "input_count"}
    };
    static const size_t inputs[INPUT_COUNT] = {INPUT_ALIGNMENTS};
    static const size_t outputs[OUTPUT_COUNT] = {OUTPUT_ALIGNMENTS};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields,
        INPUT_COUNT, OUTPUT_COUNT, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *matching_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
