#include "common.h"
#include <stdatomic.h>

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

/* The components are linked into this backend's library. */
extern COMPONENT_API TrillimStatus trillim_neon_dense_glu(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern COMPONENT_API const TrillimParameterSchema *trillim_neon_dense_glu_parameters(void);
extern COMPONENT_API const TrillimCapabilities *trillim_neon_dense_glu_capabilities(void);
extern COMPONENT_API TrillimStatus trillim_neon_dense(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern COMPONENT_API const TrillimParameterSchema *trillim_neon_dense_parameters(void);
extern COMPONENT_API const TrillimCapabilities *trillim_neon_dense_capabilities(void);

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
} Values;

static const TrillimParameter value_fields[] = {
        {"batches", "size_t", offsetof(Values, batches), NULL},
        {"m", "size_t", offsetof(Values, m), NULL},
        {"n", "size_t", offsetof(Values, n), NULL},
        {"k", "size_t", offsetof(Values, k), NULL},
        {"rank", "size_t", offsetof(Values, rank), NULL},
        {"batch_shape", "size_t", offsetof(Values, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(Values, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(Values, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(Values, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(Values, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Values, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(Values, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Values, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Values, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(Values, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(Values, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Values, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Values, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Values, b_storage_columns), NULL},
    };

enum { FIELD_COUNT = sizeof(value_fields) / sizeof(value_fields[0]), MAX_PARAMETERS = 4096 };

typedef struct {
    size_t source;
    size_t destination;
    size_t bytes;
} FieldCopy;

typedef struct {
    const TrillimParameterSchema *schema;
    FieldCopy fields[FIELD_COUNT];
} Component;

typedef struct {
    size_t inputs[4];
    size_t output;
    size_t hidden_span;
    size_t scratch;
} Layout;

static Component glu_component, dense_component;
static size_t input_alignments[] = {1, 1, 1, 1};
static size_t output_alignments[] = {4};
static TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, UINT64_MAX};
static atomic_uint initialization = 0;
static bool initialized;

static bool valid_alignment(size_t value) {
    return value != 0 && value <= 64 && (value & (value - 1)) == 0;
}

static bool map_component(Component *component, const TrillimParameterSchema *schema, size_t inputs) {
    if (schema == NULL || schema->size > MAX_PARAMETERS || schema->field_count != FIELD_COUNT ||
        schema->input_count != inputs || schema->output_count != 1 || schema->requirements == NULL ||
        schema->fields == NULL || schema->input_alignments == NULL || schema->output_alignments == NULL ||
        !valid_alignment(schema->alignment) || !valid_alignment(schema->scratch_alignment) ||
        !valid_alignment(schema->output_alignments[0])) {
        return false;
    }
    for (size_t i = 0; i < inputs; ++i) {
        if (!valid_alignment(schema->input_alignments[i])) {
            return false;
        }
    }
    bool seen[FIELD_COUNT] = {false};
    for (size_t i = 0; i < FIELD_COUNT; ++i) {
        const TrillimParameter *target = &schema->fields[i];
        if (target->name == NULL || target->type == NULL) {
            return false;
        }
        size_t source = 0;
        while (source < FIELD_COUNT && strcmp(value_fields[source].name, target->name) != 0) {
            ++source;
        }
        if (source == FIELD_COUNT || seen[source] || strcmp(value_fields[source].type, target->type) != 0 ||
            (value_fields[source].length == NULL) != (target->length == NULL) ||
            (target->length != NULL && strcmp(value_fields[source].length, target->length) != 0)) {
            return false;
        }
        seen[source] = true;
        size_t bytes = target->length != NULL ? sizeof(void *) :
            (strcmp(target->type, "size_t") == 0 ? sizeof(size_t) : sizeof(uint32_t));
        if (bytes > schema->size || target->offset > schema->size - bytes) {
            return false;
        }
        component->fields[i] = (FieldCopy){value_fields[source].offset, target->offset, bytes};
    }
    component->schema = schema;
    return true;
}

static bool initialize(void) {
    if (atomic_load_explicit(&initialization, memory_order_acquire) == 2) {
        return initialized;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&initialization, &expected, 1,
                                                memory_order_acq_rel, memory_order_acquire)) {
        const TrillimCapabilities *glu = trillim_neon_dense_glu_capabilities();
        const TrillimCapabilities *dense = trillim_neon_dense_capabilities();
        initialized = glu != NULL && dense != NULL && glu->abi_version == TRILLIM_KERNEL_ABI_VERSION &&
            dense->abi_version == TRILLIM_KERNEL_ABI_VERSION &&
            map_component(&glu_component, trillim_neon_dense_glu_parameters(), 3) &&
            map_component(&dense_component, trillim_neon_dense_parameters(), 2);
        if (initialized) {
            for (size_t i = 0; i < 3; ++i) {
                input_alignments[i] = glu_component.schema->input_alignments[i];
            }
            input_alignments[3] = dense_component.schema->input_alignments[1];
            output_alignments[0] = dense_component.schema->output_alignments[0];
            capabilities.required = glu->required | dense->required;
        }
        atomic_store_explicit(&initialization, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&initialization, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return initialized;
}

static void pack(const Component *component, const Values *values, void *parameters) {
    memset(parameters, 0, component->schema->size);
    for (size_t i = 0; i < FIELD_COUNT; ++i) {
        const FieldCopy *field = &component->fields[i];
        memcpy((uint8_t *)parameters + field->destination,
               (const uint8_t *)values + field->source, field->bytes);
    }
}

static TrillimStatus prepare(const Parameters *p, void *glu_parameters, void *dense_parameters, Layout *layout) {
    if (p->rows == 0 || p->output_width == 0) {
        memset(layout, 0, sizeof(*layout));
        return TRILLIM_OK;
    }
    if (!initialize()) {
        return TRILLIM_ERROR;
    }
    Values glu = {
        .batches = 1, .m = p->rows, .n = p->hidden_width, .k = p->input_width,
        .a_row_stride = p->x_row_stride, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->input_width,
        .a_dtype = p->x_dtype, .b_dtype = p->gate_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->x_storage_rows, .a_storage_columns = p->x_storage_columns,
        .b_storage_rows = p->hidden_width, .b_storage_columns = p->input_width
    };
    Values dense = {
        .batches = 1, .m = p->rows, .n = p->output_width, .k = p->hidden_width,
        .a_row_stride = p->hidden_width, .a_col_stride = 1,
        .b_row_stride = 1, .b_col_stride = p->hidden_width,
        .a_dtype = TRILLIM_FP32, .b_dtype = p->down_dtype, .out_dtype = TRILLIM_FP32,
        .a_storage_rows = p->rows, .a_storage_columns = p->hidden_width,
        .b_storage_rows = p->output_width, .b_storage_columns = p->hidden_width
    };
    pack(&glu_component, &glu, glu_parameters);
    pack(&dense_component, &dense, dense_parameters);
    size_t glu_inputs[3], glu_outputs[1], glu_scratch;
    size_t dense_inputs[2], dense_outputs[1], dense_scratch;
    TrillimStatus status = glu_component.schema->requirements(glu_parameters, glu_inputs, glu_outputs, &glu_scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    status = dense_component.schema->requirements(dense_parameters, dense_inputs, dense_outputs, &dense_scratch);
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
    _Alignas(64) uint8_t glu_parameters[MAX_PARAMETERS], dense_parameters[MAX_PARAMETERS];
    Layout layout;
    TrillimStatus status = prepare(parameters, glu_parameters, dense_parameters, &layout);
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

TRILLIM_KERNEL TrillimStatus trillim_neon_dense_ffn(void *const *outputs, const void *const *inputs, void *scratch,
                                    const void *parameters, const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->rows == 0 || p->output_width == 0) {
        return TRILLIM_OK;
    }
    _Alignas(64) uint8_t glu_parameters[MAX_PARAMETERS], dense_parameters[MAX_PARAMETERS];
    Layout layout;
    TrillimStatus status = prepare(p, glu_parameters, dense_parameters, &layout);
    if (status != TRILLIM_OK) {
        return status;
    }
    void *hidden = scratch;
    void *workspace = layout.hidden_span != 0 ? (uint8_t *)scratch + layout.hidden_span : scratch;
    void *glu_outputs[] = {hidden};
    const void *glu_inputs[] = {inputs[0], inputs[1], inputs[2]};
    status = trillim_neon_dense_glu(glu_outputs, glu_inputs, workspace, glu_parameters, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const void *dense_inputs[] = {hidden, inputs[3]};
    return trillim_neon_dense(outputs, dense_inputs, workspace, dense_parameters, context);
}

TRILLIM_KERNEL const char *trillim_neon_dense_ffn_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_dense_ffn_parameters(void) {
    (void)initialize();
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
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 4, 1, requirements, can_alias, input_alignments, output_alignments, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_dense_ffn_capabilities(void) {
    (void)initialize();
    return &capabilities;
}
