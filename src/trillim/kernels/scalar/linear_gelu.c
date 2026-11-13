#include "common.h"
#include <math.h>
#include <stdatomic.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(_WIN32)
#define COMPONENT_API
#else
#define COMPONENT_API __attribute__((visibility("default")))
#endif

extern COMPONENT_API TrillimStatus trillim_scalar_linear(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern COMPONENT_API const TrillimParameterSchema *trillim_scalar_linear_parameters(void);
extern COMPONENT_API const TrillimCapabilities *trillim_scalar_linear_capabilities(void);

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
} Parameters;

static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(Parameters, batches), NULL},
        {"m", "size_t", offsetof(Parameters, m), NULL},
        {"n", "size_t", offsetof(Parameters, n), NULL},
        {"k", "size_t", offsetof(Parameters, k), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(Parameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(Parameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(Parameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(Parameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(Parameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(Parameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(Parameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(Parameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(Parameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(Parameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(Parameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(Parameters, b_storage_columns), NULL},
};

enum { FIELD_COUNT = sizeof(fields) / sizeof(fields[0]) };
static atomic_uint initialization = 0;
static const TrillimParameterSchema *provider;
static TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, UINT64_MAX};

/* Availability may differ between compilation and execution. */
#define OPTIONAL_WORKSPACE (32u * 1024u * 1024u)
extern COMPONENT_API TrillimStatus trillim_scalar_blas_linear(void *const *, const void *const *, void *, const void *, const TrillimKernelContext *);
extern COMPONENT_API const TrillimParameterSchema *trillim_scalar_blas_linear_parameters(void);
extern COMPONENT_API const TrillimCapabilities *trillim_scalar_blas_linear_capabilities(void);
static const TrillimParameterSchema *preferred;


static bool compatible(const TrillimParameterSchema *schema, const TrillimCapabilities *caps) {
    bool valid = schema != NULL && caps != NULL &&
        caps->abi_version == TRILLIM_KERNEL_ABI_VERSION &&
        schema->size == sizeof(Parameters) && schema->alignment == TRILLIM_ALIGNOF(Parameters) &&
        schema->field_count == FIELD_COUNT && schema->fields != NULL &&
        schema->input_count == 2 && schema->output_count == 1 &&
        schema->requirements != NULL && schema->input_alignments != NULL &&
        schema->output_alignments != NULL && schema->scratch_alignment != 0 &&
        schema->scratch_alignment <= 64 && (64 % schema->scratch_alignment) == 0;
    for (size_t i = 0; valid && i < 2; ++i) {
        valid = schema->input_alignments[i] != 0 && (4 % schema->input_alignments[i]) == 0;
    }
    if (valid) {
        valid = schema->output_alignments[0] != 0 && (4 % schema->output_alignments[0]) == 0;
    }
    for (size_t i = 0; valid && i < FIELD_COUNT; ++i) {
        const TrillimParameter *a = &fields[i], *b = &schema->fields[i];
        valid = b->name != NULL && b->type != NULL &&
            strcmp(a->name, b->name) == 0 && strcmp(a->type, b->type) == 0 &&
            a->offset == b->offset && (a->length == NULL) == (b->length == NULL) &&
            (a->length == NULL || strcmp(a->length, b->length) == 0);
    }
    return valid;
}

static bool initialize(void) {
    if (atomic_load_explicit(&initialization, memory_order_acquire) == 2) {
        return provider != NULL;
    }
    unsigned expected = 0;
    if (atomic_compare_exchange_strong_explicit(&initialization, &expected, 1,
                                               memory_order_acq_rel, memory_order_acquire)) {
        const TrillimParameterSchema *schema = trillim_scalar_linear_parameters();
        const TrillimCapabilities *caps = trillim_scalar_linear_capabilities();
        if (compatible(schema, caps)) {
            provider = schema;
            capabilities.required = caps->required | TRILLIM_CAP_LIBM;
            const TrillimParameterSchema *blas = trillim_scalar_blas_linear_parameters();
            if (compatible(blas, trillim_scalar_blas_linear_capabilities())) {
                preferred = blas;
            }
        }
        atomic_store_explicit(&initialization, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&initialization, memory_order_acquire) != 2) {
            atomic_signal_fence(memory_order_acquire);
        }
    }
    return provider != NULL;
}

static bool matrix_fits(size_t rows, size_t columns, size_t row_stride,
                        size_t column_stride, size_t storage_rows, size_t storage_columns) {
    if (storage_columns != 0 && storage_rows > SIZE_MAX / storage_columns) {
        return false;
    }
    size_t capacity = storage_rows * storage_columns;
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (capacity == 0 || (row_stride != 0 && rows - 1 > (capacity - 1) / row_stride)) {
        return false;
    }
    size_t last_row = (rows - 1) * row_stride;
    return column_stride == 0 || columns - 1 <= (capacity - 1 - last_row) / column_stride;
}

static TrillimStatus prepare(const void *parameters, size_t *inputs,
                             size_t *outputs, size_t *scratch, TrillimKernel *projection) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    if (p->batches != 1 || p->rank != 0 || p->a_dtype != TRILLIM_FP32 ||
        !trillim_float(p->b_dtype) || p->out_dtype != TRILLIM_FP32) {
        return TRILLIM_ERROR;
    }
    if (p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / sizeof(float) / p->n || p->n > SIZE_MAX / sizeof(float)) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0 &&
        (!matrix_fits(p->m, p->k, p->a_row_stride, p->a_col_stride, p->a_storage_rows, p->a_storage_columns) ||
         !matrix_fits(p->k, p->n, p->b_row_stride, p->b_col_stride, p->b_storage_rows, p->b_storage_columns) ||
         (p->a_storage_columns != 0 && p->a_storage_rows > SIZE_MAX / sizeof(float) / p->a_storage_columns) ||
         (p->b_storage_columns != 0 && p->b_storage_rows > SIZE_MAX / trillim_itemsize(p->b_dtype) / p->b_storage_columns))) {
        return TRILLIM_ERROR;
    }
    if (!initialize()) {
        return TRILLIM_ERROR;
    }
    *projection = trillim_scalar_linear;
    size_t workspace;
    if (preferred != NULL &&
        preferred->requirements(parameters, inputs, outputs, &workspace) == TRILLIM_OK &&
        workspace <= OPTIONAL_WORKSPACE) {
        *scratch = OPTIONAL_WORKSPACE;
        inputs[2] = p->n * sizeof(float);
        *projection = trillim_scalar_blas_linear;
        return TRILLIM_OK;
    }
    TrillimStatus status = provider->requirements(parameters, inputs, outputs, scratch);
    if (status != TRILLIM_OK) {
        return status;
    }
    inputs[2] = p->n * sizeof(float);
    if (*scratch > OPTIONAL_WORKSPACE) {
        return TRILLIM_ERROR;
    }
    *scratch = OPTIONAL_WORKSPACE;
    return TRILLIM_OK;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    TrillimKernel projection;
    return prepare(parameters, inputs, outputs, scratch, &projection);
}

static float gelu(float value) {
    float argument = value * 0.7071067811865476f;
    float cdf = erff(argument);
    float half = value * 0.5f;
    float shifted = cdf + 1.0f;
    return half * shifted;
}


typedef struct {
    float *output;
    const float *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus epilogue(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t total = task->p->m * task->p->n;
    size_t begin = tile * 1024;
    size_t end = total - begin < 1024 ? total : begin + 1024;
    while (begin < end) {
        size_t column = begin % task->p->n;
        size_t count = task->p->n - column;
        if (count > end - begin) {
            count = end - begin;
        }
        size_t lane = 0;

        for (; lane < count; ++lane) {
            float value = task->output[begin + lane] + task->bias[column + lane];
            task->output[begin + lane] = gelu(value);
        }
        begin += count;
    }
    return task->context->is_cancelled(task->context->cancellation_context) ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_linear_gelu(void *const *outputs, const void *const *inputs,
                                    void *scratch, const void *parameters,
                                    const TrillimKernelContext *context) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    size_t input_spans[3], output_spans[1], scratch_span;
    TrillimKernel projection;
    TrillimStatus status = prepare(parameters, input_spans, output_spans, &scratch_span, &projection);
    if (status != TRILLIM_OK || p->m == 0 || p->n == 0) {
        return status;
    }
    status = projection(outputs, inputs, scratch, parameters, context);
    if (status != TRILLIM_OK) {
        return status;
    }
    Task task = {outputs[0], inputs[2], p, context};
    size_t count = p->m * p->n;
    size_t tiles = count / 1024 + (count % 1024 != 0);
    if (count <= 4096) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            status = epilogue(&task, tile, 0);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, epilogue, &task);
}

TRILLIM_KERNEL const char *trillim_scalar_linear_gelu_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _linear, _contiguous_strides\n"
           "\n"
           "def bind(c):\n"
           "    x, weight, bias = (c.inputs[name] for name in ('x', 'weight', 'bias'))\n"
           "    output = c.outputs[0]\n"
           "    if not x.shape or len(weight.shape) != 2 or x.shape[-1] != weight.shape[1]:\n"
           "        raise ValueError('Linear-GELU requires compatible channel and matrix dimensions')\n"
           "    if x.dtype != 'fp32' or weight.dtype not in {'fp16', 'bf16', 'fp32'} or bias.dtype != 'fp32' or output.dtype != 'fp32':\n"
           "        raise ValueError('Linear-GELU requires FP32 activations and bias with floating weights')\n"
           "    if bias.shape != weight.shape[:1] or output.shape != (*x.shape[:-1], weight.shape[0]):\n"
           "        raise ValueError('Linear-GELU requires a channel bias and the projected output shape')\n"
           "    for value in (bias, output):\n"
           "        if (value.strides or _contiguous_strides(value.shape)) != _contiguous_strides(value.shape):\n"
           "            raise ValueError('Linear-GELU requires contiguous bias and output')\n"
           "    if c.attrs.get('approximate', 'none') != 'none':\n"
           "        raise ValueError('Linear-GELU requires the erf definition')\n"
           "    return _linear(c)\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, approximate: str = 'none') -> tuple[Tensor, ...]:\n"
           "    if approximate != 'none':\n"
           "        raise ValueError('Linear-GELU requires the erf definition')\n"
           "    return (Gelu(Linear(x, weight, bias=bias)),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_linear_gelu_parameters(void) {
    static const size_t input_alignments[] = {4, 4, 4}, output_alignments[] = {4};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), FIELD_COUNT, fields,
        3, 1, requirements, NULL, input_alignments, output_alignments, 64
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_linear_gelu_capabilities(void) {
    (void)initialize();
    return &capabilities;
}
