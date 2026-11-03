#include "indexing.h"

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    return trillim_index_requirements(parameters, inputs, outputs, scratch, true, false);
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_scatter(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    return trillim_index_compute(true, outputs, inputs, scratch, parameters, context);
}

TRILLIM_KERNEL const char *trillim_scalar_scatter_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import scatter as bind\n"
           "physical_inputs = ('x', 'indices', 'values')\n"
           "def pattern(x: Tensor, indices: Tensor, values: Tensor, *, mode: str = \"replace\") -> tuple[Tensor, ...]:\n"
           "    return (Scatter(x, indices=indices, values=values, mode=mode),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_scatter_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(TrillimIndexParameters, count), NULL},
        {"items", "size_t", offsetof(TrillimIndexParameters, items), NULL},
        {"coordinates", "size_t", offsetof(TrillimIndexParameters, coordinates), NULL},
        {"index_rank", "size_t", offsetof(TrillimIndexParameters, index_rank), NULL},
        {"slice_rank", "size_t", offsetof(TrillimIndexParameters, slice_rank), NULL},
        {"slice_size", "size_t", offsetof(TrillimIndexParameters, slice_size), NULL},
        {"index_shape", "size_t", offsetof(TrillimIndexParameters, index_shape), "index_rank"},
        {"index_strides", "size_t", offsetof(TrillimIndexParameters, index_strides), "index_rank"},
        {"mask_strides", "size_t", offsetof(TrillimIndexParameters, mask_strides), "index_rank"},
        {"coordinate_stride", "size_t", offsetof(TrillimIndexParameters, coordinate_stride), NULL},
        {"indexed_shape", "size_t", offsetof(TrillimIndexParameters, indexed_shape), "coordinates"},
        {"indexed_strides", "size_t", offsetof(TrillimIndexParameters, indexed_strides), "coordinates"},
        {"output_indexed_strides", "size_t", offsetof(TrillimIndexParameters, output_indexed_strides), "coordinates"},
        {"slice_shape", "size_t", offsetof(TrillimIndexParameters, slice_shape), "slice_rank"},
        {"slice_strides", "size_t", offsetof(TrillimIndexParameters, slice_strides), "slice_rank"},
        {"values_rank", "size_t", offsetof(TrillimIndexParameters, values_rank), NULL},
        {"values_shape", "size_t", offsetof(TrillimIndexParameters, values_shape), "values_rank"},
        {"values_strides", "size_t", offsetof(TrillimIndexParameters, values_strides), "values_rank"},
        {"storage_rows", "size_t", offsetof(TrillimIndexParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimIndexParameters, storage_columns), NULL},
        {"output_rows", "size_t", offsetof(TrillimIndexParameters, output_rows), NULL},
        {"output_columns", "size_t", offsetof(TrillimIndexParameters, output_columns), NULL},
        {"dtype", "uint32_t", offsetof(TrillimIndexParameters, dtype), NULL},
        {"masked", "uint32_t", offsetof(TrillimIndexParameters, masked), NULL},
        {"mode", "uint32_t", offsetof(TrillimIndexParameters, mode), NULL},
        {"fill_bits", "uint64_t", offsetof(TrillimIndexParameters, fill_bits), NULL},
        {"fill_float", "float", offsetof(TrillimIndexParameters, fill_float), NULL},
    };
    static const size_t inputs[] = {1, 1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimIndexParameters), TRILLIM_ALIGNOF(TrillimIndexParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 3, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(size_t)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_scatter_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "scalar", "", "m"};
    return &capabilities;
}
