#include "common.h"
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))

TRILLIM_AVX2 static inline __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    __m256i bits = _mm256_cvtepu16_epi32(raw);
    if (dtype == TRILLIM_BF16) return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
    __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
    __m256i exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7c00));
    __m256 normal = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(magnitude, 13), _mm256_set1_epi32(0x38000000)));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude), _mm256_set1_ps(0x1p-24f));
    __m256 special = _mm256_castsi256_ps(_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x3ff)), 13), _mm256_set1_epi32(0x7f800000)));
    __m256 value = _mm256_blendv_ps(normal, subnormal, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256())));
    value = _mm256_blendv_ps(value, special, _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00))));
    return _mm256_xor_ps(value, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)), 16)));
}

TRILLIM_AVX2 static inline void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i absolute = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    __m256i nan = _mm256_cmpgt_epi32(absolute, _mm256_set1_epi32(0x7f800000));
    __m256i rounded;
    if (dtype == TRILLIM_BF16) {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper, _mm256_set1_epi32(1)));
        rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    } else {
        __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0xfff), _mm256_and_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(1)));
        rounded = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(absolute, bias), 13), _mm256_set1_epi32(112 << 10));
        rounded = _mm256_min_epi32(rounded, _mm256_set1_epi32(0x7c00));
        __m256 small = _mm256_mul_ps(_mm256_castsi256_ps(absolute), _mm256_set1_ps(0x1p24f));
        __m256i subnormal = _mm256_cvttps_epi32(_mm256_round_ps(small, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        rounded = _mm256_blendv_epi8(rounded, subnormal, _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), absolute));
        __m256i payload = _mm256_or_si256(_mm256_srli_epi32(absolute, 13), _mm256_set1_epi32(0x200));
        payload = _mm256_and_si256(payload, _mm256_set1_epi32(0x3ff));
        rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(payload, _mm256_set1_epi32(0x7c00)), nan);
        rounded = _mm256_or_si256(rounded, _mm256_and_si256(upper, _mm256_set1_epi32(0x8000)));
    }
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index), _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

TRILLIM_AVX2 static inline float lane8(__m256 a, size_t lane) {
    float values[8];
    _mm256_storeu_ps(values,a);
    return values[lane];
}

typedef struct {
    size_t outputs;
    size_t output_rank;
    size_t reduce_rank;
    size_t reduce_count;
    const size_t *output_shape;
    const size_t *output_strides;
    const size_t *reduce_shape;
    const size_t *reduce_strides;
    size_t storage_rows;
    size_t storage_columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} TrillimReductionParameters;

typedef struct {
    void *output;
    const void *input;
    const TrillimReductionParameters *parameters;
    const TrillimKernelContext *context;
} TrillimReductionTask;

TRILLIM_AVX2 static bool shape_product(size_t rank, const size_t *shape, size_t *product) {
    if (rank && !shape) {
        return false;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!shape[axis]) {
            *product = 0;
            return true;
        }
    }
    size_t result = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (result > SIZE_MAX / shape[axis]) {
            return false;
        }
        result *= shape[axis];
    }
    *product = result;
    return true;
}

TRILLIM_AVX2 static inline TrillimStatus trillim_reduction_requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimReductionParameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype);
    size_t output_count, reduce_count;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        (p->output_rank && !p->output_strides) || (p->reduce_rank && !p->reduce_strides) ||
        !shape_product(p->output_rank, p->output_shape, &output_count) ||
        !shape_product(p->reduce_rank, p->reduce_shape, &reduce_count) ||
        output_count != p->outputs || reduce_count != p->reduce_count ||
        p->outputs > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    outputs[0] = p->outputs * bytes;
    if (p->outputs && p->reduce_count) {
        size_t outer_span, inner_span;
        if ((p->storage_columns && p->storage_rows > SIZE_MAX / p->storage_columns) ||
            !trillim_span(p->output_rank, p->output_shape, p->output_strides, &outer_span) ||
            !trillim_span(p->reduce_rank, p->reduce_shape, p->reduce_strides, &inner_span)) {
            return TRILLIM_ERROR;
        }
        size_t capacity = p->storage_rows * p->storage_columns;
        if (!outer_span || !inner_span || outer_span - 1 > SIZE_MAX - inner_span ||
            outer_span - 1 + inner_span > capacity ||
            outer_span - 1 + inner_span > SIZE_MAX / trillim_itemsize(p->input_dtype)) {
            return TRILLIM_ERROR;
        }
        inputs[0] = (outer_span - 1 + inner_span) * trillim_itemsize(p->input_dtype);
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus sum_float(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t base = trillim_offset(index, p->output_rank, p->output_shape, p->output_strides);
    float sum = 0.0f;
    size_t k = 0;
    if (p->reduce_rank == 1 && p->reduce_strides[0] == 1) {
        for (; p->reduce_count - k >= 8; k += 8) {
            if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 values = load8(task->input, base + k, p->input_dtype);
            sum += lane8(values, 0);
            sum += lane8(values, 1);
            sum += lane8(values, 2);
            sum += lane8(values, 3);
        sum += lane8(values, 4);
        sum += lane8(values, 5);
        sum += lane8(values, 6);
        sum += lane8(values, 7);
        }
    }
    for (; k < p->reduce_count; ++k) {
        if ((k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
        sum += trillim_load(task->input, base + offset, p->input_dtype).f;
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    TrillimValue value = {.f = sum};
    trillim_store(task->output, index, p->output_dtype, value);
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus sum_columns(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const TrillimReductionTask *task = opaque;
    const TrillimReductionParameters *p = task->parameters;
    size_t columns = p->output_shape[p->output_rank - 1];
    size_t tiles = columns / 64 + (columns % 64 != 0);
    size_t first = tile % tiles * 64;
    size_t count = columns - first < 64 ? columns - first : 64;
    size_t output = tile / tiles * columns + first;
    size_t base = trillim_offset(output, p->output_rank, p->output_shape, p->output_strides);
    size_t vectors = count / 8;
    __m256 sums[16];
    float tail[7] = {0};
    for (size_t lane = 0; lane < vectors; ++lane) {
        sums[lane] = _mm256_set1_ps(0.0f);
    }
    for (size_t k = 0; k < p->reduce_count; ++k) {
        if ((k & 127) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t offset = p->reduce_rank == 1 ? k * p->reduce_strides[0] :
            trillim_offset(k, p->reduce_rank, p->reduce_shape, p->reduce_strides);
        for (size_t lane = 0; lane < vectors; ++lane) {
            sums[lane] = _mm256_add_ps(sums[lane], load8(task->input, base + offset + lane * 8, p->input_dtype));
        }
        for (size_t lane = vectors * 8; lane < count; ++lane) {
            tail[lane - vectors * 8] += trillim_load(task->input, base + offset + lane, p->input_dtype).f;
        }
    }
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    for (size_t lane = 0; lane < vectors; ++lane) {
        store8(task->output, output + lane * 8, p->output_dtype, sums[lane]);
    }
    for (size_t lane = vectors * 8; lane < count; ++lane) {
        TrillimValue value = {.f = tail[lane - vectors * 8]};
        trillim_store(task->output, output + lane, p->output_dtype, value);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_reduce_sum_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimReductionParameters *p = parameters;
    if (!p->outputs) {
        return TRILLIM_OK;
    }
    TrillimReductionTask task = {outputs[0], inputs[0], p, context};
    if (p->output_rank && p->output_strides[p->output_rank - 1] == 1) {
        size_t columns = p->output_shape[p->output_rank - 1];
        size_t tiles = columns / 64 + (columns % 64 != 0);
        size_t tasks = p->outputs / columns * tiles;
        return tasks == 1 ? sum_columns(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tasks, sum_columns, &task);
    }
    return p->outputs == 1 ? sum_float(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, p->outputs, sum_float, &task);
}

TRILLIM_KERNEL const char *trillim_x86_reduce_sum_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _bind_reduction as bind\n"
           "def pattern(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> tuple[Tensor, ...]:\n"
           "    return (Sum(x, axes=axes, keepdims=keepdims),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_reduce_sum_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"outputs", "size_t", offsetof(TrillimReductionParameters, outputs), NULL},
        {"output_rank", "size_t", offsetof(TrillimReductionParameters, output_rank), NULL},
        {"reduce_rank", "size_t", offsetof(TrillimReductionParameters, reduce_rank), NULL},
        {"reduce_count", "size_t", offsetof(TrillimReductionParameters, reduce_count), NULL},
        {"output_shape", "size_t", offsetof(TrillimReductionParameters, output_shape), "output_rank"},
        {"output_strides", "size_t", offsetof(TrillimReductionParameters, output_strides), "output_rank"},
        {"reduce_shape", "size_t", offsetof(TrillimReductionParameters, reduce_shape), "reduce_rank"},
        {"reduce_strides", "size_t", offsetof(TrillimReductionParameters, reduce_strides), "reduce_rank"},
        {"storage_rows", "size_t", offsetof(TrillimReductionParameters, storage_rows), NULL},
        {"storage_columns", "size_t", offsetof(TrillimReductionParameters, storage_columns), NULL},
        {"input_dtype", "uint32_t", offsetof(TrillimReductionParameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(TrillimReductionParameters, output_dtype), NULL}
    };
    static const size_t inputs[] = {1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimReductionParameters), TRILLIM_ALIGNOF(TrillimReductionParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 1, 1,
        trillim_reduction_requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_reduce_sum_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
