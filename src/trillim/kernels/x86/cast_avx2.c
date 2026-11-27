#include "common.h"
#include <immintrin.h>
#define SIMD __attribute__((target("avx2")))

enum { TILE_ELEMENTS = 32768, CHUNK_ELEMENTS = 4096 };

typedef struct {
    size_t count, rank;
    const size_t *shape, *strides;
    uint32_t input_dtype, output_dtype;
} Parameters;

typedef struct {
    uint8_t *output;
    const uint8_t *input;
    const Parameters *p;
    const TrillimKernelContext *context;
    size_t input_bytes, output_bytes;
    bool contiguous;
} Task;

static TrillimStatus requirements(
    const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch
) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->input_dtype) || !trillim_float(p->output_dtype) ||
        p->input_dtype == p->output_dtype || (p->rank && (!p->shape || !p->strides))) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->shape[axis] == 0) {
            return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
        }
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) {
            return TRILLIM_ERROR;
        }
        count *= p->shape[axis];
    }
    size_t span;
    size_t input_bytes = trillim_itemsize(p->input_dtype);
    size_t output_bytes = trillim_itemsize(p->output_dtype);
    if (count != p->count || count > SIZE_MAX / output_bytes ||
        !trillim_span(p->rank, p->shape, p->strides, &span) || span > SIZE_MAX / input_bytes) {
        return TRILLIM_ERROR;
    }
    inputs[0] = span * input_bytes;
    outputs[0] = count * output_bytes;
    return TRILLIM_OK;
}

SIMD static __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    if (dtype == TRILLIM_FP16) {
        __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
        __m256i h = _mm256_cvtepu16_epi32(raw);
        __m256i fraction = _mm256_and_si256(h, _mm256_set1_epi32(1023));
        __m256i exponent = _mm256_and_si256(h, _mm256_set1_epi32(0x7c00));
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
        __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x7fff)), 13),
                                          _mm256_set1_epi32(112 << 23));
        __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
        normal = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                    _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
        __m256i special = _mm256_or_si256(_mm256_set1_epi32(0x7f800000), _mm256_slli_epi32(fraction, 13));
        normal = _mm256_blendv_epi8(normal, special, _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00)));
        return _mm256_castsi256_ps(_mm256_or_si256(normal, sign));
    }
    __m128i raw = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

SIMD static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
    if (dtype == TRILLIM_FP32) {
        _mm256_storeu_ps((float *)data + index, value);
        return;
    }
    if (dtype == TRILLIM_FP16) {
        float lanes[8];
        _mm256_storeu_ps(lanes, value);
        for (size_t lane = 0; lane < 8; ++lane) {
            TrillimValue item = {.f = lanes[lane]};
            trillim_store(data, index + lane, dtype, item);
        }
        return;
    }
    __m256i bits = _mm256_castps_si256(value);
    __m256i upper = _mm256_srli_epi32(bits, 16);
    __m256i rounded =
        _mm256_srli_epi32(_mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                                                  _mm256_and_si256(upper, _mm256_set1_epi32(1)))),
                          16);
    __m256i nan =
        _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)), _mm256_set1_epi32(0x7f800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_or_si256(upper, _mm256_set1_epi32(0x40)), nan);
    _mm_storeu_si128((__m128i *)((uint16_t *)data + index),
                     _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}

SIMD static size_t convert_vectors(uint8_t *output, const uint8_t *input, size_t count,
        uint32_t input_dtype, uint32_t output_dtype) {
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        store8(output, i, output_dtype, load8(input, i, input_dtype));
    }
    return i;
}

static void convert_chunk(
    const Task *task, size_t output_index, size_t input_index, size_t count, bool vectorize
) {
    const Parameters *p = task->p;
    uint8_t *output = task->output + output_index * task->output_bytes;
    const uint8_t *input = task->input + input_index * task->input_bytes;
    size_t i = 0;
    if (p->input_dtype == TRILLIM_BF16 && p->output_dtype == TRILLIM_FP32) {
        i = convert_vectors(output, input, count, TRILLIM_BF16, TRILLIM_FP32);
    } else if (p->input_dtype == TRILLIM_FP32 && p->output_dtype == TRILLIM_BF16) {
        i = convert_vectors(output, input, count, TRILLIM_FP32, TRILLIM_BF16);
    } else if (vectorize) {
        i = convert_vectors(output, input, count, p->input_dtype, p->output_dtype);
    }
    for (; i < count; ++i) {
        trillim_store(output, i, p->output_dtype, trillim_load(input, i, p->input_dtype));
    }
}

SIMD static void convert_strided(const Task *task, size_t output_index, size_t input_index, size_t count, size_t stride) {
    const Parameters *p = task->p;
    size_t i=0;
    if (p->input_dtype == TRILLIM_FP32 && stride <= INT32_MAX / 7) {
        __m256i offsets=_mm256_mullo_epi32(_mm256_setr_epi32(0,1,2,3,4,5,6,7),_mm256_set1_epi32((int32_t)stride));
        const float *input=(const float *)task->input+input_index;
        for (;count-i>=8;i+=8) {
            store8(task->output,output_index+i,p->output_dtype,_mm256_i32gather_ps(input+i*stride,offsets,4));
        }
    }
    for (;i<count;++i) {
        TrillimValue value=trillim_load(task->input,input_index+i*stride,p->input_dtype);
        trillim_store(task->output,output_index+i,p->output_dtype,value);
    }
}

static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    bool vectorize = true;
    size_t index = tile * TILE_ELEMENTS;
    size_t limit = p->count - index < TILE_ELEMENTS ? p->count : index + TILE_ELEMENTS;
    while (index < limit) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t length = limit - index < CHUNK_ELEMENTS ? limit - index : CHUNK_ELEMENTS;
        size_t source = index, stride = 1;
        if (!task->contiguous) {
            size_t width = p->shape[p->rank - 1];
            size_t remaining = width - index % width;
            if (remaining < length) {
                length = remaining;
            }
            source = trillim_offset(index, p->rank, p->shape, p->strides);
            stride = p->strides[p->rank - 1];
        }
        if (stride == 1) convert_chunk(task,index,source,length,vectorize);
        else convert_strided(task,index,source,length,stride);
        index += length;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_cast_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    if (p->count == 0) {
        return TRILLIM_OK;
    }
    Task task = {
        outputs[0], inputs[0], p, context,
        trillim_itemsize(p->input_dtype), trillim_itemsize(p->output_dtype),
        trillim_contiguous(p->rank, p->shape, p->strides)
    };
    size_t tiles = p->count / TILE_ELEMENTS + (p->count % TILE_ELEMENTS != 0);
    if (tiles > 1 && context->thread_pool->worker_count > 1) {
        return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
    }
    for (size_t tile = 0; tile < tiles; ++tile) {
        TrillimStatus status = apply(&task, tile, 0);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_x86_cast_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _cast\n"
           "def bind(context):\n"
           "    source, output = context.inputs['x'], context.outputs[0]\n"
           "    floats = ('fp16', 'bf16', 'fp32')\n"
           "    if source.dtype not in floats or output.dtype not in floats or source.dtype == output.dtype:\n"
           "        raise ValueError('floating cast requires different floating dtypes')\n"
           "    p = _cast(context)\n"
           "    return {name: p[name] for name in ('count', 'rank', 'shape', 'strides', 'input_dtype', 'output_dtype')}\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_cast_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"strides", "size_t", offsetof(Parameters, strides), "rank"},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_cast_avx2_capabilities(void) {
    /* Keep the scalar Cast capability for default kernel selection on macOS. */
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM
    };
    return &capabilities;
}
