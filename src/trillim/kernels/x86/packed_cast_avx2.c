#define TRILLIM_ENABLE_PACKED_AVX2
#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"

#define AVX2 __attribute__((target("avx2")))

typedef struct {
    size_t rows;
    size_t columns;
    uint32_t input_dtype;
    uint32_t output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const Parameters *parameters;
    const TrillimKernelContext *context;
} Task;

static bool packed_dtype(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static bool has_direct_encoder(uint32_t dtype) {
    return dtype == TRILLIM_ROW_INT8 || dtype == TRILLIM_GRP_INT8 ||
           dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = outputs[0] = *scratch = 0;
    if (p->rows == 0 || p->columns == 0 ||
        !((packed_dtype(p->input_dtype) && trillim_float(p->output_dtype)) ||
          (trillim_float(p->input_dtype) && packed_dtype(p->output_dtype))) ||
        !trillim_packed_payload_size(p->rows, p->columns, p->input_dtype, &inputs[0]) ||
        !trillim_packed_payload_size(p->rows, p->columns, p->output_dtype, &outputs[0])) {
        return TRILLIM_ERROR;
    }
    if (!packed_dtype(p->input_dtype) && !has_direct_encoder(p->output_dtype) &&
        p->input_dtype != TRILLIM_FP32) {
        size_t count = p->rows * p->columns;
        if (count > SIZE_MAX / sizeof(float)) {
            return TRILLIM_ERROR;
        }
        *scratch = count * sizeof(float);
    }
    return TRILLIM_OK;
}

AVX2 static TrillimStatus decode(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t count = p->rows * p->columns;
    size_t index = tile * 1024;
    size_t end = count - index < 1024 ? count : index + 1024;
    while (index < end) {
        size_t row = index / p->columns;
        size_t column = index % p->columns;
        size_t limit = p->columns - column < end - index ? p->columns - column : end - index;
        size_t row_end = index + limit;
        for (; index < row_end && (column & 7) != 0; ++index, ++column) {
            TrillimValue value = {.f = trillim_packed_read_float(
                task->input, index, p->input_dtype, p->rows, p->columns)};
            trillim_store(task->output, index, p->output_dtype, value);
        }
        for (; row_end - index >= 8; index += 8, column += 8) {
            __m256 values = trillim_packed_decode8(task->input, p->rows, p->columns, p->input_dtype, row, column);
            trillim_packed_store8(task->output, index, p->output_dtype, values);
        }
        for (; index < row_end; ++index) {
            TrillimValue value = {.f = trillim_packed_read_float(
                task->input, index, p->input_dtype, p->rows, p->columns)};
            trillim_store(task->output, index, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

AVX2 static __m256 load8(const Task *task, size_t index) {
    uint32_t dtype = task->parameters->input_dtype;
    if (dtype == TRILLIM_FP32) {
        return _mm256_loadu_ps((const float *)task->input + index);
    }
    __m128i half = _mm_loadu_si128((const __m128i *)((const uint16_t *)task->input + index));
    __m256i bits = _mm256_cvtepu16_epi32(half);
    if (dtype == TRILLIM_BF16) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
    }
    __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bits, 10), _mm256_set1_epi32(31));
    __m256i mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(1023));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)), 16);
    __m256i adjusted = _mm256_add_epi32(exponent, _mm256_set1_epi32(112));
    adjusted = _mm256_blendv_epi8(adjusted, _mm256_set1_epi32(255),
                                 _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(31)));
    __m256i normal = _mm256_or_si256(_mm256_slli_epi32(adjusted, 23), _mm256_slli_epi32(mantissa, 13));
    __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_set1_ps(0x1p-24f));
    __m256i magnitude = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal),
                                          _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    return _mm256_castsi256_ps(_mm256_or_si256(sign, magnitude));
}

AVX2 static TrillimStatus validate_and_stage(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t count = p->rows * p->columns;
    size_t index = tile * 1024;
    size_t end = count - index < 1024 ? count : index + 1024;
    float *staged = task->output;
    for (; end - index >= 8; index += 8) {
        __m256 values = load8(task, index);
        if (!trillim_packed_finite8(values)) {
            return TRILLIM_ERROR;
        }
        if (staged != NULL) {
            _mm256_storeu_ps(staged + index, values);
        }
    }
    for (; index < end; ++index) {
        float value = trillim_load(task->input, index, p->input_dtype).f;
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        if (staged != NULL) {
            staged[index] = value;
        }
    }
    return TRILLIM_OK;
}

AVX2 static __m256i round8(__m256 values, int32_t lower, int32_t upper) {
    values = _mm256_min_ps(_mm256_max_ps(values, _mm256_set1_ps((float)lower)), _mm256_set1_ps((float)upper));
    return _mm256_cvttps_epi32(_mm256_round_ps(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

AVX2 static __m128i pack8(__m256i codes) {
    __m128i halves = _mm_packs_epi32(_mm256_castsi256_si128(codes), _mm256_extracti128_si256(codes, 1));
    return _mm_packs_epi16(halves, halves);
}

static int8_t round_row_value(float value) {
    float magnitude = fminf(fabsf(value), 127.0f);
    int32_t integer = (int32_t)magnitude;
    float fraction = magnitude - (float)integer;
    if (fraction > 0.5f || (fraction == 0.5f && (integer & 1))) {
        ++integer;
    }
    return (int8_t)(value < 0 ? -integer : integer);
}

AVX2 static TrillimStatus encode_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t first = row * p->columns;
    size_t column = 0;
    __m256 max8 = _mm256_set1_ps(1e-5f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (; p->columns - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 values = load8(task, first + column);
        if (!trillim_packed_finite8(values)) {
            return TRILLIM_ERROR;
        }
        max8 = _mm256_max_ps(max8, _mm256_andnot_ps(sign, values));
    }
    float maximum = 1e-5f;
    float lanes[8];
    _mm256_storeu_ps(lanes, max8);
    for (size_t i = 0; i < 8; ++i) {
        maximum = fmaxf(maximum, lanes[i]);
    }
    for (; column < p->columns; ++column) {
        float value = trillim_load(task->input, first + column, p->input_dtype).f;
        if (!isfinite(value)) {
            return TRILLIM_ERROR;
        }
        maximum = fmaxf(maximum, fabsf(value));
    }
    uint8_t *output = task->output;
    uint16_t scale = trillim_to_bf16(maximum / 127.0f);
    memcpy(output + p->rows * p->columns + row * sizeof(scale), &scale, sizeof(scale));
    float multiplier = 127.0f / maximum;
    __m256 ratio = _mm256_set1_ps(multiplier);
    column = 0;
    for (; p->columns - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256i codes = round8(_mm256_mul_ps(load8(task, first + column), ratio), -127, 127);
        _mm_storel_epi64((__m128i *)(output + first + column), pack8(codes));
    }
    for (; column < p->columns; ++column) {
        float value = trillim_load(task->input, first + column, p->input_dtype).f;
        int8_t code = round_row_value(value * multiplier);
        memcpy(output + first + column, &code, sizeof(code));
    }
    return TRILLIM_OK;
}

AVX2 static TrillimStatus encode_group(const Task *task, size_t block, size_t groups) {
    const Parameters *p = task->parameters;
    size_t row = block / groups;
    size_t column = block % groups * 32;
    size_t first = row * p->columns + column;
    size_t count = p->columns - column < 32 ? p->columns - column : 32;
    __m256 input[4];
    __m256 high = _mm256_setzero_ps(), low = _mm256_setzero_ps();
    for (size_t lane = 0; lane < 4; ++lane) {
        size_t start = lane * 8;
        if (count >= start + 8) {
            input[lane] = load8(task, first + start);
        } else {
            float tail[8] = {0};
            for (size_t i = 0; i < 8 && start + i < count; ++i) {
                tail[i] = trillim_load(task->input, first + start + i, p->input_dtype).f;
            }
            input[lane] = _mm256_loadu_ps(tail);
        }
        if (!trillim_packed_finite8(input[lane])) {
            return TRILLIM_ERROR;
        }
        high = _mm256_max_ps(high, input[lane]);
        low = _mm256_min_ps(low, input[lane]);
    }
    float positive[8], negative[8];
    _mm256_storeu_ps(positive, high);
    _mm256_storeu_ps(negative, low);
    float max_value = 0.0f, min_value = 0.0f;
    for (size_t i = 0; i < 8; ++i) {
        max_value = fmaxf(max_value, positive[i]);
        min_value = fminf(min_value, negative[i]);
    }
    bool int8 = p->output_dtype == TRILLIM_GRP_INT8;
    int32_t bound = int8 ? 127 : (p->output_dtype == TRILLIM_GRP_INT4 ? 8 : 16);
    int32_t upper = int8 ? 127 : bound - 1;
    max_value = fabsf(max_value);
    float negative_magnitude = fabsf(min_value);
    float scale = int8 ? fmaxf(max_value, negative_magnitude) / 127.0f :
        fmaxf(max_value / (float)upper, negative_magnitude / (float)bound);
    uint16_t bits = trillim_to_bf16(scale);
    size_t total = p->rows * groups;
    size_t value_width = int8 ? 32 : 16;
    uint8_t *output = task->output;
    memcpy(output + total * value_width + block * sizeof(bits), &bits, sizeof(bits));
    if (!int8) {
        scale = trillim_bf16(bits);
    }
    __m256 divisor = _mm256_set1_ps(scale == 0.0f ? 1.0f : scale);
    __m128i packed[4];
    __m256i bias = _mm256_set1_epi32(int8 ? 0 : bound);
    for (size_t lane = 0; lane < 4; ++lane) {
        __m256i codes = round8(_mm256_div_ps(input[lane], divisor), -bound, upper);
        packed[lane] = pack8(_mm256_add_epi32(codes, bias));
    }
    uint8_t *codes = output + block * value_width;
    if (int8) {
        for (size_t lane = 0; lane < 4; ++lane) {
            _mm_storel_epi64((__m128i *)(codes + lane * 8), packed[lane]);
        }
        return TRILLIM_OK;
    }
    __m128i mask = _mm_set1_epi8(15);
    for (size_t lane = 0; lane < 2; ++lane) {
        __m128i a = _mm_and_si128(packed[lane], mask);
        __m128i b = _mm_slli_epi16(_mm_and_si128(packed[lane + 2], mask), 4);
        _mm_storel_epi64((__m128i *)(codes + lane * 8), _mm_or_si128(a, b));
    }
    if (p->output_dtype == TRILLIM_GRP_INT5) {
        uint8_t *high_bits = output + total * (16 + 2) + block * 4;
        for (size_t lane = 0; lane < 4; ++lane) {
            high_bits[lane] = (uint8_t)_mm_movemask_epi8(_mm_slli_epi16(packed[lane], 3));
        }
    }
    return TRILLIM_OK;
}

static TrillimStatus encode_groups(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t groups = p->columns / 32 + (p->columns % 32 != 0);
    size_t count = p->rows * groups, first = tile * 16;
    size_t end = count - first < 16 ? count : first + 16;
    for (size_t block = first; block < end; ++block) {
        TrillimStatus status = encode_group(task, block, groups);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_packed_cast_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], p, context};
    TrillimTask operation;
    size_t tiles;
    if (packed_dtype(p->input_dtype)) {
        size_t count = p->rows * p->columns;
        tiles = count / 1024 + (count % 1024 != 0);
        operation = decode;
    } else if (p->output_dtype == TRILLIM_ROW_INT8) {
        tiles = p->rows;
        operation = encode_row;
    } else if (has_direct_encoder(p->output_dtype)) {
        size_t groups = p->columns / 32 + (p->columns % 32 != 0);
        size_t count = p->rows * groups;
        tiles = count / 16 + (count % 16 != 0);
        operation = encode_groups;
    } else {
        size_t count = p->rows * p->columns;
        size_t chunks = count / 1024 + (count % 1024 != 0);
        bool stage = p->input_dtype != TRILLIM_FP32;
        Task preparation = {stage ? scratch : NULL, inputs[0], p, context};
        TrillimStatus status = chunks == 1 ? validate_and_stage(&preparation, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, chunks, validate_and_stage, &preparation);
        if (status != TRILLIM_OK) {
            return status;
        }
        const float *source = stage ? scratch : inputs[0];
        return trillim_packed_encode_quant(source, p->rows, p->columns, p->output_dtype, outputs[0], context);
    }
    return tiles == 1 ? operation(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, operation, &task);
}

TRILLIM_KERNEL const char *trillim_x86_packed_cast_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _storage_shape\n"
           "def bind(context):\n"
           "    source, output = context.inputs['x'], context.outputs[0]\n"
           "    floats = ('fp32', 'fp16', 'bf16')\n"
           "    if not ((_DTYPE_IDS[source.dtype] >= _DTYPE_IDS['row_int8'] and output.dtype in floats) or (source.dtype in floats and _DTYPE_IDS[output.dtype] >= _DTYPE_IDS['row_int8'])):\n"
           "        raise ValueError('AVX2 packed cast requires a supported packed and floating dtype')\n"
           "    if source.shape != output.shape or _storage_shape(source) != _storage_shape(output):\n"
           "        raise ValueError('AVX2 packed cast requires matching canonical layouts')\n"
           "    if source.strides and source.strides != _contiguous_strides(source.shape):\n"
           "        raise ValueError('AVX2 packed cast requires contiguous input')\n"
           "    rows, columns = _storage_shape(output)\n"
           "    return dict(rows=rows, columns=columns, input_dtype=_DTYPE_IDS[source.dtype], output_dtype=_DTYPE_IDS[output.dtype])\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_packed_cast_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"columns", "size_t", offsetof(Parameters, columns), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {TRILLIM_ALIGNOF(float)};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]),
        fields, 1, 1, requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(float)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_packed_cast_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &capabilities;
}
