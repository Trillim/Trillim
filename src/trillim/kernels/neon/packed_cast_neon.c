#define TRILLIM_ENABLE_PACKED_NEON
#define TRILLIM_ENABLE_PACKED_ENCODER
#include "common.h"
#include <arm_neon.h>
#include <float.h>

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

static uint16x4_t to_bf16(float32x4_t values) {
    uint32x4_t bits = vreinterpretq_u32_f32(values);
    uint32x4_t high = vshrq_n_u32(bits, 16);
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(high, vdupq_n_u32(1)));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(bits, bias), 16);
    uint32x4_t quiet = vorrq_u32(high, vdupq_n_u32(0x40));
    uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(UINT32_C(0x7fffffff)));
    uint32x4_t nan = vcgtq_u32(magnitude, vdupq_n_u32(UINT32_C(0x7f800000)));
    return vmovn_u32(vbslq_u32(nan, quiet, rounded));
}

static void store4(void *output, size_t index, uint32_t dtype, float32x4_t values) {
    uint8_t *destination = (uint8_t *)output + index * trillim_itemsize(dtype);
    if (dtype == TRILLIM_FP32) {
        memcpy(destination, &values, sizeof(values));
    } else {
        uint16x4_t bits;
        if (dtype == TRILLIM_BF16) {
            bits = to_bf16(values);
        } else {
            bits = vreinterpret_u16_f16(vcvt_f16_f32(values));
        }
        memcpy(destination, &bits, sizeof(bits));
    }
}

static TrillimStatus decode(void *opaque, size_t tile, size_t worker) {
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
        for (; index < row_end && (column & 15) != 0; ++index, ++column) {
            TrillimValue value = {.f = trillim_packed_read_float(
                task->input, index, p->input_dtype, p->rows, p->columns)};
            trillim_store(task->output, index, p->output_dtype, value);
        }
        if (row_end - index >= 16) {
            TrillimPackedNeonRow packed_row = trillim_packed_neon_row(
                task->input, p->input_dtype, row, p->columns, p->rows
            );
            for (; row_end - index >= 16; index += 16, column += 16) {
                float32x4_t values[4];
                trillim_packed_neon_decode16(&packed_row, column, values);
                for (size_t lane = 0; lane < 4; ++lane) {
                    store4(task->output, index + lane * 4, p->output_dtype, values[lane]);
                }
            }
        }
        for (; index < row_end; ++index) {
            TrillimValue value = {.f = trillim_packed_read_float(
                task->input, index, p->input_dtype, p->rows, p->columns)};
            trillim_store(task->output, index, p->output_dtype, value);
        }
    }
    return TRILLIM_OK;
}

static bool finite4(float32x4_t values) {
    return vminvq_u32(vcleq_f32(vabsq_f32(values), vdupq_n_f32(FLT_MAX))) != 0;
}

static TrillimStatus validate_and_stage(void *opaque, size_t tile, size_t worker) {
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
    for (; end - index >= 4; index += 4) {
        float32x4_t values = trillim_neon_load4(task->input, index, p->input_dtype);
        if (!finite4(values)) {
            return TRILLIM_ERROR;
        }
        if (staged != NULL) {
            vst1q_f32(staged + index, values);
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

static int32x4_t round4(float32x4_t values, int32_t lower, int32_t upper) {
    values = vmaxnmq_f32(values, vdupq_n_f32((float)lower));
    return vcvtnq_s32_f32(vminq_f32(values, vdupq_n_f32((float)upper)));
}

static int8x8_t pack8(int32x4_t low, int32x4_t high) {
    return vmovn_s16(vcombine_s16(vmovn_s32(low), vmovn_s32(high)));
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

static TrillimStatus encode_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    if (task->context->is_cancelled(task->context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t first = row * p->columns;
    size_t column = 0;
    float32x4_t max4 = vdupq_n_f32(1e-5f);
    for (; p->columns - column >= 4; column += 4) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t values = trillim_neon_load4(task->input, first + column, p->input_dtype);
        if (!finite4(values)) {
            return TRILLIM_ERROR;
        }
        max4 = vmaxq_f32(max4, vabsq_f32(values));
    }
    float maximum = vmaxvq_f32(max4);
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
    column = 0;
    for (; p->columns - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        int32x4_t low = round4(vmulq_n_f32(trillim_neon_load4(task->input, first + column, p->input_dtype), multiplier), -127, 127);
        int32x4_t high = round4(vmulq_n_f32(trillim_neon_load4(task->input, first + column + 4, p->input_dtype), multiplier), -127, 127);
        vst1_s8((int8_t *)output + first + column, pack8(low, high));
    }
    for (; column < p->columns; ++column) {
        float value = trillim_load(task->input, first + column, p->input_dtype).f;
        int8_t code = round_row_value(value * multiplier);
        memcpy(output + first + column, &code, sizeof(code));
    }
    return TRILLIM_OK;
}

static TrillimStatus encode_group(const Task *task, size_t block, size_t groups) {
    const Parameters *p = task->parameters;
    size_t row = block / groups;
    size_t column = block % groups * 32;
    size_t first = row * p->columns + column;
    size_t count = p->columns - column < 32 ? p->columns - column : 32;
    float32x4_t input[8], high = vdupq_n_f32(0.0f), low = vdupq_n_f32(0.0f);
    for (size_t lane = 0; lane < 8; ++lane) {
        size_t start = lane * 4;
        if (count >= start + 4) {
            input[lane] = trillim_neon_load4(task->input, first + start, p->input_dtype);
        } else {
            float tail[4] = {0};
            for (size_t i = 0; i < 4 && start + i < count; ++i) {
                tail[i] = trillim_load(task->input, first + start + i, p->input_dtype).f;
            }
            input[lane] = vld1q_f32(tail);
        }
        if (!finite4(input[lane])) {
            return TRILLIM_ERROR;
        }
        high = vmaxq_f32(high, input[lane]);
        low = vminq_f32(low, input[lane]);
    }
    float max_value = vmaxvq_f32(high), min_value = vminvq_f32(low);
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
    float32x4_t divisor = vdupq_n_f32(scale == 0.0f ? 1.0f : scale);
    uint8x8_t packed[4];
    int32x4_t bias = vdupq_n_s32(int8 ? 0 : bound);
    for (size_t lane = 0; lane < 4; ++lane) {
        int32x4_t a = round4(vdivq_f32(input[lane * 2], divisor), -bound, upper);
        int32x4_t b = round4(vdivq_f32(input[lane * 2 + 1], divisor), -bound, upper);
        packed[lane] = vreinterpret_u8_s8(pack8(vaddq_s32(a, bias), vaddq_s32(b, bias)));
    }
    uint8_t *codes = output + block * value_width;
    if (int8) {
        for (size_t lane = 0; lane < 4; ++lane) {
            vst1_u8(codes + lane * 8, packed[lane]);
        }
        return TRILLIM_OK;
    }
    for (size_t lane = 0; lane < 2; ++lane) {
        uint8x8_t a = vand_u8(packed[lane], vdup_n_u8(15));
        uint8x8_t b = vshl_n_u8(packed[lane + 2], 4);
        vst1_u8(codes + lane * 8, vorr_u8(a, b));
    }
    if (p->output_dtype == TRILLIM_GRP_INT5) {
        static const int8_t shifts[] = {0, 1, 2, 3, 4, 5, 6, 7};
        uint8_t *high_bits = output + total * 18 + block * 4;
        for (size_t lane = 0; lane < 4; ++lane) {
            uint8x8_t high_bit = vshr_n_u8(packed[lane], 4);
            high_bits[lane] = vaddv_u8(vshl_u8(high_bit, vld1_s8(shifts)));
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

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_cast_neon(
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

TRILLIM_KERNEL const char *trillim_neon_packed_cast_neon_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _storage_shape\n"
           "def bind(context):\n"
           "    source, output = context.inputs['x'], context.outputs[0]\n"
           "    floats = ('fp32', 'fp16', 'bf16')\n"
           "    if not ((_DTYPE_IDS[source.dtype] >= _DTYPE_IDS['row_int8'] and output.dtype in floats) or (source.dtype in floats and _DTYPE_IDS[output.dtype] >= _DTYPE_IDS['row_int8'])):\n"
           "        raise ValueError('NEON packed cast requires a supported packed and floating dtype')\n"
           "    if source.shape != output.shape or _storage_shape(source) != _storage_shape(output):\n"
           "        raise ValueError('NEON packed cast requires matching canonical layouts')\n"
           "    if source.strides and source.strides != _contiguous_strides(source.shape):\n"
           "        raise ValueError('NEON packed cast requires contiguous input')\n"
           "    rows, columns = _storage_shape(output)\n"
           "    return dict(rows=rows, columns=columns, input_dtype=_DTYPE_IDS[source.dtype], output_dtype=_DTYPE_IDS[output.dtype])\n"
           "def pattern(x: Tensor, *, dtype: DType) -> tuple[Tensor, ...]:\n"
           "    return (Cast(x, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_cast_neon_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_cast_neon_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
    return &capabilities;
}
