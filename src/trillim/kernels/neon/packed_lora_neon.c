
#include "common.h"
#include <arm_neon.h>

typedef struct {
    const uint8_t *data;
    uint32_t dtype;
    unsigned group_shift;
    size_t first_element, first_block, blocks;
    float row_scale;
} TrillimPackedNeonRow;

static inline TrillimPackedNeonRow trillim_packed_neon_row(
    const void *weight, uint32_t dtype, size_t column, size_t width, size_t outputs
) {
    TrillimPackedNeonRow row = {.data = weight, .dtype = dtype, .first_element = column * width};
    if (width == 0) {
        return row;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        uint16_t bits;
        memcpy(&bits, row.data + outputs * width + column * sizeof(bits), sizeof(bits));
        row.row_scale = trillim_bf16(bits);
    } else if (dtype >= TRILLIM_MAT_TERNARY && dtype <= TRILLIM_MXFP4) {
        row.group_shift = 5;
        if (dtype == TRILLIM_DUBGRP_INT6) {
            row.group_shift = 8;
        } else if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
            row.group_shift = 7;
        }
        size_t group = (size_t)1 << row.group_shift;
        size_t groups = (width >> row.group_shift) + (width % group != 0);
        row.first_block = column * groups;
        row.blocks = outputs * groups;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&row.row_scale, row.data + row.blocks * 32, sizeof(row.row_scale));
        }
    }
    return row;
}

static inline float32x4_t trillim_neon_load4(const void *data, size_t index, uint32_t dtype) {
    const uint8_t *bytes = data;
    if (dtype == TRILLIM_FP32) {
        float32x4_t values;
        memcpy(&values, bytes + index * sizeof(float), sizeof(values));
        return values;
    }
    uint16x4_t bits;
    memcpy(&bits, bytes + index * sizeof(uint16_t), sizeof(bits));
    if (dtype == TRILLIM_BF16) {
        return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(bits));
}

static inline float32x4_t trillim_packed_neon_convert4(int16x4_t codes, float scale) {
    return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(codes)), scale);
}

/* k is sixteen-aligned and the read stays within the logical row. */
static inline __attribute__((always_inline)) void trillim_packed_neon_decode16(
    const TrillimPackedNeonRow *row, size_t k, float32x4_t result[4]
) {
    const uint8_t *values = row->data;
    uint32_t dtype = row->dtype;
    size_t total = row->blocks;
    size_t block = row->first_block + (k >> row->group_shift);
    size_t offset = k & (((size_t)1 << row->group_shift) - 1);
    int8x16_t codes;
    float scale;
    if (dtype == TRILLIM_ROW_INT8) {
        scale = row->row_scale;
        codes = vld1q_s8((const int8_t *)values + row->first_element + k);
    } else if (dtype == TRILLIM_MXFP4) {
        uint8x8_t packed = vld1_u8(values + block * 16 + offset / 2);
        uint8x8x2_t split = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
        uint8x16_t code = vcombine_u8(split.val[0], split.val[1]);
        /* Exact E2M1 values assembled from their two BF16 bytes. */
        static const uint8_t low_bits[16] = {0, 0, 128, 192, 0, 64, 128, 192, 0, 0, 128, 192, 0, 64, 128, 192};
        static const uint8_t high_bits[16] = {0, 63, 63, 63, 64, 64, 64, 64, 128, 191, 191, 191, 192, 192, 192, 192};
        uint8x16_t low = vqtbl1q_u8(vld1q_u8(low_bits), code);
        uint8x16_t high = vqtbl1q_u8(vld1q_u8(high_bits), code);
        uint16x8_t a = vreinterpretq_u16_u8(vzip1q_u8(low, high));
        uint16x8_t b = vreinterpretq_u16_u8(vzip2q_u8(low, high));
        result[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16));
        result[1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a), 16));
        result[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
        result[3] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16));
        scale = trillim_mxfp4_scale(values[total * 16 + block]);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vmulq_n_f32(result[i], scale);
        }
        return;
    } else if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        uint8x16_t packed = vld1q_u8(values + block * 32 + offset % 32);
        uint8x16_t shifted = vshlq_u8(packed, vdupq_n_s8(-(int8_t)(6 - (offset / 32) * 2)));
        codes = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(shifted, vdupq_n_u8(3))), vdupq_n_s8(1));
        if (dtype == TRILLIM_MAT_TERNARY) {
            scale = row->row_scale;
        } else {
            uint16_t bits;
            memcpy(&bits, values + total * 32 + block * 2, 2);
            scale = trillim_bf16(bits);
        }
    } else {
        size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
        size_t scale_start = total * value_width;
        uint16_t bits;
        memcpy(&bits, values + scale_start + block * 2, 2);
        scale = trillim_bf16(bits);
        if (dtype == TRILLIM_GRP_INT8) {
            codes = vld1q_s8((const int8_t *)values + block * 32 + offset);
        } else if (dtype == TRILLIM_GRP_BINARY) {
            const uint8_t *packed = values + block * 16 + offset / 8;
            uint8x16_t bytes = vcombine_u8(vdup_n_u8(packed[0]), vdup_n_u8(packed[1]));
            static const int8_t shifts[16] = {-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0};
            uint8x16_t code = vandq_u8(vshlq_u8(bytes, vld1q_s8(shifts)), vdupq_n_u8(1));
            codes = vsubq_s8(vreinterpretq_s8_u8(vshlq_n_u8(code, 1)), vdupq_n_s8(1));
        } else if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
            uint8x16_t packed = vld1q_u8(values + block * 16);
            uint8x16_t code = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(int8_t)((offset / 16) * 4))), vdupq_n_u8(15));
            if (dtype == TRILLIM_GRP_INT5) {
                const uint8_t *high = values + scale_start + total * 2 + block * 4 + offset / 8;
                uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[0]), vdup_n_u8(high[1]));
                static const int8_t shifts[16] = {0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7};
                uint8x16_t upper = vshlq_n_u8(vandq_u8(vshlq_u8(bytes, vld1q_s8(shifts)), vdupq_n_u8(1)), 4);
                code = vorrq_u8(code, upper);
            }
            codes = vsubq_s8(vreinterpretq_s8_u8(code), vdupq_n_s8(dtype == TRILLIM_GRP_INT4 ? 8 : 16));
        } else {
            size_t half = offset / 128;
            size_t lane = (offset % 128) / 32;
            uint8x16_t packed = vld1q_u8(values + block * 128 + half * 64 + (lane / 2) * 32 + offset % 32);
            uint8x16_t high = vld1q_u8(values + scale_start + total * 2 + block * 64 + half * 32 + offset % 32);
            uint8x16_t low = vandq_u8(vshlq_u8(packed, vdupq_n_s8(-(int8_t)((lane % 2) * 4))), vdupq_n_u8(15));
            uint8x16_t upper = vshlq_n_u8(vandq_u8(vshlq_u8(high, vdupq_n_s8(-(int8_t)(lane * 2))), vdupq_n_u8(3)), 4);
            codes = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, upper)), vdupq_n_s8(32));
            int8_t subscale;
            memcpy(&subscale, values + scale_start + total * (2 + 64) + block * 16 + offset / 16, 1);
            int16x8_t a = vmulq_n_s16(vmovl_s8(vget_low_s8(codes)), subscale);
            int16x8_t b = vmulq_n_s16(vmovl_s8(vget_high_s8(codes)), subscale);
            result[0] = trillim_packed_neon_convert4(vget_low_s16(a), scale);
            result[1] = trillim_packed_neon_convert4(vget_high_s16(a), scale);
            result[2] = trillim_packed_neon_convert4(vget_low_s16(b), scale);
            result[3] = trillim_packed_neon_convert4(vget_high_s16(b), scale);
            return;
        }
    }
    int16x8_t low = vmovl_s8(vget_low_s8(codes)), high = vmovl_s8(vget_high_s8(codes));
    result[0] = trillim_packed_neon_convert4(vget_low_s16(low), scale);
    result[1] = trillim_packed_neon_convert4(vget_high_s16(low), scale);
    result[2] = trillim_packed_neon_convert4(vget_low_s16(high), scale);
    result[3] = trillim_packed_neon_convert4(vget_high_s16(high), scale);
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        int32x4_t code[4] = {vmovl_s16(vget_low_s16(low)), vmovl_s16(vget_high_s16(low)),
            vmovl_s16(vget_low_s16(high)), vmovl_s16(vget_high_s16(high))};
        for (size_t i = 0; i < 4; ++i) {
            result[i] = vbslq_f32(vceqq_s32(code[i], vdupq_n_s32(2)), vdupq_n_f32(NAN), result[i]);
        }
    }
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { TILE_SIZE = 16 };

typedef struct {
    size_t tokens, width, outputs;
    size_t hidden_width;
    uint32_t input_dtype, output_dtype;
    uint32_t weight_dtype, a_dtype, b_dtype, hidden_dtype, delta_dtype, scale_dtype, scaled_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *const *inputs;
    void *hidden;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

static bool packed(uint32_t dtype) {
    return dtype >= TRILLIM_ROW_INT8 && dtype <= TRILLIM_MXFP4;
}

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    *scratch = outputs[0] = 0;
    memset(inputs, 0, 5 * sizeof(*inputs));
    if (!trillim_float(p->input_dtype) ||
        !trillim_float(p->output_dtype) ||
        !packed(p->weight_dtype) ||
        (!trillim_float(p->a_dtype) && !packed(p->a_dtype)) ||
        (!trillim_float(p->b_dtype) && !packed(p->b_dtype)) ||
        !trillim_float(p->hidden_dtype) ||
        !trillim_float(p->delta_dtype) ||
        !trillim_float(p->scale_dtype) ||
        !trillim_float(p->scaled_dtype)) {
        return TRILLIM_ERROR;
    }
    if (p->tokens == 0 || p->outputs == 0) {
        return TRILLIM_OK;
    }
    if (!trillim_packed_payload_size(p->tokens, p->outputs, p->output_dtype, outputs) ||
        !trillim_packed_payload_size(p->tokens, p->width, p->input_dtype, inputs)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->width, p->weight_dtype, inputs + 1)) {
        return TRILLIM_ERROR;
    }
    if (p->width != 0 && p->hidden_width != 0 &&
        !trillim_packed_payload_size(p->hidden_width, p->width, p->a_dtype, inputs + 2)) {
        return TRILLIM_ERROR;
    }
    if (p->hidden_width != 0 && p->outputs != 0 &&
        !trillim_packed_payload_size(p->outputs, p->hidden_width, p->b_dtype, inputs + 3)) {
        return TRILLIM_ERROR;
    }
    if (!trillim_packed_payload_size(p->tokens, p->hidden_width, p->hidden_dtype, scratch)) {
        return TRILLIM_ERROR;
    }
    inputs[4] = trillim_itemsize(p->scale_dtype);
    return TRILLIM_OK;
}

static inline __attribute__((always_inline)) TrillimStatus dot(
    const void *input, uint32_t input_dtype, size_t row,
    const void *weight, uint32_t weight_dtype, size_t column,
    size_t width, size_t outputs, bool ordered, const TrillimKernelContext *context, float *result
) {
    if (width == 0) {
        *result = 0.0f;
        return TRILLIM_OK;
    }
    size_t x_start = row * width;
    TrillimPackedNeonRow w = trillim_packed_neon_row(weight, weight_dtype, column, width, outputs);
    size_t k = 0;
    float sum = 0.0f;
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; width - k >= 16; k += 16) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float32x4_t decoded[4];
        if (packed(weight_dtype)) {
            trillim_packed_neon_decode16(&w, k, decoded);
        } else {
            for (size_t i = 0; i < 4; ++i) {
                decoded[i] = trillim_neon_load4(weight, w.first_element + k + i * 4, weight_dtype);
            }
        }
        for (size_t i = 0; i < 4; ++i) {
            float32x4_t x = trillim_neon_load4(input, x_start + k + i * 4, input_dtype);
            float32x4_t products = vmulq_f32(x, decoded[i]);
            if (ordered) {
                /* Preserve scalar addition order before narrow rounding. */
                sum += vgetq_lane_f32(products, 0);
                sum += vgetq_lane_f32(products, 1);
                sum += vgetq_lane_f32(products, 2);
                sum += vgetq_lane_f32(products, 3);
            } else {
                sums[i] = vaddq_f32(sums[i], products);
            }
        }
    }
    if (!ordered) {
        float32x4_t a = vaddq_f32(sums[0], sums[2]);
        float32x4_t b = vaddq_f32(sums[1], sums[3]);
        sum = vaddvq_f32(vaddq_f32(a, b));
    }
    for (; k < width; ++k) {
        if (k % 256 == 0 && context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        float x = trillim_load(input, x_start + k, input_dtype).f;
        float decoded = trillim_packed_read_float(weight, w.first_element + k, weight_dtype, outputs, width);
        sum += x * decoded;
    }
    *result = sum;
    return TRILLIM_OK;
}

static float round_to_dtype(float value, uint32_t dtype) {
    if (dtype == TRILLIM_BF16) {
        return trillim_bf16(trillim_to_bf16(value));
    }
    if (dtype == TRILLIM_FP16) {
        return trillim_fp16(trillim_to_fp16(value));
    }
    return value;
}

static TrillimStatus project_hidden(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->hidden_width;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->hidden_width, column = index % p->hidden_width;
        float value;
        TrillimStatus status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[2], p->a_dtype,
            column, p->width, p->hidden_width, p->hidden_dtype != TRILLIM_FP32, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        trillim_store(task->hidden, index, p->hidden_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t total = p->tokens * p->outputs;
    size_t first = tile * TILE_SIZE;
    size_t end = total - first < TILE_SIZE ? total : first + TILE_SIZE;
    bool narrow_output = p->output_dtype != TRILLIM_FP32;
    bool narrow_delta = p->delta_dtype != TRILLIM_FP32 || p->scaled_dtype != TRILLIM_FP32;
    for (size_t index = first; index < end; ++index) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t row = index / p->outputs, column = index % p->outputs;
        float value;
        TrillimStatus status = dot(
            task->inputs[0], p->input_dtype, row, task->inputs[1], p->weight_dtype,
            column, p->width, p->outputs, narrow_output, task->context, &value
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        float delta;
        status = dot(
            task->hidden, p->hidden_dtype, row, task->inputs[3], p->b_dtype,
            column, p->hidden_width, p->outputs, narrow_delta || narrow_output, task->context, &delta
        );
        if (status != TRILLIM_OK) {
            return status;
        }
        /* Matmul and Mul can each produce a narrow floating intermediate. */
        delta = round_to_dtype(delta, p->delta_dtype);
        float scale = trillim_load(task->inputs[4], 0, p->scale_dtype).f;
        value += round_to_dtype(delta * scale, p->scaled_dtype);
        trillim_store(task->output, index, p->output_dtype, (TrillimValue){.f = value});
    }
    return task->context->is_cancelled(task->context->cancellation_context)
        ? TRILLIM_CANCELLED : TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_neon_packed_lora(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    size_t input_sizes[5], output_size, scratch_size;
    TrillimStatus status = requirements(parameters, input_sizes, &output_size, &scratch_size);
    if (status != TRILLIM_OK || output_size == 0) {
        return status;
    }
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs, scratch, p, context};
    if (scratch_size != 0) {
        size_t count = p->tokens * p->hidden_width;
        size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
        status = tiles == 1 ? project_hidden(&task, 0, 0) :
            context->thread_pool->run(context->thread_pool->context, tiles, project_hidden, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
    }
    size_t count = p->tokens * p->outputs;
    size_t tiles = count / TILE_SIZE + (count % TILE_SIZE != 0);
    return tiles == 1 ? project(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, project, &task);
}

TRILLIM_KERNEL const char *trillim_neon_packed_lora_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _DTYPE_IDS, _product, _storage_shape\n"
           "from trillim.compiler.passes.infer import _promote\n"
           "physical_inputs = ('x', 'weight', 'a_weight', 'b_weight', 'scale')\n"
           "\n"
           "def bind(context):\n"
           "    x, weight, a_weight, b_weight, scale = (context.inputs[name] for name in physical_inputs)\n"
           "    output = context.outputs[0]\n"
           "    floating = {'fp16', 'bf16', 'fp32'}\n"
           "    packed = {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'}\n"
           "    if not x.shape or x.dtype not in floating or output.dtype not in floating:\n"
           "        raise ValueError('packed fusion requires floating activations and output')\n"
           "    for tensor in (*context.inputs.values(), output):\n"
           "        strides = _contiguous_strides(tensor.shape)\n"
           "        if (tensor.strides or strides) != strides:\n"
           "            raise ValueError('packed fusion requires contiguous tensors')\n"
           "    matrices = weight, a_weight, b_weight\n"
           "    for matrix in matrices:\n"
           "        if len(matrix.shape) != 2 or _storage_shape(matrix) != matrix.shape:\n"
           "            raise ValueError('packed fusion requires complete canonical matrices')\n"
           "    if weight.dtype not in packed or x.shape[-1] != weight.shape[1]:\n"
           "        raise ValueError('packed fusion requires compatible compressed base weights')\n"
           "    if any(matrix.dtype not in floating | packed for matrix in (a_weight, b_weight)):\n"
           "        raise ValueError('LoRA adapters require floating or compressed matrices')\n"
           "    if a_weight.shape[1] != x.shape[-1] or b_weight.shape != (weight.shape[0], a_weight.shape[0]):\n"
           "        raise ValueError('incompatible LoRA adapter dimensions')\n"
           "    if scale.shape or scale.dtype not in floating:\n"
           "        raise ValueError('LoRA scale must be a floating scalar tensor')\n"
           "    hidden_dtype = _promote(x.dtype, a_weight.dtype)\n"
           "    delta_dtype = _promote(hidden_dtype, b_weight.dtype)\n"
           "    if output.shape != (*x.shape[:-1], weight.shape[0]):\n"
           "        raise ValueError('incompatible packed fusion output dimensions')\n"
           "    return dict(\n"
           "        tokens=_product(x.shape[:-1]), width=x.shape[-1],\n"
           "        outputs=weight.shape[0],\n"
           "        input_dtype=_DTYPE_IDS[x.dtype], output_dtype=_DTYPE_IDS[output.dtype],\n"
           "        hidden_width=a_weight.shape[0],\n"
           "        weight_dtype=_DTYPE_IDS[weight.dtype],\n"
           "        a_dtype=_DTYPE_IDS[a_weight.dtype],\n"
           "        b_dtype=_DTYPE_IDS[b_weight.dtype],\n"
           "        hidden_dtype=_DTYPE_IDS[hidden_dtype],\n"
           "        delta_dtype=_DTYPE_IDS[delta_dtype],\n"
           "        scale_dtype=_DTYPE_IDS[scale.dtype],\n"
           "        scaled_dtype=_DTYPE_IDS[_promote(delta_dtype, scale.dtype)],\n"
           "    )\n"
           "\n"
           "def pattern(x: Tensor, weight: Tensor, a_weight: Tensor, b_weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    result = Add(Linear(x, weight), Mul(Linear(Linear(x, a_weight), b_weight), scale))\n"
           "    return (result if dtype == 'fp32' else Cast(result, dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_neon_packed_lora_parameters(void) {
    static const TrillimParameter fields[] = {
        {"tokens", "size_t", offsetof(Parameters, tokens), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"hidden_width", "size_t", offsetof(Parameters, hidden_width), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"a_dtype", "uint32_t", offsetof(Parameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(Parameters, b_dtype), NULL},
        {"hidden_dtype", "uint32_t", offsetof(Parameters, hidden_dtype), NULL},
        {"delta_dtype", "uint32_t", offsetof(Parameters, delta_dtype), NULL},
        {"scale_dtype", "uint32_t", offsetof(Parameters, scale_dtype), NULL},
        {"scaled_dtype", "uint32_t", offsetof(Parameters, scaled_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), sizeof(fields) / sizeof(fields[0]), fields,
        5, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_neon_packed_lora_capabilities(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
#else
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_NEON | TRILLIM_CAP_LIBM};
#endif
    return &caps;
}
