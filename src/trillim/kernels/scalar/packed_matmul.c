#include "common.h"
#include <fenv.h>
#include <math.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

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
} TrillimMatmulParameters;

enum { PACKED_ACTIVATION_TILE = 4, PACKED_ACTIVATION_MAX_BYTES = 32 * 1024 * 1024 };

typedef struct {
    const void *input;
    const TrillimKernelContext *context;
    float *scales;
    int8_t *codes;
    size_t width;
    size_t groups;
    const TrillimMatmulParameters *p;
    size_t first_row;
} PackedActivation;

static inline size_t packed_activation_group(uint32_t dtype) {
    switch (dtype) {
    case TRILLIM_GRP_INT4:
    case TRILLIM_GRP_INT5:
    case TRILLIM_GRP_INT8:
        return 32;
    case TRILLIM_DUBGRP_INT6:
        return 256;
    default:
        return 0;
    }
}

static inline bool packed_activation_quantize(
    const void *input, uint32_t dtype, size_t start, size_t group, float *scale, int8_t *codes
) {
    float values[256];
    float peak = 0.0f;
    for (size_t i = 0; i < group; ++i) {
        float value = trillim_load(input, start + i, dtype).f;
        if (!isfinite(value)) {
            return false;
        }
        if (fabsf(value) > fabsf(peak)) {
            peak = value;
        }
        values[i] = value;
    }
    if (peak == 0.0f) {
        *scale = 0.0f;
        memset(codes, 0, group);
        return true;
    }
    float inverse;
    if (group == 256) {
        inverse = -127.0f / peak;
        if (!isfinite(inverse) || inverse == 0.0f || fabsf(peak) > 0x1p40f) {
            return false;
        }
        *scale = 1.0f / inverse;
    } else {
        float step = fabsf(peak) / 127.0f;
        if (step > 65504.0f || (*scale = trillim_fp16(trillim_to_fp16(step))) == 0.0f) {
            return false;
        }
        inverse = 1.0f / step;
    }
    for (size_t i = 0; i < group; ++i) {
        /* Bias the significand to round to nearest, ties to even. */
        float rounded = values[i] * inverse + 12582912.0f;
        uint32_t bits;
        memcpy(&bits, &rounded, sizeof(bits));
        codes[i] = (int8_t)((int32_t)(bits & UINT32_C(0x007fffff)) - INT32_C(0x00400000));
    }
    return true;
}

static inline TrillimStatus packed_activation_row(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const PackedActivation *a = opaque;
    size_t group = a->width / a->groups;
    float *scales = a->scales + row * a->groups;
    int8_t *codes = a->codes + row * a->width;
    if (fegetround() != FE_TONEAREST) {
        scales[0] = NAN;
        return TRILLIM_OK;
    }
    const TrillimMatmulParameters *p = a->p;
    size_t input_row = a->first_row + row;
    size_t offset = trillim_offset(input_row / p->m, p->rank, p->batch_shape, p->a_batch_strides) +
        input_row % p->m * p->a_row_stride;
    for (size_t g = 0; g < a->groups; ++g) {
        if (g % 4 == 0 && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t start = g * group;
        if (!packed_activation_quantize(a->input, p->a_dtype, offset + start,
                group, scales + g, codes + start)) {
            scales[0] = NAN;
            return TRILLIM_OK;
        }
    }
    return TRILLIM_OK;
}

static inline size_t packed_activation_space(uint32_t dtype, size_t rows, size_t width, size_t columns) {
    size_t group = packed_activation_group(dtype);
    if (!group || columns < 16) {
        return 0;
    }
    size_t bounded_width = width < 65536 ? width : 65536;
    size_t row_bytes = (bounded_width / group) * (group + sizeof(float));
    if (!row_bytes) {
        return 0;
    }
    return rows > PACKED_ACTIVATION_MAX_BYTES / row_bytes ? PACKED_ACTIVATION_MAX_BYTES : rows * row_bytes;
}

static inline TrillimStatus packed_activation_prepare(
    PackedActivation *a, const void *input, const TrillimMatmulParameters *p,
    size_t first_row, size_t rows, const TrillimKernelContext *context, void *scratch
) {
    size_t width = p->k;
    size_t group = packed_activation_group(p->b_dtype);
    if (scratch == NULL || !group || width < group || width > 65536 || width % group || p->n < 16 ||
        fegetround() != FE_TONEAREST) {
        return TRILLIM_OK;
    }
    size_t groups = width / group;
    float *scales = scratch;
    int8_t *codes = (int8_t *)(scales + rows * groups);
    *a = (PackedActivation){input, context, scales, codes, width, groups, p, first_row};
    const TrillimThreadPool *pool = context->thread_pool;
    return rows == 1 ? packed_activation_row(a, 0, 0)
                     : pool->run(pool->context, rows, packed_activation_row, a);
}

static inline int32_t packed_q6_dot(
    const uint8_t *low, const uint8_t *high, const int8_t *subscales, const int8_t *codes
) {
    int32_t total = 0;
    for (size_t region = 0; region < 8; ++region) {
        size_t half = region / 4;
        size_t lane = region % 4;
        const uint8_t *l = low + half * 64 + lane / 2 * 32;
        const uint8_t *h = high + half * 32;
        for (size_t sub = 0; sub < 2; ++sub) {
            int32_t sum = 0;
            size_t first = sub * 16;
            for (size_t i = first; i < first + 16; ++i) {
                int32_t lo = (l[i] >> ((lane % 2) * 4)) & 15;
                int32_t hi = (h[i] >> (lane * 2)) & 3;
                int32_t w = (lo | hi << 4) - 32;
                sum += w * codes[region * 32 + i];
            }
            total += sum * subscales[region * 2 + sub];
        }
    }
    return total;
}

static inline bool packed_activation_ready(const PackedActivation *a, size_t row) {
    return a->scales != NULL && !isnan(a->scales[row * a->groups]);
}

static inline TrillimStatus packed_activation_dot(
    const PackedActivation *a, size_t first_row, size_t rows, const uint8_t *weight,
    uint32_t dtype, size_t storage_rows, size_t storage_groups, size_t first_group, size_t column,
    float sums[PACKED_ACTIVATION_TILE], bool *used
) {
    *used = false;
    first_row -= a->first_row;
    for (size_t row = 0; row < rows; ++row) {
        if (!packed_activation_ready(a, first_row + row)) {
            return TRILLIM_OK;
        }
    }
    size_t total = storage_rows * storage_groups;
    size_t block = column * storage_groups + first_group;
    bool q6 = dtype == TRILLIM_DUBGRP_INT6;
    size_t low_size = q6 ? 128 : (dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const uint8_t *low = weight + block * low_size;
    const uint8_t *scales = weight + total * low_size + block * 2;
    const uint8_t *high = weight + total * (low_size + 2);
    const int8_t *subscales = (const int8_t *)(high + (q6 ? total * 64 + block * 16 : 0));
    size_t high_size = q6 ? 64 : (dtype == TRILLIM_GRP_INT5 ? 4 : 0);
    high += block * high_size;
    const int8_t *codes = a->codes + first_row * a->width;
    const float *activation_scales = a->scales + first_row * a->groups;
    float output[PACKED_ACTIVATION_TILE] = {0};
    for (size_t g = 0; g < a->groups; ++g) {
        if ((q6 || g % 8 == 0) && a->context->is_cancelled(a->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        uint16_t bits;
        memcpy(&bits, scales + g * 2, sizeof(bits));
        float scale = trillim_bf16(bits);
        if (!isfinite(scale) || fabsf(scale) > 0x1p40f) {
            return TRILLIM_OK;
        }
        const uint8_t *l = low + g * low_size;
        const uint8_t *h = high + g * high_size;
        int32_t dots[PACKED_ACTIVATION_TILE] = {0};
        if (q6 && rows == 1) {
            dots[0] = packed_q6_dot(l, h, subscales + g * 16, codes + g * 256);
        } else if (q6) {
            for (size_t region = 0; region < 8; ++region) {
                size_t half = region / 4, lane = region % 4;
                const uint8_t *lower = l + half * 64 + lane / 2 * 32;
                const uint8_t *upper = h + half * 32;
                const int8_t *s = subscales + g * 16 + region * 2;
                int16_t decoded[32];
                for (size_t i = 0; i < 32; ++i) {
                    int32_t lo = (lower[i] >> (lane % 2 * 4)) & 15;
                    int32_t hi = (upper[i] >> (lane * 2)) & 3;
                    decoded[i] = (int16_t)(((lo | hi << 4) - 32) * s[i / 16]);
                }
                for (size_t row = 0; row < rows; ++row) {
                    const int8_t *x = codes + row * a->width + g * 256 + region * 32;
                    for (size_t i = 0; i < 32; ++i) {
                        dots[row] += decoded[i] * x[i];
                    }
                }
            }
        } else {
            int8_t decoded[32];
            const int8_t *w = (const int8_t *)l;
            if (dtype != TRILLIM_GRP_INT8) {
                uint32_t fifth = 0;
                int32_t center = 8;
                if (dtype == TRILLIM_GRP_INT5) {
                    fifth = (uint32_t)h[0] | (uint32_t)h[1] << 8 |
                        (uint32_t)h[2] << 16 | (uint32_t)h[3] << 24;
                    center = 16;
                }
                for (size_t i = 0; i < 16; ++i) {
                    decoded[i] = (int8_t)((int32_t)((l[i] & 15) | ((fifth >> i & 1) << 4)) - center);
                    decoded[i + 16] = (int8_t)((int32_t)((l[i] >> 4) | ((fifth >> (i + 16) & 1) << 4)) - center);
                }
                w = decoded;
            }
            for (size_t row = 0; row < rows; ++row) {
                const int8_t *x = codes + row * a->width + g * 32;
                for (size_t i = 0; i < 32; ++i) {
                    dots[row] += w[i] * x[i];
                }
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            float combined_scale = scale * activation_scales[row * a->groups + g];
            output[row] += (float)dots[row] * combined_scale;
        }
    }
    memcpy(sums, output, rows * sizeof(*sums));
    *used = true;
    return TRILLIM_OK;
}

static inline bool trillim_payload_size(size_t rows, size_t columns, uint32_t dtype, size_t *bytes) {
    size_t itemsize = trillim_itemsize(dtype);
    if (columns != 0 && rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (itemsize != 0) {
        if (count > SIZE_MAX / itemsize) {
            return false;
        }
        *bytes = count * itemsize;
        return true;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        if (rows > (SIZE_MAX - count) / 2) {
            return false;
        }
        *bytes = count + rows * 2;
        return true;
    }
    size_t group;
    size_t record;
    switch (dtype) {
    case TRILLIM_MAT_TERNARY: group = 128; record = 32; break;
    case TRILLIM_GRP_INT8: group = 32; record = 34; break;
    case TRILLIM_GRP_INT4: group = 32; record = 18; break;
    case TRILLIM_GRP_INT5: group = 32; record = 22; break;
    case TRILLIM_DUBGRP_INT6: group = 256; record = 210; break;
    case TRILLIM_GRP_BINARY: group = 128; record = 18; break;
    case TRILLIM_GRP_TERNARY: group = 128; record = 34; break;
    case TRILLIM_MXFP4: group = 32; record = 17; break;
    default: return false;
    }
    size_t groups = columns / group + (columns % group != 0);
    if (rows != 0 && groups > SIZE_MAX / record / rows) {
        return false;
    }
    *bytes = rows * groups * record;
    if (dtype == TRILLIM_MAT_TERNARY) {
        if (*bytes > SIZE_MAX - sizeof(float)) {
            return false;
        }
        *bytes += sizeof(float);
    }
    return true;
}

typedef struct {
    void *output;
    const void *a;
    const uint8_t *b;
    const TrillimMatmulParameters *p;
    size_t columns;
    const TrillimKernelContext *context;
    PackedActivation activation;
    size_t first_tile;
} Task;

static size_t group_width(uint32_t dtype) {
    if (dtype == TRILLIM_DUBGRP_INT6) {
        return 256;
    }
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY) {
        return 128;
    }
    return 32;
}

static inline bool trillim_matrix_span(size_t base, size_t rows, size_t columns, size_t row_stride, size_t col_stride, size_t capacity) {
    if (rows == 0 || columns == 0) {
        return true;
    }
    if (row_stride != 0 && rows - 1 > (SIZE_MAX - base) / row_stride) {
        return false;
    }
    base += (rows - 1) * row_stride;
    if (col_stride != 0 && columns - 1 > (SIZE_MAX - base) / col_stride) {
        return false;
    }
    return base + (columns - 1) * col_stride <= capacity;
}

static inline TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    if (!trillim_float(p->a_dtype) || !trillim_float(p->out_dtype) ||
        p->b_dtype < TRILLIM_ROW_INT8 || p->b_dtype > TRILLIM_MXFP4 ||
        !p->k || p->k % 16 || p->a_col_stride != 1 || p->b_row_stride != 1 ||
        p->b_col_stride != p->k || !p->b_storage_columns || p->b_storage_columns % p->k) {
        return TRILLIM_ERROR;
    }
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (p->b_batch_strides[axis] % p->k) {
            return TRILLIM_ERROR;
        }
    }
    size_t bytes = trillim_itemsize(p->out_dtype);
    size_t a_span, b_span;
    inputs[0] = 0;
    inputs[1] = 0;
    outputs[0] = 0;
    *scratch = 0;
    if (bytes == 0 || !trillim_span(p->rank, p->batch_shape, p->a_batch_strides, &a_span) ||
        !trillim_span(p->rank, p->batch_shape, p->b_batch_strides, &b_span)) {
        return TRILLIM_ERROR;
    }
    if (p->batches == 0 || p->m == 0 || p->n == 0) {
        return TRILLIM_OK;
    }
    if (p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / (p->m * p->n) ||
        p->batches * p->m * p->n > SIZE_MAX / bytes) {
        return TRILLIM_ERROR;
    }
    if (p->k != 0) {
        if (p->a_storage_columns == 0 || p->b_storage_columns == 0 ||
            p->a_storage_rows > SIZE_MAX / p->a_storage_columns ||
            p->b_storage_rows > SIZE_MAX / p->b_storage_columns ||
            !trillim_matrix_span(a_span, p->m, p->k, p->a_row_stride, p->a_col_stride, p->a_storage_rows * p->a_storage_columns) ||
            !trillim_matrix_span(b_span, p->k, p->n, p->b_row_stride, p->b_col_stride, p->b_storage_rows * p->b_storage_columns) ||
            !trillim_payload_size(p->a_storage_rows, p->a_storage_columns, p->a_dtype, &inputs[0]) ||
            !trillim_payload_size(p->b_storage_rows, p->b_storage_columns, p->b_dtype, &inputs[1])) {
            return TRILLIM_ERROR;
        }
    }
    outputs[0] = p->batches * p->m * p->n * bytes;
    if (p->out_dtype == TRILLIM_FP32) {
        *scratch = packed_activation_space(p->b_dtype, p->batches * p->m, p->k, p->n);
    }
    return TRILLIM_OK;
}

static float multiply_add(float a, float b, float sum) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF)
    return fmaf(a, b, sum);
#else
    return a * b + sum;
#endif
}

static float read_bf16(const uint8_t *data) {
    uint16_t bits;
    memcpy(&bits, data, sizeof(bits));
    return trillim_bf16(bits);
}

static float load_fp32(const void *data, size_t index) {
    float value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return value;
}

static float load_fp16(const void *data, size_t index) {
    uint16_t value;
    memcpy(&value, (const uint8_t *)data + index * sizeof(value), sizeof(value));
    return trillim_fp16(value);
}

static float load_bf16(const void *data, size_t index) {
    return read_bf16((const uint8_t *)data + index * sizeof(uint16_t));
}

static void mxfp4_palette(uint8_t exponent, float palette[16]) {
    static const float magnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float scale = trillim_mxfp4_scale(exponent);
    for (size_t i = 0; i < 8; ++i) {
        palette[i] = exponent == 255 ? NAN : magnitudes[i] * scale;
        palette[i + 8] = exponent == 255 ? NAN : -palette[i];
    }
}

static void decode_block(const Task *task, size_t column, size_t k, size_t count, float *decoded) {
    const TrillimMatmulParameters *p = task->p;
    const uint8_t *values = task->b;
    uint32_t dtype = p->b_dtype;
    if (dtype == TRILLIM_ROW_INT8) {
        float scale = read_bf16(values + p->b_storage_rows * p->b_storage_columns + column * 2);
        const int8_t *codes = (const int8_t *)values + column * p->b_storage_columns + k;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    size_t group = group_width(dtype);
    size_t groups = p->b_storage_columns / group + (p->b_storage_columns % group != 0);
    size_t total = p->b_storage_rows * groups;
    size_t block = column * groups + k / group;
    size_t offset = k % group;
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        float scale;
        if (dtype == TRILLIM_MAT_TERNARY) {
            memcpy(&scale, values + total * 32, sizeof(scale));
        } else {
            scale = read_bf16(values + total * 32 + block * 2);
        }
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 32 + (offset + i) % 32] >> (6 - ((offset + i) / 32) * 2)) & 3;
            decoded[i] = code == 3 ? NAN : (float)(code - 1) * scale;
        }
        return;
    }
    if (dtype == TRILLIM_MXFP4) {
        float palette[16];
        mxfp4_palette(values[total * 16 + block], palette);
        for (size_t i = 0; i < count; ++i) {
            uint8_t code = (values[block * 16 + (offset + i) / 2] >> (((offset + i) % 2) * 4)) & 15;
            decoded[i] = palette[code];
        }
        return;
    }
    size_t value_width = dtype == TRILLIM_GRP_INT8 ? 32 : (dtype == TRILLIM_DUBGRP_INT6 ? 128 : 16);
    size_t scale_start = total * value_width;
    float scale = read_bf16(values + scale_start + block * 2);
    if (dtype == TRILLIM_GRP_INT8) {
        const int8_t *codes = (const int8_t *)values + block * 32 + offset;
        for (size_t i = 0; i < count; ++i) {
            decoded[i] = (float)codes[i] * scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_BINARY) {
        for (size_t i = 0; i < count; ++i) {
            bool positive = (values[block * 16 + (offset + i) / 8] >> (7 - (offset + i) % 8)) & 1;
            decoded[i] = positive ? scale : -scale;
        }
        return;
    }
    if (dtype == TRILLIM_GRP_INT4 || dtype == TRILLIM_GRP_INT5) {
        int32_t zero = dtype == TRILLIM_GRP_INT4 ? 8 : 16;
        for (size_t i = 0; i < count; ++i) {
            int32_t code = (values[block * 16 + (offset + i) % 16] >> (((offset + i) / 16) * 4)) & 15;
            if (dtype == TRILLIM_GRP_INT5) {
                code |= ((values[scale_start + total * 2 + block * 4 + (offset + i) / 8] >> ((offset + i) % 8)) & 1) << 4;
            }
            decoded[i] = (float)(code - zero) * scale;
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t position = offset + i;
        size_t half = position / 128;
        size_t lane = position % 128 / 32;
        size_t low = block * 128 + half * 64 + lane / 2 * 32 + position % 32;
        size_t high = scale_start + total * 2 + block * 64 + half * 32 + position % 32;
        int32_t code = (values[low] >> ((lane % 2) * 4)) & 15;
        code |= ((values[high] >> (lane * 2)) & 3) << 4;
        int8_t subscale;
        memcpy(&subscale, values + scale_start + total * 66 + block * 16 + position / 16, sizeof(subscale));
        decoded[i] = (float)((code - 32) * subscale) * scale;
    }
}

#define TRILLIM_PACKED_MATMUL(NAME, LOAD) \
static TrillimStatus project_##NAME(void *opaque, size_t index, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    index += task->first_tile; \
    const TrillimMatmulParameters *p = task->p; \
    size_t row_tiles = p->m / 4 + (p->m % 4 != 0); \
    size_t columns = p->n / task->columns + (p->n % task->columns != 0); \
    size_t batch = index / columns / row_tiles; \
    size_t row = index / columns % row_tiles * 4; \
    size_t rows = p->m - row < 4 ? p->m - row : 4; \
    size_t first = index % columns * task->columns; \
    size_t end = p->n - first < task->columns ? p->n : first + task->columns; \
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride; \
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides); \
    size_t output = (batch * p->m + row) * p->n; \
    size_t group = group_width(p->b_dtype); \
    for (size_t column = first; column < end; ++column) { \
        size_t flat = b_base + column * p->k; \
        size_t storage_row = flat / p->b_storage_columns; \
        size_t storage_column = flat % p->b_storage_columns; \
        float sums[4] = {0}; \
        bool quantized = false; \
        if (task->activation.scales != NULL) { \
            size_t storage_groups = p->b_storage_columns / group; \
            TrillimStatus status = packed_activation_dot(&task->activation, batch * p->m + row, rows, task->b, \
                p->b_dtype, p->b_storage_rows, storage_groups, storage_column / group, \
                storage_row, sums, &quantized); \
            if (status != TRILLIM_OK) { \
                return status; \
            } \
        } \
        for (size_t k = 0; !quantized && k < p->k;) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            size_t count = group - (storage_column + k) % group; \
            if (count > p->k - k) { \
                count = p->k - k; \
            } \
            float decoded[256]; \
            decode_block(task, storage_row, storage_column + k, count, decoded); \
            for (size_t i = 0; i < count; ++i) { \
                float weight = load_fp32(decoded, i); \
                sums[0] = multiply_add(LOAD(task->a, a_base + k + i), weight, sums[0]); \
                if (rows > 1) { \
                    sums[1] = multiply_add(LOAD(task->a, a_base + p->a_row_stride + k + i), weight, sums[1]); \
                } \
                if (rows > 2) { \
                    sums[2] = multiply_add(LOAD(task->a, a_base + 2 * p->a_row_stride + k + i), weight, sums[2]); \
                } \
                if (rows > 3) { \
                    sums[3] = multiply_add(LOAD(task->a, a_base + 3 * p->a_row_stride + k + i), weight, sums[3]); \
                } \
            } \
            k += count; \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t r = 0; r < rows; ++r) { \
            TrillimValue result = {.f = sums[r]}; \
            trillim_store(task->output, output + r * p->n + column, p->out_dtype, result); \
        } \
    } \
    return TRILLIM_OK; \
}

TRILLIM_PACKED_MATMUL(fp32, load_fp32)
TRILLIM_PACKED_MATMUL(fp16, load_fp16)
TRILLIM_PACKED_MATMUL(bf16, load_bf16)

#undef TRILLIM_PACKED_MATMUL

/* Group-aligned weight vectors can be addressed directly across storage rows. */
#define TRILLIM_MXFP4_MATVEC(NAME, LOAD) \
static TrillimStatus project_mxfp4_##NAME(void *opaque, size_t tile, size_t worker) { \
    (void)worker; \
    const Task *task = opaque; \
    const TrillimMatmulParameters *p = task->p; \
    size_t tiles = p->n / task->columns + (p->n % task->columns != 0); \
    size_t batch = tile / tiles; \
    size_t first = tile % tiles * task->columns; \
    size_t end = p->n - first < task->columns ? p->n : first + task->columns; \
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides); \
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides) / 32; \
    const uint8_t *scales = task->b + p->b_storage_rows * (p->b_storage_columns / 32) * 16; \
    size_t groups = p->k / 32; \
    for (size_t column = first; column < end; column += 4) { \
        size_t count = end - column < 4 ? end - column : 4; \
        size_t blocks[4]; \
        float sums[4] = {0}; \
        for (size_t j = 0; j < 4; ++j) { \
            blocks[j] = b_base + (column + (j < count ? j : 0)) * groups; \
        } \
        for (size_t group = 0; group < groups; ++group) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) { \
                return TRILLIM_CANCELLED; \
            } \
            float palette[4][16]; \
            const uint8_t *codes[4]; \
            for (size_t j = 0; j < 4; ++j) { \
                size_t block = blocks[j] + group; \
                codes[j] = task->b + block * 16; \
                mxfp4_palette(scales[block], palette[j]); \
            } \
            size_t offset = a_base + group * 32; \
            for (size_t i = 0; i < 16; ++i) { \
                float even = LOAD(task->a, offset + i * 2); \
                float odd = LOAD(task->a, offset + i * 2 + 1); \
                for (size_t j = 0; j < 4; ++j) { \
                    uint8_t code = codes[j][i]; \
                    sums[j] = multiply_add(even, palette[j][code & 15], sums[j]); \
                    sums[j] = multiply_add(odd, palette[j][code >> 4], sums[j]); \
                } \
            } \
        } \
        if (task->context->is_cancelled(task->context->cancellation_context)) { \
            return TRILLIM_CANCELLED; \
        } \
        for (size_t j = 0; j < count; ++j) { \
            TrillimValue result = {.f = sums[j]}; \
            trillim_store(task->output, batch * p->n + column + j, p->out_dtype, result); \
        } \
    } \
    return TRILLIM_OK; \
}

TRILLIM_MXFP4_MATVEC(fp32, load_fp32)
TRILLIM_MXFP4_MATVEC(fp16, load_fp16)
TRILLIM_MXFP4_MATVEC(bf16, load_bf16)

#undef TRILLIM_MXFP4_MATVEC

TRILLIM_KERNEL TrillimStatus trillim_scalar_block_matmul(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimMatmulParameters *p = parameters;
    if (!p->batches || !p->m || !p->n) {
        return TRILLIM_OK;
    }
    size_t workers = context->thread_pool->worker_count;
    if (!workers) {
        return TRILLIM_ERROR;
    }
    size_t columns = p->n / workers / 4;
    columns = columns / 16 * 16;
    if (columns < 16) {
        columns = 16;
    }
    Task task = {outputs[0], inputs[0], inputs[1], p, columns, context, {0}, 0};
    size_t rows = p->m / 4 + (p->m % 4 != 0);
    size_t tiles = p->n / columns + (p->n % columns != 0);
    bool mxfp4_matvec = p->b_dtype == TRILLIM_MXFP4 && p->m == 1 && p->n >= 4 && p->k % 32 == 0;
    TrillimTask project;
    switch (p->a_dtype) {
    case TRILLIM_FP32: project = mxfp4_matvec ? project_mxfp4_fp32 : project_fp32; break;
    case TRILLIM_BF16: project = mxfp4_matvec ? project_mxfp4_bf16 : project_bf16; break;
    case TRILLIM_FP16: project = mxfp4_matvec ? project_mxfp4_fp16 : project_fp16; break;
    default: return TRILLIM_ERROR;
    }
    size_t row_tiles = p->batches * rows;
    size_t panel_tiles = row_tiles;
    size_t group = packed_activation_group(p->b_dtype);
    if (p->a_col_stride == 1 && p->out_dtype == TRILLIM_FP32 && scratch != NULL && group != 0 &&
        p->k >= group && p->k <= 65536 && p->k % group == 0 && p->n >= 16 &&
        fegetround() == FE_TONEAREST) {
        size_t capacity = PACKED_ACTIVATION_MAX_BYTES / (p->k / group * sizeof(float) + p->k);
        if (p->batches * p->m > capacity) {
            panel_tiles = capacity / PACKED_ACTIVATION_TILE;
        }
    }
    for (size_t first = 0; first < row_tiles;) {
        size_t count = row_tiles - first < panel_tiles ? row_tiles - first : panel_tiles;
        size_t end = first + count;
        size_t first_row = first / rows * p->m + first % rows * PACKED_ACTIVATION_TILE;
        size_t end_row = end / rows * p->m + end % rows * PACKED_ACTIVATION_TILE;
        task.first_tile = first * tiles;
        task.activation = (PackedActivation){0};
        TrillimStatus status = TRILLIM_OK;
        if (p->a_col_stride == 1 && p->out_dtype == TRILLIM_FP32) {
            status = packed_activation_prepare(&task.activation, inputs[0], p,
                first_row, end_row - first_row, context, scratch);
            if (status != TRILLIM_OK) {
                return status;
            }
        }
        size_t tasks = count * tiles;
        status = tasks == 1 ? project(&task, 0, 0)
            : context->thread_pool->run(context->thread_pool->context, tasks, project, &task);
        if (status != TRILLIM_OK) {
            return status;
        }
        first = end;
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_block_matmul_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _matmul\n"
           "def bind(c):\n"
           "    if c.inputs['a'].dtype not in {'fp16', 'bf16', 'fp32'} or c.inputs['b'].dtype not in {'row_int8', 'mat_ternary', 'grp_int8', 'grp_int4', 'grp_int5', 'dubgrp_int6', 'grp_binary', 'grp_ternary', 'mxfp4'} or c.outputs[0].dtype not in {'fp16', 'bf16', 'fp32'}:\n"
           "        raise ValueError('packed matmul requires floating activations and compressed weights')\n"
           "    p = _matmul(c)\n"
           "    width = p['k'].value\n"
           "    columns = p['b_storage_columns'].value\n"
           "    if width is None or width <= 0 or width % 16 or columns is None or columns <= 0 or columns % width:\n"
           "        raise ValueError('packed matmul requires aligned fixed-width rows')\n"
           "    if p['a_col_stride'].value != 1 or p['b_row_stride'].value != 1 or p['b_col_stride'] != p['k']:\n"
           "        raise ValueError('packed matmul requires contiguous reduction axes and output-major weights')\n"
           "    def aligned(stride):\n"
           "        if stride.value is not None: return stride.value % width == 0\n"
           "        if stride.op == 'broadcast_stride': return aligned(stride.args[1])\n"
           "        if stride.op == 'mul': return any(aligned(part) for part in stride.args)\n"
           "        return False\n"
           "    if any(not aligned(stride) for stride in p['b_batch_strides']):\n"
           "        raise ValueError('packed matmul batch strides must preserve complete weight vectors')\n"
           "    return p\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_block_matmul_parameters(void) {
    static const TrillimParameter fields[] = {
        {"batches", "size_t", offsetof(TrillimMatmulParameters, batches), NULL},
        {"m", "size_t", offsetof(TrillimMatmulParameters, m), NULL},
        {"n", "size_t", offsetof(TrillimMatmulParameters, n), NULL},
        {"k", "size_t", offsetof(TrillimMatmulParameters, k), NULL},
        {"rank", "size_t", offsetof(TrillimMatmulParameters, rank), NULL},
        {"batch_shape", "size_t", offsetof(TrillimMatmulParameters, batch_shape), "rank"},
        {"a_batch_strides", "size_t", offsetof(TrillimMatmulParameters, a_batch_strides), "rank"},
        {"b_batch_strides", "size_t", offsetof(TrillimMatmulParameters, b_batch_strides), "rank"},
        {"a_row_stride", "size_t", offsetof(TrillimMatmulParameters, a_row_stride), NULL},
        {"a_col_stride", "size_t", offsetof(TrillimMatmulParameters, a_col_stride), NULL},
        {"b_row_stride", "size_t", offsetof(TrillimMatmulParameters, b_row_stride), NULL},
        {"b_col_stride", "size_t", offsetof(TrillimMatmulParameters, b_col_stride), NULL},
        {"a_dtype", "uint32_t", offsetof(TrillimMatmulParameters, a_dtype), NULL},
        {"b_dtype", "uint32_t", offsetof(TrillimMatmulParameters, b_dtype), NULL},
        {"out_dtype", "uint32_t", offsetof(TrillimMatmulParameters, out_dtype), NULL},
        {"a_storage_rows", "size_t", offsetof(TrillimMatmulParameters, a_storage_rows), NULL},
        {"a_storage_columns", "size_t", offsetof(TrillimMatmulParameters, a_storage_columns), NULL},
        {"b_storage_rows", "size_t", offsetof(TrillimMatmulParameters, b_storage_rows), NULL},
        {"b_storage_columns", "size_t", offsetof(TrillimMatmulParameters, b_storage_columns), NULL}
    };
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(TrillimMatmulParameters), TRILLIM_ALIGNOF(TrillimMatmulParameters),
        sizeof(fields) / sizeof(fields[0]), fields, 2, 1, requirements, NULL, inputs, outputs, 4
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_block_matmul_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_LIBM};
    return &capabilities;
}
