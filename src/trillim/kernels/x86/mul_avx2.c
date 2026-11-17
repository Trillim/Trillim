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

typedef struct {
    size_t count, rank;
    const size_t *shape, *a_strides, *b_strides;
    uint32_t dtype;
} Parameters;
typedef struct {
    void *output;
    const void *a, *b;
    const Parameters *p;
    const TrillimKernelContext *context;
    bool flat;
    size_t a_step, b_step;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!trillim_float(p->dtype) || (p->rank && (!p->shape || !p->a_strides || !p->b_strides))) return TRILLIM_ERROR;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (!p->shape[axis]) return p->count == 0 ? TRILLIM_OK : TRILLIM_ERROR;
    }
    size_t count = 1;
    for (size_t axis = 0; axis < p->rank; ++axis) {
        if (count > SIZE_MAX / p->shape[axis]) return TRILLIM_ERROR;
        count *= p->shape[axis];
    }
    size_t a_span, b_span;
    size_t bytes = trillim_itemsize(p->dtype);
    if (count != p->count || p->count > SIZE_MAX / bytes ||
        !trillim_span(p->rank, p->shape, p->a_strides, &a_span) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b_span) ||
        a_span > SIZE_MAX / bytes || b_span > SIZE_MAX / bytes) return TRILLIM_ERROR;
    inputs[0] = a_span * bytes;
    inputs[1] = b_span * bytes;
    outputs[0] = p->count * bytes;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static TrillimStatus apply(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t i = tile * 4096;
    size_t end = p->count - i < 4096 ? p->count : i + 4096;
    while (i < end) {
        size_t limit = end;
        size_t a = i * task->a_step, b = i * task->b_step;
        size_t a_step = task->a_step, b_step = task->b_step;
        if (!task->flat) {
            size_t width = p->shape[p->rank - 1];
            size_t remaining = width - i % width;
            if (remaining < end - i) limit = i + remaining;
            a = trillim_offset(i, p->rank, p->shape, p->a_strides);
            b = trillim_offset(i, p->rank, p->shape, p->b_strides);
            a_step = p->a_strides[p->rank - 1];
            b_step = p->b_strides[p->rank - 1];
        }
        if (a_step <= 1 && b_step <= 1) {
            for (; limit - i >= 8; i += 8, a += a_step * 8, b += b_step * 8) {
                __m256 x = a_step ? load8(task->a, a, p->dtype) : _mm256_set1_ps(trillim_load(task->a, a, p->dtype).f);
                __m256 y = b_step ? load8(task->b, b, p->dtype) : _mm256_set1_ps(trillim_load(task->b, b, p->dtype).f);
                store8(task->output, i, p->dtype, _mm256_mul_ps(x, y));
            }
        }
        for (; i < limit; ++i, a += a_step, b += b_step) {
            if ((i & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            TrillimValue value = {.f = trillim_load(task->a, a, p->dtype).f * trillim_load(task->b, b, p->dtype).f};
            trillim_store(task->output, i, p->dtype, value);
        }
    }
    return TRILLIM_OK;
}

TRILLIM_AVX2 static bool can_alias(const void *parameters, size_t output, size_t input) {
    if (output != 0 || input >= 2) return false;
    if (!parameters) return true;
    const Parameters *p = parameters;
    return trillim_contiguous(p->rank, p->shape, input == 0 ? p->a_strides : p->b_strides);
}

TRILLIM_KERNEL TrillimStatus trillim_x86_mul_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (!p->count) return TRILLIM_OK;
    size_t a_span, b_span;
    if (!trillim_span(p->rank, p->shape, p->a_strides, &a_span) ||
        !trillim_span(p->rank, p->shape, p->b_strides, &b_span)) return TRILLIM_ERROR;
    bool a_contiguous = trillim_contiguous(p->rank, p->shape, p->a_strides);
    bool b_contiguous = trillim_contiguous(p->rank, p->shape, p->b_strides);
    bool flat = (a_contiguous || a_span == 1) && (b_contiguous || b_span == 1);
    Task task = {outputs[0], inputs[0], inputs[1], p, context, flat,
                 a_contiguous ? 1 : 0, b_contiguous ? 1 : 0};
    size_t tiles = p->count / 4096 + (p->count % 4096 != 0);
    if (p->count <= 16384) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            TrillimStatus status = apply(&task, tile, 0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context, tiles, apply, &task);
}

TRILLIM_KERNEL const char *trillim_x86_mul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _binary\n"
           "def bind(c):\n"
           "    p = _binary(c)\n"
           "    if p['dtype'] not in (9,10,11):\n"
           "        raise ValueError('AVX2 binary kernel requires floating operands')\n"
           "    return p\n"
           "def pattern(a: Tensor,b: Tensor) -> tuple[Tensor,...]:\n"
           "    return (Mul(a,b),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_mul_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"count", "size_t", offsetof(Parameters, count), NULL},
        {"rank", "size_t", offsetof(Parameters, rank), NULL},
        {"shape", "size_t", offsetof(Parameters, shape), "rank"},
        {"a_strides", "size_t", offsetof(Parameters, a_strides), "rank"},
        {"b_strides", "size_t", offsetof(Parameters, b_strides), "rank"},
        {"dtype", "uint32_t", offsetof(Parameters, dtype), NULL}
    };
    static const size_t inputs[] = {1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 6, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_mul_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
