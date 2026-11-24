#include "common.h"
#include <immintrin.h>
#define TRILLIM_AVX2 __attribute__((target("avx2")))
TRILLIM_AVX2 static __m256 load8(const void *data, size_t index, uint32_t dtype) {
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

TRILLIM_AVX2 static void store8(void *data, size_t index, uint32_t dtype, __m256 value) {
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


typedef struct {
    size_t rows, width;
    float eps, weight_offset;
    uint32_t weight_dtype;
} Parameters;
typedef struct {
    const uint16_t *input;
    const void *weight;
    uint16_t *output;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static TrillimStatus requirements(const void *opaque, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = opaque;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if ((p->weight_dtype != TRILLIM_FP16 && p->weight_dtype != TRILLIM_BF16) ||
        p->width > SIZE_MAX / 2 || (p->width && p->rows > SIZE_MAX / 2 / p->width)) return TRILLIM_ERROR;
    inputs[0] = outputs[0] = p->rows * p->width * 2;
    inputs[1] = p->width * 2;
    return TRILLIM_OK;
}
static bool can_alias(const void *opaque, size_t output, size_t input) {
    (void)opaque;
    return output == 0 && input == 0;
}
TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t pair, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    size_t first = pair * 2, rows = p->rows - first < 2 ? 1 : 2;
    float denominator[2];
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *x = t->input + (first + row) * p->width;
        float sum = 0;
        size_t col = 0;
        __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (; p->width - col >= 32; col += 32) {
            if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            for (size_t lane = 0; lane < 4; ++lane) {
                __m256 v = load8(x, col + lane * 8, TRILLIM_BF16);
                sums[lane] = _mm256_add_ps(sums[lane], _mm256_mul_ps(v, v));
            }
        }
        __m256 v = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        sum = _mm_cvtss_f32(s);
        for (; col < p->width; ++col) {
            if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            float x0 = trillim_bf16(x[col]);
            sum += x0 * x0;
        }
        denominator[row] = sqrtf(sum / (float)p->width + p->eps);
    }
    size_t col = 0;
    for (; p->width - col >= 8; col += 8) {
        if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 w = _mm256_add_ps(load8(t->weight, col, p->weight_dtype), _mm256_set1_ps(p->weight_offset));
        for (size_t row = 0; row < rows; ++row) {
            size_t i = (first + row) * p->width + col;
            __m256 x = _mm256_div_ps(load8(t->input, i, TRILLIM_BF16), _mm256_set1_ps(denominator[row]));
            store8(t->output, i, TRILLIM_BF16, _mm256_mul_ps(x, w));
        }
    }
    for (; col < p->width; ++col) {
        if ((col & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        float w = trillim_load(t->weight, col, p->weight_dtype).f + p->weight_offset;
        for (size_t row = 0; row < rows; ++row) {
            size_t i = (first + row) * p->width + col;
            t->output[i] = trillim_to_bf16((trillim_bf16(t->input[i]) / denominator[row]) * w);
        }
    }
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_bf16_rmsnorm(void *const *outputs, const void *const *inputs,
        void *scratch, const void *opaque, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = opaque;
    if (!p->width || !p->rows) return TRILLIM_OK;
    Task task = {inputs[0], inputs[1], outputs[0], p, context};
    size_t count = p->rows / 2 + (p->rows % 2 != 0);
    return count == 1 ? normalize(&task, 0, 0) : context->thread_pool->run(context->thread_pool->context, count, normalize, &task);
}
TRILLIM_KERNEL const char *trillim_x86_bf16_rmsnorm_tdl(void) {
    return "from trillim.tdl import DType, RMSNorm, Tensor\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides, _product, _DTYPE_IDS\n"
           "def bind(c):\n"
           "    x, weight, output = c.inputs['x'], c.inputs['weight'], c.outputs[0]\n"
           "    if not x.shape or output.shape != x.shape or weight.shape != x.shape[-1:]:\n"
           "        raise ValueError('BF16 RMSNorm requires a matching channel vector')\n"
           "    if x.dtype != 'bf16' or output.dtype != 'bf16' or weight.dtype not in ('bf16', 'fp16'):\n"
           "        raise ValueError('requires BF16 activations and BF16 or FP16 weights')\n"
           "    for value in (x, weight, output):\n"
           "        if value.strides and value.strides != _contiguous_strides(value.shape):\n"
           "            raise ValueError('BF16 RMSNorm requires contiguous BF16 tensors')\n"
           "    eps, offset = c.attrs.get('eps', 1e-6), c.attrs.get('weight_offset', 0.0)\n"
           "    if not isinstance(eps, (int, float)) or not isinstance(offset, (int, float)):\n"
           "        raise TypeError('RMSNorm parameters must be numeric')\n"
           "    return dict(rows=_product(x.shape[:-1]), width=x.shape[-1], eps=float(eps), weight_offset=float(offset), weight_dtype=_DTYPE_IDS[weight.dtype])\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_bf16_rmsnorm_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows", "size_t", offsetof(Parameters, rows), NULL},
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"eps", "float", offsetof(Parameters, eps), NULL},
        {"weight_offset", "float", offsetof(Parameters, weight_offset), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
    };
    static const size_t inputs[] = {2, 1}, outputs[] = {2};
    static const TrillimParameterSchema schema = {sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 5, fields, 2, 1,
        requirements, can_alias, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_bf16_rmsnorm_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_LIBM};
    return &caps;
}
