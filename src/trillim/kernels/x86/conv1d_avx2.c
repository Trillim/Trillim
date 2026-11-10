#include "common.h"
#include <immintrin.h>
#define SIMD __attribute__((target("avx2,fma")))
typedef struct {
    size_t channels, length, outputs, stride, output_length;
    uint32_t input_dtype, weight_dtype, bias_dtype, output_dtype;
} Parameters;
typedef struct {
    void *output;
    const void *input, *weight, *bias;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;
static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    uint32_t types[] = {p->input_dtype, p->weight_dtype, p->bias_dtype, p->output_dtype};
    for (size_t i = 0; i < 4; ++i)
        if (types[i] != TRILLIM_FP32 && types[i] != TRILLIM_BF16) return TRILLIM_ERROR;
    if ((p->stride != 1 && p->stride != 2) || !p->length || p->output_length != (p->length - 1) / p->stride + 1 ||
        p->channels > SIZE_MAX / 4 / p->length || p->channels > SIZE_MAX / 12 ||
        (p->channels && p->outputs > SIZE_MAX / 12 / p->channels) || p->outputs > SIZE_MAX / 4 / p->output_length)
        return TRILLIM_ERROR;
    inputs[0] = p->channels * p->length * trillim_itemsize(p->input_dtype);
    inputs[1] = p->outputs * p->channels * 3 * trillim_itemsize(p->weight_dtype);
    inputs[2] = p->outputs * trillim_itemsize(p->bias_dtype);
    outputs[0] = p->outputs * p->output_length * trillim_itemsize(p->output_dtype);
    return TRILLIM_OK;
}
SIMD static __m256 load8(const Task *t, size_t index) {
    const Parameters *p = t->p;
    if (p->input_dtype == TRILLIM_FP32) {
        const float *data = (const float *)t->input + index;
        if (p->stride == 1) return _mm256_loadu_ps(data);
        return _mm256_i32gather_ps(data, _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), 4);
    }
    const uint16_t *data = (const uint16_t *)t->input + index;
    __m256i bits = p->stride == 1 ? _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)data))
                                  : _mm256_loadu_si256((const __m256i *)data);
    return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
}
static void store(const Task *t, size_t index, float value) {
    if (t->p->output_dtype == TRILLIM_FP32)
        memcpy((float *)t->output + index, &value, 4);
    else {
        uint16_t bits = trillim_to_bf16(value);
        memcpy((uint16_t *)t->output + index, &bits, 2);
    }
}
SIMD static TrillimStatus convolve(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const Parameters *p = t->p;
    size_t tiles = p->output_length / 128 + (p->output_length % 128 != 0), column = tile / tiles;
    size_t first = tile % tiles * 128, end = p->output_length - first < 128 ? p->output_length : first + 128;
    float bias = trillim_load(t->bias, column, p->bias_dtype).f;
    size_t position = first;
    for (; position < end;) {
        if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        if (position == 0 || end - position < 32 || (position + 31) * p->stride + p->stride >= p->length) {
            float sum = bias;
            for (size_t c = 0; c < p->channels; ++c) {
                if ((c & 63) == 0 && t->context->is_cancelled(t->context->cancellation_context))
                    return TRILLIM_CANCELLED;
                for (size_t k = 0; k < 3; ++k) {
                    size_t ix = position * p->stride + k;
                    if (ix == 0 || ix > p->length) continue;
                    sum += trillim_load(t->input, c * p->length + ix - 1, p->input_dtype).f *
                           trillim_load(t->weight, (column * p->channels + c) * 3 + k, p->weight_dtype).f;
                }
            }
            store(t, column * p->output_length + position, sum);
            ++position;
            continue;
        }
        __m256 a = _mm256_set1_ps(bias), b = a, c = a, d = a;
        for (size_t channel = 0; channel < p->channels; ++channel) {
            if ((channel & 63) == 0 && t->context->is_cancelled(t->context->cancellation_context))
                return TRILLIM_CANCELLED;
            for (size_t k = 0; k < 3; ++k) {
                __m256 weight = _mm256_set1_ps(
                    trillim_load(t->weight, (column * p->channels + channel) * 3 + k, p->weight_dtype).f);
                size_t ix = channel * p->length + position * p->stride + k - 1;
                a = _mm256_fmadd_ps(load8(t, ix), weight, a);
                b = _mm256_fmadd_ps(load8(t, ix + 8 * p->stride), weight, b);
                c = _mm256_fmadd_ps(load8(t, ix + 16 * p->stride), weight, c);
                d = _mm256_fmadd_ps(load8(t, ix + 24 * p->stride), weight, d);
            }
        }
        float values[32];
        _mm256_storeu_ps(values, a);
        _mm256_storeu_ps(values + 8, b);
        _mm256_storeu_ps(values + 16, c);
        _mm256_storeu_ps(values + 24, d);
        for (size_t i = 0; i < 32; ++i)
            store(t, column * p->output_length + position + i, values[i]);
        position += 32;
    }
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_conv1d_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                     const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const Parameters *p = parameters;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, context};
    size_t tiles = p->output_length / 128 + (p->output_length % 128 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->outputs * tiles, convolve, &task);
}
TRILLIM_KERNEL const char *trillim_x86_conv1d_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _conv1d as bind\n"
           "def pattern(x: Tensor, weight: Tensor, bias: Tensor, *, kernel: tuple[int, ...] = (3,), stride = 1, "
           "padding = 1, dilation = 1, groups: int = 1, dtype: DType | None = None) -> tuple[Tensor, ...]:\n"
           "    return (Conv(x, weight, bias=bias, kernel=kernel, stride=stride, padding=padding, dilation=dilation, "
           "groups=groups) if dtype is None else Cast(Conv(x, weight, bias=bias, kernel=kernel, stride=stride, "
           "padding=padding, dilation=dilation, groups=groups), dtype),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_conv1d_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"channels", "size_t", offsetof(Parameters, channels), NULL},
        {"length", "size_t", offsetof(Parameters, length), NULL},
        {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
        {"stride", "size_t", offsetof(Parameters, stride), NULL},
        {"output_length", "size_t", offsetof(Parameters, output_length), NULL},
        {"input_dtype", "uint32_t", offsetof(Parameters, input_dtype), NULL},
        {"weight_dtype", "uint32_t", offsetof(Parameters, weight_dtype), NULL},
        {"bias_dtype", "uint32_t", offsetof(Parameters, bias_dtype), NULL},
        {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL},
    };
    static const size_t inputs[] = {1, 1, 1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 9, fields, 3, 1, requirements, NULL, inputs, outputs, 1};
    return &schema;
}
TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_conv1d_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA};
    return &caps;
}
