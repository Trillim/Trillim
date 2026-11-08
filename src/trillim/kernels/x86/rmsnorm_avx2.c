#include "avx2_helpers.h"
#include "normalization.h"

TRILLIM_AVX2 static TrillimStatus normalize(void *opaque, size_t row, size_t worker) {
    (void)worker;
    const TrillimRMSNormTask *task = (const TrillimRMSNormTask *)opaque;
    const TrillimRMSNormParameters *p = task->parameters;
    const float *input = (const float *)task->input + row * p->width;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    size_t column = 0;
    for (; p->width - column >= 32; column += 32) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 a = _mm256_loadu_ps(input + column);
        __m256 b = _mm256_loadu_ps(input + column + 8);
        __m256 c = _mm256_loadu_ps(input + column + 16);
        __m256 d = _mm256_loadu_ps(input + column + 24);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a, a));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(b, b));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(c, c));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(d, d));
    }
    float sum = trillim_avx2_sum(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
    for (; column < p->width; ++column) {
        sum += input[column] * input[column];
    }
    float denominator = sqrtf(sum / (float)p->width + p->eps);
    __m256 divisor = _mm256_set1_ps(denominator);
    __m256 offset = _mm256_set1_ps(p->weight_offset);
    column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        __m256 value = _mm256_div_ps(_mm256_loadu_ps(input + column), divisor);
        __m256 weight = trillim_avx2_load8(task->weight, column, p->weight_dtype, 1, p->width);
        value = _mm256_mul_ps(value, _mm256_add_ps(weight, offset));
        _mm256_storeu_ps((float *)task->output + row * p->width + column, value);
    }
    for (; column < p->width; ++column) {
        float weight = trillim_read_float(task->weight, column, p->weight_dtype, 1, p->width);
        ((float *)task->output)[row * p->width + column] = (input[column] / denominator) * (weight + p->weight_offset);
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_rmsnorm_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimRMSNormParameters *p = (const TrillimRMSNormParameters *)parameters;
    TrillimRMSNormTask task = {outputs[0], inputs[0], inputs[1], p, context, true};
    return context->thread_pool->run(context->thread_pool->context, p->rows, normalize, &task);
}

TRILLIM_KERNEL const char *trillim_x86_rmsnorm_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _avx2_rmsnorm as bind\n"
           "def pattern(x: Tensor, weight: Tensor, *, eps: float = 1e-6, weight_offset: float = 0.0) -> tuple[Tensor, ...]:\n"
           "    return (RMSNorm(x, weight=weight, eps=eps, weight_offset=weight_offset),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_rmsnorm_avx2_parameters(void) {
    return trillim_rms_schema(true);
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_rmsnorm_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "x86", "avx2", "m"};
    return &capabilities;
}
