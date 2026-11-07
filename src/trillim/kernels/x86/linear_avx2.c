#include "avx2_helpers.h"
#include "matmul.h"

TRILLIM_AVX2 static TrillimStatus linear_task(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const TrillimMatmulTask *task = (const TrillimMatmulTask *)opaque;
    const TrillimMatmulParameters *p = task->p;
    size_t tiles = p->n / 64 + (p->n % 64 != 0);
    size_t token = index / tiles;
    size_t batch = token / p->m;
    size_t row = token % p->m;
    size_t first = (index % tiles) * 64;
    size_t last = p->n - first < 64 ? p->n : first + 64;
    size_t a_base = trillim_offset(batch, p->rank, p->batch_shape, p->a_batch_strides) + row * p->a_row_stride;
    size_t b_base = trillim_offset(batch, p->rank, p->batch_shape, p->b_batch_strides);
    for (size_t column = first; column < last; column += 4) {
        if (task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t lanes = last - column < 4 ? last - column : 4;
        size_t b0 = b_base + column * p->b_col_stride;
        size_t b1 = lanes > 1 ? b0 + p->b_col_stride : b0;
        size_t b2 = lanes > 2 ? b1 + p->b_col_stride : b0;
        size_t b3 = lanes > 3 ? b2 + p->b_col_stride : b0;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        size_t k = 0;
        for (; p->k - k >= 8; k += 8) {
            if (p->k > 32768 && (k & 1023) == 0 &&
                task->context->is_cancelled(task->context->cancellation_context)) {
                return TRILLIM_CANCELLED;
            }
            __m256 a = _mm256_loadu_ps((const float *)task->a + a_base + k);
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a, trillim_avx2_load8(task->b, b0 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns)));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a, trillim_avx2_load8(task->b, b1 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns)));
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a, trillim_avx2_load8(task->b, b2 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns)));
            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a, trillim_avx2_load8(task->b, b3 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns)));
        }
        float values[] = {trillim_avx2_sum(sum0), trillim_avx2_sum(sum1), trillim_avx2_sum(sum2), trillim_avx2_sum(sum3)};
        for (; k < p->k; ++k) {
            float a = ((const float *)task->a)[a_base + k];
            values[0] += a * trillim_read_float(task->b, b0 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns);
            values[1] += a * trillim_read_float(task->b, b1 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns);
            values[2] += a * trillim_read_float(task->b, b2 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns);
            values[3] += a * trillim_read_float(task->b, b3 + k, p->b_dtype, p->b_storage_rows, p->b_storage_columns);
        }
        for (size_t lane = 0; lane < lanes; ++lane) {
            memcpy((uint8_t *)task->output + (token * p->n + column + lane) * sizeof(float), &values[lane], sizeof(float));
        }
    }
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_linear_avx2(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    const TrillimMatmulParameters *p = (const TrillimMatmulParameters *)parameters;
    TrillimMatmulTask task = {outputs[0], inputs[0], inputs[1], p, context};
    size_t tiles = p->n / 64 + (p->n % 64 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->batches * p->m * tiles, linear_task, &task);
}

TRILLIM_KERNEL const char *trillim_x86_linear_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import avx2_linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Linear(x, weight),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_linear_avx2_parameters(void) {
    return trillim_matmul_schema();
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_linear_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {1, "x86", "avx2", "m"};
    return &capabilities;
}
