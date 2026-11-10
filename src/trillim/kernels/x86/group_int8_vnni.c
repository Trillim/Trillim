#include "common.h"
#include <immintrin.h>

#define SIMD __attribute__((target("avx2,fma,avxvnni")))

typedef struct {
    size_t tokens, width, outputs;
    uint32_t output_dtype;
} Parameters;
typedef struct {
    void *output;
    const int8_t *input, *weight;
    const float *scales;
    const Parameters *p;
    size_t columns;
    const TrillimKernelContext *context;
} Task;

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = inputs[2] = outputs[0] = *scratch = 0;
    size_t bytes = trillim_itemsize(p->output_dtype), groups = p->width / 32;
    if (!trillim_float(p->output_dtype) || !p->width || p->width % 32 || groups > SIZE_MAX / 34 ||
        p->outputs > SIZE_MAX / (groups * 34) || p->tokens > SIZE_MAX / p->width ||
        p->tokens > SIZE_MAX / sizeof(float) || (p->outputs && p->tokens > SIZE_MAX / bytes / p->outputs))
        return TRILLIM_ERROR;
    inputs[0] = p->tokens * p->width;
    inputs[1] = p->outputs * groups * 34;
    inputs[2] = p->tokens * sizeof(float);
    outputs[0] = p->tokens * p->outputs * bytes;
    return TRILLIM_OK;
}

SIMD static __m256 product(__m256i magnitude, __m256i x, __m256i weight) {
    __m256i signed_weight = _mm256_sign_epi8(weight, x);
    __m256i sum = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), magnitude, signed_weight);
    return _mm256_cvtepi32_ps(sum);
}

SIMD static float sum8(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

static float weight_scale(const Task *task, size_t column, size_t block) {
    uint16_t bits;
    memcpy(&bits,
           (const uint8_t *)task->weight + task->p->outputs * task->p->width +
               (column * (task->p->width / 32) + block) * 2,
           sizeof(bits));
    return trillim_bf16(bits);
}

static void store(const Task *task, size_t row, size_t column, float value) {
    TrillimValue result = {.f = value / task->scales[row]};
    size_t index = row * task->p->outputs + column;
    if (task->p->output_dtype == TRILLIM_BF16) {
        uint16_t bits = trillim_to_bf16(result.f);
        memcpy((uint8_t *)task->output + index * 2, &bits, sizeof(bits));
    } else if (task->p->output_dtype == TRILLIM_FP32) {
        memcpy((uint8_t *)task->output + index * 4, &result.f, sizeof(result.f));
    } else
        trillim_store(task->output, index, task->p->output_dtype, result);
}

SIMD static TrillimStatus decode(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t begin = tile * task->columns;
    size_t end = p->outputs - begin < task->columns ? p->outputs : begin + task->columns;
    for (size_t column = begin; column < end; column += 4) {
        if (((column - begin) & 31) == 0 && task->context->is_cancelled(task->context->cancellation_context))
            return TRILLIM_CANCELLED;
        size_t count = end - column < 4 ? end - column : 4;
        size_t c1 = count > 1 ? column + 1 : column, c2 = count > 2 ? column + 2 : column,
               c3 = count > 3 ? column + 3 : column;
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (size_t k = 0; k < p->width; k += 32) {
            if (p->width > 32768 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                return TRILLIM_CANCELLED;
            __m256i x = _mm256_loadu_si256((const __m256i *)(task->input + k)), magnitude = _mm256_abs_epi8(x);
            a = _mm256_fmadd_ps(
                product(magnitude, x, _mm256_loadu_si256((const __m256i *)(task->weight + column * p->width + k))),
                _mm256_set1_ps(weight_scale(task, column, k / 32)), a);
            b = _mm256_fmadd_ps(
                product(magnitude, x, _mm256_loadu_si256((const __m256i *)(task->weight + c1 * p->width + k))),
                _mm256_set1_ps(weight_scale(task, c1, k / 32)), b);
            c = _mm256_fmadd_ps(
                product(magnitude, x, _mm256_loadu_si256((const __m256i *)(task->weight + c2 * p->width + k))),
                _mm256_set1_ps(weight_scale(task, c2, k / 32)), c);
            d = _mm256_fmadd_ps(
                product(magnitude, x, _mm256_loadu_si256((const __m256i *)(task->weight + c3 * p->width + k))),
                _mm256_set1_ps(weight_scale(task, c3, k / 32)), d);
        }
        store(task, 0, column, sum8(a));
        if (count > 1) store(task, 0, c1, sum8(b));
        if (count > 2) store(task, 0, c2, sum8(c));
        if (count > 3) store(task, 0, c3, sum8(d));
    }
    return TRILLIM_OK;
}

#define ACCUMULATE(row_index, acc0, acc1)                                                                              \
    do {                                                                                                               \
        __m256i x = _mm256_loadu_si256((const __m256i *)(task->input + (row + (row_index)) * p->width + k));           \
        __m256i magnitude = _mm256_abs_epi8(x);                                                                        \
        acc0 = _mm256_fmadd_ps(product(magnitude, x, w0), scale0, acc0);                                               \
        acc1 = _mm256_fmadd_ps(product(magnitude, x, w1), scale1, acc1);                                               \
    } while (0)

SIMD static TrillimStatus prefill4(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 64 + (p->outputs % 64 != 0);
    size_t row = tile / tiles * 4, first = tile % tiles * 64;
    size_t rows = p->tokens - row < 4 ? p->tokens - row : 4;
    size_t end = p->outputs - first < 64 ? p->outputs : first + 64;
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        size_t second = column + 1 < end ? column + 1 : column;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        for (size_t k = 0; k < p->width; k += 32) {
            if (p->width > 32768 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                return TRILLIM_CANCELLED;
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(task->weight + column * p->width + k));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(task->weight + second * p->width + k));
            __m256 scale0 = _mm256_set1_ps(weight_scale(task, column, k / 32));
            __m256 scale1 = _mm256_set1_ps(weight_scale(task, second, k / 32));
            ACCUMULATE(0, a0, a1);
            if (rows > 1) {
                ACCUMULATE(1, b0, b1);
            }
            if (rows > 2) {
                ACCUMULATE(2, c0, c1);
            }
            if (rows > 3) {
                ACCUMULATE(3, d0, d1);
            }
        }
        store(task, row, column, sum8(a0));
        if (second != column) store(task, row, second, sum8(a1));
        if (rows > 1) {
            store(task, row + 1, column, sum8(b0));
            if (second != column) store(task, row + 1, second, sum8(b1));
        }
        if (rows > 2) {
            store(task, row + 2, column, sum8(c0));
            if (second != column) store(task, row + 2, second, sum8(c1));
        }
        if (rows > 3) {
            store(task, row + 3, column, sum8(d0));
            if (second != column) store(task, row + 3, second, sum8(d1));
        }
    }
    return TRILLIM_OK;
}
SIMD static TrillimStatus prefill(void *opaque, size_t tile, size_t worker) {
    const Task *task = opaque;
    const Parameters *p = task->p;
    size_t tiles = p->outputs / 64 + (p->outputs % 64 != 0);
    size_t row = tile / tiles * 8, first = tile % tiles * 64;
    size_t count = p->tokens - row;
    size_t end = p->outputs - first < 64 ? p->outputs : first + 64;
    if (count < 8 || (end - first) % 2) {
        size_t base = tile / tiles * 2 * tiles + tile % tiles;
        TrillimStatus status = prefill4(opaque, base, worker);
        if (status != TRILLIM_OK || count <= 4) return status;
        return prefill4(opaque, base + tiles, worker);
    }
    for (size_t column = first; column < end; column += 2) {
        if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        __m256 a0 = _mm256_setzero_ps(), b0 = a0;
        __m256 a1 = _mm256_setzero_ps(), b1 = a1;
        __m256 a2 = _mm256_setzero_ps(), b2 = a2;
        __m256 a3 = _mm256_setzero_ps(), b3 = a3;
        __m256 a4 = _mm256_setzero_ps(), b4 = a4;
        __m256 a5 = _mm256_setzero_ps(), b5 = a5;
        __m256 a6 = _mm256_setzero_ps(), b6 = a6;
        __m256 a7 = _mm256_setzero_ps(), b7 = a7;
        for (size_t k = 0; k < p->width; k += 32) {
            if (p->width > 32768 && (k & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context))
                return TRILLIM_CANCELLED;
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(task->weight + column * p->width + k));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(task->weight + (column + 1) * p->width + k));
            __m256 scale0 = _mm256_set1_ps(weight_scale(task, column, k / 32));
            __m256 scale1 = _mm256_set1_ps(weight_scale(task, column + 1, k / 32));
            ACCUMULATE(0, a0, b0);
            ACCUMULATE(1, a1, b1);
            ACCUMULATE(2, a2, b2);
            ACCUMULATE(3, a3, b3);
            ACCUMULATE(4, a4, b4);
            ACCUMULATE(5, a5, b5);
            ACCUMULATE(6, a6, b6);
            ACCUMULATE(7, a7, b7);
        }
        store(task, row + 0, column, sum8(a0));
        store(task, row + 0, column + 1, sum8(b0));
        store(task, row + 1, column, sum8(a1));
        store(task, row + 1, column + 1, sum8(b1));
        store(task, row + 2, column, sum8(a2));
        store(task, row + 2, column + 1, sum8(b2));
        store(task, row + 3, column, sum8(a3));
        store(task, row + 3, column + 1, sum8(b3));
        store(task, row + 4, column, sum8(a4));
        store(task, row + 4, column + 1, sum8(b4));
        store(task, row + 5, column, sum8(a5));
        store(task, row + 5, column + 1, sum8(b5));
        store(task, row + 6, column, sum8(a6));
        store(task, row + 6, column + 1, sum8(b6));
        store(task, row + 7, column, sum8(a7));
        store(task, row + 7, column + 1, sum8(b7));
    }
    return TRILLIM_OK;
}
#undef ACCUMULATE

TRILLIM_KERNEL TrillimStatus trillim_x86_group_int8_vnni(void *const *outputs, const void *const *inputs, void *scratch,
                                                         const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    if (!p->tokens || !p->outputs) return TRILLIM_OK;
    size_t workers = context->thread_pool->worker_count;
    if (!workers) return TRILLIM_ERROR;
    size_t per_worker = p->outputs / workers + (p->outputs % workers != 0);
    size_t columns = (per_worker / 16 + (per_worker % 16 != 0)) * 4;
    Task task = {outputs[0], inputs[0], inputs[1], inputs[2], p, columns, context};
    if (p->tokens == 1)
        return context->thread_pool->run(context->thread_pool->context,
                                         p->outputs / columns + (p->outputs % columns != 0), decode, &task);
    size_t rows = p->tokens / 8 + (p->tokens % 8 != 0), tiles = p->outputs / 64 + (p->outputs % 64 != 0);
    return context->thread_pool->run(context->thread_pool->context, rows * tiles, prefill, &task);
}

TRILLIM_KERNEL const char *trillim_x86_group_int8_vnni_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _group_int8_linear as bind\n"
           "def pattern(x: Tensor, weight: Tensor, scale: Tensor, *, dtype: DType = 'fp32') -> tuple[Tensor, ...]:\n"
           "    return (QuantizedLinear(x, weight, scale, dtype=dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_group_int8_vnni_parameters(void) {
    static const TrillimParameter fields[] = {{"tokens", "size_t", offsetof(Parameters, tokens), NULL},
                                              {"width", "size_t", offsetof(Parameters, width), NULL},
                                              {"outputs", "size_t", offsetof(Parameters, outputs), NULL},
                                              {"output_dtype", "uint32_t", offsetof(Parameters, output_dtype), NULL}};
    static const size_t inputs[] = {1, 1, 4}, outputs[] = {1};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 4, fields, 3, 1, requirements, NULL, inputs, outputs, 1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_group_int8_vnni_capabilities(void) {
    static const TrillimCapabilities capabilities = {
        TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_X86_AVX_VNNI};
    return &capabilities;
}
