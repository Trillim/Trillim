#include "common.h"
#include <cpuid.h>
#include <immintrin.h>
#include <math.h>
#define FAST __attribute__((target("avx2,fma,f16c")))
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

typedef struct {
    void *output;
    const void *input, *weight;
    const TrillimMatmulParameters *p;
    const TrillimKernelContext *context;
} Task;
static bool floating(uint32_t dtype) { return dtype == TRILLIM_FP16 || dtype == TRILLIM_BF16 || dtype == TRILLIM_FP32; }
static bool extent(size_t base, size_t rows, size_t columns, size_t row_stride, size_t column_stride, size_t capacity) {
    if (!rows || !columns) return true;
    if (row_stride && rows - 1 > (SIZE_MAX - base) / row_stride) return false;
    base += (rows - 1) * row_stride;
    if (column_stride && columns - 1 > (SIZE_MAX - base) / column_stride) return false;
    return base + (columns - 1) * column_stride < capacity;
}
static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const TrillimMatmulParameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!floating(p->a_dtype) ||
        !floating(p->b_dtype) || !floating(p->out_dtype))
        return TRILLIM_ERROR;
    if (!p->batches || !p->m || !p->n) return TRILLIM_OK;
    if (p->m > SIZE_MAX / p->n || p->batches > SIZE_MAX / p->m / p->n / trillim_itemsize(p->out_dtype)) return TRILLIM_ERROR;
    size_t rows[] = {p->a_storage_rows, p->b_storage_rows}, columns[] = {p->a_storage_columns, p->b_storage_columns};
    uint32_t types[] = {p->a_dtype, p->b_dtype};
    for (size_t i = 0; i < 2; ++i) {
        if (columns[i] && rows[i] > SIZE_MAX / trillim_itemsize(types[i]) / columns[i]) return TRILLIM_ERROR;
        inputs[i] = rows[i] * columns[i] * trillim_itemsize(types[i]);
    }
    size_t a_span, b_span;
    if (!trillim_span(p->rank,p->batch_shape,p->a_batch_strides,&a_span) ||
        !trillim_span(p->rank,p->batch_shape,p->b_batch_strides,&b_span)) return TRILLIM_ERROR;
    size_t a_base = a_span ? a_span - 1 : 0, b_base = b_span ? b_span - 1 : 0;
    if (!extent(a_base,p->m,p->k,p->a_row_stride,p->a_col_stride,rows[0] * columns[0]) ||
        !extent(b_base,p->k,p->n,p->b_row_stride,p->b_col_stride,rows[1] * columns[1])) return TRILLIM_ERROR;
    outputs[0] = p->batches * p->m * p->n * trillim_itemsize(p->out_dtype);
    return TRILLIM_OK;
}
FAST static inline __attribute__((always_inline)) __m256 load8(const void *data, size_t index, uint32_t dtype) {
    if (dtype == TRILLIM_FP32) return _mm256_loadu_ps((const float *)data + index);
    __m128i v = _mm_loadu_si128((const __m128i *)((const uint16_t *)data + index));
    if (dtype == TRILLIM_FP16) return _mm256_cvtph_ps(v);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}
FAST static float sum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(_mm_hadd_ps(s, s));
}
static void store(const Task *t, size_t row, size_t col, float x) {
    TrillimValue value = {.f = x};
    trillim_store(t->output, row * t->p->n + col, t->p->out_dtype, value);
}
FAST static TrillimStatus project(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const TrillimMatmulParameters *p = t->p;
    size_t tiles = p->n / 64 + (p->n % 64 != 0), row = tile / tiles * 4, first = tile % tiles * 64;
    size_t rows = p->m - row < 4 ? p->m - row : 4, end = p->n - first < 64 ? p->n : first + 64;
    for (size_t column = first; column < end; column += 2) {
        if (t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
        size_t second = column + 1 < end ? column + 1 : column;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        size_t k = 0;
        for (; p->k - k >= 8; k += 8) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            __m256 w0 = load8(t->weight, column * p->b_col_stride + k, p->b_dtype),
                   w1 = load8(t->weight, second * p->b_col_stride + k, p->b_dtype);
            __m256 x = load8(t->input, row * p->a_row_stride + k, p->a_dtype);
            a0 = _mm256_fmadd_ps(x, w0, a0);
            a1 = _mm256_fmadd_ps(x, w1, a1);
            if (rows > 1) {
                x = load8(t->input, (row + 1) * p->a_row_stride + k, p->a_dtype);
                b0 = _mm256_fmadd_ps(x, w0, b0);
                b1 = _mm256_fmadd_ps(x, w1, b1);
            }
            if (rows > 2) {
                x = load8(t->input, (row + 2) * p->a_row_stride + k, p->a_dtype);
                c0 = _mm256_fmadd_ps(x, w0, c0);
                c1 = _mm256_fmadd_ps(x, w1, c1);
            }
            if (rows > 3) {
                x = load8(t->input, (row + 3) * p->a_row_stride + k, p->a_dtype);
                d0 = _mm256_fmadd_ps(x, w0, d0);
                d1 = _mm256_fmadd_ps(x, w1, d1);
            }
        }
        float sums[4][2] = {{sum8(a0), sum8(a1)}, {sum8(b0), sum8(b1)}, {sum8(c0), sum8(c1)}, {sum8(d0), sum8(d1)}};
        for (; k < p->k; ++k)
            for (size_t r = 0; r < rows; ++r) {
                float x = trillim_load(t->input, (row + r) * p->a_row_stride + k, p->a_dtype).f;
                sums[r][0] += x * trillim_load(t->weight, column * p->b_col_stride + k, p->b_dtype).f;
                sums[r][1] += x * trillim_load(t->weight, second * p->b_col_stride + k, p->b_dtype).f;
            }
        for (size_t r = 0; r < rows; ++r) {
            store(t, row + r, column, sums[r][0]);
            if (second != column) store(t, row + r, second, sums[r][1]);
        }
    }
    return TRILLIM_OK;
}
static TrillimStatus portable(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *t = opaque;
    const TrillimMatmulParameters *p = t->p;
    size_t tiles = p->n / 16 + (p->n % 16 != 0), row = tile / tiles, first = tile % tiles * 16,
           end = p->n - first < 16 ? p->n : first + 16;
    for (size_t col = first; col < end; ++col) {
        float value = 0;
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 1023) == 0 && t->context->is_cancelled(t->context->cancellation_context)) return TRILLIM_CANCELLED;
            value += trillim_load(t->input, row * p->a_row_stride + k * p->a_col_stride, p->a_dtype).f *
                     trillim_load(t->weight, col * p->b_col_stride + k * p->b_row_stride, p->b_dtype).f;
        }
        store(t, row, col, value);
    }
    return TRILLIM_OK;
}
static TrillimStatus project_batch(void *opaque, size_t index, size_t worker) {
    const Task *t = opaque;
    const TrillimMatmulParameters *p = t->p;
    size_t rows = p->m / 4 + (p->m % 4 != 0), tiles = p->n / 64 + (p->n % 64 != 0);
    size_t count = rows * tiles, batch = index / count;
    Task task = *t;
    task.input = (const uint8_t *)t->input + trillim_offset(batch,p->rank,p->batch_shape,p->a_batch_strides) * trillim_itemsize(p->a_dtype);
    task.weight = (const uint8_t *)t->weight + trillim_offset(batch,p->rank,p->batch_shape,p->b_batch_strides) * trillim_itemsize(p->b_dtype);
    task.output = (uint8_t *)t->output + batch * p->m * p->n * trillim_itemsize(p->out_dtype);
    return project(&task,index % count,worker);
}
static TrillimStatus portable_batch(void *opaque, size_t index, size_t worker) {
    const Task *t = opaque;
    const TrillimMatmulParameters *p = t->p;
    size_t tiles = p->n / 16 + (p->n % 16 != 0), count = p->m * tiles, batch = index / count;
    Task task = *t;
    task.input = (const uint8_t *)t->input + trillim_offset(batch,p->rank,p->batch_shape,p->a_batch_strides) * trillim_itemsize(p->a_dtype);
    task.weight = (const uint8_t *)t->weight + trillim_offset(batch,p->rank,p->batch_shape,p->b_batch_strides) * trillim_itemsize(p->b_dtype);
    task.output = (uint8_t *)t->output + batch * p->m * p->n * trillim_itemsize(p->out_dtype);
    return portable(&task,index % count,worker);
}
FAST static TrillimStatus project_columns(void *opaque, size_t index, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const TrillimMatmulParameters *p = task->p;
    size_t tiles = p->n / 64 + (p->n % 64 != 0), groups = p->m / 4 + (p->m % 4 != 0);
    size_t batch = index / (tiles * groups), tile = index % (tiles * groups);
    size_t row = tile / tiles * 4, first = tile % tiles * 64;
    size_t rows = p->m - row < 4 ? p->m - row : 4, end = p->n - first < 64 ? p->n : first + 64;
    size_t a_base = trillim_offset(batch,p->rank,p->batch_shape,p->a_batch_strides);
    size_t b_base = trillim_offset(batch,p->rank,p->batch_shape,p->b_batch_strides);
    Task output_task = *task;
    output_task.output = (uint8_t *)task->output + batch * p->m * p->n * trillim_itemsize(p->out_dtype);
    if (rows == 1 && end - first == 64) {
        __m256 sums[8] = {_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),
                          _mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
        size_t input = a_base + row * p->a_row_stride;
        uint32_t input_dtype = p->a_dtype;
        size_t input_stride = p->a_col_stride, weight_stride = p->b_row_stride;
#define ACCUMULATE_DECODE(TYPE) \
        for (size_t start = 0; start < p->k; start += 256) { \
            if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED; \
            size_t stop = p->k - start < 256 ? p->k : start + 256; \
            for (size_t k = start; k < stop; ++k) { \
                __m256 x = _mm256_set1_ps(trillim_load(task->input,input + k * input_stride,input_dtype).f); \
                size_t weight = b_base + k * weight_stride + first; \
                sums[0] = _mm256_fmadd_ps(x,load8(task->weight,weight + 0,TYPE),sums[0]); \
                sums[1] = _mm256_fmadd_ps(x,load8(task->weight,weight + 8,TYPE),sums[1]); \
                sums[2] = _mm256_fmadd_ps(x,load8(task->weight,weight + 16,TYPE),sums[2]); \
                sums[3] = _mm256_fmadd_ps(x,load8(task->weight,weight + 24,TYPE),sums[3]); \
                sums[4] = _mm256_fmadd_ps(x,load8(task->weight,weight + 32,TYPE),sums[4]); \
                sums[5] = _mm256_fmadd_ps(x,load8(task->weight,weight + 40,TYPE),sums[5]); \
                sums[6] = _mm256_fmadd_ps(x,load8(task->weight,weight + 48,TYPE),sums[6]); \
                sums[7] = _mm256_fmadd_ps(x,load8(task->weight,weight + 56,TYPE),sums[7]); \
            } \
        }
        if (p->b_dtype == TRILLIM_FP32) {
            ACCUMULATE_DECODE(TRILLIM_FP32)
        } else if (p->b_dtype == TRILLIM_BF16) {
            ACCUMULATE_DECODE(TRILLIM_BF16)
        } else {
            ACCUMULATE_DECODE(TRILLIM_FP16)
        }
#undef ACCUMULATE_DECODE
        for (size_t block = 0; block < 8; ++block) {
            if (p->out_dtype == TRILLIM_FP32) {
                _mm256_storeu_ps((float *)output_task.output + row * p->n + first + block * 8,sums[block]);
            } else {
                float lanes[8]; _mm256_storeu_ps(lanes,sums[block]);
                for (size_t lane = 0; lane < 8; ++lane) store(&output_task,row,first + block * 8 + lane,lanes[lane]);
            }
        }
        return TRILLIM_OK;
    }
    size_t column = first;
    for (; end - column >= 8;) {
        size_t width = end - column >= 16 ? 16 : 8;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
        for (size_t k = 0; k < p->k; ++k) {
            if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
            size_t weight = b_base + k * p->b_row_stride + column;
            __m256 w0 = load8(task->weight,weight,p->b_dtype);
            __m256 w1 = width == 16 ? load8(task->weight,weight + 8,p->b_dtype) : _mm256_setzero_ps();
            size_t input = a_base + row * p->a_row_stride + k * p->a_col_stride;
            __m256 x = _mm256_set1_ps(trillim_load(task->input,input,p->a_dtype).f);
            a0 = _mm256_fmadd_ps(x,w0,a0); a1 = _mm256_fmadd_ps(x,w1,a1);
            if (rows > 1) {
                x = _mm256_set1_ps(trillim_load(task->input,input + p->a_row_stride,p->a_dtype).f);
                b0 = _mm256_fmadd_ps(x,w0,b0); b1 = _mm256_fmadd_ps(x,w1,b1);
            }
            if (rows > 2) {
                x = _mm256_set1_ps(trillim_load(task->input,input + 2 * p->a_row_stride,p->a_dtype).f);
                c0 = _mm256_fmadd_ps(x,w0,c0); c1 = _mm256_fmadd_ps(x,w1,c1);
            }
            if (rows > 3) {
                x = _mm256_set1_ps(trillim_load(task->input,input + 3 * p->a_row_stride,p->a_dtype).f);
                d0 = _mm256_fmadd_ps(x,w0,d0); d1 = _mm256_fmadd_ps(x,w1,d1);
            }
        }
        __m256 values[4][2] = {{a0,a1},{b0,b1},{c0,c1},{d0,d1}};
        for (size_t r = 0; r < rows; ++r) {
            for (size_t half = 0; half < width / 8; ++half) {
                if (p->out_dtype == TRILLIM_FP32) {
                    _mm256_storeu_ps((float *)output_task.output + (row+r)*p->n + column + half*8,values[r][half]);
                } else {
                    float lanes[8]; _mm256_storeu_ps(lanes,values[r][half]);
                    for (size_t lane = 0; lane < 8; ++lane) store(&output_task,row+r,column+half*8+lane,lanes[lane]);
                }
            }
        }
        column += width;
    }
    for (; column < end; ++column) {
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0;
            for (size_t k = 0; k < p->k; ++k) {
                if ((k & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
                sum = fmaf(trillim_load(task->input,a_base+(row+r)*p->a_row_stride+k*p->a_col_stride,p->a_dtype).f,
                    trillim_load(task->weight,b_base+k*p->b_row_stride+column,p->b_dtype).f,sum);
            }
            store(&output_task,row+r,column,sum);
        }
    }
    return TRILLIM_OK;
}
TRILLIM_KERNEL TrillimStatus trillim_x86_matmul_avx2(void *const *outputs, const void *const *inputs, void *scratch,
                                                    const void *parameters, const TrillimKernelContext *context) {
    (void)scratch;
    const TrillimMatmulParameters *p = parameters;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    if (!p->batches || !p->m || !p->n) return TRILLIM_OK;
    Task task = {outputs[0], inputs[0], inputs[1], p, context};
    // F16C is optional; the portable path keeps the advertised requirements
    // sufficient.
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 29))) {
        size_t rows = p->m / 4 + (p->m % 4 != 0), tiles = p->n / 64 + (p->n % 64 != 0);
        if (p->b_col_stride == 1) return context->thread_pool->run(context->thread_pool->context, p->batches * rows * tiles, project_columns, &task);
        if (p->a_col_stride == 1 && p->b_row_stride == 1) return context->thread_pool->run(context->thread_pool->context, p->batches * rows * tiles, project_batch, &task);
    }
    size_t tiles = p->n / 16 + (p->n % 16 != 0);
    return context->thread_pool->run(context->thread_pool->context, p->batches * p->m * tiles, portable_batch, &task);
}
TRILLIM_KERNEL const char *trillim_x86_matmul_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _matmul as bind\n"
           "def pattern(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:\n"
           "    return (Matmul(a, b),)\n";
}
TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_matmul_avx2_parameters(void) {
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
        {"b_storage_columns", "size_t", offsetof(TrillimMatmulParameters, b_storage_columns), NULL}};
    static const size_t inputs[] = {1, 1};
    static const size_t outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(TrillimMatmulParameters),
                                                  TRILLIM_ALIGNOF(TrillimMatmulParameters),
                                                  sizeof(fields) / sizeof(fields[0]),
                                                  fields,
                                                  2,
                                                  1,
                                                  requirements,
                                                  NULL,
                                                  inputs,
                                                  outputs,
                                                  1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_matmul_avx2_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,
                                                     TRILLIM_CAP_X86_AVX2 | TRILLIM_CAP_X86_FMA | TRILLIM_CAP_LIBM};
    return &capabilities;
}
