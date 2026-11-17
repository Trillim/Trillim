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

TRILLIM_AVX2 static inline __m256 select8(__m256i m, __m256 a, __m256 b) {
    return _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m));
}

TRILLIM_AVX2 static inline __m256i selecti8(__m256i m, __m256i a, __m256i b) {
    return _mm256_blendv_epi8(b, a, m);
}

TRILLIM_AVX2 static inline __m256i gt8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_GT_OQ));
}

TRILLIM_AVX2 static inline __m256i eq8(__m256 a, __m256 b) {
    return _mm256_castps_si256(_mm256_cmp_ps(a,b,_CMP_EQ_OQ));
}

TRILLIM_AVX2 static inline uint32_t any8(__m256i a) {
    return _mm256_testz_si256(a,a) ? 0 : UINT32_MAX;
}

TRILLIM_AVX2 static inline void storei8(void *data, __m256i a) {
    _mm256_storeu_si256((__m256i *)data,a);
}

TRILLIM_AVX2 static inline __m256i loadi8(const void *data) {
    return _mm256_loadu_si256((const __m256i *)data);
}

TRILLIM_AVX2 static inline __m256i not8(__m256i a) {
    return _mm256_xor_si256(a,_mm256_set1_epi32(-1));
}

TRILLIM_AVX2 static inline __m256i andnot8(__m256i a, __m256i b) {
    return _mm256_andnot_si256(b,a);
}

typedef struct {
    size_t rows, width, rank;
    const size_t *shape, *mask_strides;
    uint32_t input_dtype, output_dtype;
} Parameters;

typedef struct {
    void *output;
    const void *input;
    const uint8_t *mask;
    const Parameters *p;
    const TrillimKernelContext *context;
} Task;

TRILLIM_AVX2 static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    inputs[0] = inputs[1] = outputs[0] = *scratch = 0;
    if (!p->rank || !p->shape || !p->mask_strides || !trillim_float(p->input_dtype) ||
        (p->output_dtype != TRILLIM_INT32 && p->output_dtype != TRILLIM_INT64) ||
        p->shape[p->rank - 1] != p->width || p->width > INT32_MAX) return TRILLIM_ERROR;
    size_t rows = 1;
    bool empty = false;
    for (size_t axis = 0; axis + 1 < p->rank; ++axis) empty |= p->shape[axis] == 0;
    if (empty) rows = 0;
    else for (size_t axis = 0; axis + 1 < p->rank; ++axis) {
        if (rows > SIZE_MAX / p->shape[axis]) return TRILLIM_ERROR;
        rows *= p->shape[axis];
    }
    size_t output_bytes = trillim_itemsize(p->output_dtype);
    size_t input_bytes = trillim_itemsize(p->input_dtype);
    if (rows != p->rows || rows > SIZE_MAX / output_bytes ||
        (p->width && rows > SIZE_MAX / input_bytes / p->width)) return TRILLIM_ERROR;
    outputs[0] = rows * output_bytes;
    inputs[0] = rows * p->width * input_bytes;
    if (rows && p->width && !trillim_span(p->rank,p->shape,p->mask_strides,&inputs[1])) return TRILLIM_ERROR;
    return TRILLIM_OK;
}

TRILLIM_AVX2 static int64_t scalar_argmax(const Task *task, size_t row) {
    const Parameters *p = task->p;
    size_t base = row * p->width;
    size_t mask_base = p->width ? trillim_offset(base,p->rank,p->shape,p->mask_strides) : 0;
    size_t mask_step = p->mask_strides[p->rank - 1];
    int64_t selected = INT64_MAX;
    float best = 0;
    for (size_t column = 0; column < p->width; ++column) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return -1;
        if (!task->mask[mask_base + column * mask_step]) continue;
        float value = trillim_load(task->input,base + column,p->input_dtype).f;
        if (selected == INT64_MAX || !(best >= value)) {
            best = value;
            selected = (int64_t)column;
        }
    }
    return selected;
}

TRILLIM_AVX2 static TrillimStatus apply(void *opaque,size_t row,size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->p;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    size_t base = row * p->width;
    size_t mask_base = p->width ? trillim_offset(base,p->rank,p->shape,p->mask_strides) : 0;
    size_t mask_step = p->mask_strides[p->rank - 1];
    __m256 best = _mm256_set1_ps(0);
    __m256i valid = _mm256_set1_epi32(0);
    __m256i selected = _mm256_set1_epi32(INT32_MAX);
    const int32_t positions[] = {0,1,2,3,4,5,6,7};
    __m256i lanes = loadi8(positions);
    bool nan = false;
    size_t column = 0;
    for (; p->width - column >= 8; column += 8) {
        if ((column & 1023) == 0 && task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
        uint32_t masks[8];
        for (size_t lane = 0; lane < 8; ++lane) masks[lane] = task->mask[mask_base + (column + lane) * mask_step] ? UINT32_MAX : 0;
        __m256i active = loadi8(masks);
        __m256 value = load8(task->input,base + column,p->input_dtype);
        nan |= any8(andnot8(active,eq8(value,value))) != 0;
        __m256i choose = _mm256_and_si256(active,_mm256_or_si256(not8(valid),gt8(value,best)));
        best = select8(choose,value,best);
        selected = selecti8(choose,_mm256_add_epi32(_mm256_set1_epi32((int32_t)column),lanes),selected);
        valid = _mm256_or_si256(valid,active);
    }
    int64_t index = INT64_MAX;
    float maximum = 0;
    if (!nan) {
        float values[8];
        int32_t indices[8];
        uint32_t active[8];
        _mm256_storeu_ps(values,best);
        storei8(indices,selected);
        storei8(active,valid);
        for (size_t lane = 0; lane < 8; ++lane) {
            if (active[lane] && (index == INT64_MAX || values[lane] > maximum ||
                (values[lane] == maximum && indices[lane] < index))) {
                maximum = values[lane];
                index = indices[lane];
            }
        }
        for (; column < p->width; ++column) {
            if (!task->mask[mask_base + column * mask_step]) continue;
            float value = trillim_load(task->input,base + column,p->input_dtype).f;
            if (index == INT64_MAX || !(maximum >= value)) {
                maximum = value;
                index = (int64_t)column;
            }
        }
    } else index = scalar_argmax(task,row);
    if (index == -1) return TRILLIM_CANCELLED;
    TrillimValue result = {.i = index};
    trillim_store(task->output,row,p->output_dtype,result);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_x86_argmax_masked_avx2(void *const *outputs,const void *const *inputs,void *scratch,
    const void *parameters,const TrillimKernelContext *context) {
    (void)scratch;
    if (context->is_cancelled(context->cancellation_context)) return TRILLIM_CANCELLED;
    const Parameters *p = parameters;
    Task task = {outputs[0],inputs[0],inputs[1],p,context};
    if (p->rows <= 4 || (p->width <= 4096 && p->rows <= 4096 / (p->width ? p->width : 1))) {
        for (size_t row = 0; row < p->rows; ++row) {
            TrillimStatus status = apply(&task,row,0);
            if (status != TRILLIM_OK) return status;
        }
        return TRILLIM_OK;
    }
    return context->thread_pool->run(context->thread_pool->context,p->rows,apply,&task);
}

TRILLIM_KERNEL const char *trillim_x86_argmax_masked_avx2_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _broadcast_strides, _contiguous_strides, _product, _DTYPE_IDS\n"
           "physical_inputs = ('x','mask')\n"
           "def bind(c):\n"
           "    x, mask, o = c.inputs['x'], c.inputs['mask'], c.outputs[0]\n"
           "    rank = c.attrs['rank']\n"
           "    axis = c.attrs.get('axis',-1)\n"
           "    if not isinstance(rank,int) or rank != len(x.shape) or rank < 1 or not isinstance(axis,int) or axis not in (-1,rank-1):\n"
           "        raise ValueError('masked Argmax requires the final axis')\n"
           "    if x.dtype not in ('fp16','bf16','fp32') or mask.dtype != 'bool' or o.dtype not in ('int32','int64'):\n"
           "        raise ValueError('masked Argmax requires native floating scores, a boolean mask and integer indices')\n"
           "    if (x.strides or _contiguous_strides(x.shape)) != _contiguous_strides(x.shape) or o.shape != x.shape[:-1]:\n"
           "        raise ValueError('masked Argmax requires contiguous score rows and an axis-reduced output')\n"
           "    return dict(rows=_product(x.shape[:-1]),width=x.shape[-1],rank=rank,shape=x.shape,\n"
           "                mask_strides=_broadcast_strides(mask.shape,x.shape,mask.strides),\n"
           "                input_dtype=_DTYPE_IDS[x.dtype],output_dtype=_DTYPE_IDS[o.dtype])\n"
           "def pattern(x: Tensor,mask: Tensor,*,axis: int = -1,rank: int,dtype: DType | None = None) -> tuple[Tensor,...]:\n"
           "    result=Argmax(x,axis=axis,rank=rank,mask=mask)\n"
           "    return (result if dtype is None else Cast(result,dtype),)\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_x86_argmax_masked_avx2_parameters(void) {
    static const TrillimParameter fields[] = {
        {"rows","size_t",offsetof(Parameters,rows),NULL},
        {"width","size_t",offsetof(Parameters,width),NULL},
        {"rank","size_t",offsetof(Parameters,rank),NULL},
        {"shape","size_t",offsetof(Parameters,shape),"rank"},
        {"mask_strides","size_t",offsetof(Parameters,mask_strides),"rank"},
        {"input_dtype","uint32_t",offsetof(Parameters,input_dtype),NULL},
        {"output_dtype","uint32_t",offsetof(Parameters,output_dtype),NULL}
    };
    static const size_t inputs[] = {1,1}, outputs[] = {1};
    static const TrillimParameterSchema schema = {sizeof(Parameters),TRILLIM_ALIGNOF(Parameters),7,fields,
        2,1,requirements,NULL,inputs,outputs,1};
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_x86_argmax_masked_avx2_capabilities(void) {
    static const TrillimCapabilities caps = {TRILLIM_KERNEL_ABI_VERSION, TRILLIM_CAP_X86_AVX2};
    return &caps;
}
