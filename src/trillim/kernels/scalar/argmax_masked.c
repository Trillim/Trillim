#include "common.h"

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

static TrillimStatus requirements(const void *parameters, size_t *inputs, size_t *outputs, size_t *scratch) {
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

static int64_t scalar_argmax(const Task *task, size_t row) {
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

static TrillimStatus apply(void *opaque,size_t row,size_t worker) {
    (void)worker;
    const Task *task = opaque;
    if (task->context->is_cancelled(task->context->cancellation_context)) return TRILLIM_CANCELLED;
    int64_t selected = scalar_argmax(task,row);
    if (selected == -1) return TRILLIM_CANCELLED;
    TrillimValue result = {.i = selected};
    trillim_store(task->output,row,task->p->output_dtype,result);
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_argmax_masked(void *const *outputs,const void *const *inputs,void *scratch,
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

TRILLIM_KERNEL const char *trillim_scalar_argmax_masked_tdl(void) {
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

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_argmax_masked_parameters(void) {
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

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_argmax_masked_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION,0};
    return &capabilities;
}
