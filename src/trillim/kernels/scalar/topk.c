#include "common.h"

enum { MAX_K = 64, TILE = 4096 };

typedef struct {
    size_t width, k, mask_stride;
} Parameters;

typedef struct {
    float score;
    uint32_t index, segment;
} Candidate;

typedef struct {
    Candidate values[MAX_K];
    size_t count;
} Heap;

typedef struct {
    Heap heap;
    uint32_t last_nan;
} TileResult;

typedef struct {
    const float *scores;
    const uint8_t *mask;
    const Parameters *parameters;
    const TrillimKernelContext *context;
    TileResult *tiles;
} Task;

static size_t tile_count(size_t width) {
    return width / TILE + (width % TILE != 0);
}

static TrillimStatus requirements(const void *parameters, size_t *inputs,
                                  size_t *outputs, size_t *scratch) {
    const Parameters *p = parameters;
    if (!p->k || p->k > MAX_K || p->k > p->width || p->width > INT32_MAX ||
        p->width > SIZE_MAX / sizeof(float) || p->mask_stride > 1) {
        return TRILLIM_ERROR;
    }
    size_t tiles = tile_count(p->width);
    if (tiles > SIZE_MAX / sizeof(TileResult)) {
        return TRILLIM_ERROR;
    }
    inputs[0] = p->width * sizeof(float);
    inputs[1] = p->mask_stride ? p->width : 1;
    outputs[0] = sizeof(int64_t);
    outputs[1] = p->width;
    outputs[2] = p->k * sizeof(float);
    outputs[3] = p->k * sizeof(int64_t);
    *scratch = tiles > 1 ? tiles * sizeof(TileResult) : 0;
    return TRILLIM_OK;
}

static bool better(Candidate a, Candidate b) {
    /* An active NaN resets the ordered Argmax scan. Its suffix ranks first. */
    if (a.segment != b.segment) {
        return a.segment > b.segment;
    }
    bool a_nan = a.index + 1 == a.segment;
    bool b_nan = b.index + 1 == b.segment;
    if (a_nan || b_nan) {
        return b_nan && !a_nan;
    }
    return a.score > b.score || (a.score == b.score && a.index < b.index);
}

static void sift_down(Heap *heap, Candidate value) {
    size_t index = 0;
    size_t child = 1;
    while (child < heap->count) {
        if (child + 1 < heap->count && better(heap->values[child], heap->values[child + 1])) {
            ++child;
        }
        if (!better(value, heap->values[child])) {
            break;
        }
        heap->values[index] = heap->values[child];
        index = child;
        child = index * 2 + 1;
    }
    heap->values[index] = value;
}

static void insert(Heap *heap, size_t limit, Candidate value) {
    if (heap->count == limit) {
        if (better(value, heap->values[0])) {
            sift_down(heap, value);
        }
        return;
    }
    size_t index = heap->count++;
    while (index != 0) {
        size_t parent = (index - 1) / 2;
        if (!better(heap->values[parent], value)) {
            break;
        }
        heap->values[index] = heap->values[parent];
        index = parent;
    }
    heap->values[index] = value;
}

static TrillimStatus collect(void *opaque, size_t tile, size_t worker) {
    (void)worker;
    const Task *task = opaque;
    const Parameters *p = task->parameters;
    TileResult *result = task->tiles + tile;
    Heap *heap = &result->heap;
    heap->count = 0;
    uint32_t segment = 0;
    size_t first = tile * TILE;
    size_t end = p->width - first < TILE ? p->width : first + TILE;
    for (size_t i = first; i < end; ++i) {
        if ((i & 255) == 0 && task->context->is_cancelled(task->context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        if (p->mask_stride && task->mask[i] == 0) {
            continue;
        }
        float score = task->scores[i];
        if (isnan(score)) {
            segment = (uint32_t)i + 1;
        } else if (heap->count == p->k && segment == heap->values[0].segment &&
                   score <= heap->values[0].score) {
            continue;
        }
        insert(heap, p->k, (Candidate){score, (uint32_t)i, segment});
    }
    result->last_nan = segment;
    return TRILLIM_OK;
}

TRILLIM_KERNEL TrillimStatus trillim_scalar_topk(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {
    const Parameters *p = parameters;
    const uint8_t *mask = inputs[1];
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    if (!p->mask_stride && !mask[0]) {
        return TRILLIM_ERROR;
    }
    size_t tiles = tile_count(p->width);
    TileResult local;
    Task task = {inputs[0], mask, p, context, tiles == 1 ? &local : scratch};
    TrillimStatus status = tiles == 1 ? collect(&task, 0, 0) :
        context->thread_pool->run(context->thread_pool->context, tiles, collect, &task);
    if (status != TRILLIM_OK) {
        return status;
    }

    Heap selected;
    selected.count = 0;
    uint32_t previous = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        const TileResult *current = task.tiles + tile;
        for (size_t i = 0; i < current->heap.count; ++i) {
            Candidate value = current->heap.values[i];
            if (!value.segment) {
                value.segment = previous;
            }
            insert(&selected, p->k, value);
        }
        if (current->last_nan) {
            previous = current->last_nan;
        }
    }
    if (selected.count < p->k) {
        return TRILLIM_ERROR;
    }

    uint8_t *remaining = outputs[1];
    for (size_t first = 0; first < p->width; first += TILE) {
        if (context->is_cancelled(context->cancellation_context)) {
            return TRILLIM_CANCELLED;
        }
        size_t count = p->width - first < TILE ? p->width - first : TILE;
        if (p->mask_stride) {
            for (size_t i = 0; i < count; ++i) {
                remaining[first + i] = mask[first + i] != 0;
            }
        } else {
            memset(remaining + first, 1, count);
        }
    }
    float *scores = outputs[2];
    int64_t *indices = outputs[3];
    while (selected.count) {
        size_t index = selected.values[0].index;
        size_t position = --selected.count;
        memcpy(scores + position, task.scores + index, sizeof(float));
        indices[position] = (int64_t)index;
        remaining[index] = 0;
        if (position) {
            sift_down(&selected, selected.values[position]);
        }
    }
    if (context->is_cancelled(context->cancellation_context)) {
        return TRILLIM_CANCELLED;
    }
    *(int64_t *)outputs[0] = (int64_t)p->k;
    return TRILLIM_OK;
}

TRILLIM_KERNEL const char *trillim_scalar_topk_tdl(void) {
    return "from trillim.tdl import *\n"
           "from trillim.compiler.kernels.bindings import _contiguous_strides\n"
           "physical_inputs = ('x', 'initial.1')\n"
           "\n"
           "def pattern(initial: tuple, x: Tensor) -> tuple[Tensor, ...]:\n"
           "    scores, _ = TopK.__wrapped__(x, k=Minimum(Dim(x, -1), Dim(initial[2], 0)), rank=1)\n"
           "    loop = scores._producer\n"
           "    if loop is None or loop.op != 'while':\n"
           "        raise ValueError('TopK requires a bounded While')\n"
           "    seed = loop.args.initial\n"
           "    loop.args = loop.args.model_copy(update={'initial': (seed[0], initial[1], *seed[2:])})\n"
           "    return loop.outputs\n"
           "\n"
           "def bind(context):\n"
           "    x, mask = (context.inputs[name] for name in physical_inputs)\n"
           "    count, remaining, scores, indices = context.outputs\n"
           "    if x.dtype != 'fp32' or len(x.shape) != 1 or x.strides != _contiguous_strides(x.shape):\n"
           "        raise ValueError('TopK requires a contiguous FP32 vector')\n"
           "    if mask.dtype != 'bool' or mask.shape != x.shape or not mask.strides or mask.strides[0].value not in (0, 1):\n"
           "        raise ValueError('TopK requires a contiguous or broadcast boolean mask')\n"
           "    if count.shape or count.dtype != 'int64' or remaining.shape != x.shape or remaining.dtype != 'bool':\n"
           "        raise ValueError('TopK requires matching loop carries')\n"
           "    if len(scores.shape) != 1 or scores.shape != indices.shape or scores.dtype != 'fp32' or indices.dtype != 'int64':\n"
           "        raise ValueError('TopK requires FP32 scores and int64 indices')\n"
           "    return {'width': x.shape[0], 'k': scores.shape[0], 'mask_stride': mask.strides[0]}\n";
}

TRILLIM_KERNEL const TrillimParameterSchema *trillim_scalar_topk_parameters(void) {
    static const TrillimParameter fields[] = {
        {"width", "size_t", offsetof(Parameters, width), NULL},
        {"k", "size_t", offsetof(Parameters, k), NULL},
        {"mask_stride", "size_t", offsetof(Parameters, mask_stride), NULL},
    };
    static const size_t inputs[] = {4, 1};
    static const size_t outputs[] = {8, 1, 4, 8};
    static const TrillimParameterSchema schema = {
        sizeof(Parameters), TRILLIM_ALIGNOF(Parameters), 3, fields, 2, 4,
        requirements, NULL, inputs, outputs, TRILLIM_ALIGNOF(TileResult)
    };
    return &schema;
}

TRILLIM_KERNEL const TrillimCapabilities *trillim_scalar_topk_capabilities(void) {
    static const TrillimCapabilities capabilities = {TRILLIM_KERNEL_ABI_VERSION, 0};
    return &capabilities;
}
