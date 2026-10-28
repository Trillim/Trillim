from __future__ import annotations

from .activation import Tanh
from .core import (
    Add,
    Broadcast,
    Cache,
    Compare,
    Constant,
    Construct,
    Div,
    Exp,
    FloorDiv,
    Gather,
    Iota,
    Matmul,
    Mul,
    Permute,
    Reshape,
    Select,
    Shape,
    Sqrt,
    Sub,
    Tensor,
    While,
)
from .math import Dim, FullLike, Softmax, Stack, Sum, Take, Value, _tensor
from .norm import L2Norm


@Construct
def Attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    mask: Tensor | None = None,
    query_positions: Tensor | None = None,
    key_positions: Tensor | None = None,
    window: int | None = None,
    softcap: float | None = None,
    sinks: Value | None = None,
    softmax: bool = True,
) -> Tensor:
    """Grouped-query attention on [heads, tokens, channels] for one sequence.

    Masks add to scores. Explicit positions are absolute one-dimensional token
    indices; defaults align queries to the right edge of keys for cached decode.
    """
    groups = FloorDiv(Dim(q, 0), Dim(k, 0))
    head_indices = FloorDiv(Iota(Dim(q, 0)), groups)
    k = Take(k, head_indices, axis=0, rank=3)
    v = Take(v, head_indices, axis=0, rank=3)
    scores = Matmul(q, Permute(k, (0, 2, 1)))
    factor = Div(Constant(1.0), Sqrt(Dim(q, -1))) if scale is None else Constant(scale)
    scores = Mul(scores, factor)
    if softcap is not None:
        scores = Mul(Tanh(Div(scores, softcap)), softcap)
    if causal or window is not None:
        qp = (
            Add(Iota(Dim(q, 1)), Sub(Dim(k, 1), Dim(q, 1)))
            if query_positions is None
            else query_positions
        )
        kp = Iota(Dim(k, 1)) if key_positions is None else key_positions
        distance = Sub(Reshape(qp, (-1, 1)), kp)
        visible = (
            Compare(distance, 0, predicate="ge")
            if causal
            else Broadcast(Constant(True), Shape(distance))
        )
        if window is not None:
            visible = Select(Compare(distance, window, predicate="lt"), visible, False)
        scores = Select(visible, scores, float("-inf"))
    if mask is not None:
        scores = Add(scores, mask)
    sink_values = None if sinks is None else Reshape(_tensor(sinks), (-1, 1, 1))
    probabilities = Softmax(scores, sinks=sink_values) if softmax else scores
    return Matmul(probabilities, v)


@Construct
def GatedDelta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    beta: Tensor,
    log_decay: Tensor,
    state: Cache,
) -> Tensor:
    """Delta rule; Q/K [T,H,K], V [T,H,V], persistent state [H,K,V]."""
    q = Div(L2Norm(q), Sqrt(Dim(q, -1)))
    k = L2Norm(k)
    initial_state = state.read()
    output = FullLike(v)
    positions = Reshape(Iota(Dim(v, 0)), (-1, 1, 1))

    def step(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        index, memory, result = values
        qt = Take(q, index, rank=3, index_rank=0)
        kt = Take(k, index, rank=3, index_rank=0)
        vt = Take(v, index, rank=3, index_rank=0)
        bt = Take(beta, index, rank=2, index_rank=0)
        gt = Take(log_decay, index, rank=2, index_rank=0)
        memory = Mul(memory, Reshape(Exp(gt), (-1, 1, 1)))
        key = Reshape(kt, (Dim(kt, 0), Dim(kt, 1), 1))
        residual = Sub(vt, Sum(Mul(memory, key), axes=(-2,)))
        update = Mul(residual, Reshape(bt, (-1, 1)))
        memory = Add(
            memory, Mul(key, Reshape(update, (Dim(update, 0), 1, Dim(update, 1))))
        )
        token = Sum(Mul(memory, Reshape(qt, (Dim(qt, 0), Dim(qt, 1), 1))), axes=(-2,))
        result = Select(Compare(positions, index, predicate="eq"), token, result)
        return Add(index, 1), memory, result

    result = While(
        (Constant(0), initial_state, output),
        condition=lambda s: Compare(s[0], Dim(v, 0), predicate="lt"),
        body=step,
    )
    state.assign(result[1])
    return result[2]


@Construct
def RelativeAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    relative_keys: Tensor,
    *,
    content_bias: Tensor,
    position_bias: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Transformer-XL scores; relative keys [H,2T-1,D], descending positions."""

    content = Matmul(
        Add(q, Reshape(content_bias, (Dim(q, 0), 1, Dim(q, 2)))), Permute(k, (0, 2, 1))
    )
    relative = Matmul(
        Add(q, Reshape(position_bias, (Dim(q, 0), 1, Dim(q, 2)))),
        Permute(relative_keys, (0, 2, 1)),
    )
    length = Dim(q, 1)
    row = Broadcast(Reshape(Iota(length), (-1, 1)), (length, length))
    column = Add(Sub(Sub(length, 1), row), Iota(length))
    coordinates = Stack((row, column), axis=2, rank=2)
    shifted = Permute(
        Gather(Permute(relative, (1, 2, 0)), indices=coordinates), (2, 0, 1)
    )
    scores = Div(Add(content, shifted), Sqrt(Dim(q, -1)))
    if mask is not None:
        scores = Add(scores, mask)
    return Matmul(Softmax(scores), v)


@Construct
def SplitHeads(x: Tensor, *, heads: int, head_dim: int | None = None) -> Tensor:
    """Convert [T,H*D] to [H,T,D]."""
    width = FloorDiv(Dim(x, -1), heads) if head_dim is None else head_dim
    return Permute(Reshape(x, (Dim(x, 0), heads, width)), (1, 0, 2))


@Construct
def MergeHeads(x: Tensor) -> Tensor:
    """Convert [H,T,D] to [T,H*D]."""
    return Reshape(Permute(x, (1, 0, 2)), (Dim(x, 1), -1))
