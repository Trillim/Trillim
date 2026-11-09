from __future__ import annotations

from .activation import Sigmoid, Tanh
from .conv import Conv
from .core import (
    Add,
    Cache,
    Compare,
    Constant,
    Construct,
    Exp,
    FloorDiv,
    Iota,
    Mul,
    Permute,
    Reshape,
    Scatter,
    Tensor,
    While,
)
from .linear import Linear
from .math import Concat, Dim, FullLike, Slice, Sum, Take, _Value


@Construct
def CausalConv(
    x: Tensor,
    weight: _Value,
    *,
    kernel: int,
    state: Cache,
    bias: _Value | None = None,
    groups: int = 1,
) -> Tensor:
    """Streaming depthwise or grouped convolution; X [T,C], history [C,K-1]."""
    channels_first = Permute(x, (1, 0))
    previous = state.read()
    sequence = Concat((previous, channels_first), axis=1, rank=2)
    result = Conv(sequence, weight, kernel=(kernel,), bias=bias, groups=groups)
    next_state = (
        Slice(sequence, axis=1, start=-(kernel - 1), rank=2)
        if kernel > 1
        else Slice(sequence, axis=1, stop=0, rank=2)
    )
    state.assign(next_state)
    return Permute(result, (1, 0))


@Construct
def Mamba(
    x: Tensor,
    *,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    skip: Tensor,
    state: Cache,
) -> Tensor:
    """Selective scan. X/delta [T,D], A [D,N], B/C [T,N], cache [D,N]."""

    initial_state = state.read()
    output = FullLike(x)

    def step(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        i, memory, result = values
        xt = Take(x, i, rank=2, index_rank=0)
        dt = Reshape(Take(delta, i, rank=2, index_rank=0), (-1, 1))
        bt = Take(b, i, rank=2, index_rank=0)
        ct = Take(c, i, rank=2, index_rank=0)
        update = Mul(Mul(dt, Reshape(xt, (-1, 1))), bt)
        memory = Add(Mul(memory, Exp(Mul(dt, a))), update)
        token = Add(Sum(Mul(memory, ct), axes=(-1,)), Mul(xt, skip))
        result = Scatter(result, indices=Reshape(i, (1,)), values=token)
        return Add(i, 1), memory, result

    result = While(
        (Constant(0), initial_state, output),
        condition=lambda s: Compare(s[0], Dim(x, 0), predicate="lt"),
        body=step,
    )
    state.assign(result[1])
    return result[2]


@Construct
def Mamba2(
    x: Tensor,
    *,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    skip: Tensor,
    state: Cache,
) -> Tensor:
    """SSD recurrence. X [T,H,P], B/C [T,G,N], delta [T,H], cache [H,P,N]."""

    heads = Dim(x, 1)
    indices = FloorDiv(Iota(heads), FloorDiv(heads, Dim(b, 1)))
    b, c = Take(b, indices, axis=1, rank=3), Take(c, indices, axis=1, rank=3)
    initial_state = state.read()
    output = FullLike(x)

    def step(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        i, memory, result = values
        xt = Take(x, i, rank=3, index_rank=0)
        dt = Take(delta, i, rank=2, index_rank=0)
        bt = Take(b, i, rank=3, index_rank=0)
        ct = Take(c, i, rank=3, index_rank=0)
        decay = Reshape(Exp(Mul(dt, a)), (heads, 1, 1))
        update = Mul(
            Reshape(Mul(xt, Reshape(dt, (-1, 1))), (heads, Dim(x, 2), 1)),
            Reshape(bt, (heads, 1, Dim(bt, -1))),
        )
        memory = Add(Mul(memory, decay), update)
        token = Add(
            Sum(Mul(memory, Reshape(ct, (heads, 1, Dim(ct, -1)))), axes=(-1,)),
            Mul(xt, Reshape(skip, (-1, 1))),
        )
        result = Scatter(result, indices=Reshape(i, (1,)), values=token)
        return Add(i, 1), memory, result

    result = While(
        (Constant(0), initial_state, output),
        condition=lambda s: Compare(s[0], Dim(x, 0), predicate="lt"),
        body=step,
    )
    state.assign(result[1])
    return result[2]


@Construct
def LSTMCell(
    x: Tensor,
    hidden: Tensor,
    cell: Tensor,
    *,
    input_weight: _Value,
    recurrent_weight: _Value,
    input_bias: _Value | None = None,
    recurrent_bias: _Value | None = None,
) -> tuple[Tensor, Tensor]:
    """One inference step with PyTorch/ONNX gate order input, forget, candidate, output."""

    gates = Add(
        Linear(x, input_weight, bias=input_bias),
        Linear(hidden, recurrent_weight, bias=recurrent_bias),
    )
    width = Dim(hidden, -1)
    i = Sigmoid(Slice(gates, axis=0, stop=width, rank=1))
    f = Sigmoid(Slice(gates, axis=0, start=width, stop=Mul(width, 2), rank=1))
    g = Tanh(Slice(gates, axis=0, start=Mul(width, 2), stop=Mul(width, 3), rank=1))
    o = Sigmoid(Slice(gates, axis=0, start=Mul(width, 3), rank=1))
    cell = Add(Mul(f, cell), Mul(i, g))
    return Mul(o, Tanh(cell)), cell
