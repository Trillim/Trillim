from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .activation import Sigmoid
from .core import (
    Add,
    Compare,
    Construct,
    Div,
    Gather,
    Matmul,
    Mul,
    Permute,
    Reshape,
    Select,
    Tensor,
)
from .linear import Linear
from .math import (
    Clip,
    Dim,
    FullLike,
    Minimum,
    Slice,
    Softmax,
    Sum,
    TopK,
    _dims,
    _tensor,
    _Value,
)


@Construct
def MoE(
    x: Tensor,
    *,
    router: _Value,
    router_bias: _Value | None = None,
    experts: tuple[Callable[[Tensor], Tensor], ...],
    top_k: int,
    scoring: Literal["softmax", "sigmoid"] = "softmax",
    normalize: bool = True,
    shared: tuple[Callable[[Tensor], Tensor], ...] = (),
    rank: int = 2,
) -> Tensor:
    """Route to pure tensor functions; fallback evaluates all experts then weights them."""
    if not 0 < top_k <= len(experts):
        raise ValueError("invalid expert routing configuration")
    logits = Linear(x, router, bias=router_bias)
    probabilities = Softmax(logits) if scoring == "softmax" else Sigmoid(logits)
    weights, selected = TopK(probabilities, k=top_k, rank=rank)
    if normalize:
        weights = Div(weights, Sum(weights, axes=(-1,), keepdims=True))
    output = FullLike(x)
    for index, expert in enumerate(experts):
        coefficient = Sum(
            Select(Compare(selected, index, predicate="eq"), weights, 0),
            axes=(-1,),
            keepdims=True,
        )
        output = Add(output, Mul(expert(x), coefficient))
    for expert in shared:
        output = Add(output, expert(x))
    return output


@Construct
def GptOssMoE(
    x: Tensor,
    *,
    router: _Value,
    router_bias: _Value | None = None,
    gate_up: _Value,
    down: _Value,
    gate_up_bias: _Value,
    down_bias: _Value,
    top_k: int,
    limit: float = 7.0,
) -> Tensor:
    """Logical expert banks [E,2F,D], [E,D,F]; packed formats stay storage metadata."""
    logits = Linear(x, router, bias=router_bias)
    scores, selected = TopK(logits, k=top_k, rank=2)
    weights = Softmax(scores)
    indices = Reshape(selected, (*_dims(selected, 2), 1))
    up_bank = Gather(_tensor(gate_up), indices=indices)
    down_bank = Gather(_tensor(down), indices=indices)
    up_bias = Gather(_tensor(gate_up_bias), indices=indices)
    out_bias = Gather(_tensor(down_bias), indices=indices)
    hidden = Add(
        Reshape(
            Matmul(
                Reshape(x, (Dim(x, 0), 1, 1, Dim(x, 1))),
                Permute(up_bank, (0, 1, 3, 2)),
            ),
            (Dim(x, 0), top_k, Dim(up_bias, -1)),
        ),
        up_bias,
    )
    gate = Slice(hidden, axis=2, start=0, step=2, rank=3)
    up = Slice(hidden, axis=2, start=1, step=2, rank=3)
    gate = Minimum(gate, limit)
    hidden = Mul(Mul(gate, Sigmoid(Mul(gate, 1.702))), Add(Clip(up, -limit, limit), 1))
    output = Matmul(
        Reshape(hidden, (*_dims(hidden, 2), 1, Dim(hidden, -1))),
        Permute(down_bank, (0, 1, 3, 2)),
    )
    output = Add(Reshape(output, (*_dims(x, 1), top_k, Dim(x, -1))), out_bias)
    return Sum(Mul(output, Reshape(weights, (*_dims(weights, 2), 1))), axes=(1,))
