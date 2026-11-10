from __future__ import annotations

from typing import Literal

from .core import (
    Add,
    Cast,
    Compare,
    Constant,
    Construct,
    Div,
    DType,
    Erf,
    Exp,
    FloorDiv,
    Log,
    Mul,
    Select,
    Tensor,
)
from .math import Abs, Dim, Maximum, Slice


@Construct
def Relu(x: Tensor) -> Tensor:
    return Maximum(x, 0)


@Construct
def Sigmoid(x: Tensor) -> Tensor:
    e = Exp(Mul(Abs(x), -1))
    return Select(
        Compare(x, 0, predicate="ge"), Div(Constant(1.0), Add(e, 1)), Div(e, Add(e, 1))
    )


@Construct
def Tanh(x: Tensor) -> Tensor:
    return Add(Mul(Sigmoid(Mul(x, 2)), 2), -1)


@Construct
def Gelu(x: Tensor, *, approximate: Literal["none", "tanh"] = "none") -> Tensor:
    if approximate == "tanh":
        cdf = Tanh(Mul(Add(x, Mul(Mul(Mul(x, x), x), 0.044715)), 0.7978845608028654))
    else:
        cdf = Erf(Mul(x, 0.7071067811865476))
    return Mul(Mul(x, 0.5), Add(cdf, 1))


@Construct
def Silu(x: Tensor) -> Tensor:
    return Mul(x, Sigmoid(x))


@Construct
def SiluMul(a: Tensor, b: Tensor, *, dtype: DType = "fp32") -> Tensor:
    return Cast(Mul(Silu(a), b), dtype)


@Construct
def ReluSquared(x: Tensor) -> Tensor:
    value = Relu(x)
    return Mul(value, value)


@Construct
def Softplus(x: Tensor) -> Tensor:
    return Add(Maximum(x, 0), Log(Add(Exp(Mul(Abs(x), -1)), 1)))


@Construct
def Elu(x: Tensor, *, alpha: float = 1.0) -> Tensor:
    return Select(Compare(x, 0, predicate="ge"), x, Mul(Add(Exp(x), -1), alpha))


@Construct
def SwiGLU(x: Tensor, *, rank: int = 2) -> Tensor:
    half = FloorDiv(Dim(x, -1), 2)
    return Mul(
        Silu(Slice(x, axis=rank - 1, stop=half, rank=rank)),
        Slice(x, axis=rank - 1, start=half, rank=rank),
    )
