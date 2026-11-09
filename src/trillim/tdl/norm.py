from __future__ import annotations

from .core import Add, Cast, Construct, Div, DType, Mul, Reshape, Sqrt, Sub, Tensor
from .math import Dim, Mean, Sum, _dims, _tensor, _Value


def _affine(x: Tensor, weight: _Value | None, bias: _Value | None) -> Tensor:
    if weight is not None:
        x = Mul(x, _tensor(weight))
    return x if bias is None else Add(x, _tensor(bias))


@Construct
def LayerNorm(
    x: Tensor,
    *,
    weight: _Value | None = None,
    bias: _Value | None = None,
    axes: tuple[int, ...] = (-1,),
    eps: float = 1e-5,
) -> Tensor:
    centered = Sub(x, Mean(x, axes=axes, keepdims=True))
    variance = Mean(Mul(centered, centered), axes=axes, keepdims=True)
    return _affine(Div(centered, Sqrt(Add(variance, eps))), weight, bias)


@Construct
def RMSNorm(
    x: Tensor,
    *,
    weight: _Value | None = None,
    eps: float = 1e-6,
    weight_offset: float = 0.0,
    dtype: DType | None = None,
) -> Tensor:
    result = Div(x, Sqrt(Add(Mean(Mul(x, x), axes=(-1,), keepdims=True), eps)))
    result = (
        result if weight is None else Mul(result, Add(_tensor(weight), weight_offset))
    )
    return result if dtype is None else Cast(result, dtype)


@Construct
def L2Norm(x: Tensor, *, axis: int = -1, eps: float = 1e-6) -> Tensor:
    return Div(x, Sqrt(Add(Sum(Mul(x, x), axes=(axis,), keepdims=True), eps)))


@Construct
def BatchNorm(
    x: Tensor,
    *,
    running_mean: _Value,
    running_var: _Value,
    weight: _Value | None = None,
    bias: _Value | None = None,
    eps: float = 1e-5,
    rank: int,
) -> Tensor:
    """Channel-first inference normalization of one input using saved statistics."""
    shape = (-1, *((1,) * (rank - 1)))
    mean = Reshape(_tensor(running_mean), shape)
    var = Reshape(_tensor(running_var), shape)
    result = Div(Sub(x, mean), Sqrt(Add(var, eps)))
    if weight is not None:
        result = Mul(result, Reshape(_tensor(weight), shape))
    if bias is not None:
        result = Add(result, Reshape(_tensor(bias), shape))
    return result


@Construct
def GroupNorm(
    x: Tensor,
    *,
    groups: int,
    weight: _Value | None = None,
    bias: _Value | None = None,
    eps: float = 1e-5,
    rank: int,
) -> Tensor:
    if groups <= 0:
        raise ValueError("groups must be positive")
    dims = _dims(x, rank)
    grouped = Reshape(x, (groups, -1))
    result = Reshape(LayerNorm(grouped, eps=eps), dims)
    shape = (-1, *((1,) * (rank - 1)))
    if weight is not None:
        result = Mul(result, Reshape(_tensor(weight), shape))
    return result if bias is None else Add(result, Reshape(_tensor(bias), shape))


@Construct
def VarianceNorm(
    x: Tensor, *, weight: _Value, correction: int = 1, eps: float = 1e-5
) -> Tensor:
    """Scale by variance without subtracting the mean, as PocketTTS's flow head does."""

    centered = Sub(x, Mean(x, axes=(-1,), keepdims=True))
    variance = Div(
        Sum(Mul(centered, centered), axes=(-1,), keepdims=True),
        Sub(Dim(x, -1), correction),
    )
    return Mul(Div(x, Sqrt(Add(variance, eps))), _tensor(weight))


@Construct
def AdaptiveLayerNorm(
    x: Tensor, *, shift: Tensor, scale: Tensor, eps: float = 1e-6
) -> Tensor:
    return Add(Mul(LayerNorm(x, eps=eps), Add(scale, 1)), shift)
