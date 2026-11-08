from __future__ import annotations

from .core import (
    Add,
    Cast,
    Compare,
    Constant,
    Construct,
    Div,
    Matmul,
    Mod,
    Mul,
    Permute,
    Reshape,
    Scatter,
    Select,
    Shape,
    Sub,
    Tensor,
)
from .math import Abs, Clip, Dim, Max, Maximum, _tensor, _Value


@Construct
def Linear(
    x: Tensor,
    weight: _Value,
    *,
    bias: _Value | None = None,
) -> Tensor:
    """Apply [out, in] logical weights and optional bias."""
    matrix = _tensor(weight)
    shape = Shape(x)
    shape = Scatter(
        shape,
        indices=Reshape(Sub(Dim(shape, 0), 1), (1, 1)),
        values=Reshape(Dim(matrix, 0), (1,)),
    )
    output = Matmul(Reshape(x, (-1, Dim(x, -1))), Permute(matrix, (1, 0)))
    output = Reshape(output, shape)
    if bias is not None:
        output = Add(output, _tensor(bias))
    return output


@Construct
def BitLinear(x: Tensor, weight: _Value) -> Tensor:
    """Project nearest-even INT8 activation codes and rescale the output to FP32."""
    scale = Div(Constant(127.0), Maximum(Max(Abs(x), axes=(-1,), keepdims=True), 1e-5))
    scaled = Mul(x, scale)
    magnitude = Abs(scaled)
    integer = Cast(magnitude, "int8")
    fraction = Sub(magnitude, Cast(integer, "fp32"))
    increment = Select(
        Compare(fraction, 0.5, predicate="eq"),
        Compare(Mod(integer, 2), 1, predicate="eq"),
        Compare(fraction, 0.5, predicate="gt"),
    )
    rounded = Add(integer, Cast(increment, "int8"))
    signed = Select(Compare(scaled, 0.0, predicate="lt"), Mul(rounded, -1), rounded)
    quantized = Cast(Clip(signed, -128, 127), "int8")
    return Div(Linear(quantized, weight), scale)
