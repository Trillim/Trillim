from __future__ import annotations

from .core import (
    Add,
    Cast,
    Compare,
    Constant,
    Construct,
    Div,
    DType,
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
def QuantizeInt8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return nearest-even INT8 codes and the FP32 per-row multiplier."""
    x = Cast(x, "fp32")
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
    return quantized, scale


@Construct
def QuantizedLinear(
    x: Tensor, weight: _Value, scale: Tensor, *, dtype: DType = "fp32"
) -> Tensor:
    """Project shared activation codes, then undo their quantization multiplier."""
    return Cast(Div(Linear(x, weight), scale), dtype)


@Construct
def BitLinear(x: Tensor, weight: _Value) -> Tensor:
    """Quantize activations and project them to FP32 in one composition."""
    codes, scale = QuantizeInt8(x)
    return QuantizedLinear(codes, weight, scale)
