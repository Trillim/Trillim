from __future__ import annotations

from typing import Literal

from .core_dataclasses import _Scalar
from .core_pydantic_models import (
    _BinaryArgs,
    _CastArgs,
    _CompareArgs,
    _MatmulArgs,
    _SelectArgs,
    _UnaryArgs,
)
from .operator import _Operator
from .tensor import Tensor
from .types import DType


def Add(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    """Numeric addition with broadcasting and compiler-resolved dtype promotion.

    Boolean values represent zero and one; adding two booleans yields int64.
    The compiler inserts conversions required by numeric promotion.
    """
    return _Operator("add", _BinaryArgs(a=a, b=b)).outputs[0]


def Sub(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    return _Operator("sub", _BinaryArgs(a=a, b=b)).outputs[0]


def Mul(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    return _Operator("mul", _BinaryArgs(a=a, b=b)).outputs[0]


def Div(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    return _Operator("div", _BinaryArgs(a=a, b=b)).outputs[0]


def FloorDiv(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    """Divide integer values, rounding toward negative infinity."""
    return _Operator("floor_div", _BinaryArgs(a=a, b=b)).outputs[0]


def Mod(a: Tensor, b: Tensor | _Scalar) -> Tensor:
    """Return a - FloorDiv(a, b) * b for integer values."""
    return _Operator("mod", _BinaryArgs(a=a, b=b)).outputs[0]


def Exp(x: Tensor) -> Tensor:
    return _Operator("exp", _UnaryArgs(x=x)).outputs[0]


def Log(x: Tensor) -> Tensor:
    return _Operator("log", _UnaryArgs(x=x)).outputs[0]


def Sqrt(x: Tensor) -> Tensor:
    return _Operator("sqrt", _UnaryArgs(x=x)).outputs[0]


def Sin(x: Tensor) -> Tensor:
    return _Operator("sin", _UnaryArgs(x=x)).outputs[0]


def Cos(x: Tensor) -> Tensor:
    return _Operator("cos", _UnaryArgs(x=x)).outputs[0]


def Erf(x: Tensor) -> Tensor:
    return _Operator("erf", _UnaryArgs(x=x)).outputs[0]


def Compare(
    a: Tensor,
    b: Tensor | _Scalar,
    *,
    predicate: Literal["eq", "ne", "lt", "le", "gt", "ge"],
) -> Tensor:
    return _Operator(
        "compare",
        _CompareArgs(a=a, b=b, predicate=predicate),
        outputs=((None, "bool"),),
    ).outputs[0]


def Select(
    condition: Tensor, when_true: Tensor | _Scalar, when_false: Tensor | _Scalar
) -> Tensor:
    """Select elementwise with broadcasting; both value expressions are evaluated."""
    return _Operator(
        "select",
        _SelectArgs(condition=condition, when_true=when_true, when_false=when_false),
    ).outputs[0]


def Cast(x: Tensor, dtype: DType) -> Tensor:
    """Convert logical values, including quantization into the target storage dtype."""
    return _Operator(
        "cast", _CastArgs(x=x, dtype=dtype), outputs=((None, dtype),)
    ).outputs[0]


def Matmul(a: Tensor, b: Tensor) -> Tensor:
    """Compute [..., M, K] @ [..., K, N]; physical packing is a backend choice."""
    return _Operator("matmul", _MatmulArgs(a=a, b=b)).outputs[0]
