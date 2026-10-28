from __future__ import annotations

from typing import Literal

from .core_dataclasses import Scalar
from .core_pydantic_models import (
    BinaryArgs,
    CastArgs,
    CompareArgs,
    MatmulArgs,
    SelectArgs,
    UnaryArgs,
)
from .operator import Operator
from .tensor import Tensor
from .types import DType


def Add(a: Tensor, b: Tensor | Scalar) -> Tensor:
    """Numeric addition with broadcasting and compiler-resolved dtype promotion.

    Boolean values represent zero and one; adding two booleans yields int64.
    The compiler inserts conversions required by numeric promotion.
    """
    return Operator("add", BinaryArgs(a=a, b=b)).outputs[0]


def Sub(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return Operator("sub", BinaryArgs(a=a, b=b)).outputs[0]


def Mul(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return Operator("mul", BinaryArgs(a=a, b=b)).outputs[0]


def Div(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return Operator("div", BinaryArgs(a=a, b=b)).outputs[0]


def FloorDiv(a: Tensor, b: Tensor | Scalar) -> Tensor:
    """Divide integer values, rounding toward negative infinity."""
    return Operator("floor_div", BinaryArgs(a=a, b=b)).outputs[0]


def Mod(a: Tensor, b: Tensor | Scalar) -> Tensor:
    """Return a - FloorDiv(a, b) * b for integer values."""
    return Operator("mod", BinaryArgs(a=a, b=b)).outputs[0]


def Exp(x: Tensor) -> Tensor:
    return Operator("exp", UnaryArgs(x=x)).outputs[0]


def Log(x: Tensor) -> Tensor:
    return Operator("log", UnaryArgs(x=x)).outputs[0]


def Sqrt(x: Tensor) -> Tensor:
    return Operator("sqrt", UnaryArgs(x=x)).outputs[0]


def Sin(x: Tensor) -> Tensor:
    return Operator("sin", UnaryArgs(x=x)).outputs[0]


def Cos(x: Tensor) -> Tensor:
    return Operator("cos", UnaryArgs(x=x)).outputs[0]


def Erf(x: Tensor) -> Tensor:
    return Operator("erf", UnaryArgs(x=x)).outputs[0]


def Compare(
    a: Tensor,
    b: Tensor | Scalar,
    *,
    predicate: Literal["eq", "ne", "lt", "le", "gt", "ge"],
) -> Tensor:
    return Operator(
        "compare", CompareArgs(a=a, b=b, predicate=predicate), outputs=((None, "bool"),)
    ).outputs[0]


def Select(
    condition: Tensor, when_true: Tensor | Scalar, when_false: Tensor | Scalar
) -> Tensor:
    """Select elementwise with broadcasting; both value expressions are evaluated."""
    return Operator(
        "select",
        SelectArgs(condition=condition, when_true=when_true, when_false=when_false),
    ).outputs[0]


def Cast(x: Tensor, dtype: DType) -> Tensor:
    """Convert logical values, including quantization into the target storage dtype."""
    return Operator(
        "cast", CastArgs(x=x, dtype=dtype), outputs=((None, dtype),)
    ).outputs[0]


def Matmul(a: Tensor, b: Tensor) -> Tensor:
    """Compute [..., M, K] @ [..., K, N]; physical packing is a backend choice."""
    return Operator("matmul", MatmulArgs(a=a, b=b)).outputs[0]
