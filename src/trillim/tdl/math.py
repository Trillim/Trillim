from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import add_args, concat_args, mul_args, op_args, softmax_args
from .tensor import Tensor

MathOp = Literal["add", "mul", "concat", "softmax"]
MathArgs = add_args | mul_args | concat_args | softmax_args

_MATH_ARGS: dict[str, type[op_args]] = {
    "add": add_args,
    "mul": mul_args,
    "concat": concat_args,
    "softmax": softmax_args,
}


def Math(*, op: MathOp, args: MathArgs) -> Tensor:
    _validate_variant(
        family="Math",
        op=op,
        args=args,
        registry=_MATH_ARGS,
    )
    return _create_operator(
        family="math",
        op=op,
        args=args,
    )
