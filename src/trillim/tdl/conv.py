from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import conv_args, op_args
from .tensor import Tensor

ConvOp = Literal["conv"]
ConvArgs = conv_args

_CONV_ARGS: dict[str, type[op_args]] = {
    "conv": conv_args,
}


def Conv(*, op: ConvOp, args: ConvArgs) -> Tensor:
    _validate_variant(
        family="Conv",
        op=op,
        args=args,
        registry=_CONV_ARGS,
    )
    return _create_operator(
        family="conv",
        op=op,
        args=args,
    )
