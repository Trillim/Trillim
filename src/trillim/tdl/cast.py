from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import cast_args, op_args
from .tensor import Tensor

CastOp = Literal["cast"]
CastArgs = cast_args

_CAST_ARGS: dict[str, type[op_args]] = {
    "cast": cast_args,
}


def Cast(*, op: CastOp, args: CastArgs) -> Tensor:
    _validate_variant(
        family="Cast",
        op=op,
        args=args,
        registry=_CAST_ARGS,
    )
    return _create_operator(
        family="cast",
        op=op,
        args=args,
    )
