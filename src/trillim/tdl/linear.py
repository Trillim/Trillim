from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import linear_args, op_args
from .tensor import Tensor

LinearOp = Literal["linear"]
LinearArgs = linear_args

_LINEAR_ARGS: dict[str, type[op_args]] = {
    "linear": linear_args,
}


def Linear(*, op: LinearOp, args: LinearArgs) -> Tensor:
    _validate_variant(
        family="Linear",
        op=op,
        args=args,
        registry=_LINEAR_ARGS,
    )
    return _create_operator(
        family="linear",
        op=op,
        args=args,
    )
