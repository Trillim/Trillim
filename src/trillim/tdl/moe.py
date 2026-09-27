from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import moe_args, op_args
from .tensor import Tensor

MoEOp = Literal["moe"]
MoEArgs = moe_args

_MOE_ARGS: dict[str, type[op_args]] = {
    "moe": moe_args,
}


def MoE(*, op: MoEOp, args: MoEArgs) -> Tensor:
    _validate_variant(
        family="MoE",
        op=op,
        args=args,
        registry=_MOE_ARGS,
    )
    return _create_operator(
        family="moe",
        op=op,
        args=args,
    )
