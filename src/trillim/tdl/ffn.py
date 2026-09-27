from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import ffn_args, op_args
from .tensor import Tensor

FFNOp = Literal["ffn"]
FFNArgs = ffn_args

_FFN_ARGS: dict[str, type[op_args]] = {
    "ffn": ffn_args,
}


def FFN(*, op: FFNOp, args: FFNArgs) -> Tensor:
    _validate_variant(
        family="FFN",
        op=op,
        args=args,
        registry=_FFN_ARGS,
    )
    return _create_operator(
        family="ffn",
        op=op,
        args=args,
    )
