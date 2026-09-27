from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import gqa_args, mla_args, op_args
from .tensor import Tensor

AttentionOp = Literal["gqa", "mla"]
AttentionArgs = gqa_args | mla_args

_ATTENTION_ARGS: dict[str, type[op_args]] = {
    "gqa": gqa_args,
    "mla": mla_args,
}


def Attention(*, op: AttentionOp, args: AttentionArgs) -> Tensor:
    _validate_variant(
        family="Attention",
        op=op,
        args=args,
        registry=_ATTENTION_ARGS,
    )
    return _create_operator(
        family="attention",
        op=op,
        args=args,
    )
