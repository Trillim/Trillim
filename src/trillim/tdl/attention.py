from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import (
    gated_delta_args,
    gated_full_attn_args,
    gqa_args,
    gqa_norm_args,
    mla_args,
    op_args,
)
from .tensor import Tensor

AttentionOp = Literal["gqa", "gqa_norm", "gated_delta", "gated_full_attn", "mla"]
AttentionArgs = (
    gqa_args | gqa_norm_args | gated_delta_args | gated_full_attn_args | mla_args
)

_ATTENTION_ARGS: dict[str, type[op_args]] = {
    "gqa": gqa_args,
    "gqa_norm": gqa_norm_args,
    "gated_delta": gated_delta_args,
    "gated_full_attn": gated_full_attn_args,
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
