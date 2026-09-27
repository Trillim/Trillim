from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import (
    batchnorm_args,
    groupnorm_args,
    layernorm_args,
    op_args,
    rmsnorm_args,
)
from .tensor import Tensor

NormOp = Literal["layernorm", "rmsnorm", "batchnorm", "groupnorm"]
NormArgs = layernorm_args | rmsnorm_args | batchnorm_args | groupnorm_args

_NORM_ARGS: dict[str, type[op_args]] = {
    "layernorm": layernorm_args,
    "rmsnorm": rmsnorm_args,
    "batchnorm": batchnorm_args,
    "groupnorm": groupnorm_args,
}


def Norm(*, op: NormOp, args: NormArgs) -> Tensor:
    _validate_variant(
        family="Norm",
        op=op,
        args=args,
        registry=_NORM_ARGS,
    )
    return _create_operator(
        family="norm",
        op=op,
        args=args,
    )
