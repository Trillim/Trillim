from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import mamba2_args, mamba_args, op_args
from .tensor import Tensor

SSMOp = Literal["mamba", "mamba2"]
SSMArgs = mamba_args | mamba2_args

_SSM_ARGS: dict[str, type[op_args]] = {
    "mamba": mamba_args,
    "mamba2": mamba2_args,
}


def SSM(*, op: SSMOp, args: SSMArgs) -> Tensor:
    _validate_variant(
        family="SSM",
        op=op,
        args=args,
        registry=_SSM_ARGS,
    )
    return _create_operator(
        family="ssm",
        op=op,
        args=args,
    )
