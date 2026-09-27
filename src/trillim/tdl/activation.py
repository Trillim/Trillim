from __future__ import annotations

from typing import Literal

from .tensor import Tensor

ActivationOp = Literal["relu", "sigmoid", "tanh", "gelu", "silu", "relu_sqr"]

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import (
    gelu_args,
    op_args,
    relu_args,
    relu_sqr_args,
    sigmoid_args,
    silu_args,
    tanh_args,
)

ActivationArgs = (
    relu_args | sigmoid_args | tanh_args | gelu_args | silu_args | relu_sqr_args
)

_ACTIVATION_ARGS: dict[str, type[op_args]] = {
    "relu": relu_args,
    "sigmoid": sigmoid_args,
    "tanh": tanh_args,
    "gelu": gelu_args,
    "silu": silu_args,
    "relu_sqr": relu_sqr_args,
}


def Activation(*, op: ActivationOp, args: ActivationArgs) -> Tensor:
    _validate_variant(
        family="Activation",
        op=op,
        args=args,
        registry=_ACTIVATION_ARGS,
    )
    return _create_operator(
        family="activation",
        op=op,
        args=args,
    )
