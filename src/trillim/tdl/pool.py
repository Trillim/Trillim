from __future__ import annotations

from typing import Literal

from .operator import _create_operator, _validate_variant
from .tdl_pydantic_models import (
    adaptive_avg_pool_args,
    adaptive_max_pool_args,
    avg_pool_args,
    max_pool_args,
    op_args,
)
from .tensor import Tensor

PoolOp = Literal["max_pool", "avg_pool", "adaptive_avg_pool", "adaptive_max_pool"]
PoolArgs = (
    max_pool_args | avg_pool_args | adaptive_avg_pool_args | adaptive_max_pool_args
)

_POOL_ARGS: dict[str, type[op_args]] = {
    "max_pool": max_pool_args,
    "avg_pool": avg_pool_args,
    "adaptive_avg_pool": adaptive_avg_pool_args,
    "adaptive_max_pool": adaptive_max_pool_args,
}


def Pool(*, op: PoolOp, args: PoolArgs) -> Tensor:
    _validate_variant(
        family="Pool",
        op=op,
        args=args,
        registry=_POOL_ARGS,
    )
    return _create_operator(
        family="pool",
        op=op,
        args=args,
    )
