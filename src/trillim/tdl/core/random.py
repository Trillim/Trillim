from __future__ import annotations

from typing import Literal

from .core_dataclasses import _Dimensions
from .core_pydantic_models import _RandomArgs
from .operator import _Operator
from .tensor import Tensor


def Random(
    shape: _Dimensions,
    *,
    seed: int | Tensor,
    offset: int | Tensor = 0,
    distribution: Literal["uniform", "normal"] = "uniform",
    dtype: Literal["fp16", "bf16", "fp32"] = "fp32",
) -> Tensor:
    """Describe seeded draws with no shared generator state.

    Uniform draws are in [0, 1); normal draws have mean zero and variance one.
    Offset counts logical output samples, independent of worker scheduling.
    Advance it in repeated calls to obtain fresh draws. Backend algorithms need
    not generate identical samples across different backends or dtypes.
    """
    args = _RandomArgs(
        shape=shape, seed=seed, offset=offset, distribution=distribution, dtype=dtype
    )
    return _Operator("random", args, outputs=((None, dtype),)).outputs[0]
