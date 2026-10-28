from __future__ import annotations

import math

from .core import Construct, Cos, Exp, Iota, Mul, Reshape, Sin, Tensor
from .math import Concat, Stack, _dims


@Construct
def Sinusoidal(
    positions: Tensor,
    *,
    dim: int,
    theta: float = 10000.0,
    interleaved: bool = False,
    cos_first: bool = True,
    frequency_shift: float = 0.0,
    rank: int = 1,
) -> Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("sinusoidal dimension must be positive and even")
    frequencies = Exp(
        Mul(Iota(dim // 2), -math.log(theta) / (dim // 2 - frequency_shift))
    )
    angles = Mul(Reshape(positions, (*_dims(positions, rank), 1)), frequencies)
    first, second = (
        (Cos(angles), Sin(angles)) if cos_first else (Sin(angles), Cos(angles))
    )
    if interleaved:
        return Reshape(
            Stack((first, second), axis=rank + 1, rank=rank + 1),
            (*_dims(positions, rank), dim),
        )
    return Concat((first, second), axis=rank, rank=rank + 1)
