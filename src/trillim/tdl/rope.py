from __future__ import annotations

import math

import numpy as np

from .core import (
    Add,
    Constant,
    Construct,
    Cos,
    Exp,
    Gather,
    Iota,
    Mul,
    Permute,
    Reshape,
    Sin,
    Sub,
    Tensor,
)
from .math import Concat, Dim, Slice, Stack, Take


@Construct
def Rotary(
    x: Tensor,
    positions: Tensor,
    *,
    dim: int,
    theta: float = 10000.0,
    interleaved: bool = False,
    frequencies: Tensor | None = None,
    amplitude: float = 1.0,
) -> Tensor:
    """Rotate a prefix of [H,T,D]; positions have shape [T]."""
    if dim <= 0 or dim % 2 or theta <= 0:
        raise ValueError("rotary dimension must be positive/even and theta positive")
    frequency = (
        Exp(Mul(Iota(dim // 2), -2 * math.log(theta) / dim))
        if frequencies is None
        else frequencies
    )
    angles = Mul(Reshape(positions, (-1, 1)), frequency)
    angles = Reshape(angles, (1, -1, dim // 2))
    return RotaryFromAngles(
        x, angles, dim=dim, interleaved=interleaved, amplitude=amplitude
    )


@Construct
def RotaryFromAngles(
    x: Tensor,
    angles: Tensor,
    *,
    dim: int,
    interleaved: bool = False,
    amplitude: float = 1.0,
) -> Tensor:
    """Apply a supplied [1,T,dim/2] frequency phase to a prefix of X."""
    cosine = Mul(Cos(angles), amplitude)
    sine = Mul(Sin(angles), amplitude)
    prefix = Slice(x, axis=2, stop=dim, rank=3)
    if interleaved:
        a = Slice(prefix, axis=2, start=0, step=2, rank=3)
        b = Slice(prefix, axis=2, start=1, step=2, rank=3)
    else:
        a = Slice(prefix, axis=2, stop=dim // 2, rank=3)
        b = Slice(prefix, axis=2, start=dim // 2, rank=3)
    first = Sub(Mul(a, cosine), Mul(b, sine))
    second = Add(Mul(b, cosine), Mul(a, sine))
    if interleaved:
        rotated = Reshape(
            Stack((first, second), axis=3, rank=3),
            (Dim(x, 0), Dim(x, 1), dim),
        )
    else:
        rotated = Concat((first, second), axis=2, rank=3)
    return Concat((rotated, Slice(x, axis=2, start=dim, rank=3)), axis=2, rank=3)


@Construct
def YaRN(
    x: Tensor,
    positions: Tensor,
    *,
    dim: int,
    theta: float,
    factor: float,
    original_context: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float | None = None,
    truncate: bool = True,
) -> Tensor:
    def correction(rotations: float) -> float:
        return (
            dim
            * math.log(original_context / (rotations * 2 * math.pi))
            / (2 * math.log(theta))
        )

    low = correction(beta_fast)
    high = correction(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 0.001), 0, 1
    )
    inverse = np.exp(
        np.arange(dim // 2, dtype=np.float32) * (-2 * math.log(theta) / dim)
    )
    frequencies = Constant((inverse * (1 - ramp + ramp / factor)).astype(np.float32))
    amplitude = (
        0.1 * math.log(factor) + 1 if attention_factor is None else attention_factor
    )
    return Rotary(
        x, positions, dim=dim, theta=theta, frequencies=frequencies, amplitude=amplitude
    )


@Construct
def AxialRotary(
    x: Tensor,
    positions: Tensor,
    *,
    sections: tuple[int, ...],
    theta: float = 10000.0,
    interleaved: bool = True,
) -> Tensor:
    """Positions [T,axes]; sections specify full rotary dimensions per axis."""
    pieces: list[Tensor] = []
    offset = 0
    for axis, dim in enumerate(sections):
        part = Slice(x, axis=2, start=offset, stop=offset + dim, rank=3)
        position = Take(positions, Constant(axis), axis=1, rank=2, index_rank=0)
        pieces.append(
            Rotary(part, position, dim=dim, theta=theta, interleaved=interleaved)
        )
        offset += dim
    pieces.append(Slice(x, axis=2, start=offset, rank=3))
    return Concat(tuple(pieces), axis=2, rank=3)


@Construct
def MRope(
    x: Tensor,
    positions: Tensor,
    *,
    dim: int,
    sections: tuple[int, ...],
    theta: float = 10000.0,
    interleaved_frequencies: bool = True,
) -> Tensor:
    """Qwen multimodal positions [T,axes] with shared inverse frequencies."""

    if sum(sections) != dim // 2:
        raise ValueError("MRoPE sections must partition half of the rotary dimension")
    if interleaved_frequencies:
        axes = np.zeros(dim // 2, dtype=np.int64)
        for axis, section in enumerate(sections[1:], start=1):
            axes[axis : section * len(sections) : len(sections)] = axis
    else:
        axes = np.repeat(np.arange(len(sections), dtype=np.int64), sections)
    selected = Gather(
        Permute(positions, (1, 0)), indices=Reshape(Constant(axes), (-1, 1))
    )
    selected = Permute(selected, (1, 0))
    frequencies = Exp(Mul(Iota(dim // 2), -2 * math.log(theta) / dim))
    angles = Reshape(Mul(selected, frequencies), (1, Dim(positions, 0), dim // 2))
    return RotaryFromAngles(x, angles, dim=dim)
