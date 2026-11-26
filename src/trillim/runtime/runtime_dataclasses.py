from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

from ..tdl.core.types import DType


@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: DType
    shape: tuple[int | None, ...]
    maximum_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One generated token and its position in the named output."""

    output: str
    index: int
    token: int


@dataclass(slots=True)
class _RunArguments(Namespace):
    directory: Path = field(init=False)
    threads: int | None = None
    output: Path | None = None
