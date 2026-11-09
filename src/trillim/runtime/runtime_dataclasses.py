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


@dataclass(slots=True)
class _RunArguments(Namespace):
    directory: Path = field(init=False)
    threads: int | None = None
