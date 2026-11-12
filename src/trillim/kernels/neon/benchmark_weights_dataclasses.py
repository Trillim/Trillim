from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trillim.compiler.weights.weights_dataclasses import _WeightSource
from trillim.tdl.core.types import DType


@dataclass(frozen=True, slots=True)
class _ReplayCheckpoint:
    family: str
    relative_path: str
    tensors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Projection:
    family: str
    checkpoint: Path
    source: _WeightSource
    dtype: DType
    sha256: str
