from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trillim.compiler.weights.weights_dataclasses import _WeightSource


@dataclass(frozen=True, slots=True)
class _Mxfp4Expert:
    checkpoint: Path
    layer: int
    projection: str
    expert: int
    rows: int
    width: int
    blocks: _WeightSource
    scales: _WeightSource
    block_offset: int
    scale_offset: int
    block_bytes: int
    scale_bytes: int
    block_sha256: str
    scale_sha256: str
    payload: Path
    payload_sha256: str
