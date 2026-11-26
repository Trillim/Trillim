from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue

from ...tdl.core.types import DType

if TYPE_CHECKING:
    from ...tdl.core.tensor import Tensor
    from .weights_pydantic_models import _WeightRecord

_SourceDType = Literal[
    "BOOL",
    "I8",
    "U8",
    "I16",
    "U16",
    "I32",
    "U32",
    "I64",
    "U64",
    "F16",
    "BF16",
    "F32",
    "F8_E4M3",
]
_ImportMethod = Literal[
    "native", "bitnet_absmean", "bitnet_packed", "mxfp4", "fp8_e4m3"
]


@dataclass(frozen=True, slots=True)
class _WeightSource:
    checkpoint: str
    tensor: str
    path: Path
    dtype: _SourceDType
    shape: tuple[int, ...]
    offset: int
    size: int
    file_size: int
    modified_ns: int


@dataclass(slots=True)
class _Checkpoint:
    name: str
    config: dict[str, JsonValue]
    tensors: dict[str, _WeightSource]
    metadata: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _WeightEntry:
    identity: str
    shape: tuple[int, ...]
    dtype: DType
    source: _WeightSource
    method: _ImportMethod = "native"
    components: tuple[_WeightSource, ...] = ()
    aliases: tuple[str, ...] = ()
    tied_sources: tuple[_WeightSource, ...] = ()
    linear_class: str = ""


@dataclass(frozen=True, slots=True)
class _WeightBinding:
    logical: str
    source: _WeightEntry
    dtype: DType
    packing: str = "canonical"


@dataclass(slots=True)
class _WeightIndex:
    sources: dict[str, _WeightEntry] = field(default_factory=dict)
    bindings: dict[Tensor, _WeightBinding] = field(default_factory=dict)
    checkpoints: dict[str, _Checkpoint] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _WeightRequest:
    logical: str
    dtype: DType
    packing: str = "canonical"
    name: str | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedWeight:
    identity: str
    record: _WeightRecord
    sources: tuple[_WeightSource, ...]
    chunks: Iterator[tuple[int, bytes]]
