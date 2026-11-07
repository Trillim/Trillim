from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ...tdl.core.types import DType

if TYPE_CHECKING:
    from ...tdl.core.tensor import Tensor
    from .weights_pydantic_models import CheckpointConfig

SourceDType = Literal[
    "BOOL", "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "F16", "BF16", "F32"
]
ImportMethod = Literal["native", "bitnet_absmean", "bitnet_packed", "mxfp4"]


@dataclass(frozen=True, slots=True)
class WeightSource:
    checkpoint: str
    tensor: str
    path: Path
    dtype: SourceDType
    shape: tuple[int, ...]
    offset: int
    size: int
    file_size: int
    modified_ns: int


@dataclass(slots=True)
class Checkpoint:
    name: str
    config: CheckpointConfig
    tensors: dict[str, WeightSource]
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WeightEntry:
    identity: str
    shape: tuple[int, ...]
    dtype: DType
    source: WeightSource
    method: ImportMethod = "native"
    components: tuple[WeightSource, ...] = ()
    aliases: tuple[str, ...] = ()
    tied_sources: tuple[WeightSource, ...] = ()
    linear_class: str = ""


@dataclass(frozen=True, slots=True)
class WeightBinding:
    logical: str
    source: WeightEntry
    dtype: DType


@dataclass(slots=True)
class WeightIndex:
    sources: dict[str, WeightEntry] = field(default_factory=dict)
    bindings: dict[Tensor, WeightBinding] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WeightRequest:
    logical: str
    dtype: DType
    packing: str = "canonical"
    name: str | None = None
