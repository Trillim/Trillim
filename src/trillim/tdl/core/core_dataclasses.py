from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .tensor import Tensor
from .types import DType

if TYPE_CHECKING:
    from .operator import Operator

Scalar = bool | int | float
TensorTuple = tuple[Tensor, ...]
Dimensions = Tensor | tuple[int | Tensor, ...]
CacheMode = Literal["overwrite", "append", "sliding"]


@dataclass(frozen=True, slots=True)
class Shape:
    dimensions: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if any(
            dimension is not None and dimension < 0 for dimension in self.dimensions
        ):
            raise ValueError("fixed dimensions cannot be negative")


@dataclass(frozen=True, slots=True)
class CacheSpec:
    name: str
    shape: Shape
    dtype: DType
    mode: CacheMode
    axis: int | None


@dataclass(slots=True)
class Region:
    parameters: TensorTuple
    outputs: TensorTuple
    operators: tuple[Operator, ...]
    captures: TensorTuple
    reads: tuple[CacheSpec, ...]
    writes: tuple[CacheSpec, ...]


@dataclass(eq=False, slots=True)
class _Scope:
    parent: _Scope | None = None
    parameters: TensorTuple = ()
    operators: list[Operator] = field(default_factory=list)
    cache_accesses: dict[CacheSpec, Operator] = field(default_factory=dict)
    reads: dict[CacheSpec, None] = field(default_factory=dict)
    writes: dict[CacheSpec, None] = field(default_factory=dict)
