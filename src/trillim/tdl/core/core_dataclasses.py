from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .tensor import Tensor
from .types import DType

if TYPE_CHECKING:
    from .operator import _Operator

_Scalar = bool | int | float
_TensorTuple = tuple[Tensor, ...]
_Dimensions = Tensor | tuple[int | Tensor, ...]
_CacheMode = Literal["overwrite", "append", "sliding"]


@dataclass(frozen=True, slots=True)
class Shape:
    dimensions: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if any(
            dimension is not None and dimension < 0 for dimension in self.dimensions
        ):
            raise ValueError("fixed dimensions cannot be negative")


@dataclass(frozen=True, slots=True)
class _CacheSpec:
    name: str
    shape: Shape
    dtype: DType
    mode: _CacheMode
    axis: int | None


@dataclass(slots=True)
class _Region:
    parameters: _TensorTuple
    outputs: _TensorTuple
    operators: tuple[_Operator, ...]
    captures: _TensorTuple
    reads: tuple[_CacheSpec, ...]
    writes: tuple[_CacheSpec, ...]


@dataclass(eq=False, slots=True)
class _Scope:
    parent: _Scope | None = None
    parameters: _TensorTuple = ()
    operators: list[_Operator] = field(default_factory=list)
    cache_accesses: dict[_CacheSpec, _Operator] = field(default_factory=dict)
    reads: dict[_CacheSpec, None] = field(default_factory=dict)
    writes: dict[_CacheSpec, None] = field(default_factory=dict)
