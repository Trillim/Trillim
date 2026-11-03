from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..tdl.core.core_dataclasses import CacheSpec
    from ..tdl.core.graph import Graph
    from ..tdl.core.operator import Operator
    from ..tdl.core.tensor import Tensor
    from ..tdl.core.types import DType


@dataclass(slots=True)
class CompileArguments(Namespace):
    tdl_path: Path = field(init=False)
    checkpoints: list[tuple[str, Path]] = field(default_factory=list)
    backend: str = "scalar"
    output: Path = Path("build/model")
    bounds: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    dtypes: list[tuple[str, str]] = field(default_factory=list)
    depth: str = "none"
    threads: int = 1
    memory_budget: int = 8 * 1024**3
    beam_width: int = 4
    search_seconds: float = 10
    value_bounds: list[tuple[str, tuple[int, int]]] = field(default_factory=list)


@dataclass(slots=True)
class ValidationState:
    graph: Graph
    regions: set[int] = field(default_factory=set)
    operators: set[Operator] = field(default_factory=set)
    consumers: dict[Tensor, set[Operator]] = field(default_factory=dict)
    declarations: dict[CacheSpec, Operator] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationScope:
    tensors: set[Tensor] = field(default_factory=set)
    operators: set[Operator] = field(default_factory=set)
    writers: dict[CacheSpec, Operator] = field(default_factory=dict)
    readers: dict[CacheSpec, list[Operator]] = field(default_factory=dict)


ShapeOperation = Literal[
    "const",
    "scalar",
    "dimension",
    "cache_length",
    "add",
    "sub",
    "mul",
    "floor_div",
    "mod",
    "min",
    "max",
    "ceil_div",
    "broadcast",
    "broadcast_stride",
]


@dataclass(frozen=True, slots=True)
class ShapeExpression:
    op: ShapeOperation
    args: tuple[ShapeExpression, ...] = ()
    value: int | None = None
    tensor: Tensor | None = None
    axis: int | None = None
    cache: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.op,
                    self.args,
                    self.value,
                    self.tensor,
                    self.axis,
                    self.cache,
                    self.minimum,
                    self.maximum,
                )
            ),
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapeExpression):
            return NotImplemented
        pending = [(self, other)]
        visited: set[tuple[int, int]] = set()
        while pending:
            left, right = pending.pop()
            if left is right or (id(left), id(right)) in visited:
                continue
            if (
                left._hash != right._hash
                or left.op != right.op
                or left.value != right.value
                or left.tensor is not right.tensor
                or left.axis != right.axis
                or left.cache != right.cache
                or left.minimum != right.minimum
                or left.maximum != right.maximum
                or len(left.args) != len(right.args)
            ):
                return False
            visited.add((id(left), id(right)))
            pending.extend(zip(left.args, right.args, strict=True))
        return True


@dataclass(frozen=True, slots=True)
class TensorInfo:
    shape: tuple[ShapeExpression, ...] | None
    maximum_shape: tuple[int | None, ...] | None
    dtype: DType | None
    values: tuple[ShapeExpression, ...] | None = None


ShapePredicate = Literal["eq", "le", "ge", "divisible", "broadcast"]


@dataclass(frozen=True, slots=True)
class ShapeConstraint:
    predicate: ShapePredicate
    left: ShapeExpression
    right: ShapeExpression
    reason: str = ""
    operator: Operator | None = None


@dataclass(slots=True)
class Inference:
    tensors: dict[Tensor, TensorInfo] = field(default_factory=dict)
    constraints: list[ShapeConstraint] = field(default_factory=list)
    value_ranges: dict[Tensor, tuple[int | None, int | None]] = field(
        default_factory=dict
    )
    loop_iterations: dict[Operator, int] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceState:
    graph: Graph
    bounds: Mapping[str, tuple[int, ...]]
    dtypes: Mapping[str, DType]
    value_bounds: Mapping[str, tuple[int, int]]
    result: Inference = field(default_factory=Inference)
    constraint_keys: set[ShapeConstraint] = field(default_factory=set)
    operators: list[Operator] = field(default_factory=list)


@dataclass(slots=True)
class OptimizationState:
    inference: Inference | None
    constants: dict[Tensor, NDArray[np.generic]] = field(default_factory=dict)
    replacements: dict[Tensor, Tensor] = field(default_factory=dict)
    changed: bool = False
