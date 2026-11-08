from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..tdl.core.core_dataclasses import _CacheSpec
    from ..tdl.core.graph import Graph
    from ..tdl.core.operator import _Operator
    from ..tdl.core.tensor import Tensor
    from ..tdl.core.types import DType


@dataclass(slots=True)
class _CompileArguments(Namespace):
    tdl_path: Path = field(init=False)
    checkpoints: list[tuple[str, Path]] = field(init=False)
    backend: str | None = None
    output: Path | None = None
    depth: str = "none"
    threads: int | None = None
    max_input_size: int = 4096


@dataclass(slots=True)
class _ValidationState:
    graph: Graph
    regions: set[int] = field(default_factory=set)
    operators: set[_Operator] = field(default_factory=set)
    consumers: dict[Tensor, set[_Operator]] = field(default_factory=dict)
    declarations: dict[_CacheSpec, _Operator] = field(default_factory=dict)


@dataclass(slots=True)
class _ValidationScope:
    tensors: set[Tensor] = field(default_factory=set)
    operators: set[_Operator] = field(default_factory=set)
    writers: dict[_CacheSpec, _Operator] = field(default_factory=dict)
    readers: dict[_CacheSpec, list[_Operator]] = field(default_factory=dict)


_ShapeOperation = Literal[
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
class _ShapeExpression:
    op: _ShapeOperation
    args: tuple[_ShapeExpression, ...] = ()
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
        if not isinstance(other, _ShapeExpression):
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
class _TensorInfo:
    shape: tuple[_ShapeExpression, ...] | None
    maximum_shape: tuple[int | None, ...] | None
    dtype: DType | None
    values: tuple[_ShapeExpression, ...] | None = None


_ShapePredicate = Literal["eq", "le", "ge", "divisible", "broadcast"]


@dataclass(frozen=True, slots=True)
class _ShapeConstraint:
    predicate: _ShapePredicate
    left: _ShapeExpression
    right: _ShapeExpression
    reason: str = ""
    operator: _Operator | None = None


@dataclass(slots=True)
class _Inference:
    tensors: dict[Tensor, _TensorInfo] = field(default_factory=dict)
    constraints: list[_ShapeConstraint] = field(default_factory=list)
    value_ranges: dict[Tensor, tuple[int | None, int | None]] = field(
        default_factory=dict
    )
    loop_iterations: dict[_Operator, int] = field(default_factory=dict)


@dataclass(slots=True)
class _InferenceState:
    graph: Graph
    max_input_size: int | None
    result: _Inference = field(default_factory=_Inference)
    constraint_keys: set[_ShapeConstraint] = field(default_factory=set)
    operators: list[_Operator] = field(default_factory=list)


@dataclass(slots=True)
class _OptimizationState:
    inference: _Inference | None
    constants: dict[Tensor, NDArray[np.generic]] = field(default_factory=dict)
    replacements: dict[Tensor, Tensor] = field(default_factory=dict)
    changed: bool = False
