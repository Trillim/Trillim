from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import _Inference, _ShapeExpression
from ..kernels.kernel_dataclasses import _Kernel, _ParameterValue, _TensorBinding


@dataclass(frozen=True, slots=True)
class _KernelMatch:
    kernel: _Kernel
    anchor: _Operator
    covered: frozenset[_Operator]
    bindings: dict[str, Tensor]
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    parameters: dict[str, _ParameterValue]
    attributes: dict[str, object] = field(default_factory=dict)
    packings: dict[Tensor, str] = field(default_factory=dict)
    loop_dimensions: dict[Tensor, tuple[_ShapeExpression, ...]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class _KernelPattern:
    inputs: dict[Tensor, str]
    outputs: tuple[Tensor, ...]
    inference: _Inference


@dataclass(slots=True)
class _SymbolicPattern:
    inputs: dict[Tensor, str]
    outputs: tuple[Tensor, ...]
    region: _Region
    roots: tuple[frozenset[str], ...]
    inference: _Inference | None = None


@dataclass(frozen=True, slots=True)
class _PatternBindings:
    values: dict[str, Tensor]
    pairs: frozenset[tuple[Tensor, Tensor]] = frozenset()


@dataclass(slots=True)
class _MatchSet:
    candidates: dict[_Operator, tuple[_KernelMatch, ...]] = field(default_factory=dict)
    views: set[_Operator] = field(default_factory=set)
    rejections: dict[_Operator, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class _Selection:
    calls: dict[_Operator, _KernelMatch] = field(default_factory=dict)
    covered: set[_Operator] = field(default_factory=set)
    cost: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class _Canonical:
    """Temporary comparison form, with links to the unchanged executable graph."""

    op: str
    arguments: tuple[_Canonical, ...]
    value: Hashable | None
    key: int
    sources: tuple[Tensor, ...] = ()
    origins: frozenset[_Operator] = frozenset()
    lower: float | None = None
    upper: float | None = None
    finite: bool = False


@dataclass(slots=True)
class _MatchingState:
    inference: _Inference
    canonical: bool = False
    remaining: int = 0
    purity: dict[int, bool] = field(default_factory=dict)
    pattern_purity: dict[int, bool] = field(default_factory=dict)
    pattern_values: dict[Tensor, Tensor] = field(default_factory=dict)
    pattern_roots: dict[Tensor, frozenset[str]] = field(default_factory=dict)
    unwrapped: dict[Tensor, tuple[Tensor, tuple[_Operator, ...]]] = field(
        default_factory=dict
    )
    tensor_bindings: dict[Tensor, _TensorBinding] = field(default_factory=dict)
    normal: dict[tuple[Tensor, tuple[Tensor, ...]], _Canonical] = field(
        default_factory=dict
    )
    shape_normal: dict[
        tuple[_ShapeExpression, int, tuple[Tensor, ...]], tuple[_Inference, _Canonical]
    ] = field(default_factory=dict)
    normal_keys: dict[tuple[str, tuple[int, ...], Hashable | None], int] = field(
        default_factory=dict
    )
    normal_budget: int = 4096
    normal_active: set[tuple[Tensor, tuple[Tensor, ...]]] = field(default_factory=set)
