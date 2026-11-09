from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import _Inference
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


@dataclass(slots=True)
class _KernelPattern:
    inputs: dict[Tensor, str]
    outputs: tuple[Tensor, ...]
    inference: _Inference


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
    unwrapped: dict[Tensor, tuple[Tensor, tuple[_Operator, ...]]] = field(
        default_factory=dict
    )
    tensor_bindings: dict[Tensor, _TensorBinding] = field(default_factory=dict)
    normal: dict[tuple[Tensor, tuple[Tensor, ...]], _Canonical] = field(
        default_factory=dict
    )
    normal_keys: dict[tuple[str, tuple[int, ...], Hashable | None], int] = field(
        default_factory=dict
    )
    normal_budget: int = 4096
    normal_active: set[tuple[Tensor, tuple[Tensor, ...]]] = field(default_factory=set)
