from __future__ import annotations

from dataclasses import dataclass, field

from ...tdl.core.core_dataclasses import Region
from ...tdl.core.graph import Graph
from ...tdl.core.operator import Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import Inference
from ..kernels.kernel_dataclasses import Kernel, ParameterValue, TensorBinding


@dataclass(frozen=True, slots=True)
class KernelMatch:
    kernel: Kernel
    anchor: Operator
    covered: frozenset[Operator]
    bindings: dict[str, Tensor]
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    parameters: dict[str, ParameterValue]


@dataclass(slots=True)
class KernelPattern:
    graph: Graph
    inputs: dict[Tensor, str]
    outputs: tuple[Tensor, ...]
    inference: Inference


@dataclass(slots=True)
class MatchSet:
    candidates: dict[Operator, tuple[KernelMatch, ...]] = field(default_factory=dict)
    owners: dict[Operator, Region] = field(default_factory=dict)
    views: set[Operator] = field(default_factory=set)
    rejections: dict[Operator, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class Selection:
    calls: dict[Operator, KernelMatch] = field(default_factory=dict)
    covered: set[Operator] = field(default_factory=set)
    cost: float = 0.0


@dataclass(slots=True)
class MatchingState:
    inference: Inference
    purity: dict[int, bool] = field(default_factory=dict)
    unwrapped: dict[Tensor, tuple[Tensor, tuple[Operator, ...]]] = field(
        default_factory=dict
    )
    tensor_bindings: dict[Tensor, TensorBinding] = field(default_factory=dict)
