from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ...tdl.core.types import DType

if TYPE_CHECKING:
    import numpy as np

    from ...tdl.core.operator import _Operator
    from ..compiler_dataclasses import _Inference, _ShapeExpression
    from ..compiler_pydantic_models import _CompileRequest
    from ..passes.passes_dataclasses import _KernelMatch, _Selection
    from ..weights.weights_dataclasses import _WeightIndex
    from ..weights.weights_pydantic_models import _WeightManifest

_BenchmarkLevel = Literal["none", "low", "high"]
_ConcreteParameter = int | float | tuple[int | float, ...]


@dataclass(frozen=True, slots=True)
class _BenchmarkOutput:
    dtype: DType
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BenchmarkInput:
    path: Path
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class _Invocation:
    library: Path
    kernel: str
    parameters: dict[str, _ConcreteParameter]
    inputs: tuple[bytes | _BenchmarkInput, ...]
    outputs: tuple[_BenchmarkOutput, ...]
    threads: int = 1


@dataclass(frozen=True, slots=True)
class _BenchmarkResult:
    samples: tuple[float, ...]
    mean: float
    median: float
    variance: float
    correct: bool
    maximum_error: float
    allocated_bytes: int
    normalized_error: float = 0.0


@dataclass(frozen=True, slots=True)
class _BenchmarkFailure:
    message: str


@dataclass(slots=True)
class _NativeAllocation:
    owner: object
    address: int
    size: int


@dataclass(slots=True)
class _Tuning:
    selection: _Selection
    measured: int = 0
    accepted: int = 0
    propagated: int = 0


@dataclass(frozen=True, slots=True)
class _TuningRegion:
    original: _KernelMatch
    choices: dict[Hashable, tuple[_KernelMatch, ...]]


@dataclass(slots=True)
class _TuningState:
    inference: _Inference
    weights: _WeightIndex
    manifest: _WeightManifest
    bundle: Path
    request: _CompileRequest
    rng: np.random.Generator
    dimensions: dict[Hashable, int] = field(default_factory=dict)
    views: frozenset[_Operator] = frozenset()
    history: int = 0
    decisions: dict[Hashable, Hashable] = field(default_factory=dict)
    rewrite: Callable[[_ShapeExpression], _ShapeExpression] | None = None
