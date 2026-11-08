from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ...tdl.core.types import DType

if TYPE_CHECKING:
    import numpy as np

    from ...tdl.core.operator import _Operator
    from ..artifacts.artifact_dataclasses import _JSON
    from ..compiler_dataclasses import _Inference
    from ..compiler_pydantic_models import _CompileRequest
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
    reference_samples: tuple[float, ...] = ()


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
    costs: dict[_Operator, dict[str, float]] = field(default_factory=dict)
    workloads: list[dict[str, _JSON]] = field(default_factory=list)
    per_workload: dict[int, dict[_Operator, dict[str, float]]] = field(
        default_factory=dict
    )


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
