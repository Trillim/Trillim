from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ...dtype import DType

if TYPE_CHECKING:
    import numpy as np

    from ...tdl.core.operator import Operator
    from ..artifacts.artifact_dataclasses import JSON
    from ..compiler_dataclasses import Inference
    from ..compiler_pydantic_models import CompileRequest
    from ..weights.weights_dataclasses import WeightIndex
    from ..weights.weights_pydantic_models import WeightManifest

BenchmarkLevel = Literal["none", "low", "medium", "high"]
ConcreteParameter = int | float | tuple[int | float, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkOutput:
    dtype: DType
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkInput:
    path: Path
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class Invocation:
    library: Path
    kernel: str
    parameters: dict[str, ConcreteParameter]
    inputs: tuple[bytes | BenchmarkInput, ...]
    outputs: tuple[BenchmarkOutput, ...]
    threads: int = 1


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
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
class BenchmarkFailure:
    message: str


@dataclass(slots=True)
class NativeAllocation:
    owner: object
    address: int
    size: int


@dataclass(slots=True)
class Tuning:
    costs: dict[Operator, dict[str, float]] = field(default_factory=dict)
    workloads: list[dict[str, JSON]] = field(default_factory=list)
    per_workload: dict[int, dict[Operator, dict[str, float]]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class TuningState:
    inference: Inference
    weights: WeightIndex
    manifest: WeightManifest
    bundle: Path
    request: CompileRequest
    deadline: float
    rng: np.random.Generator
    dimensions: dict[Hashable, int] = field(default_factory=dict)
    views: frozenset[Operator] = frozenset()
