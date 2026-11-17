from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trillim.compiler.artifacts.artifact_dataclasses import _Layout
from trillim.compiler.compiler_dataclasses import _Inference
from trillim.compiler.passes.passes_dataclasses import _Selection
from trillim.tdl import Graph


@dataclass(slots=True)
class ModelCase:
    graph: Graph
    inputs: dict[str, NDArray[np.generic]]
    output_shape: tuple[int, ...]
    oracle: NDArray[np.float32] | None = None


@dataclass(slots=True)
class PlannedCase:
    case: ModelCase
    inference: _Inference
    selection: _Selection
    layout: _Layout
    directory: Path
