from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    directory: Path
    embeddings: NDArray[np.float32]
