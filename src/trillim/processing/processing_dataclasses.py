from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

type TensorInputs = Mapping[str, NDArray[np.generic] | JsonValue]
type TensorOutputs = Mapping[str, NDArray[np.generic]]
type Preprocessor[T] = Callable[[T], TensorInputs]
type Postprocessor[T] = Callable[[TensorOutputs], T]


@dataclass(frozen=True, slots=True)
class Audio:
    samples: NDArray[np.float32]
    sample_rate: int
