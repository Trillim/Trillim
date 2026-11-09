from __future__ import annotations

import ctypes as c
from dataclasses import dataclass

from ...compiler.artifacts.artifact_dataclasses import _PackedParameters, _Requirements
from ...compiler.kernels.kernel_dataclasses import _ParameterValue


@dataclass(slots=True)
class _Allocation:
    owner: c.Array[c.c_char]
    address: int
    size: int


@dataclass(slots=True)
class _PreparedCall:
    parameters: dict[str, _ParameterValue]
    packed: _PackedParameters
    requirements: _Requirements
