from __future__ import annotations

import ctypes as c
from dataclasses import dataclass, field

from pydantic import JsonValue

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


@dataclass(slots=True)
class _BoundCall:
    kernel: str
    pointer: int
    source: c.Array[c.c_void_p]
    target: c.Array[c.c_void_p]
    scratch: int
    prepared: _PreparedCall
    key: tuple[int, ...] | None = None
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    varying: dict[str, JsonValue] = field(default_factory=dict)
