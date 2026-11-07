from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...tdl.core.core_dataclasses import TensorTuple
from ...tdl.core.types import DType
from ..compiler_dataclasses import ShapeExpression

Backend = Literal["scalar", "x86", "neon"]
ParameterValue = (
    int | float | ShapeExpression | tuple[int | float | ShapeExpression, ...]
)


@dataclass(frozen=True, slots=True)
class TensorBinding:
    shape: tuple[ShapeExpression, ...]
    maximum_shape: tuple[int | None, ...]
    dtype: DType
    strides: tuple[ShapeExpression, ...] | None = None
    storage_shape: tuple[ShapeExpression, ShapeExpression] | None = None
    values: tuple[ShapeExpression, ...] | None = None


@dataclass(frozen=True, slots=True)
class KernelBindingContext:
    inputs: Mapping[str, TensorBinding]
    outputs: tuple[TensorBinding, ...]
    attrs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ParameterField:
    name: str
    type: str
    offset: int
    length: str | None


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    size: int
    alignment: int
    fields: tuple[ParameterField, ...]
    input_count: int
    output_count: int
    input_alignments: tuple[int, ...]
    output_alignments: tuple[int, ...]
    scratch_alignment: int
    requirements_address: int
    can_alias_address: int | None
    native_type: type[ctypes.Structure]


@dataclass(frozen=True, slots=True)
class Capabilities:
    abi_version: int
    backend: str
    hardware: tuple[str, ...]
    software: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    backend: Backend
    pattern: Callable[..., TensorTuple]
    parameter_schema: ParameterSchema
    capabilities: Capabilities
    bind: Callable[[KernelBindingContext], dict[str, ParameterValue]]
    library: Path
    physical_inputs: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class BuildArtifact:
    backend: Backend
    library: Path
    symbols: tuple[str, ...]
    object_path: Path
