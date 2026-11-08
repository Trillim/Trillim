from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...tdl.core.core_dataclasses import _TensorTuple
from ...tdl.core.types import DType
from ..compiler_dataclasses import _ShapeExpression

_Backend = Literal["scalar", "x86", "neon"]
_ParameterValue = (
    int | float | _ShapeExpression | tuple[int | float | _ShapeExpression, ...]
)


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    shape: tuple[_ShapeExpression, ...]
    maximum_shape: tuple[int | None, ...]
    dtype: DType
    strides: tuple[_ShapeExpression, ...] | None = None
    storage_shape: tuple[_ShapeExpression, _ShapeExpression] | None = None
    values: tuple[_ShapeExpression, ...] | None = None


@dataclass(frozen=True, slots=True)
class _KernelBindingContext:
    inputs: Mapping[str, _TensorBinding]
    outputs: tuple[_TensorBinding, ...]
    attrs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ParameterField:
    name: str
    type: str
    offset: int
    length: str | None


@dataclass(frozen=True, slots=True)
class _ParameterSchema:
    size: int
    alignment: int
    fields: tuple[_ParameterField, ...]
    input_count: int
    output_count: int
    input_alignments: tuple[int, ...]
    output_alignments: tuple[int, ...]
    scratch_alignment: int
    requirements_address: int
    can_alias_address: int | None
    native_type: type[ctypes.Structure]


@dataclass(frozen=True, slots=True)
class _Kernel:
    name: str
    backend: _Backend
    pattern: Callable[..., _TensorTuple]
    parameter_schema: _ParameterSchema
    capabilities: int
    bind: Callable[[_KernelBindingContext], dict[str, _ParameterValue]]
    library: Path
    physical_inputs: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _BuildArtifact:
    backend: _Backend
    library: Path
    symbols: tuple[str, ...]
    object_path: Path
