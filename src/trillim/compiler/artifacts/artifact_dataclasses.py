from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import _Inference, _ShapeExpression
from ..kernels.kernel_dataclasses import _Kernel, _ParameterValue, _TensorBinding

type _JSON = None | bool | int | float | str | list[_JSON] | dict[str, _JSON]


@dataclass(slots=True)
class _Expressions:
    names: dict[_ShapeExpression, str] = field(default_factory=dict)
    nodes: dict[str, _JSON] = field(default_factory=dict)


@dataclass(slots=True)
class _Buffer:
    name: str
    dtype: DType
    shape: tuple[_ShapeExpression, ...]
    maximum_shape: tuple[int, ...]
    strides: tuple[_ShapeExpression, ...]
    storage: Literal["arena", "input", "weight", "constant", "alias", "cache"] = "arena"
    source: str | None = None
    offset: int = 0
    size: int = 0
    alignment: int = 64
    first: int = 0
    last: int = 0
    last_read: int = 0
    storage_shape: tuple[_ShapeExpression, _ShapeExpression] | None = None
    byte_offset: _ShapeExpression | None = None
    values: tuple[_ShapeExpression, ...] | None = None

    def _binding(self) -> _TensorBinding:
        return _TensorBinding(
            self.shape,
            self.maximum_shape,
            self.dtype,
            self.strides,
            self.storage_shape,
            self.values,
        )


@dataclass(slots=True)
class _Workspace:
    offset: int
    size: int
    alignment: int
    first: int
    last: int


@dataclass(slots=True)
class _Layout:
    inference: _Inference
    buffers: dict[Tensor, _Buffer] = field(default_factory=dict)
    parameters: dict[_Operator, dict[str, _ParameterValue]] = field(
        default_factory=dict
    )
    scratch: dict[_Operator, _Workspace] = field(default_factory=dict)
    arena_size: int = 0
    cache_size: int = 0
    constants: bytes = b""
    positions: dict[_Operator, tuple[int, int]] = field(default_factory=dict)
    direct_writes: set[_Operator] = field(default_factory=set)
    cache_guards: dict[_Operator, list[tuple[_ShapeExpression, int]]] = field(
        default_factory=dict
    )
    transfer_kernel: _Kernel | None = None
    transfer_scratch: dict[_Operator, _Workspace] = field(default_factory=dict)
    shared_transfers: set[tuple[Tensor, Tensor]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _Requirements:
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    scratch: int


@dataclass(slots=True)
class _PackedParameters:
    address: int
    owners: list[object]
