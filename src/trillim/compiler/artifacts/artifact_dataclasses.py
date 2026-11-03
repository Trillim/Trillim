from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...dtype import DType
from ...tdl.core.operator import Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import Inference, ShapeExpression
from ..kernels.kernel_dataclasses import Kernel, ParameterValue, TensorBinding

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]


@dataclass(slots=True)
class Expressions:
    names: dict[ShapeExpression, str] = field(default_factory=dict)
    nodes: dict[str, JSON] = field(default_factory=dict)


@dataclass(slots=True)
class Buffer:
    name: str
    dtype: DType
    shape: tuple[ShapeExpression, ...]
    maximum_shape: tuple[int, ...]
    strides: tuple[ShapeExpression, ...]
    storage: Literal["arena", "input", "weight", "constant", "alias", "cache"] = "arena"
    source: str | None = None
    offset: int = 0
    size: int = 0
    alignment: int = 64
    first: int = 0
    last: int = 0
    storage_shape: tuple[ShapeExpression, ShapeExpression] | None = None
    byte_offset: ShapeExpression | None = None
    values: tuple[ShapeExpression, ...] | None = None

    def binding(self) -> TensorBinding:
        return TensorBinding(
            self.shape,
            self.maximum_shape,
            self.dtype,
            self.strides,
            self.storage_shape,
            self.values,
        )


@dataclass(slots=True)
class Workspace:
    offset: int
    size: int
    alignment: int
    first: int
    last: int


@dataclass(slots=True)
class Layout:
    inference: Inference
    buffers: dict[Tensor, Buffer] = field(default_factory=dict)
    parameters: dict[Operator, dict[str, ParameterValue]] = field(default_factory=dict)
    scratch: dict[Operator, Workspace] = field(default_factory=dict)
    arena_size: int = 0
    cache_size: int = 0
    constants: bytes = b""
    positions: dict[Operator, tuple[int, int]] = field(default_factory=dict)
    direct_writes: set[Operator] = field(default_factory=set)
    cache_guards: dict[Operator, list[tuple[ShapeExpression, int]]] = field(
        default_factory=dict
    )
    transfer_kernel: Kernel | None = None
    transfer_scratch: dict[Operator, Workspace] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Requirements:
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    scratch: int


@dataclass(slots=True)
class PackedParameters:
    value: object
    address: int
    owners: list[object]
