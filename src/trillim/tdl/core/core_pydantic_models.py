from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from .core_dataclasses import (
    Shape,
    _CacheMode,
    _CacheSpec,
    _Dimensions,
    _Scalar,
    _TensorTuple,
)
from .tensor import Tensor
from .types import DType


class Weight(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid", strict=True)

    root: str
    shape: Shape | None = None
    dtype: DType | None = None

    def __init__(self, root: str, *, dtype: DType | None = None) -> None:
        super().__init__(root=root, dtype=dtype)

    @field_validator("root")
    @classmethod
    def _validate_reference(cls, value: str) -> str:
        model_name, separator, tensor_name = value.partition(".")

        if not model_name.isalpha():
            raise ValueError("model names must contain only letters")

        if not separator or not tensor_name:
            raise ValueError("weight must name one logical tensor as Model.tensor_name")

        return value

    @property
    def _model_name(self) -> str:
        return self.root.partition(".")[0]

    @property
    def _tensor_name(self) -> str:
        return self.root.partition(".")[2]

    def __str__(self) -> str:
        return self.root


class _OperatorArgs(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        strict=True,
    )


class _ConstructArgs(_OperatorArgs):
    arguments: dict[str, object]


class Model(RootModel[str]):
    model_config = ConfigDict(frozen=True, strict=True)

    @field_validator("root")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("model names must contain only letters")

        return value

    def __getitem__(self, tensor_name: str) -> Weight:
        return Weight(f"{self.root}.{tensor_name}")


class _UnaryArgs(_OperatorArgs):
    x: Tensor


class _BinaryArgs(_OperatorArgs):
    a: Tensor
    b: Tensor | _Scalar


class _MatmulArgs(_OperatorArgs):
    a: Tensor
    b: Tensor


class _CompareArgs(_BinaryArgs):
    predicate: Literal["eq", "ne", "lt", "le", "gt", "ge"]


class _SelectArgs(_OperatorArgs):
    condition: Tensor
    when_true: Tensor | _Scalar
    when_false: Tensor | _Scalar


class _CastArgs(_UnaryArgs):
    dtype: DType


class _InputArgs(_OperatorArgs):
    name: str = Field(min_length=1)
    shape: Shape
    dtype: DType
    maximum_shape: tuple[int, ...] | None = None
    minimum: int | None = None
    maximum: int | None = None
    multiple_of: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> _InputArgs:
        if self.maximum_shape is not None and (
            len(self.maximum_shape) != len(self.shape.dimensions)
            or any(
                limit < 0 or dimension is not None and dimension != limit
                for dimension, limit in zip(
                    self.shape.dimensions, self.maximum_shape, strict=True
                )
            )
        ):
            raise ValueError(
                "maximum_shape must match fixed dimensions and bound each variable dimension"
            )
        if any(
            value is not None
            for value in (self.minimum, self.maximum, self.multiple_of)
        ):
            if self.shape.dimensions or not self.dtype.startswith(("int", "uint")):
                raise ValueError("value bounds require a scalar integer input")
            bits = int(
                self.dtype.removeprefix("uint")
                if self.dtype.startswith("uint")
                else self.dtype.removeprefix("int")
            )
            low, high = (
                (0, 2**bits - 1)
                if self.dtype.startswith("uint")
                else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            )
            minimum = low if self.minimum is None else self.minimum
            maximum = high if self.maximum is None else self.maximum
            if not low <= minimum <= maximum <= high:
                raise ValueError("input bounds must be ordered and fit its dtype")
            if (
                self.multiple_of is not None
                and -(-minimum // self.multiple_of) * self.multiple_of > maximum
            ):
                raise ValueError("input bounds contain no permitted multiple")
        return self


class _OutputArgs(_UnaryArgs):
    name: str = Field(min_length=1)


class _ConstantArgs(_OperatorArgs):
    value: _Scalar | NDArray[np.generic] | Weight
    dtype: DType | None = None


class _IotaArgs(_OperatorArgs):
    stop: int | Tensor
    start: int | Tensor = 0
    step: int = 1
    dtype: Literal["int32", "int64"] = "int64"

    @field_validator("step")
    @classmethod
    def _validate_step(cls, value: int) -> int:
        if value == 0:
            raise ValueError("iota step cannot be zero")
        return value


class _ReshapeArgs(_UnaryArgs):
    shape: _Dimensions

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: _Dimensions) -> _Dimensions:
        if isinstance(value, tuple):
            fixed = [dimension for dimension in value if isinstance(dimension, int)]
            if fixed.count(-1) > 1 or any(dimension < -1 for dimension in fixed):
                raise ValueError("reshape permits at most one inferred dimension")
        return value


class _PermuteArgs(_UnaryArgs):
    axes: tuple[int, ...]

    @field_validator("axes")
    @classmethod
    def _validate_axes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if sorted(value) != list(range(len(value))):
            raise ValueError("axes must be a permutation of 0 through rank - 1")
        return value


class _BroadcastArgs(_UnaryArgs):
    shape: _Dimensions

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: _Dimensions) -> _Dimensions:
        if isinstance(value, tuple) and any(
            isinstance(dimension, int) and dimension < 0 for dimension in value
        ):
            raise ValueError("broadcast dimensions cannot be negative")
        return value


class _GatherArgs(_UnaryArgs):
    indices: Tensor
    mask: Tensor | None = None
    fill: _Scalar = 0


class _ScatterArgs(_UnaryArgs):
    indices: Tensor
    values: Tensor
    mode: Literal["replace", "add"] = "replace"
    mask: Tensor | None = None


class _ReduceArgs(_OperatorArgs):
    values: _TensorTuple = Field(min_length=1)
    initial: _TensorTuple = Field(min_length=1)
    axes: tuple[int, ...]
    keepdims: bool = False

    @model_validator(mode="after")
    def _validate_reduction(self) -> _ReduceArgs:
        if len(self.values) != len(self.initial):
            raise ValueError("reduce values and initial values must have equal arity")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("reduction axes cannot repeat")
        return self


class _WhileArgs(_OperatorArgs):
    initial: _TensorTuple


class _IfArgs(_OperatorArgs):
    condition: Tensor


class _RandomArgs(_OperatorArgs):
    shape: _Dimensions
    seed: int | Tensor
    offset: int | Tensor = 0
    distribution: Literal["uniform", "normal"] = "uniform"
    dtype: Literal["fp16", "bf16", "fp32"] = "fp32"

    @model_validator(mode="after")
    def _validate_random(self) -> _RandomArgs:
        if isinstance(self.seed, int) and not 0 <= self.seed < 2**64:
            raise ValueError("random seed must fit uint64")
        if isinstance(self.offset, int) and self.offset < 0:
            raise ValueError("random offset cannot be negative")
        if isinstance(self.shape, tuple) and any(
            isinstance(dimension, int) and dimension < 0 for dimension in self.shape
        ):
            raise ValueError("random dimensions cannot be negative")
        return self


class _CacheArgs(_OperatorArgs):
    name: str = Field(min_length=1)
    shape: Shape
    dtype: DType
    mode: _CacheMode = "overwrite"
    axis: int | None = None
    initial: _Scalar = 0

    @model_validator(mode="after")
    def _validate_cache(self) -> _CacheArgs:
        if any(dimension is None for dimension in self.shape.dimensions):
            raise ValueError("cache capacity must have fixed dimensions")
        if self.mode == "overwrite":
            if self.axis is not None:
                raise ValueError("overwrite caches do not use an append axis")
        else:
            rank = len(self.shape.dimensions)
            if self.axis is None or not -rank <= self.axis < rank:
                raise ValueError("append and sliding caches require a valid axis")
            capacity = self.shape.dimensions[self.axis]
            if capacity is None or capacity <= 0:
                raise ValueError(
                    "cache capacity along its append axis must be positive"
                )
        return self


class _CacheReadArgs(_OperatorArgs):
    cache: _CacheSpec


class _CacheAssignArgs(_CacheReadArgs):
    value: Tensor
    action: Literal["assign", "append"]
