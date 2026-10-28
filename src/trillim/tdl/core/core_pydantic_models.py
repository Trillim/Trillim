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
    CacheMode,
    CacheSpec,
    Dimensions,
    Scalar,
    Shape,
    TensorTuple,
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
    def validate_reference(cls, value: str) -> str:
        model_name, separator, tensor_name = value.partition(".")

        if not model_name.isalpha():
            raise ValueError("model names must contain only letters")

        if not separator or not tensor_name:
            raise ValueError("weight must name one logical tensor as Model.tensor_name")

        return value

    @property
    def model_name(self) -> str:
        return self.root.partition(".")[0]

    @property
    def tensor_name(self) -> str:
        return self.root.partition(".")[2]

    def __str__(self) -> str:
        return self.root


class OperatorArgs(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        strict=True,
    )


class ConstructArgs(OperatorArgs):
    arguments: dict[str, object]


class Model(RootModel[str]):
    model_config = ConfigDict(frozen=True, strict=True)

    @field_validator("root")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("model names must contain only letters")

        return value

    @property
    def name(self) -> str:
        return self.root

    def __getitem__(self, tensor_name: str) -> Weight:
        return Weight(f"{self.name}.{tensor_name}")


class UnaryArgs(OperatorArgs):
    x: Tensor


class BinaryArgs(OperatorArgs):
    a: Tensor
    b: Tensor | Scalar


class MatmulArgs(OperatorArgs):
    a: Tensor
    b: Tensor


class CompareArgs(BinaryArgs):
    predicate: Literal["eq", "ne", "lt", "le", "gt", "ge"]


class SelectArgs(OperatorArgs):
    condition: Tensor
    when_true: Tensor | Scalar
    when_false: Tensor | Scalar


class CastArgs(UnaryArgs):
    dtype: DType


class InputArgs(OperatorArgs):
    name: str = Field(min_length=1)
    shape: Shape
    dtype: DType


class OutputArgs(UnaryArgs):
    name: str = Field(min_length=1)


class ConstantArgs(OperatorArgs):
    value: Scalar | NDArray[np.generic] | Weight
    dtype: DType | None = None


class IotaArgs(OperatorArgs):
    stop: int | Tensor
    start: int | Tensor = 0
    step: int = 1
    dtype: Literal["int32", "int64"] = "int64"

    @field_validator("step")
    @classmethod
    def validate_step(cls, value: int) -> int:
        if value == 0:
            raise ValueError("iota step cannot be zero")
        return value


class ReshapeArgs(UnaryArgs):
    shape: Dimensions

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: Dimensions) -> Dimensions:
        if isinstance(value, tuple):
            fixed = [dimension for dimension in value if isinstance(dimension, int)]
            if fixed.count(-1) > 1 or any(dimension < -1 for dimension in fixed):
                raise ValueError("reshape permits at most one inferred dimension")
        return value


class PermuteArgs(UnaryArgs):
    axes: tuple[int, ...]

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if sorted(value) != list(range(len(value))):
            raise ValueError("axes must be a permutation of 0 through rank - 1")
        return value


class BroadcastArgs(UnaryArgs):
    shape: Dimensions

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: Dimensions) -> Dimensions:
        if isinstance(value, tuple) and any(
            isinstance(dimension, int) and dimension < 0 for dimension in value
        ):
            raise ValueError("broadcast dimensions cannot be negative")
        return value


class GatherArgs(UnaryArgs):
    indices: Tensor
    mask: Tensor | None = None
    fill: Scalar = 0


class ScatterArgs(UnaryArgs):
    indices: Tensor
    values: Tensor
    mode: Literal["replace", "add"] = "replace"
    mask: Tensor | None = None


class ReduceArgs(OperatorArgs):
    values: TensorTuple = Field(min_length=1)
    initial: TensorTuple = Field(min_length=1)
    axes: tuple[int, ...]
    keepdims: bool = False

    @model_validator(mode="after")
    def validate_reduction(self) -> ReduceArgs:
        if len(self.values) != len(self.initial):
            raise ValueError("reduce values and initial values must have equal arity")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("reduction axes cannot repeat")
        return self


class WhileArgs(OperatorArgs):
    initial: TensorTuple


class IfArgs(OperatorArgs):
    condition: Tensor


class RandomArgs(OperatorArgs):
    shape: Dimensions
    seed: int | Tensor
    offset: int | Tensor = 0
    distribution: Literal["uniform", "normal"] = "uniform"
    dtype: Literal["fp16", "bf16", "fp32"] = "fp32"

    @model_validator(mode="after")
    def validate_random(self) -> RandomArgs:
        if isinstance(self.seed, int) and not 0 <= self.seed < 2**64:
            raise ValueError("random seed must fit uint64")
        if isinstance(self.offset, int) and self.offset < 0:
            raise ValueError("random offset cannot be negative")
        if isinstance(self.shape, tuple) and any(
            isinstance(dimension, int) and dimension < 0 for dimension in self.shape
        ):
            raise ValueError("random dimensions cannot be negative")
        return self


class CacheArgs(OperatorArgs):
    name: str = Field(min_length=1)
    shape: Shape
    dtype: DType
    mode: CacheMode = "overwrite"
    axis: int | None = None
    initial: Scalar = 0

    @model_validator(mode="after")
    def validate_cache(self) -> CacheArgs:
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


class CacheReadArgs(OperatorArgs):
    cache: CacheSpec


class CacheAssignArgs(CacheReadArgs):
    value: Tensor
    action: Literal["assign", "append"]
