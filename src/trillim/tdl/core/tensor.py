from __future__ import annotations

from typing import TYPE_CHECKING

from .types import DType, validate_dtype

if TYPE_CHECKING:
    from .core_dataclasses import Shape
    from .operator import Operator


class Tensor:
    def __init__(
        self,
        producer: Operator | None = None,
        shape: Shape | None = None,
        dtype: DType | None = None,
        *,
        _parameter: bool = False,
    ) -> None:
        if producer is None and not _parameter:
            if dtype is None:
                raise ValueError("input tensors require a dtype")
            if shape is None:
                raise ValueError("input tensors require a shape")

        self.producer = producer
        self.shape = shape
        self.dtype = validate_dtype(dtype) if dtype is not None else None
        self.consumers: list[Operator] = []

    def __bool__(self) -> bool:
        raise TypeError("TDL conditions must use If, While or Select")
