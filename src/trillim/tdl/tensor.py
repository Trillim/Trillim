from __future__ import annotations

from typing import TYPE_CHECKING

from .tdl_dataclasses import Shape
from .types import DType, validate_dtype

if TYPE_CHECKING:
    from .operator import Operator


class Tensor:
    def __init__(
        self,
        producer: Operator | None = None,
        shape: Shape | None = None,
        dtype: DType | None = None,
    ) -> None:
        if producer is None:
            if dtype is None:
                raise ValueError("input tensors require a dtype")
            if shape is None:
                raise ValueError("input tensors require a shape")

        self.producer = producer
        self.shape = shape
        self.dtype = validate_dtype(dtype) if dtype is not None else None
        self.consumers: list[Operator] = []
