from __future__ import annotations

from typing import TYPE_CHECKING

from .types import DType, _validate_dtype

if TYPE_CHECKING:
    from .core_dataclasses import Shape
    from .operator import _Operator


class Tensor:
    def __init__(
        self,
        _producer: _Operator | None = None,
        shape: Shape | None = None,
        dtype: DType | None = None,
        *,
        _parameter: bool = False,
    ) -> None:
        if _producer is None and not _parameter:
            if dtype is None:
                raise ValueError("input tensors require a dtype")
            if shape is None:
                raise ValueError("input tensors require a shape")

        self._producer = _producer
        self.shape = shape
        self.dtype = _validate_dtype(dtype) if dtype is not None else None
        self._consumers: list[_Operator] = []

    def __bool__(self) -> bool:
        raise TypeError("TDL conditions must use If, While or Select")
