from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...tdl.core.types import DType


def _validate_packing(dtype: DType, shape: tuple[int, ...], packing: str) -> None:
    if packing == "canonical":
        return
    if (
        packing != "row_int8_block32"
        or dtype != "row_int8"
        or len(shape) != 2
        or any(dimension <= 0 or dimension % 32 for dimension in shape)
    ):
        raise ValueError(f"unsupported weight packing {packing!r} for {dtype} {shape}")


def _repack_row_int8(
    payload: NDArray[np.uint8], shape: tuple[int, ...], source: str, target: str
) -> NDArray[np.uint8]:
    _validate_packing("row_int8", shape, source)
    _validate_packing("row_int8", shape, target)
    if source == target:
        return payload
    rows, columns = shape
    count = rows * columns
    values = payload[:count].reshape(
        (rows // 32, 32, columns // 32, 32)
        if source == "canonical"
        else (rows // 32, columns // 32, 32, 32)
    )
    return np.concatenate((values.transpose(0, 2, 1, 3).reshape(-1), payload[count:]))
