from __future__ import annotations

from typing import Literal

from .conv import _windows
from .core import (
    Add,
    Broadcast,
    Compare,
    Constant,
    Construct,
    Div,
    FloorDiv,
    Iota,
    Mul,
    Reshape,
    Select,
    Sub,
    Tensor,
)
from .math import Dim, Max, Sum, _dims, _tuple


@Construct
def Pool(
    x: Tensor,
    *,
    kernel: tuple[int, ...],
    mode: Literal["max", "avg"],
    stride: int | tuple[int, ...] | None = None,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """Pool one channel-first input; ceil-mode tails exclude undeclared padding."""
    n = len(kernel)
    strides = kernel if stride is None else _tuple(stride, n)
    pads, dilations = _tuple(padding, n), _tuple(dilation, n)
    windows, output_sizes = _windows(
        x,
        kernel=kernel,
        stride=strides,
        padding=pads,
        dilation=dilations,
        fill=float("-inf") if mode == "max" else 0,
        ceil_mode=ceil_mode,
    )
    if mode == "max":
        return Max(windows, axes=(-1,))
    divisor = Constant(1.0)
    for axis, (size, k, s, p, d) in enumerate(
        zip(output_sizes, kernel, strides, pads, dilations, strict=True)
    ):
        coordinates = Add(Reshape(Mul(Iota(size), s), (-1, 1)), Sub(Mul(Iota(k), d), p))
        lower = -p if count_include_pad else 0
        upper = Add(Dim(x, axis + 1), p if count_include_pad else 0)
        valid = Select(
            Compare(coordinates, lower, predicate="ge"),
            Compare(coordinates, upper, predicate="lt"),
            False,
        )
        count = Sum(Select(valid, 1.0, 0.0), axes=(-1,))
        shape: list[int | Tensor] = [1] * (n + 1)
        shape[axis + 1] = size
        divisor = Mul(divisor, Reshape(count, tuple(shape)))
    return Div(Sum(windows, axes=(-1,)), divisor)


@Construct
def AdaptivePool(
    x: Tensor, *, sizes: tuple[int, ...], mode: Literal["max", "avg"]
) -> Tensor:
    """Adaptive pooling with floor starts and ceil ends, including overlapping bins."""
    n = len(sizes)
    shape: tuple[int | Tensor, ...] = (
        Dim(x, 0),
        *sizes,
        *_dims(x, n + 1)[1:],
    )
    valid = Broadcast(Constant(True), shape)
    for axis, output_size in enumerate(sizes):
        output_shape = [1] * (1 + 2 * n)
        output_shape[1 + axis] = output_size
        input_shape: list[int | Tensor] = [1] * (1 + 2 * n)
        input_size = Dim(x, 1 + axis)
        input_shape[1 + n + axis] = input_size
        output_index = Reshape(Iota(output_size), tuple(output_shape))
        input_index = Reshape(Iota(input_size), tuple(input_shape))
        start = FloorDiv(Mul(output_index, input_size), output_size)
        end = FloorDiv(
            Add(Mul(Add(output_index, 1), input_size), output_size - 1), output_size
        )
        valid = Select(Compare(input_index, start, predicate="ge"), valid, False)
        valid = Select(Compare(input_index, end, predicate="lt"), valid, False)
    expanded = Reshape(x, (*_dims(x, n + 1)[:1], *((1,) * n), *_dims(x, n + 1)[1:]))
    axes = tuple(range(1 + n, 1 + 2 * n))
    if mode == "max":
        return Max(Select(valid, expanded, float("-inf")), axes=axes)
    return Div(
        Sum(Select(valid, expanded, 0), axes=axes),
        Sum(Select(valid, 1.0, 0.0), axes=axes),
    )
