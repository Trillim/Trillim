from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np

from .core import (
    Add,
    Broadcast,
    Cast,
    Compare,
    Constant,
    Construct,
    Div,
    FloorDiv,
    Gather,
    Iota,
    Matmul,
    Mul,
    Permute,
    Reshape,
    Scatter,
    Select,
    Shape,
    Sub,
    Tensor,
)
from .math import (
    Abs,
    Clip,
    Dim,
    FullLike,
    Maximum,
    Stack,
    Take,
    _dims,
    _tensor,
    _tuple,
    _Value,
)


def _grid(shape: tuple[int | Tensor, ...]) -> tuple[Tensor, ...]:
    coordinates: list[Tensor] = []
    for axis, size in enumerate(shape):
        expanded: list[int | Tensor] = [1] * len(shape)
        expanded[axis] = size
        coordinates.append(Broadcast(Reshape(Iota(size), tuple(expanded)), shape))
    return tuple(coordinates)


def _windows(
    x: Tensor,
    *,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    fill: float = 0.0,
    ceil_mode: bool = False,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    dimensions = len(kernel)
    sizes = tuple(Dim(x, i + 1) for i in range(dimensions))
    output: list[Tensor] = []
    for size, k, s, p, d in zip(sizes, kernel, stride, padding, dilation, strict=True):
        length = Add(
            FloorDiv(
                Add(size, 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)), s
            ),
            1,
        )
        if ceil_mode:
            length = Select(
                Compare(Mul(Sub(length, 1), s), Add(size, p), predicate="ge"),
                Sub(length, 1),
                length,
            )
        output.append(length)
    shape = (Dim(x, 0), *output)
    grid = _grid(shape)
    pieces: list[Tensor] = []
    for offsets in product(*(range(k) for k in kernel)):
        coords = [grid[0]]
        valid = Broadcast(Constant(True), shape)
        for axis, offset in enumerate(offsets):
            index = Add(
                Mul(grid[axis + 1], stride[axis]),
                offset * dilation[axis] - padding[axis],
            )
            valid = Select(Compare(index, 0, predicate="ge"), valid, False)
            valid = Select(Compare(index, sizes[axis], predicate="lt"), valid, False)
            coords.append(index)
        pieces.append(
            Gather(
                x,
                indices=Stack(tuple(coords), axis=dimensions + 1, rank=dimensions + 1),
                mask=valid,
                fill=fill,
            )
        )
    return Stack(tuple(pieces), axis=dimensions + 1, rank=dimensions + 1), tuple(output)


@Construct
def Conv(
    x: Tensor,
    weight: _Value,
    *,
    kernel: tuple[int, ...],
    bias: _Value | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    """Channel-first convolution of one input [C,spatial...]; weights [out, in/groups, spatial...]."""
    n = len(kernel)
    if not 1 <= n <= 3 or groups <= 0 or any(k <= 0 for k in kernel):
        raise ValueError("invalid convolution kernel or groups")
    strides, pads, dilations = (
        _tuple(stride, n),
        _tuple(padding, n),
        _tuple(dilation, n),
    )
    bank = _tensor(weight)
    windows, output = _windows(
        x, kernel=kernel, stride=strides, padding=pads, dilation=dilations
    )
    in_group = FloorDiv(Dim(x, 0), groups)
    out_group = FloorDiv(Dim(bank, 0), groups)
    part = Reshape(windows, (groups, in_group, -1, Dim(windows, -1)))
    part = Permute(part, (0, 2, 1, 3))
    part = Reshape(part, (groups, -1, Mul(in_group, Dim(windows, -1))))
    weights = Permute(Reshape(bank, (groups, out_group, -1)), (0, 2, 1))
    result = Permute(Matmul(part, weights), (0, 2, 1))
    result = Reshape(result, (Dim(bank, 0), *output))
    return (
        result
        if bias is None
        else Add(result, Reshape(_tensor(bias), (-1, *((1,) * n))))
    )


@Construct
def ConvTranspose(
    x: Tensor,
    weight: _Value,
    *,
    kernel: tuple[int, ...],
    bias: _Value | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
) -> Tensor:
    """Transposed convolution; checkpoint weights [in, out/groups, spatial...]."""
    n = len(kernel)
    if not 1 <= n <= 3 or groups <= 0 or any(k <= 0 for k in kernel):
        raise ValueError("invalid convolution kernel or groups")
    strides, pads = _tuple(stride, n), _tuple(padding, n)
    dilations, extra = _tuple(dilation, n), _tuple(output_padding, n)
    bank = _tensor(weight)
    in_group = FloorDiv(Dim(x, 0), groups)
    out_group = Dim(bank, 1)
    input_sizes = _dims(x, n + 1)[1:]
    output_sizes = tuple(
        Add(Mul(Sub(size, 1), s), -2 * p + d * (k - 1) + e + 1)
        for size, s, p, d, k, e in zip(
            input_sizes, strides, pads, dilations, kernel, extra, strict=True
        )
    )
    result = FullLike(x, shape=(Mul(out_group, groups), *output_sizes))
    grid = _grid((Mul(out_group, groups), *input_sizes))
    source = Reshape(x, (groups, in_group, -1))
    source = Permute(source, (0, 2, 1))
    spatial_first = Permute(bank, (*range(2, n + 2), 0, 1))
    for offsets in product(*(range(k) for k in kernel)):
        matrix = Gather(
            spatial_first, indices=Constant(np.array(offsets, dtype=np.int64))
        )
        matrix = Reshape(matrix, (groups, in_group, out_group))
        values = Reshape(
            Permute(Matmul(source, matrix), (0, 2, 1)),
            (Mul(out_group, groups), *input_sizes),
        )
        coordinates = [grid[0]]
        valid = Broadcast(Constant(True), Shape(grid[0]))
        for axis, offset in enumerate(offsets):
            index = Add(
                Mul(grid[axis + 1], strides[axis]),
                offset * dilations[axis] - pads[axis],
            )
            coordinates.append(index)
            valid = Select(Compare(index, 0, predicate="ge"), valid, False)
            valid = Select(
                Compare(index, output_sizes[axis], predicate="lt"), valid, False
            )
        result = Scatter(
            result,
            indices=Stack(tuple(coordinates), axis=n + 1, rank=n + 1),
            values=values,
            mask=valid,
            mode="add",
        )
    return (
        result
        if bias is None
        else Add(result, Reshape(_tensor(bias), (-1, *((1,) * n))))
    )


@Construct
def Resize(
    x: Tensor,
    *,
    sizes: tuple[int | Tensor, ...],
    mode: Literal["nearest", "linear", "cubic"] = "nearest",
    align_corners: bool = False,
) -> Tensor:
    """Separable spatial interpolation on channel-first tensors; cubic uses coefficient -0.75."""

    result = x
    rank = len(sizes) + 1
    for axis, size in enumerate(sizes, start=1):
        source_size = Dim(result, axis)
        position = Iota(size)
        if align_corners:
            denominator = Maximum(
                Sub(Constant(size) if isinstance(size, int) else size, 1), 1
            )
            position = Div(Mul(position, Sub(source_size, 1)), denominator)
        else:
            position = Sub(Div(Mul(Add(position, 0.5), source_size), size), 0.5)
        if mode == "nearest":
            index = FloorDiv(Mul(Iota(size), source_size), size)
            result = Take(result, index, axis=axis, rank=rank)
            continue
        base = Cast(position, "int64")
        base = Select(Compare(position, base, predicate="lt"), Sub(base, 1), base)
        delta = Sub(position, base)
        shape: list[int | Tensor] = [1] * rank
        shape[axis] = size
        if mode == "linear":
            lo = Take(
                result,
                Clip(base, 0, Sub(source_size, 1)),
                axis=axis,
                rank=rank,
            )
            hi = Take(
                result,
                Clip(Add(base, 1), 0, Sub(source_size, 1)),
                axis=axis,
                rank=rank,
            )
            fraction = Reshape(delta, tuple(shape))
            result = Add(lo, Mul(Sub(hi, lo), fraction))
        else:
            total: Tensor | None = None
            for offset in (-1, 0, 1, 2):
                distance = Abs(Sub(delta, offset))
                d2 = Mul(distance, distance)
                d3 = Mul(d2, distance)
                first = Add(Sub(Mul(d3, 1.25), Mul(d2, 2.25)), 1)
                second = Add(
                    Add(Mul(d3, -0.75), Mul(d2, 3.75)), Add(Mul(distance, -6), 3)
                )
                coefficient = Select(
                    Compare(distance, 1, predicate="le"),
                    first,
                    Select(Compare(distance, 2, predicate="lt"), second, 0),
                )
                index = Clip(Add(base, offset), 0, Sub(source_size, 1))
                value = Mul(
                    Take(result, index, axis=axis, rank=rank),
                    Reshape(coefficient, tuple(shape)),
                )
                total = value if total is None else Add(total, value)
            assert total is not None
            result = total
    return result
