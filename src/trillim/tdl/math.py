from __future__ import annotations

from collections.abc import Sequence

from .core import (
    Add,
    Broadcast,
    Compare,
    Constant,
    Construct,
    Div,
    Exp,
    Gather,
    Iota,
    Mul,
    Permute,
    Reduce,
    Reshape,
    Scatter,
    Select,
    Shape,
    Sub,
    Tensor,
    Weight,
    While,
)
from .core.core_dataclasses import Dimensions, Scalar

Value = Tensor | Weight


def _tensor(value: Value) -> Tensor:
    return Constant(value) if isinstance(value, Weight) else value


@Construct
def Dim(x: Tensor, axis: int) -> Tensor:
    index: Tensor = Constant(axis)
    if axis < 0:
        rank = Gather(Shape(Shape(x)), indices=Iota(1))
        index = Add(rank, axis)
    return Gather(Shape(x), indices=Reshape(index, (1,)))


def _dims(x: Tensor, rank: int) -> tuple[Tensor, ...]:
    return tuple(Dim(x, axis) for axis in range(rank))


def _axis(axis: int, rank: int) -> int:
    if rank <= 0 or not -rank <= axis < rank:
        raise ValueError("axis is out of range for the tensor rank")
    return axis % rank


@Construct
def FullLike(x: Tensor, *, shape: Dimensions | None = None, fill: Scalar = 0) -> Tensor:
    """Allocate a filled tensor with X's dtype, including before dtype inference."""
    scalar = Gather(
        Reshape(x, (-1,)),
        indices=Iota(1),
        mask=Constant(False),
        fill=fill,
    )
    return Broadcast(scalar, Shape(x) if shape is None else shape)


@Construct
def Maximum(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return Select(Compare(a, b, predicate="ge"), a, b)


@Construct
def Minimum(a: Tensor, b: Tensor | Scalar) -> Tensor:
    return Select(Compare(a, b, predicate="le"), a, b)


@Construct
def Clip(x: Tensor, low: Tensor | Scalar, high: Tensor | Scalar) -> Tensor:
    return Minimum(Maximum(x, low), high)


@Construct
def Abs(x: Tensor) -> Tensor:
    return Maximum(x, Mul(x, -1))


@Construct
def Sum(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> Tensor:
    """Numeric sum; the compiler infers accumulator promotion from the reducer."""
    return Reduce(
        (x,),
        initial=(FullLike(x, shape=()),),
        axes=axes,
        reducer=lambda a, b: (Add(a[0], b[0]),),
        keepdims=keepdims,
    )[0]


@Construct
def Max(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> Tensor:
    """Maximum over a nonempty domain, without a dtype-specific infinity sentinel."""
    valid = Broadcast(Constant(True), Shape(x))

    def reducer(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        maximum = Maximum(a[0], b[0])
        value = Select(a[1], Select(b[1], maximum, a[0]), b[0])
        return value, Select(a[1], True, b[1])

    return Reduce(
        (x, valid),
        initial=(FullLike(x, shape=()), Constant(False)),
        axes=axes,
        reducer=reducer,
        keepdims=keepdims,
    )[0]


@Construct
def Mean(x: Tensor, *, axes: tuple[int, ...], keepdims: bool = False) -> Tensor:
    count = Constant(1)
    for axis in axes:
        count = Mul(count, Dim(x, axis))
    return Div(Sum(x, axes=axes, keepdims=keepdims), count)


@Construct
def Softmax(x: Tensor, *, axis: int = -1, sinks: Tensor | None = None) -> Tensor:
    maximum = Max(x, axes=(axis,), keepdims=True)
    if sinks is not None:
        maximum = Maximum(maximum, sinks)
    finite = Compare(maximum, float("-inf"), predicate="ne")
    maximum = Select(finite, maximum, 0.0)
    numerator = Exp(Sub(x, maximum))
    denominator = Sum(numerator, axes=(axis,), keepdims=True)
    if sinks is not None:
        denominator = Add(denominator, Exp(Sub(sinks, maximum)))
    return Div(
        numerator, Select(Compare(denominator, 0, predicate="gt"), denominator, 1)
    )


@Construct
def Take(
    x: Tensor, indices: Tensor, *, axis: int = 0, rank: int = 1, index_rank: int = 1
) -> Tensor:
    """Gather one axis; indices replace that axis with their own dimensions."""
    axis = _axis(axis, rank)
    others = tuple(i for i in range(rank) if i != axis)
    front = Permute(x, (axis, *others))
    coordinates = Reshape(indices, (*_dims(indices, index_rank), 1))
    result = Gather(front, indices=coordinates)
    order = (
        *range(index_rank, index_rank + axis),
        *range(index_rank),
        *range(index_rank + axis, rank - 1 + index_rank),
    )
    return Permute(result, order)


@Construct
def Slice(
    x: Tensor,
    *,
    axis: int,
    start: int | Tensor = 0,
    stop: int | Tensor | None = None,
    step: int = 1,
    rank: int,
) -> Tensor:
    """Positive-stride slicing with clamped bounds; negative bounds count backwards."""
    if step <= 0:
        raise ValueError("Slice requires a positive step")
    axis = _axis(axis, rank)
    size = Dim(x, axis)
    begin = Constant(start) if isinstance(start, int) else start
    end = size if stop is None else Constant(stop) if isinstance(stop, int) else stop
    begin = Select(Compare(begin, 0, predicate="lt"), Add(begin, size), begin)
    end = Select(Compare(end, 0, predicate="lt"), Add(end, size), end)
    begin = Clip(begin, 0, size)
    end = Maximum(begin, Clip(end, 0, size))
    return Take(x, Iota(end, start=begin, step=step), axis=axis, rank=rank)


@Construct
def Concat(tensors: tuple[Tensor, ...], *, axis: int, rank: int) -> Tensor:
    if not tensors:
        raise ValueError("Concat requires at least one tensor")
    axis = _axis(axis, rank)
    order = (axis, *(i for i in range(rank) if i != axis))
    inverse = tuple(order.index(i) for i in range(rank))
    fronts = tuple(Permute(x, order) for x in tensors)
    length = Constant(0)
    for value in fronts:
        length = Add(length, Dim(value, 0))
    output = FullLike(tensors[0], shape=(length, *_dims(fronts[0], rank)[1:]))
    offset = Constant(0)
    for value in fronts:
        end = Add(offset, Dim(value, 0))
        output = Scatter(
            output, indices=Reshape(Iota(end, start=offset), (-1, 1)), values=value
        )
        offset = end
    return Permute(output, inverse)


@Construct
def Stack(tensors: tuple[Tensor, ...], *, axis: int = 0, rank: int) -> Tensor:
    axis = _axis(axis, rank + 1)
    expanded = tuple(
        Reshape(x, (*_dims(x, rank)[:axis], 1, *_dims(x, rank)[axis:])) for x in tensors
    )
    return Concat(expanded, axis=axis, rank=rank + 1)


@Construct
def Argmax(
    x: Tensor, *, axis: int = -1, rank: int, mask: Tensor | None = None
) -> Tensor:
    """Choose the lowest active index on ties; empty domains return int64's maximum."""
    axis = _axis(axis, rank)
    shape: list[int | Tensor] = [1] * rank
    shape[axis] = Dim(x, axis)
    indices = Broadcast(Reshape(Iota(Dim(x, axis)), tuple(shape)), Shape(x))
    valid = (
        Broadcast(Constant(True), Shape(x))
        if mask is None
        else Broadcast(mask, Shape(x))
    )

    def reducer(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        tied = Select(
            Compare(a[0], b[0], predicate="eq"),
            Compare(a[1], b[1], predicate="le"),
            False,
        )
        better = Select(Compare(a[0], b[0], predicate="gt"), True, tied)
        choose = Select(a[2], Select(b[2], better, True), False)
        return (
            Select(choose, a[0], b[0]),
            Select(a[2], Select(choose, a[1], b[1]), Select(b[2], b[1], 2**63 - 1)),
            Select(a[2], True, b[2]),
        )

    return Reduce(
        (x, indices, valid),
        initial=(FullLike(x, shape=()), Constant(2**63 - 1), Constant(False)),
        axes=(axis,),
        reducer=reducer,
    )[1]


@Construct
def TopK(x: Tensor, *, k: int | Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Sort descending on the last axis with unique indices; require 0 <= k <= width."""
    count = Constant(k) if isinstance(k, int) else k
    shape = (*_dims(x, rank)[:-1], count)
    values = FullLike(x, shape=shape)
    indices = Broadcast(Constant(0), shape)
    available = Broadcast(Constant(True), Shape(x))
    width = Dim(x, -1)
    flat_x = Reshape(x, (-1,))
    vocab = Iota(width)
    positions = Iota(count)

    def step(state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        i, remaining, output, selected = state
        index = Argmax(x, rank=rank, mask=remaining)
        index = Reshape(index, (*_dims(x, rank)[:-1], 1))
        match = Compare(vocab, index, predicate="eq")
        flat_index = Reshape(index, (-1,))
        coordinates = Add(Mul(Iota(Dim(flat_index, 0)), width), flat_index)
        value = Reshape(
            Gather(flat_x, indices=Reshape(coordinates, (-1, 1))), Shape(index)
        )
        remaining = Select(match, False, remaining)
        output = Select(Compare(positions, i, predicate="eq"), value, output)
        selected = Select(Compare(positions, i, predicate="eq"), index, selected)
        return Add(i, 1), remaining, output, selected

    result = While(
        (Constant(0), available, values, indices),
        condition=lambda s: Compare(s[0], count, predicate="lt"),
        body=step,
    )
    return result[2], result[3]


@Construct
def CumSum(x: Tensor, *, rank: int) -> Tensor:
    """Inclusive last-axis prefix sum; fallback uses a loop with one-slice updates."""
    order = (rank - 1, *range(rank - 1))
    inverse = tuple(order.index(i) for i in range(rank))
    front = Permute(x, order)
    carry = FullLike(x, shape=_dims(x, rank)[:-1])
    output = FullLike(front)

    def step(state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        index, total, result = state
        coordinates = Reshape(index, (1,))
        total = Add(total, Gather(front, indices=coordinates))
        result = Scatter(result, indices=coordinates, values=total)
        return Add(index, 1), total, result

    result = While(
        (Constant(0), carry, output),
        condition=lambda s: Compare(s[0], Dim(x, -1), predicate="lt"),
        body=step,
    )
    return Permute(result[2], inverse)


def _tuple(value: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    result = (value,) * dimensions if isinstance(value, int) else tuple(value)
    if len(result) != dimensions:
        raise ValueError("spatial parameter has incorrect rank")
    return result


@Construct
def Pad(x: Tensor, *, padding: tuple[tuple[int, int], ...], fill: Scalar = 0) -> Tensor:
    """Constant padding; each pair gives before/after widths for one axis."""
    result = x
    rank = len(padding)
    for axis, (before, after) in enumerate(padding):
        if before < 0 or after < 0:
            raise ValueError("padding widths must be nonnegative")
        if before == 0 and after == 0:
            continue
        order = (axis, *(i for i in range(rank) if i != axis))
        inverse = tuple(order.index(i) for i in range(rank))
        front = Permute(result, order)
        size = Dim(front, 0)
        indices = Iota(Add(size, after), start=-before)
        valid = Select(
            Compare(indices, 0, predicate="ge"),
            Compare(indices, size, predicate="lt"),
            False,
        )
        front = Gather(front, indices=Reshape(indices, (-1, 1)), mask=valid, fill=fill)
        result = Permute(front, inverse)
    return result
