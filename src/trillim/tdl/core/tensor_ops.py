from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .core_dataclasses import Shape as TensorShape
from .core_dataclasses import _Dimensions, _Scalar
from .core_pydantic_models import (
    Weight,
    _BroadcastArgs,
    _ConstantArgs,
    _GatherArgs,
    _InputArgs,
    _IotaArgs,
    _OutputArgs,
    _PermuteArgs,
    _ReshapeArgs,
    _ScatterArgs,
    _UnaryArgs,
)
from .graph import _get_graph, _get_scope
from .operator import _Operator
from .tensor import Tensor
from .types import DType


def Input(
    *, name: str, shape: TensorShape | tuple[int | None, ...], dtype: DType
) -> Tensor:
    shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
    args = _InputArgs(name=name, shape=shape, dtype=dtype)
    graph = _get_graph()
    if _get_scope() is not graph._root:
        raise ValueError("inputs must be declared in the outer graph")
    if name in graph._inputs:
        raise ValueError(f'duplicate input name "{name}"')
    tensor = _Operator("input", args, outputs=((shape, dtype),)).outputs[0]
    graph._inputs[name] = tensor
    return tensor


def Output(x: Tensor, *, name: str) -> None:
    args = _OutputArgs(x=x, name=name)
    graph = _get_graph()
    if _get_scope() is not graph._root:
        raise ValueError("outputs must be declared in the outer graph")
    if name in graph._outputs:
        raise ValueError(f'duplicate output name "{name}"')
    _Operator("output", args, outputs=())
    graph._outputs[name] = x


def Constant(
    value: _Scalar | NDArray[np.generic] | Weight, *, dtype: DType | None = None
) -> Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("constant arrays must contain booleans, integers or reals")
        value = value.copy()
        value.setflags(write=False)
    args = _ConstantArgs(value=value, dtype=dtype)
    return _Operator("constant", args, outputs=((None, dtype),)).outputs[0]


def Shape(x: Tensor) -> Tensor:
    """Return the runtime dimensions as a rank-one int64 tensor."""
    return _Operator("shape", _UnaryArgs(x=x), outputs=((None, "int64"),)).outputs[0]


def Iota(
    stop: int | Tensor,
    *,
    start: int | Tensor = 0,
    step: int = 1,
    dtype: Literal["int32", "int64"] = "int64",
) -> Tensor:
    """Generate indices in the half-open range [start, stop) with a fixed step."""
    return _Operator(
        "iota",
        _IotaArgs(stop=stop, start=start, step=step, dtype=dtype),
        outputs=((None, dtype),),
    ).outputs[0]


def Reshape(x: Tensor, shape: _Dimensions) -> Tensor:
    return _Operator("reshape", _ReshapeArgs(x=x, shape=shape)).outputs[0]


def Permute(x: Tensor, axes: tuple[int, ...]) -> Tensor:
    return _Operator("permute", _PermuteArgs(x=x, axes=axes)).outputs[0]


def Broadcast(x: Tensor, shape: _Dimensions) -> Tensor:
    """Broadcast with trailing-axis alignment; insert axes with Reshape first."""
    return _Operator("broadcast", _BroadcastArgs(x=x, shape=shape)).outputs[0]


def Gather(
    x: Tensor,
    *,
    indices: Tensor,
    mask: Tensor | None = None,
    fill: _Scalar = 0,
) -> Tensor:
    """Index leading axes with [..., K] coordinates and retain trailing slice axes.

    A false mask entry yields fill without accessing its coordinates. Active
    coordinates must be in bounds; negative coordinates are invalid.
    """
    return _Operator(
        "gather", _GatherArgs(x=x, indices=indices, mask=mask, fill=fill)
    ).outputs[0]


def Scatter(
    x: Tensor,
    *,
    indices: Tensor,
    values: Tensor,
    mode: Literal["replace", "add"] = "replace",
    mask: Tensor | None = None,
) -> Tensor:
    """Functionally update slices using Gather's indexing convention.

    Masked coordinates are ignored. Replace requires unique active coordinates;
    add accumulates repeated coordinates. The input retains its previous value.
    The compiler promotes X and values to a common result dtype. Persistent
    cache storage conversion belongs to a subsequent cache write.
    """
    return _Operator(
        "scatter",
        _ScatterArgs(x=x, indices=indices, values=values, mode=mode, mask=mask),
    ).outputs[0]
