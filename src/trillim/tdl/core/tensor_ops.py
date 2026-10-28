from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .core_dataclasses import Dimensions, Scalar
from .core_dataclasses import Shape as TensorShape
from .core_pydantic_models import (
    BroadcastArgs,
    ConstantArgs,
    GatherArgs,
    InputArgs,
    IotaArgs,
    OutputArgs,
    PermuteArgs,
    ReshapeArgs,
    ScatterArgs,
    UnaryArgs,
    Weight,
)
from .graph import _get_graph, _get_scope
from .operator import Operator
from .tensor import Tensor
from .types import DType, validate_dtype


def Input(
    *, name: str, shape: TensorShape | tuple[int | None, ...], dtype: DType
) -> Tensor:
    shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
    args = InputArgs(name=name, shape=shape, dtype=dtype)
    graph = _get_graph()
    if _get_scope() is not graph._root:
        raise ValueError("inputs must be declared in the outer graph")
    if name in graph.inputs:
        raise ValueError(f'duplicate input name "{name}"')
    tensor = Operator("input", args, outputs=((shape, dtype),)).outputs[0]
    graph.inputs[name] = tensor
    return tensor


def Output(x: Tensor, *, name: str) -> None:
    args = OutputArgs(x=x, name=name)
    graph = _get_graph()
    if _get_scope() is not graph._root:
        raise ValueError("outputs must be declared in the outer graph")
    if name in graph.outputs:
        raise ValueError(f'duplicate output name "{name}"')
    Operator("output", args, outputs=())
    graph.outputs[name] = x


def Constant(
    value: Scalar | NDArray[np.generic] | Weight, *, dtype: DType | None = None
) -> Tensor:
    shape: TensorShape | None
    if isinstance(value, Weight):
        shape = value.shape
        dtype = dtype if dtype is not None else value.dtype
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("constant arrays must contain booleans, integers or reals")
        value = value.copy()
        value.setflags(write=False)
        shape = TensorShape(value.shape)
        if dtype is None:
            name = {"float16": "fp16", "float32": "fp32"}.get(
                value.dtype.name, value.dtype.name
            )
            dtype = validate_dtype(name)
    else:
        shape = TensorShape(())
        if dtype is None:
            if isinstance(value, bool):
                dtype = "bool"
            elif isinstance(value, int):
                dtype = "int64"
            else:
                dtype = "fp32"
    args = ConstantArgs(value=value, dtype=dtype)
    return Operator("constant", args, outputs=((shape, dtype),)).outputs[0]


def Shape(x: Tensor) -> Tensor:
    """Return the runtime dimensions as a rank-one int64 tensor."""
    return Operator("shape", UnaryArgs(x=x), outputs=((None, "int64"),)).outputs[0]


def Iota(
    stop: int | Tensor,
    *,
    start: int | Tensor = 0,
    step: int = 1,
    dtype: Literal["int32", "int64"] = "int64",
) -> Tensor:
    """Generate indices in the half-open range [start, stop) with a fixed step."""
    return Operator(
        "iota",
        IotaArgs(stop=stop, start=start, step=step, dtype=dtype),
        outputs=((None, dtype),),
    ).outputs[0]


def Reshape(x: Tensor, shape: Dimensions) -> Tensor:
    return Operator("reshape", ReshapeArgs(x=x, shape=shape)).outputs[0]


def Permute(x: Tensor, axes: tuple[int, ...]) -> Tensor:
    return Operator("permute", PermuteArgs(x=x, axes=axes)).outputs[0]


def Broadcast(x: Tensor, shape: Dimensions) -> Tensor:
    """Broadcast with trailing-axis alignment; insert axes with Reshape first."""
    return Operator("broadcast", BroadcastArgs(x=x, shape=shape)).outputs[0]


def Gather(
    x: Tensor,
    *,
    indices: Tensor,
    mask: Tensor | None = None,
    fill: Scalar = 0,
) -> Tensor:
    """Index leading axes with [..., K] coordinates and retain trailing slice axes.

    A false mask entry yields fill without accessing its coordinates. Active
    coordinates must be in bounds; negative coordinates are invalid.
    """
    return Operator(
        "gather", GatherArgs(x=x, indices=indices, mask=mask, fill=fill)
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
    return Operator(
        "scatter",
        ScatterArgs(x=x, indices=indices, values=values, mode=mode, mask=mask),
    ).outputs[0]
