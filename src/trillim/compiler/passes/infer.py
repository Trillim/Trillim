from __future__ import annotations

from itertools import zip_longest

import numpy as np

from ...tdl.core.core_dataclasses import Shape
from ...tdl.core.core_pydantic_models import (
    BinaryArgs,
    ConstantArgs,
    ConstructArgs,
    InputArgs,
    Weight,
)
from ...tdl.core.graph import Graph
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType, validate_dtype
from .walk import walk


def infer(graph: Graph) -> None:
    """Populate supported metadata in place; unresolved facts remain unknown."""
    if graph.region is None:
        raise ValueError("graph must be completed before inference")
    shape: Shape | None
    dtype: DType | None
    for _, operator in walk(graph.region):
        args = operator.args
        if isinstance(args, InputArgs):
            shape, dtype = args.shape, args.dtype
        elif isinstance(args, ConstantArgs):
            value = args.value
            dtype = args.dtype
            if isinstance(value, Weight):
                shape = value.shape
                if dtype is None:
                    dtype = value.dtype
            else:
                shape = Shape(value.shape if isinstance(value, np.ndarray) else ())
                if dtype is None:
                    if isinstance(value, np.ndarray):
                        name = {"float16": "fp16", "float32": "fp32"}.get(
                            value.dtype.name, value.dtype.name
                        )
                        dtype = validate_dtype(name)
                    elif isinstance(value, bool):
                        dtype = "bool"
                    elif isinstance(value, int):
                        dtype = "int64"
                    else:
                        dtype = "fp32"
        elif isinstance(args, ConstructArgs):
            for output, result in zip(
                operator.outputs, operator.regions[0].outputs, strict=True
            ):
                output.shape, output.dtype = result.shape, result.dtype
            continue
        elif operator.op == "add" and isinstance(args, BinaryArgs):
            right = args.b
            shape = _broadcast(
                args.a.shape, right.shape if isinstance(right, Tensor) else Shape(())
            )
            dtype = None
            if args.a.dtype == "fp32" and (
                not isinstance(right, Tensor) or right.dtype == "fp32"
            ):
                dtype = "fp32"
        else:
            continue
        output = operator.outputs[0]
        output.shape, output.dtype = shape, dtype


def _broadcast(left: Shape | None, right: Shape | None) -> Shape | None:
    if left is None or right is None:
        return None
    dimensions: list[int | None] = []
    for a, b in zip_longest(
        reversed(left.dimensions), reversed(right.dimensions), fillvalue=1
    ):
        if a == 1:
            dimensions.append(b)
        elif b == 1 or a == b:
            dimensions.append(a)
        elif a is None or b is None:
            dimensions.append(None)
        else:
            raise ValueError(
                f"cannot broadcast shapes {left.dimensions} and {right.dimensions}"
            )
    return Shape(tuple(reversed(dimensions)))
