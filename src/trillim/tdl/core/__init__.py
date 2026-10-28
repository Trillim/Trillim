"""Core constructors record an inference graph for the compiler.

Build operators inside ``with Graph() as graph``. Use TensorShape for declared
dimensions and Shape(x) for runtime dimensions. Shape/dtype inference and
execution belong to the compiler and runtime, respectively.

Loop and branch callbacks build regions once and return tuples of tensors.
For example::

    with Graph() as graph:
        limit = Input(name="limit", shape=TensorShape(()), dtype="int64")
        history = Cache(
            name="history", shape=TensorShape((128,)), dtype="int64",
            mode="append", axis=0,
        )

        def step(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            history.append(Reshape(values[0], (1,)))
            return (Add(values[0], 1),)

        While(
            (Constant(0),),
            condition=lambda values: Compare(values[0], limit, predicate="lt"),
            body=step,
        )
        Output(history.read(), name="history")
"""

from .arithmetic import (
    Add,
    Cast,
    Compare,
    Cos,
    Div,
    Erf,
    Exp,
    FloorDiv,
    Log,
    Matmul,
    Mod,
    Mul,
    Select,
    Sin,
    Sqrt,
    Sub,
)
from .cache import Cache
from .construct import Construct
from .control import If, Reduce, While
from .core_dataclasses import Shape as TensorShape
from .core_pydantic_models import Model, Weight
from .graph import Graph
from .random import Random
from .tensor import Tensor
from .tensor_ops import (
    Broadcast,
    Constant,
    Gather,
    Input,
    Iota,
    Output,
    Permute,
    Reshape,
    Scatter,
    Shape,
)
from .types import DType

__all__ = [
    "Add",
    "Broadcast",
    "Cache",
    "Cast",
    "Compare",
    "Constant",
    "Construct",
    "Cos",
    "DType",
    "Div",
    "Erf",
    "Exp",
    "FloorDiv",
    "Gather",
    "Graph",
    "If",
    "Input",
    "Iota",
    "Log",
    "Matmul",
    "Mod",
    "Model",
    "Mul",
    "Output",
    "Permute",
    "Random",
    "Reduce",
    "Reshape",
    "Scatter",
    "Select",
    "Shape",
    "Sin",
    "Sqrt",
    "Sub",
    "Tensor",
    "TensorShape",
    "Weight",
    "While",
]
