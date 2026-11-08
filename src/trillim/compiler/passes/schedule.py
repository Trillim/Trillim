from __future__ import annotations

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import _ConstantArgs, _InputArgs
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..artifacts.expressions import _byte_size
from ..compiler_dataclasses import _Inference
from .walk import _predecessors


def _size(tensor: Tensor, inference: _Inference) -> int:
    if tensor._producer is not None and isinstance(
        tensor._producer.args, (_ConstantArgs, _InputArgs)
    ):
        return 0
    info = inference.tensors[tensor]
    if (
        info.dtype is None
        or info.maximum_shape is None
        or any(item is None for item in info.maximum_shape)
    ):
        return 0
    return _byte_size(
        info.dtype, tuple(item for item in info.maximum_shape if item is not None)
    )


def _region(region: _Region, inference: _Inference) -> None:
    for operator in region.operators:
        for body in operator.regions:
            _region(body, inference)
    local = set(region.operators)
    dependencies = {
        operator: set(_predecessors(operator)) & local for operator in region.operators
    }
    followers: dict[_Operator, list[_Operator]] = {operator: [] for operator in local}
    uses: dict[Tensor, int] = {}
    for operator in region.operators:
        for dependency in dependencies[operator]:
            followers[dependency].append(operator)
        for tensor in operator.inputs:
            uses[tensor] = uses.get(tensor, 0) + 1
    for tensor in region.outputs:
        uses[tensor] = uses.get(tensor, 0) + 1
    indices = {operator: index for index, operator in enumerate(region.operators)}
    ready = [operator for operator in region.operators if not dependencies[operator]]
    result: list[_Operator] = []
    sizes = {
        tensor: _size(tensor, inference)
        for operator in region.operators
        for tensor in (*operator.inputs, *operator.outputs)
    }

    def score(operator: _Operator) -> tuple[int, int]:
        released = sum(sizes[tensor] for tensor in operator.inputs if uses[tensor] == 1)
        allocated = sum(sizes[tensor] for tensor in operator.outputs)
        return allocated - released, indices[operator]

    while ready:
        operator = min(ready, key=score)
        ready.remove(operator)
        result.append(operator)
        for tensor in operator.inputs:
            uses[tensor] -= 1
        for follower in followers[operator]:
            dependencies[follower].remove(operator)
            if not dependencies[follower]:
                ready.append(follower)
    if len(result) != len(local):
        raise ValueError("cannot schedule a cyclic graph")
    region.operators = tuple(result)


def _schedule(graph: Graph, inference: _Inference) -> None:
    """Prefer ready work that releases large intermediates while preserving effects."""
    if graph._region is None:
        raise ValueError("graph is not complete")
    _region(graph._region, inference)
