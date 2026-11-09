from __future__ import annotations

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    _BinaryArgs,
    _CastArgs,
    _ConstructArgs,
    _PermuteArgs,
    _ReshapeArgs,
)
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import _Inference
from .passes_dataclasses import _MatchingState


def _pure(region: _Region, state: _MatchingState) -> bool:
    identity = id(region)
    if identity not in state.purity:
        state.purity[identity] = (
            not region.reads
            and not region.writes
            and all(
                operator.op
                not in {"cache", "cache_assign", "cache_read", "while", "if"}
                and all(_pure(body, state) for body in operator.regions)
                for operator in region.operators
            )
        )
    return state.purity[identity]


def _unwrapped(
    tensor: Tensor, state: _MatchingState, covered: set[_Operator] | None = None
) -> Tensor:
    original = tensor
    if original not in state.unwrapped:
        wrappers: list[_Operator] = []
        while (
            (producer := tensor._producer) is not None
            and isinstance(producer.args, _ConstructArgs)
            and _pure(producer.regions[0], state)
        ):
            wrappers.append(producer)
            tensor = producer.regions[0].outputs[producer.outputs.index(tensor)]
        state.unwrapped[original] = tensor, tuple(wrappers)
    result, wrappers_tuple = state.unwrapped[original]
    if covered is not None:
        covered.update(wrappers_tuple)
    return result


def _canonical(
    tensor: Tensor,
    inference: _Inference,
    state: _MatchingState,
    covered: set[_Operator] | None = None,
) -> Tensor:
    while True:
        tensor = _unwrapped(tensor, state, covered)
        operator = tensor._producer
        if operator is None:
            return tensor
        args = operator.args
        nodes = [operator]
        wrappers: set[_Operator] = set()
        if isinstance(args, _CastArgs):
            source = args.x
            if inference.tensors[source].dtype != inference.tensors[tensor].dtype:
                return tensor
        elif isinstance(args, _ReshapeArgs):
            source = _unwrapped(args.x, state, wrappers)
            while source._producer is not None and isinstance(
                source._producer.args, _ReshapeArgs
            ):
                nodes.append(source._producer)
                source = _unwrapped(source._producer.args.x, state, wrappers)
            if inference.tensors[source].shape != inference.tensors[tensor].shape:
                return tensor
        elif isinstance(args, _PermuteArgs):
            source = _unwrapped(args.x, state, wrappers)
            axes = args.axes
            while source._producer is not None and isinstance(
                source._producer.args, _PermuteArgs
            ):
                previous = source._producer.args
                axes = tuple(previous.axes[axis] for axis in axes)
                nodes.append(source._producer)
                source = _unwrapped(previous.x, state, wrappers)
            if axes != tuple(range(len(axes))):
                return tensor
        else:
            return tensor
        if covered is not None:
            covered.update(nodes)
            covered.update(wrappers)
        tensor = source


def _terms(
    tensor: Tensor,
    operation: str,
    dtype: DType,
    inference: _Inference,
    state: _MatchingState,
    leaves: dict[Tensor, str],
    covered: set[_Operator],
) -> tuple[Tensor, ...] | None:
    pending = [tensor]
    result: list[Tensor] = []
    while pending:
        value = pending.pop()
        if value not in leaves:
            value = _canonical(value, inference, state, covered)
        operator = value._producer
        if (
            value not in leaves
            and operator is not None
            and operator.op == operation
            and inference.tensors[value].dtype == dtype
            and isinstance(operator.args, _BinaryArgs)
            and isinstance(operator.args.b, Tensor)
        ):
            covered.add(operator)
            pending.extend((operator.args.b, operator.args.a))
        else:
            result.append(value)
        if len(result) + len(pending) > 16:
            return None
    return tuple(result)
