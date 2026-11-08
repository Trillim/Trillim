from __future__ import annotations

from collections.abc import Iterator, Mapping
from graphlib import TopologicalSorter
from pathlib import Path

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import Weight, _ConstantArgs
from ...tdl.core.operator import _Operator


def _predecessors(operator: _Operator) -> Iterator[_Operator]:
    for value in operator.inputs:
        if value._producer is not None:
            yield value._producer
    yield from operator.dependencies


def _ordered_operators(region: _Region) -> Iterator[_Operator]:
    local = set(region.operators)
    dependencies = {
        operator: [
            dependency for dependency in _predecessors(operator) if dependency in local
        ]
        for operator in region.operators
    }
    yield from TopologicalSorter(dependencies).static_order()


def _walk(region: _Region) -> Iterator[tuple[_Region, _Operator]]:
    """Visit dependencies and nested bodies before their operators, retaining scope."""
    for operator in _ordered_operators(region):
        for body in operator.regions:
            yield from _walk(body)
        yield region, operator


def _collect_checkpoints(
    region: _Region, bindings: Mapping[str, Path]
) -> tuple[str, ...]:
    """Collect checkpoint names and check their path bindings."""
    checkpoints: dict[str, None] = {}
    for _, operator in _walk(region):
        args = operator.args
        if isinstance(args, _ConstantArgs) and isinstance(args.value, Weight):
            name = args.value._model_name
            if name not in bindings:
                raise ValueError(f"missing checkpoint binding: {name}=PATH")
            checkpoints[name] = None
    return tuple(checkpoints)
