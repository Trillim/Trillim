from __future__ import annotations

from collections.abc import Iterator, Mapping
from graphlib import TopologicalSorter
from pathlib import Path

from ...tdl.core.core_dataclasses import Region
from ...tdl.core.core_pydantic_models import ConstantArgs, Weight
from ...tdl.core.operator import Operator


def walk(region: Region) -> Iterator[tuple[Region, Operator]]:
    """Visit dependencies and nested bodies before their operators, retaining scope."""
    local = set(region.operators)
    dependencies: dict[Operator, list[Operator]] = {}
    for operator in region.operators:
        predecessors = [
            value.producer
            for value in operator.inputs
            if value.producer is not None and value.producer in local
        ]
        predecessors.extend(
            dependency for dependency in operator.dependencies if dependency in local
        )
        dependencies[operator] = predecessors

    for operator in TopologicalSorter(dependencies).static_order():
        for body in operator.regions:
            yield from walk(body)
        yield region, operator


def collect_checkpoints(
    region: Region, bindings: Mapping[str, Path]
) -> tuple[str, ...]:
    """Collect names with path bindings already validated by CompileRequest."""
    checkpoints: dict[str, None] = {}
    for _, operator in walk(region):
        args = operator.args
        if isinstance(args, ConstantArgs) and isinstance(args.value, Weight):
            name = args.value.model_name
            if name not in bindings:
                raise ValueError(
                    f"missing checkpoint binding: --checkpoint {name}=PATH"
                )
            checkpoints[name] = None
    return tuple(checkpoints)
