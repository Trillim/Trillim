from __future__ import annotations

from collections.abc import Iterator
from graphlib import TopologicalSorter

from ...tdl.core.core_dataclasses import Region
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
