from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from time import monotonic

from ...tdl.core.core_dataclasses import Region
from ...tdl.core.core_pydantic_models import (
    BroadcastArgs,
    CacheArgs,
    CacheAssignArgs,
    CacheReadArgs,
    ConstantArgs,
    ConstructArgs,
    IfArgs,
    InputArgs,
    OutputArgs,
    PermuteArgs,
    ReduceArgs,
    WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import Operator
from .passes_dataclasses import KernelMatch, MatchSet, Selection
from .walk import ordered_operators

CostModel = Mapping[Operator, Mapping[str, float]] | Callable[[KernelMatch], float]
_INTRINSICS = (
    InputArgs,
    OutputArgs,
    ConstantArgs,
    CacheArgs,
    CacheReadArgs,
    CacheAssignArgs,
    ConstructArgs,
    IfArgs,
    WhileArgs,
    ReduceArgs,
    PermuteArgs,
    BroadcastArgs,
)


def _anchors(region: Region) -> Iterator[Operator]:
    for operator in reversed(tuple(ordered_operators(region))):
        yield operator
        for body in operator.regions:
            yield from _anchors(body)


def _cost(candidate: KernelMatch, costs: CostModel | None) -> float:
    if costs is None:
        return 1.0
    if callable(costs):
        result = costs(candidate)
    else:
        result = costs.get(candidate.anchor, {}).get(candidate.kernel.name, 1.0)
    if math.isnan(result) or result < 0:
        raise ValueError("kernel costs must be nonnegative")
    return result


def _choices(
    operator: Operator, matches: MatchSet, state: Selection, costs: CostModel | None
) -> list[tuple[KernelMatch, float]]:
    if operator in matches.views:
        return []
    result = [
        (candidate, _cost(candidate, costs))
        for candidate in matches.candidates.get(operator, ())
        if not candidate.covered.intersection(state.covered)
    ]
    return sorted(
        ((candidate, cost) for candidate, cost in result if math.isfinite(cost)),
        key=lambda item: (
            item[1] / len(item[0].covered),
            -len(item[0].covered),
            -len(item[0].kernel.capabilities.hardware),
            -len(item[0].kernel.capabilities.software),
            item[0].kernel.name,
        ),
    )


def _extend(state: Selection, candidate: KernelMatch, cost: float) -> Selection:
    return Selection(
        {**state.calls, candidate.anchor: candidate},
        state.covered | candidate.covered,
        state.cost + cost,
    )


def _greedy(
    operators: tuple[Operator, ...], matches: MatchSet, costs: CostModel | None
) -> Selection:
    state = Selection()
    for operator in operators:
        if operator in state.covered:
            continue
        choices = _choices(operator, matches, state, costs)
        if choices:
            candidate, cost = choices[0]
            state.calls[candidate.anchor] = candidate
            state.covered.update(candidate.covered)
            state.cost += cost
        elif operator not in matches.views and not isinstance(
            operator.args, _INTRINSICS
        ):
            error = ValueError(
                f"no executable kernel covers TDL operator {operator.op!r}"
            )
            for kernel, reason in matches.rejections.get(operator, {}).items():
                error.add_note(f"{kernel}: {reason}")
            raise error
    return state


def select(
    graph: Graph,
    matches: MatchSet,
    *,
    beam_width: int = 4,
    search_seconds: float = 10.0,
    fast: bool = False,
    costs: CostModel | None = None,
) -> Selection:
    """Cover outer results before their implementations, retaining a complete fallback."""
    if graph.region is None:
        raise ValueError("cannot select kernels for an incomplete graph")
    if beam_width < 1 or not math.isfinite(search_seconds) or search_seconds < 0:
        raise ValueError(
            "kernel search requires a positive beam width and finite time budget"
        )
    operators = tuple(_anchors(graph.region))
    best = _greedy(operators, matches, costs)
    if fast or beam_width == 1 or search_seconds == 0:
        return best
    deadline = monotonic() + search_seconds
    states = [Selection()]
    for operator in operators:
        if monotonic() >= deadline:
            return best
        following: list[Selection] = []
        for state in states:
            if operator in state.covered:
                following.append(state)
                continue
            following.extend(
                _extend(state, candidate, cost)
                for candidate, cost in _choices(operator, matches, state, costs)
            )
            if operator in matches.views or isinstance(operator.args, _INTRINSICS):
                following.append(state)
        if not following:
            return best
        states = sorted(
            following,
            key=lambda state: (
                state.cost,
                -len(state.covered),
                tuple(candidate.kernel.name for candidate in state.calls.values()),
            ),
        )[:beam_width]
    return min((best, *states), key=lambda state: (state.cost, len(state.calls)))
