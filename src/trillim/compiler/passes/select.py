from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    _BroadcastArgs,
    _CacheArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _ConstantArgs,
    _ConstructArgs,
    _IfArgs,
    _InputArgs,
    _OutputArgs,
    _PermuteArgs,
    _ReduceArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from .passes_dataclasses import _KernelMatch, _MatchSet, _Selection
from .walk import _ordered_operators

_CostModel = Mapping[_Operator, Mapping[str, float]] | Callable[[_KernelMatch], float]
_INTRINSICS = (
    _InputArgs,
    _OutputArgs,
    _ConstantArgs,
    _CacheArgs,
    _CacheReadArgs,
    _CacheAssignArgs,
    _ConstructArgs,
    _IfArgs,
    _WhileArgs,
    _ReduceArgs,
    _PermuteArgs,
    _BroadcastArgs,
)


def _specialization(candidate: _KernelMatch) -> tuple[int, int, int]:
    kernel = candidate.kernel
    native = kernel.backend != "scalar" and not kernel.name.startswith(
        f"trillim_{kernel.backend}_scalar_"
    )
    varying_dtypes = sum(
        field.name.endswith("dtype") for field in kernel.parameter_schema.fields
    )
    return (
        int(native),
        (kernel.capabilities & 0xFFFFFFFF).bit_count(),
        -varying_dtypes,
    )


def _anchors(region: _Region) -> Iterator[_Operator]:
    for operator in reversed(tuple(_ordered_operators(region))):
        yield operator
        for body in operator.regions:
            yield from _anchors(body)


def _cost(candidate: _KernelMatch, costs: _CostModel | None) -> float:
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
    operator: _Operator, matches: _MatchSet, state: _Selection, costs: _CostModel | None
) -> list[tuple[_KernelMatch, float]]:
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
            tuple(-rank for rank in _specialization(item[0])),
        ),
    )


def _greedy(
    operators: tuple[_Operator, ...], matches: _MatchSet, costs: _CostModel | None
) -> _Selection:
    state = _Selection()
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


def _select(
    graph: Graph,
    matches: _MatchSet,
    *,
    fast: bool = False,
    costs: _CostModel | None = None,
) -> _Selection:
    """Cover outer results before their implementations, retaining a complete fallback."""
    if graph._region is None:
        raise ValueError("cannot select kernels for an incomplete graph")
    operators = tuple(_anchors(graph._region))
    best = _greedy(operators, matches, costs)
    if fast:
        return best
    states = [_Selection()]
    for operator in operators:
        following: list[_Selection] = []
        for state in states:
            if operator in state.covered:
                following.append(state)
                continue
            following.extend(
                _Selection(
                    {**state.calls, candidate.anchor: candidate},
                    state.covered | candidate.covered,
                    state.cost + cost,
                )
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
            ),
        )[:4]
    return min((best, *states), key=lambda state: (state.cost, len(state.calls)))
