from __future__ import annotations

from dataclasses import replace

from ...tdl.core.core_pydantic_models import (
    _BroadcastArgs,
    _CacheAssignArgs,
    _ConstructArgs,
    _IfArgs,
    _PermuteArgs,
    _ReduceArgs,
    _ReshapeArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.tensor import Tensor
from ..weights.weights_dataclasses import _WeightIndex
from .passes_dataclasses import _MatchSet, _Selection
from .select import _CostModel, _select
from .walk import _walk


def _weight_root(tensor: Tensor, weights: _WeightIndex) -> Tensor | None:
    while tensor not in weights.bindings:
        producer = tensor._producer
        if producer is None:
            return None
        args = producer.args
        if isinstance(args, _ConstructArgs):
            tensor = producer.regions[0].outputs[producer.outputs.index(tensor)]
        elif isinstance(args, (_PermuteArgs, _ReshapeArgs, _BroadcastArgs)):
            tensor = args.x
        else:
            return None
    return tensor


def _set_packings(
    graph: Graph,
    matches: _MatchSet,
    selection: _Selection,
    weights: _WeightIndex,
    costs: _CostModel | None = None,
    *,
    fast: bool = True,
) -> _Selection:
    required: dict[Tensor, set[str]] = {}
    for call in selection.calls.values():
        for tensor in call.inputs:
            root = _weight_root(tensor, weights)
            if root is not None:
                required.setdefault(root, set()).add(
                    call.packings.get(tensor, "canonical")
                )
    public = list(graph._outputs.values())
    assert graph._region is not None
    for _, operator in _walk(graph._region):
        if operator not in selection.covered and isinstance(
            operator.args, (_CacheAssignArgs, _IfArgs, _WhileArgs, _ReduceArgs)
        ):
            public.extend(operator.inputs)
            public.extend(
                output for body in operator.regions for output in body.outputs
            )
    for tensor in public:
        root = _weight_root(tensor, weights)
        if root is not None:
            required.setdefault(root, set()).add("canonical")
    if any(len(packings) > 1 for packings in required.values()):
        canonical = replace(
            matches,
            candidates={
                operator: tuple(call for call in calls if not call.packings)
                for operator, calls in matches.candidates.items()
            },
        )
        selection = _select(graph, canonical, costs=costs, fast=fast)
        required.clear()
    for tensor, binding in weights.bindings.items():
        packing = next(iter(required.get(tensor, ("canonical",))))
        weights.bindings[tensor] = replace(binding, packing=packing)
    return selection
