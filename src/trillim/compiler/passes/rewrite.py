from __future__ import annotations

from collections.abc import Mapping

from ...tdl.core.core_dataclasses import _CacheSpec, _Region, _Scope
from ...tdl.core.core_pydantic_models import (
    _CacheArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _InputArgs,
    _OperatorArgs,
    _OutputArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _inputs, _Operator
from ...tdl.core.tensor import Tensor
from .walk import _ordered_operators, _predecessors


def _value(value: object, replacements: Mapping[Tensor, Tensor]) -> object:
    if isinstance(value, Tensor):
        visited: set[Tensor] = set()
        while value in replacements:
            if value in visited:
                raise ValueError("cyclic tensor replacement")
            visited.add(value)
            value = replacements[value]
        return value
    if isinstance(value, _OperatorArgs):
        return value.model_copy(
            update={
                name: _value(item, replacements) for name, item in vars(value).items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_value(item, replacements) for item in value)
    if isinstance(value, list):
        return [_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _value(item, replacements) for key, item in value.items()}
    return value


def _replace_arguments(
    args: _OperatorArgs, replacements: Mapping[Tensor, Tensor]
) -> _OperatorArgs:
    result = _value(args, replacements)
    assert isinstance(result, _OperatorArgs)
    return result


def _replace(region: _Region, replacements: Mapping[Tensor, Tensor]) -> None:
    for operator in region.operators:
        operator.args = _replace_arguments(operator.args, replacements)
        for body in operator.regions:
            _replace(body, replacements)
    region.outputs = tuple(_tensor(value, replacements) for value in region.outputs)


def _tensor(value: Tensor, replacements: Mapping[Tensor, Tensor]) -> Tensor:
    result = _value(value, replacements)
    assert isinstance(result, Tensor)
    return result


def _replace_tensors(graph: Graph, replacements: Mapping[Tensor, Tensor]) -> None:
    if graph._region is None:
        raise ValueError("graph must be completed before rewriting")
    _replace(graph._region, replacements)
    _rebuild(graph)


def _summarize(
    region: _Region, tensors: set[Tensor], operators: set[_Operator]
) -> None:
    tensors.update(region.parameters)
    tensors.update(region.outputs)
    tensors.update(region.captures)
    reads: dict[_CacheSpec, None] = {}
    writes: dict[_CacheSpec, None] = {}
    for operator in region.operators:
        operators.add(operator)
        tensors.update(operator.inputs)
        tensors.update(operator.outputs)
        for body in operator.regions:
            _summarize(body, tensors, operators)
            reads.update(dict.fromkeys(body.reads))
            writes.update(dict.fromkeys(body.writes))
        args = operator.args
        if isinstance(args, _CacheAssignArgs):
            writes[args.cache] = None
        elif isinstance(args, _CacheReadArgs):
            reads[args.cache] = None
        operator.inputs = _inputs(args, operator.regions)
        tensors.update(operator.inputs)
        for output in operator.outputs:
            output._producer = operator
    local = set(region.parameters)
    local.update(output for operator in region.operators for output in operator.outputs)
    captures = dict.fromkeys(
        value
        for operator in region.operators
        for value in operator.inputs
        if value not in local
    )
    captures.update(
        dict.fromkeys(value for value in region.outputs if value not in local)
    )
    region.captures = tuple(captures)
    region.reads, region.writes = tuple(reads), tuple(writes)


def _dependencies(
    dependencies: tuple[_Operator, ...], live: set[_Operator], visible: set[_Operator]
) -> tuple[_Operator, ...]:
    result: dict[_Operator, None] = {}
    pending = list(dependencies)
    visited: set[_Operator] = set()
    while pending:
        operator = pending.pop()
        if operator in visited:
            continue
        visited.add(operator)
        if operator in live:
            if operator not in visible:
                raise ValueError("rewritten dependency lies outside its region scope")
            result[operator] = None
        else:
            pending.extend(_predecessors(operator))
    return tuple(result)


def _order(
    graph: Graph,
    region: _Region,
    parent: _Scope | None,
    outer: set[_Operator],
    live: set[_Operator],
) -> _Scope:
    # Removed snapshot barriers can still connect surviving cache effects.
    allowed = outer | set(region.operators)
    for operator in region.operators:
        operator.dependencies = _dependencies(operator.dependencies, live, allowed)
    region.operators = tuple(_ordered_operators(region))
    scope = _Scope(
        parent=parent, parameters=region.parameters, operators=list(region.operators)
    )
    scope.reads.update(dict.fromkeys(region.reads))
    scope.writes.update(dict.fromkeys(region.writes))
    visible = outer.copy()
    for parameter in region.parameters:
        graph._tensor_scopes[parameter] = scope
    for operator in region.operators:
        dependencies = list(_dependencies(operator.dependencies, live, visible))
        reads = dict.fromkeys(
            cache for body in operator.regions for cache in body.reads
        )
        writes = dict.fromkeys(
            cache for body in operator.regions for cache in body.writes
        )
        args = operator.args
        if isinstance(args, _CacheAssignArgs):
            writes[args.cache] = None
        elif isinstance(args, _CacheReadArgs):
            reads[args.cache] = None
        for cache in dict.fromkeys((*reads, *writes)):
            previous = scope.cache_accesses.get(cache, graph._cache_declarations[cache])
            if previous not in visible:
                raise ValueError(f"cache {cache.name!r} is used before its declaration")
            dependencies.append(previous)
            scope.cache_accesses[cache] = operator
        operator.dependencies = tuple(dict.fromkeys(dependencies))
        for body in operator.regions:
            _order(graph, body, scope, visible, live)
        for value in operator.inputs:
            value._consumers.append(operator)
        for output in operator.outputs:
            graph._tensor_scopes[output] = scope
        visible.add(operator)
    return scope


def _rebuild(graph: Graph) -> None:
    """Recompute graph links, region captures and ordered cache effects after edits."""
    if graph._region is None:
        raise ValueError("graph must be completed before rewriting")
    tensors = set(graph._tensor_scopes)
    operators: set[_Operator] = set()
    _summarize(graph._region, tensors, operators)
    for tensor in tensors:
        tensor._consumers.clear()
    graph._inputs = {}
    graph._outputs = {}
    graph._cache_declarations = {}
    for operator in graph._region.operators:
        args = operator.args
        if isinstance(args, _InputArgs):
            graph._inputs[args.name] = operator.outputs[0]
        elif isinstance(args, _OutputArgs):
            graph._outputs[args.name] = args.x
        elif isinstance(args, _CacheArgs):
            graph._cache_declarations[graph._caches[args.name]] = operator
    graph._region.outputs = tuple(graph._outputs.values())
    graph._tensor_scopes = {}
    graph._root = _order(graph, graph._region, None, set(), operators)
