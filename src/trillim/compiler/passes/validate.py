from __future__ import annotations

from ...tdl.core.core_dataclasses import _CacheSpec, _Region
from ...tdl.core.core_pydantic_models import (
    _CacheArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _ConstructArgs,
    _IfArgs,
    _InputArgs,
    _OutputArgs,
    _ReduceArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _inputs, _Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import _ValidationScope, _ValidationState
from .walk import _ordered_operators, _predecessors


def _validate(graph: Graph) -> None:
    """Check graph structure and effects without requiring inferred metadata."""
    region = graph._region
    if region is None:
        raise ValueError("graph must be completed before validation")
    if region.parameters or region.captures:
        raise ValueError("outer graph cannot have parameters or captures")
    state = _ValidationState(graph)
    _register(region, state)
    _validate_interface(region, state)
    _validate_region(region, state, _ValidationScope())
    if region.outputs != tuple(graph._outputs.values()):
        raise ValueError("outer region results do not match graph outputs")
    for tensor, consumers in state.consumers.items():
        if (
            len(tensor._consumers) != len(consumers)
            or set(tensor._consumers) != consumers
        ):
            raise ValueError("tensor consumer links do not match operator inputs")


def _register(region: _Region, state: _ValidationState) -> None:
    if id(region) in state.regions:
        raise ValueError("region is reused or recursively contains itself")
    state.regions.add(id(region))
    for parameter in region.parameters:
        if parameter._producer is not None:
            raise ValueError("region parameter cannot have a producer")
        _define(parameter, state)
    for operator in region.operators:
        if operator in state.operators:
            raise ValueError(f"operator {operator.op!r} belongs to multiple positions")
        state.operators.add(operator)
        for tensor in operator.outputs:
            if tensor._producer is not operator:
                raise ValueError(
                    f"operator {operator.op!r} has an incorrect producer link"
                )
            _define(tensor, state)
        if region is not state.graph._region and isinstance(
            operator.args, (_InputArgs, _OutputArgs, _CacheArgs)
        ):
            raise ValueError(
                "input, output and cache declarations belong in the outer graph"
            )
        for body in operator.regions:
            _register(body, state)


def _define(tensor: Tensor, state: _ValidationState) -> None:
    if tensor in state.consumers:
        raise ValueError("tensor has multiple definitions")
    state.consumers[tensor] = set()


def _validate_interface(region: _Region, state: _ValidationState) -> None:
    inputs: dict[str, Tensor] = {}
    outputs: dict[str, Tensor] = {}
    for operator in region.operators:
        args = operator.args
        if isinstance(args, _InputArgs):
            if args.name in inputs or len(operator.outputs) != 1:
                raise ValueError(f"invalid input declaration {args.name!r}")
            inputs[args.name] = operator.outputs[0]
        elif isinstance(args, _OutputArgs):
            if args.name in outputs or operator.outputs:
                raise ValueError(f"invalid output declaration {args.name!r}")
            outputs[args.name] = args.x
        elif isinstance(args, _CacheArgs):
            cache = state.graph._caches.get(args.name)
            axis = (
                args.axis % len(args.shape.dimensions)
                if args.axis is not None
                else None
            )
            expected = _CacheSpec(args.name, args.shape, args.dtype, args.mode, axis)
            if cache != expected or cache in state.declarations or operator.outputs:
                raise ValueError(f"invalid cache declaration {args.name!r}")
            assert cache is not None
            state.declarations[cache] = operator
    if inputs != state.graph._inputs or outputs != state.graph._outputs:
        raise ValueError("graph input/output bindings do not match declarations")
    if len(state.declarations) != len(state.graph._caches):
        raise ValueError("graph cache bindings do not match declarations")


def _validate_bindings(operator: _Operator) -> None:
    args = operator.args
    count = len(operator.outputs)
    bindings: tuple[tuple[int, int], ...] = ()
    if isinstance(args, _IfArgs):
        bindings = ((0, count), (0, count))
    elif isinstance(args, _WhileArgs):
        carried = len(args.initial)
        bindings = ((carried, 1), (carried, carried))
        if count != carried:
            raise ValueError("while outputs must match carried arity")
    elif isinstance(args, _ReduceArgs):
        carried = len(args.initial)
        bindings = ((2 * carried, carried),)
        if count != carried:
            raise ValueError("reduce outputs must match accumulator arity")
    elif isinstance(args, _ConstructArgs):
        bindings = ((0, count),)
    if len(operator.regions) != len(bindings):
        raise ValueError(f"operator {operator.op!r} has an invalid number of regions")
    for body, (parameters, results) in zip(operator.regions, bindings, strict=True):
        if len(body.parameters) != parameters or len(body.outputs) != results:
            raise ValueError(f"operator {operator.op!r} has invalid region bindings")
    if isinstance(args, _WhileArgs) and operator.regions[0].writes:
        raise ValueError("while condition cannot update caches")
    if isinstance(args, _ReduceArgs) and (
        operator.regions[0].reads or operator.regions[0].writes
    ):
        raise ValueError("reduce body cannot access caches")


def _validate_region(
    region: _Region, state: _ValidationState, outer: _ValidationScope
) -> None:
    local = set(region.parameters)
    local.update(tensor for operator in region.operators for tensor in operator.outputs)
    captures = {
        tensor
        for operator in region.operators
        for tensor in operator.inputs
        if tensor not in local
    }
    captures.update(tensor for tensor in region.outputs if tensor not in local)
    if len(region.captures) != len(captures) or set(region.captures) != captures:
        raise ValueError("region captures do not match external tensor uses")
    if not captures <= outer.tensors:
        raise ValueError("region captures a tensor outside its enclosing scope")
    scope = _ValidationScope(
        tensors=outer.tensors | set(region.parameters),
        operators=outer.operators.copy(),
    )
    reads: set[_CacheSpec] = set()
    writes: set[_CacheSpec] = set()
    for operator in _ordered_operators(region):
        if operator.inputs != _inputs(operator.args, operator.regions):
            raise ValueError(f"operator {operator.op!r} has stale input links")
        for tensor in operator.inputs:
            if tensor not in scope.tensors:
                raise ValueError(
                    f"operator {operator.op!r} uses a tensor outside its scope"
                )
            state.consumers[tensor].add(operator)
        if not set(operator.dependencies) <= scope.operators:
            raise ValueError(
                f"operator {operator.op!r} has a dependency outside its scope"
            )
        _validate_bindings(operator)
        for body in operator.regions:
            _validate_region(body, state, scope)
        op_reads, op_writes = _effects(operator, state)
        _validate_effect_order(operator, op_reads, op_writes, state, scope)
        reads.update(op_reads)
        writes.update(op_writes)
        scope.tensors.update(operator.outputs)
        scope.operators.add(operator)
    if not set(region.outputs) <= scope.tensors:
        raise ValueError("region returns a tensor outside its scope")
    for summary, expected in ((region.reads, reads), (region.writes, writes)):
        for cache in summary:
            _validate_cache(cache, state)
        if len(summary) != len(expected) or set(summary) != expected:
            raise ValueError("region cache effects do not match its operators")


def _validate_cache(cache: _CacheSpec, state: _ValidationState) -> None:
    if state.graph._caches.get(cache.name) is not cache:
        raise ValueError(f"cache {cache.name!r} belongs to another graph")


def _effects(
    operator: _Operator, state: _ValidationState
) -> tuple[set[_CacheSpec], set[_CacheSpec]]:
    reads = {cache for body in operator.regions for cache in body.reads}
    writes = {cache for body in operator.regions for cache in body.writes}
    args = operator.args
    if isinstance(args, _CacheAssignArgs):
        if operator.outputs:
            raise ValueError("cache writes cannot produce tensors")
        if args.action == "append" and args.cache.mode == "overwrite":
            raise ValueError("cannot append to an overwrite cache")
        writes.add(args.cache)
    elif isinstance(args, _CacheReadArgs):
        if len(operator.outputs) != 1:
            raise ValueError("cache reads must produce one tensor")
        reads.add(args.cache)
    for cache in reads | writes:
        _validate_cache(cache, state)
    return reads, writes


def _validate_effect_order(
    operator: _Operator,
    reads: set[_CacheSpec],
    writes: set[_CacheSpec],
    state: _ValidationState,
    scope: _ValidationScope,
) -> None:
    for cache in sorted(reads | writes, key=lambda cache: cache.name):
        declaration = state.declarations[cache]
        if declaration not in scope.operators:
            raise ValueError(f"cache {cache.name!r} is used before its declaration")
        writer = scope.writers.get(cache, declaration)
        required = [writer]
        if cache in writes:
            required.extend(scope.readers.get(cache, ()))
        if not _ordered_after(operator, set(required)):
            raise ValueError(
                f"operator {operator.op!r} has unordered cache effects on {cache.name!r}"
            )
        if cache in writes:
            scope.writers[cache] = operator
            scope.readers[cache] = []
        else:
            scope.readers.setdefault(cache, []).append(operator)


def _ordered_after(operator: _Operator, required: set[_Operator]) -> bool:
    pending = [operator]
    visited: set[_Operator] = set()
    while pending and required:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            dependencies = tuple(_predecessors(current))
            required.difference_update(dependencies)
            pending.extend(dependencies)
    return not required
