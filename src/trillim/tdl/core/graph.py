from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from types import TracebackType
from typing import TYPE_CHECKING, Self

from pydantic import ConfigDict, TypeAdapter

from .core_dataclasses import _CacheSpec, _Region, _Scope, _TensorTuple
from .tensor import Tensor

if TYPE_CHECKING:
    from .operator import _Operator

_current_graph: ContextVar[Graph | None] = ContextVar("tdl_core_graph", default=None)
_current_scope: ContextVar[_Scope | None] = ContextVar("tdl_core_scope", default=None)
_outputs = TypeAdapter[_TensorTuple](
    _TensorTuple, config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)


class Graph:
    """Build an inference program with independent state for each invocation.

    Serving-time batching must preserve these logical inputs, outputs and cache
    histories. Otherwise-unused state updates remain part of the program.
    After tracing, compiler passes edit the region directly.
    """

    def __init__(self) -> None:
        self._inputs: dict[str, Tensor] = {}
        self._outputs: dict[str, Tensor] = {}
        self._caches: dict[str, _CacheSpec] = {}
        self._region: _Region | None = None
        self._root = _Scope()
        self._tensor_scopes: dict[Tensor, _Scope] = {}
        self._cache_declarations: dict[_CacheSpec, _Operator] = {}
        self._tokens: tuple[Token[Graph | None], Token[_Scope | None]] | None = None
        self._sealed = False

    def __enter__(self) -> Self:
        if _current_graph.get() is not None:
            raise ValueError(
                "graphs cannot be nested; compose tensor functions instead"
            )
        if self._sealed:
            raise ValueError("a completed graph cannot be retraced")
        if self._tokens is not None:
            raise ValueError("this graph is already being constructed")
        self._tokens = (_current_graph.set(self), _current_scope.set(self._root))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._tokens is None:
            raise RuntimeError("graph context was not entered")
        try:
            if exc_type is None:
                self._region = _finish_region(self._root, tuple(self._outputs.values()))
        finally:
            graph_token, scope_token = self._tokens
            _current_scope.reset(scope_token)
            _current_graph.reset(graph_token)
            self._tokens = None
            self._sealed = True


def _get_graph() -> Graph:
    graph = _current_graph.get()
    if graph is None:
        raise RuntimeError("this operation requires an active Graph tracing context")
    return graph


def _get_scope() -> _Scope:
    scope = _current_scope.get()
    if scope is None:
        raise RuntimeError("this operation requires an active Graph tracing context")
    return scope


def _validate_tensor(tensor: Tensor) -> None:
    owner = _get_graph()._tensor_scopes.get(tensor)
    scope: _Scope | None = _get_scope()
    while scope is not None:
        if scope is owner:
            return
        scope = scope.parent
    raise ValueError("tensor belongs to another graph or an inaccessible region")


def _record_cache_access(
    operator: _Operator, cache: _CacheSpec, *, write: bool
) -> None:
    graph = _get_graph()
    scope = _get_scope()
    declaration = graph._cache_declarations.get(cache)
    if declaration is None or graph._caches.get(cache.name) is not cache:
        raise ValueError("cache belongs to another graph")
    previous = scope.cache_accesses.get(cache, declaration)
    operator.dependencies = tuple(dict.fromkeys((*operator.dependencies, previous)))
    scope.cache_accesses[cache] = operator
    if write:
        scope.writes[cache] = None
    else:
        scope.reads[cache] = None


def _record_operator(operator: _Operator) -> None:
    graph = _current_graph.get()
    if graph is None:
        return
    scope = _get_scope()
    for tensor in operator.inputs:
        _validate_tensor(tensor)
    for tensor in operator.outputs:
        graph._tensor_scopes[tensor] = scope
    scope.operators.append(operator)
    reads = dict.fromkeys(
        cache for region in operator.regions for cache in region.reads
    )
    writes = dict.fromkeys(
        cache for region in operator.regions for cache in region.writes
    )
    for cache in dict.fromkeys((*reads, *writes)):
        _record_cache_access(operator, cache, write=cache in writes)
        if cache in reads:
            scope.reads[cache] = None


def _finish_region(scope: _Scope, outputs: _TensorTuple) -> _Region:
    local = set(scope.parameters)
    for operator in scope.operators:
        local.update(operator.outputs)
    captures: dict[Tensor, None] = {}
    for operator in scope.operators:
        for tensor in operator.inputs:
            if tensor not in local:
                captures[tensor] = None
    for tensor in outputs:
        if tensor not in local:
            captures[tensor] = None
    return _Region(
        parameters=scope.parameters,
        outputs=outputs,
        operators=tuple(scope.operators),
        captures=tuple(captures),
        reads=tuple(scope.reads),
        writes=tuple(scope.writes),
    )


def _build_region(
    parameters: _TensorTuple, build: Callable[[_TensorTuple], _TensorTuple]
) -> _Region:
    graph = _get_graph()
    scope = _Scope(parent=_get_scope(), parameters=parameters)
    for parameter in parameters:
        graph._tensor_scopes[parameter] = scope
    token = _current_scope.set(scope)
    try:
        result = _outputs.validate_python(build(parameters))
        for tensor in result:
            _validate_tensor(tensor)
        return _finish_region(scope, result)
    finally:
        _current_scope.reset(token)
