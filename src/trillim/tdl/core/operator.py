from __future__ import annotations

from pydantic import BaseModel

from .core_dataclasses import CacheSpec, Region, Shape
from .core_pydantic_models import OperatorArgs
from .graph import _get_graph, _get_scope, _validate_tensor
from .tensor import Tensor
from .types import DType


class Operator:
    def __init__(
        self,
        op: str,
        args: OperatorArgs,
        *,
        outputs: tuple[tuple[Shape | None, DType | None], ...] = ((None, None),),
        regions: tuple[Region, ...] = (),
    ) -> None:
        self.op = op
        self.args = args
        self.regions = regions
        self.dependencies: tuple[Operator, ...] = ()
        graph = _get_graph()
        scope = _get_scope()
        inputs: dict[Tensor, None] = {}

        def collect(value: object) -> None:
            if isinstance(value, Tensor):
                _validate_tensor(value)
                inputs[value] = None
            elif isinstance(value, BaseModel):
                for item in vars(value).values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(args)
        for region in regions:
            collect(region.captures)
        self.inputs = tuple(inputs)
        self.outputs = tuple(
            Tensor(producer=self, shape=shape, dtype=dtype) for shape, dtype in outputs
        )
        for tensor in inputs:
            tensor.consumers.append(self)
        for tensor in self.outputs:
            graph._tensor_scopes[tensor] = scope
        scope.operators.append(self)
        reads = dict.fromkeys(cache for region in regions for cache in region.reads)
        writes = dict.fromkeys(cache for region in regions for cache in region.writes)
        for cache in dict.fromkeys((*reads, *writes)):
            self._access_cache(cache, write=cache in writes)
            if cache in reads:
                scope.reads[cache] = None

    def _access_cache(self, cache: CacheSpec, *, write: bool) -> None:
        graph = _get_graph()
        scope = _get_scope()
        declaration = graph._cache_declarations.get(cache)
        if declaration is None or graph.caches.get(cache.name) is not cache:
            raise ValueError("cache belongs to another graph")
        previous = scope.cache_accesses.get(cache, declaration)
        self.dependencies = tuple(dict.fromkeys((*self.dependencies, previous)))
        scope.cache_accesses[cache] = self
        if write:
            scope.writes[cache] = None
        else:
            scope.reads[cache] = None
