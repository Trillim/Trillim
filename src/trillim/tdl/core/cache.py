from __future__ import annotations

from .core_dataclasses import Shape, _CacheMode, _CacheSpec, _Scalar
from .core_pydantic_models import _CacheArgs, _CacheAssignArgs, _CacheReadArgs
from .graph import _get_graph, _get_scope, _record_cache_access
from .operator import _Operator
from .tensor import Tensor
from .types import DType


class Cache:
    """Session storage with fixed capacity, initialized once per inference run.

    Overwrite caches start filled with initial. Append and sliding caches start
    logically empty. Sliding reads expose retained entries in chronological order.
    Dtype specifies storage. The compiler converts writes to that dtype and
    resolves how reads are consumed without imposing it on intermediate results.
    """

    def __init__(
        self,
        *,
        name: str,
        shape: Shape | tuple[int, ...],
        dtype: DType,
        mode: _CacheMode = "overwrite",
        axis: int | None = None,
        initial: _Scalar = 0,
    ) -> None:
        shape = shape if isinstance(shape, Shape) else Shape(shape)
        args = _CacheArgs(
            name=name, shape=shape, dtype=dtype, mode=mode, axis=axis, initial=initial
        )
        graph = _get_graph()
        if _get_scope() is not graph._root:
            raise ValueError("caches must be declared in the outer graph")
        if name in graph._caches:
            raise ValueError(f'duplicate cache name "{name}"')
        if args.axis is not None:
            axis = args.axis % len(shape.dimensions)
        self._spec = _CacheSpec(
            name=name, shape=shape, dtype=dtype, mode=mode, axis=axis
        )
        declaration = _Operator("cache", args, outputs=())
        graph._caches[name] = self._spec
        graph._cache_declarations[self._spec] = declaration

    def read(self) -> Tensor:
        """Read a logical snapshot; later updates cannot change this tensor's value."""
        shape = self._spec.shape
        if self._spec.axis is not None:
            dimensions = list(shape.dimensions)
            dimensions[self._spec.axis] = None
            shape = Shape(tuple(dimensions))
        operator = _Operator(
            "cache_read",
            _CacheReadArgs(cache=self._spec),
            outputs=((shape, self._spec.dtype),),
        )
        _record_cache_access(operator, self._spec, write=False)
        return operator.outputs[0]

    def assign(self, value: Tensor) -> None:
        """Replace logical contents, erroring if they exceed capacity.

        Overwrite caches require their full declared shape. Append/sliding caches
        permit a shorter append axis and set their logical length to that size.
        Values may have a different dtype; the compiler converts at the write.
        """
        operator = _Operator(
            "cache_assign",
            _CacheAssignArgs(cache=self._spec, value=value, action="assign"),
            outputs=(),
        )
        _record_cache_access(operator, self._spec, write=True)

    def append(self, value: Tensor) -> None:
        """Append with compiler-inserted storage conversion; sliding evicts on overflow."""
        if self._spec.mode == "overwrite":
            raise ValueError("overwrite caches use assign rather than append")
        operator = _Operator(
            "cache_assign",
            _CacheAssignArgs(cache=self._spec, value=value, action="append"),
            outputs=(),
        )
        _record_cache_access(operator, self._spec, write=True)
