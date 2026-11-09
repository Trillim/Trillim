from __future__ import annotations

import math

from ...tdl.core.core_pydantic_models import (
    _BroadcastArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _PermuteArgs,
    _ReshapeArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ..artifacts.artifact_dataclasses import _Layout
from ..artifacts.expressions import _byte_size
from ..compiler_dataclasses import _ShapeExpression
from ..dtype import _QUANTIZED_DTYPES
from ..kernels.bindings import (
    _broadcast_strides,
    _contiguous_strides,
    _product,
    _reshape_strides,
)
from .infer import _constant_expression, _expression
from .passes_dataclasses import _Selection
from .walk import _walk


def _bind_caches(graph: Graph, layout: _Layout, selection: _Selection) -> None:
    """Use persistent storage directly only when snapshot and alias safety are proved."""
    if graph._region is None:
        raise ValueError("graph is not complete")
    operators = [
        operator for _, operator in _walk(graph._region) if operator in layout.positions
    ]
    writes: dict[str, list[_Operator]] = {}
    for operator in operators:
        if isinstance(operator.args, _CacheAssignArgs):
            writes.setdefault(operator.args.cache.name, []).append(operator)
    loop_ends: dict[_Operator, list[tuple[int, int, set[str]]]] = {}

    def scopes(
        region_operators: tuple[_Operator, ...],
        enclosing: list[tuple[int, int, set[str]]],
    ) -> None:
        for operator in region_operators:
            loop_ends[operator] = enclosing
            nested = enclosing
            if operator.op == "while" and operator in layout.positions:
                first, last = layout.positions[operator]
                nested = [
                    *enclosing,
                    (
                        first,
                        last,
                        {
                            cache.name
                            for body in operator.regions
                            for cache in body.writes
                        },
                    ),
                ]
            for body in operator.regions:
                scopes(body.operators, nested)

    scopes(graph._region.operators, [])
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}
    for operator in operators:
        args = operator.args
        if not isinstance(args, _CacheReadArgs) or isinstance(args, _CacheAssignArgs):
            continue
        cache = args.cache
        buffer = layout.buffers[operator.outputs[0]]
        if cache.mode == "sliding" or cache.dtype in _QUANTIZED_DTYPES:
            continue
        if any(
            layout.positions[write][0] >= buffer.first
            and layout.positions[write][0] <= buffer.last
            for write in writes.get(cache.name, ())
        ):
            continue
        if any(
            cache.name in modified and buffer.last >= end
            for _, end, modified in loop_ends[operator]
        ):
            continue
        shape = tuple(
            _constant_expression(dimension)
            for dimension in cache.shape.dimensions
            if dimension is not None
        )
        changed = {buffer.name: _contiguous_strides(shape)}
        compatible = True
        for _, view in _walk(graph._region):
            for output in view.outputs:
                alias = layout.buffers[output]
                if alias.storage != "alias" or alias.source not in changed:
                    continue
                original = by_name[alias.source]
                strides = changed[alias.source]
                if isinstance(view.args, _ReshapeArgs):
                    reshaped = _reshape_strides(original.shape, strides, alias.shape)
                    if reshaped is None:
                        compatible = False
                        break
                    strides = reshaped
                elif isinstance(view.args, _PermuteArgs):
                    strides = tuple(strides[axis] for axis in view.args.axes)
                elif isinstance(view.args, _BroadcastArgs):
                    strides = _broadcast_strides(original.shape, alias.shape, strides)
                changed[alias.name] = strides
            if not compatible:
                break
        if not compatible:
            continue
        buffer.storage, buffer.source = "cache", cache.name
        storage_shape = (
            (shape[0], _product(shape[1:]))
            if len(shape) > 1
            else (_constant_expression(1), _product(shape))
        )
        for name, strides in changed.items():
            by_name[name].strides = strides
            by_name[name].storage_shape = storage_shape
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}

    tensors_by_name = {buffer.name: tensor for tensor, buffer in layout.buffers.items()}

    def cache_source(name: str) -> str | None:
        buffer = by_name[name]
        while buffer.storage == "alias" and buffer.source is not None:
            buffer = by_name[buffer.source]
        return buffer.source if buffer.storage == "cache" else None

    for operator in operators:
        args = operator.args
        if (
            not isinstance(args, _CacheAssignArgs)
            or args.cache.dtype in _QUANTIZED_DTYPES
            or args.cache.mode == "sliding"
        ):
            continue
        source = args.value
        source_buffer = layout.buffers[source]
        buffer = source_buffer
        while buffer.storage == "alias" and buffer.source is not None:
            buffer = by_name[buffer.source]
        output = tensors_by_name[buffer.name]
        producer = output._producer
        call = selection.calls.get(producer) if producer is not None else None
        if call is None or buffer.last > layout.positions[operator][0]:
            continue
        if (
            buffer.storage != "arena"
            or buffer.dtype != args.cache.dtype
            or source_buffer.byte_offset is not None
            or source_buffer.strides != _contiguous_strides(source_buffer.shape)
        ):
            continue
        capacity_shape = tuple(
            dimension
            for dimension in args.cache.shape.dimensions
            if dimension is not None
        )
        axis = args.cache.axis
        if args.action == "assign" and source_buffer.maximum_shape != capacity_shape:
            continue
        if args.action == "append":
            if axis is None or math.prod(capacity_shape[:axis]) != 1:
                continue
            if (
                any(
                    dimension != capacity_shape[index]
                    for index, dimension in enumerate(source_buffer.maximum_shape)
                    if index != axis
                )
                or source_buffer.maximum_shape[axis] > capacity_shape[axis]
            ):
                continue
            slot = call.outputs.index(output)
            alignment = call.kernel.parameter_schema.output_alignments[slot]
            if (
                alignment > 64
                or _byte_size(args.cache.dtype, capacity_shape[axis + 1 :]) % alignment
            ):
                continue
        if any(
            cache_source(layout.buffers[tensor].name) == args.cache.name
            for tensor in call.inputs
        ):
            continue
        start, stop = layout.positions[call.anchor][0], layout.positions[operator][0]
        if any(
            other.storage == "cache"
            and other.source == args.cache.name
            and other.first <= stop
            and other.last >= start
            for other in layout.buffers.values()
        ):
            continue
        if any(
            isinstance(other.args, _CacheReadArgs)
            and not isinstance(other.args, _CacheAssignArgs)
            and other.args.cache.name == args.cache.name
            and start <= layout.positions[other][0] < stop
            for other in operators
        ):
            continue
        if any(
            start <= layout.positions[other][0] < stop
            for other in writes.get(args.cache.name, ())
            if other is not operator
        ):
            continue
        buffer.storage, buffer.source = "cache", args.cache.name
        if args.action == "append":
            assert axis is not None
            capacity = capacity_shape[axis]
            length = _ShapeExpression(
                "cache_length", cache=args.cache.name, minimum=0, maximum=capacity
            )
            buffer.byte_offset = _expression(
                "mul",
                length,
                _constant_expression(
                    _byte_size(args.cache.dtype, capacity_shape[axis + 1 :])
                ),
            )
            layout.cache_guards.setdefault(call.anchor, []).append(
                (_expression("add", length, source_buffer.shape[axis]), capacity)
            )
        layout.direct_writes.add(operator)
