from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    Weight,
    _BroadcastArgs,
    _ConstantArgs,
    _ConstructArgs,
    _InputArgs,
    _PermuteArgs,
    _ReduceArgs,
    _ReshapeArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.tensor import Tensor
from ..artifacts import transfers as _transfers
from ..artifacts.artifact_dataclasses import _Buffer, _Layout, _Workspace
from ..artifacts.expressions import (
    _align,
    _byte_size,
    _parameter_references,
    _references,
)
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import _Inference, _ShapeExpression
from ..dtype import _encode
from ..kernels.bindings import (
    _broadcast_strides,
    _contiguous_strides,
    _product,
    _reshape_strides,
)
from ..kernels.kernel_dataclasses import _Kernel, _KernelBindingContext, _TensorBinding
from ..weights.weights_dataclasses import _WeightIndex
from ..weights.weights_pydantic_models import _WeightManifest
from .cache import _bind_caches
from .infer import _constant_expression
from .passes_dataclasses import _Selection
from .reuse import _bind_loops, _inplace, _reuse_loops
from .walk import _walk


def _schedule(
    region: _Region,
    selection: _Selection,
    layout: _Layout,
    live: set[Tensor],
    clock: list[int],
) -> None:
    def touch(tensors: Iterable[Tensor], position: int) -> None:
        for tensor in tensors:
            if tensor not in layout.buffers:
                continue
            live.add(tensor)
            buffer = layout.buffers[tensor]
            buffer.first = min(buffer.first, position)
            buffer.last = max(buffer.last, position)
            buffer.last_read = max(buffer.last_read, position)

    touch(region.parameters, clock[0])
    for operator in region.operators:
        call = selection.calls.get(operator)
        if operator in selection.covered and call is None:
            continue
        start = clock[0]
        clock[0] += 1
        touch(call.bindings.values() if call else operator.inputs, start)
        if call is not None:
            for parameter in call.parameters.values():
                touch(_parameter_references(parameter), start)
        if call is None:
            for body in operator.regions:
                _schedule(body, selection, layout, live, clock)
        stop = clock[0]
        clock[0] += 1
        for constraint in layout.inference.constraints:
            if (
                constraint.operator is operator
                or call is not None
                and constraint.operator in call.covered
            ):
                references = _references(constraint.left) | _references(
                    constraint.right
                )
                touch(
                    references,
                    stop if references & set(operator.outputs) else start,
                )
        touch(operator.outputs, start)
        if (
            operator.regions
            and call is None
            and not isinstance(operator.args, _ConstructArgs)
        ):
            touch(operator.outputs, stop)
        for body in operator.regions if call is None else ():
            touch(body.outputs, stop)
            # Captures must survive every iteration, even after their final textual use.
            touch(body.captures, stop)
            if isinstance(operator.args, _ReduceArgs):
                touch(body.parameters, stop)
        layout.positions[operator] = start, stop
    touch(region.outputs, clock[0])


def _arena(items: list[_Buffer | _Workspace]) -> int:
    pending = sorted(items, key=lambda item: (item.first, -item.size))
    active: list[_Buffer | _Workspace] = []
    free: list[tuple[int, int]] = []
    size = 0
    for item in pending:
        for previous in active[:]:
            if previous.last < item.first:
                free.append((previous.offset, previous.size))
                active.remove(previous)
        merged: list[tuple[int, int]] = []
        for offset, length in sorted(free):
            if merged and sum(merged[-1]) == offset:
                begin, old = merged[-1]
                merged[-1] = begin, old + length
            else:
                merged.append((offset, length))
        free = merged
        candidates = [
            (length - item.size, index, _align(offset, item.alignment))
            for index, (offset, length) in enumerate(free)
            if _align(offset, item.alignment) + item.size <= offset + length
        ]
        if candidates:
            _, index, begin = min(candidates)
            offset, length = free.pop(index)
            if begin > offset:
                free.append((offset, begin - offset))
            if begin + item.size < offset + length:
                free.append((begin + item.size, offset + length - begin - item.size))
            item.offset = begin
        else:
            item.offset = _align(size, item.alignment)
            size = item.offset + item.size
        active.append(item)
    return _align(size, 64)


def _views(region: _Region, selection: _Selection, layout: _Layout) -> None:
    for _, operator in _walk(region):
        if operator in selection.calls:
            continue
        args = operator.args
        source: Tensor | None = None
        if isinstance(args, _ConstructArgs):
            pairs = zip(operator.outputs, operator.regions[0].outputs, strict=True)
            for output, body_output in pairs:
                result, original = layout.buffers[output], layout.buffers[body_output]
                result.storage, result.source = "alias", original.name
                result.strides, result.storage_shape, result.size = (
                    original.strides,
                    original.storage_shape,
                    original.size,
                )
            continue
        if isinstance(args, (_PermuteArgs, _BroadcastArgs, _ReshapeArgs)):
            source = args.x
        if source is None:
            continue
        result, original = layout.buffers[operator.outputs[0]], layout.buffers[source]
        reshaped = (
            _reshape_strides(original.shape, original.strides, result.shape)
            if isinstance(args, _ReshapeArgs)
            else None
        )
        if isinstance(args, _ReshapeArgs) and reshaped is None:
            result.storage, result.source = "arena", None
            result.strides = _contiguous_strides(result.shape)
            result.storage_shape = (
                (result.shape[0], _product(result.shape[1:]))
                if len(result.shape) > 1
                else (_constant_expression(1), _product(result.shape))
            )
            result.size = _byte_size(result.dtype, result.maximum_shape)
            continue
        result.storage, result.source = "alias", original.name
        result.size, result.storage_shape = original.size, original.storage_shape
        if isinstance(args, _PermuteArgs):
            result.strides = tuple(original.strides[index] for index in args.axes)
        elif isinstance(args, _ReshapeArgs):
            assert reshaped is not None
            result.strides = reshaped
        elif isinstance(args, _BroadcastArgs):
            result.strides = _broadcast_strides(
                original.shape, result.shape, original.strides
            )


def _lifetimes(layout: _Layout, live: set[Tensor]) -> None:
    by_name = {buffer.name: tensor for tensor, buffer in layout.buffers.items()}
    dependencies: dict[Tensor, set[Tensor]] = {}
    value_dependencies: dict[Tensor, set[Tensor]] = {}
    for tensor, buffer in layout.buffers.items():
        values = (
            *buffer.shape,
            *buffer.strides,
            *((buffer.byte_offset,) if buffer.byte_offset is not None else ()),
        )
        related: set[Tensor] = set()
        for value in values:
            related.update(_references(value))
        if buffer.storage == "alias":
            if buffer.source is None or buffer.source not in by_name:
                raise ValueError("view has no backing storage")
            related.add(by_name[buffer.source])
        related.discard(tensor)
        if not related <= layout.buffers.keys():
            raise ValueError("tensor metadata references an eliminated tensor")
        dependencies[tensor] = related
        # Dimensions can outlive tensor contents; scalar expressions still read data.
        pending_values = list(values)
        visited_values: set[_ShapeExpression] = set()
        dimensions: set[Tensor] = set()
        scalar_reads: set[Tensor] = set()
        while pending_values:
            value = pending_values.pop()
            if value in visited_values:
                continue
            visited_values.add(value)
            pending_values.extend(value.args)
            if value.op == "scalar" and value.tensor is not None:
                scalar_reads.add(value.tensor)
            elif (
                value.op == "dimension"
                and value.tensor is not None
                and value.tensor not in dimensions
            ):
                dimensions.add(value.tensor)
                pending_values.extend(layout.buffers[value.tensor].shape)
        scalar_reads.discard(tensor)
        value_dependencies[tensor] = scalar_reads
    pending = deque(live)
    queued = set(live)
    while pending:
        tensor = pending.popleft()
        queued.discard(tensor)
        buffer = layout.buffers[tensor]
        for dependency in dependencies[tensor]:
            target = layout.buffers[dependency]
            first, last = min(target.first, buffer.first), max(target.last, buffer.last)
            if (first, last) != (target.first, target.last):
                target.first, target.last = first, last
                live.add(dependency)
                if dependency not in queued:
                    pending.append(dependency)
                    queued.add(dependency)
        for dependency in value_dependencies[tensor]:
            target = layout.buffers[dependency]
            if target.last_read < buffer.last_read:
                target.last_read = buffer.last_read
                live.add(dependency)
                if dependency not in queued:
                    pending.append(dependency)
                    queued.add(dependency)


def _allocate(
    graph: Graph,
    inference: _Inference,
    selection: _Selection,
    weights: _WeightIndex,
    manifest: _WeightManifest,
    *,
    transfer_kernel: _Kernel | None = None,
) -> _Layout:
    if graph._region is None:
        raise ValueError("graph is not complete")
    layout = _Layout(inference, transfer_kernel=transfer_kernel)
    tensors: dict[Tensor, None] = {}
    for region, operator in _walk(graph._region):
        tensors.update(dict.fromkeys(region.parameters))
        tensors.update(dict.fromkeys(operator.outputs))
    constants = bytearray()
    for index, tensor in enumerate(tensors):
        info = inference.tensors[tensor]
        if (
            info.dtype is None
            or info.shape is None
            or info.maximum_shape is None
            or any(dimension is None for dimension in info.maximum_shape)
        ):
            raise ValueError(f"tensor {index} has unresolved dtype or unbounded shape")
        maximum = tuple(
            int(dimension) for dimension in info.maximum_shape if dimension is not None
        )
        buffer = _Buffer(
            f"t{index}",
            info.dtype,
            info.shape,
            maximum,
            _contiguous_strides(info.shape),
            size=_byte_size(info.dtype, maximum),
            first=2**63 - 1,
            values=info.values,
        )
        if (
            info.values is None
            and info.shape == ()
            and (info.dtype == "bool" or info.dtype.startswith(("int", "uint")))
        ):
            lower, upper = inference.value_ranges.get(tensor, (None, None))
            if lower is not None and upper is not None:
                buffer.values = (
                    _ShapeExpression(
                        "scalar", tensor=tensor, minimum=lower, maximum=upper
                    ),
                )
        buffer.storage_shape = (
            (info.shape[0], _product(info.shape[1:]))
            if len(info.shape) > 1
            else (_constant_expression(1), _product(info.shape))
        )
        layout.buffers[tensor] = buffer
        producer = tensor._producer
        if producer is None:
            continue
        args = producer.args
        if isinstance(args, _InputArgs):
            buffer.storage, buffer.source = "input", args.name
        elif isinstance(args, _ConstantArgs):
            if isinstance(args.value, Weight):
                binding = weights.bindings[tensor]
                entry = next(
                    (
                        record
                        for record in manifest.entries.values()
                        if record.logical == binding.logical
                        and record.dtype == binding.dtype
                    ),
                    None,
                )
                if entry is None:
                    raise ValueError(
                        f"weight representation missing: {binding.logical}"
                    )
                buffer.storage, buffer.source, buffer.offset, buffer.size = (
                    "weight",
                    entry.name,
                    entry.offset,
                    entry.size,
                )
            else:
                buffer.storage = "constant"
                buffer.offset = _align(len(constants), 64)
                constants.extend(bytes(buffer.offset - len(constants)))
                payload = _encode(np.asarray(args.value), info.dtype)[0].tobytes()
                buffer.size = len(payload)
                constants.extend(payload)
    _views(graph._region, selection, layout)
    live: set[Tensor] = set()
    _schedule(graph._region, selection, layout, live, [0])
    _lifetimes(layout, live)
    _bind_caches(graph, layout, selection)
    _views(graph._region, selection, layout)
    _lifetimes(layout, live)
    _bind_loops(graph._region, layout, selection)
    _views(graph._region, selection, layout)
    _lifetimes(layout, live)
    for operator, call in selection.calls.items():
        context = _KernelBindingContext(
            {
                name: layout.buffers[tensor]._binding()
                for name, tensor in call.bindings.items()
            },
            tuple(layout.buffers[tensor]._binding() for tensor in call.outputs),
            dict(vars(operator.args)),
        )
        if isinstance(operator.args, _ConstructArgs):
            context = _KernelBindingContext(
                context.inputs, context.outputs, operator.args.arguments
            )
        parameters = call.kernel.bind(context)
        required = _requirements(call.kernel.parameter_schema, parameters)
        for tensor, size in zip(call.inputs, required.inputs, strict=True):
            if size > layout.buffers[tensor].size:
                raise ValueError(
                    f"{call.kernel.name}: input span exceeds allocated capacity"
                )
        for tensor, size in zip(call.outputs, required.outputs, strict=True):
            buffer = layout.buffers[tensor]
            buffer.size = max(buffer.size, size)
        layout.parameters[operator] = parameters
        for parameter in parameters.values():
            for tensor in _parameter_references(parameter):
                if tensor not in layout.buffers:
                    raise ValueError("kernel parameter references an eliminated tensor")
                live.add(tensor)
                buffer = layout.buffers[tensor]
                buffer.first = min(buffer.first, layout.positions[operator][0])
                buffer.last = max(buffer.last, layout.positions[operator][1])
                buffer.last_read = max(buffer.last_read, layout.positions[operator][0])
        start, stop = layout.positions[operator]
        layout.scratch[operator] = _Workspace(
            0,
            required.scratch,
            call.kernel.parameter_schema.scratch_alignment,
            start,
            stop,
        )
    _lifetimes(layout, live)
    _reuse_loops(graph._region, layout, selection)
    if transfer_kernel is not None:
        for _, operator in _walk(graph._region):
            if (
                operator not in layout.positions
                or operator in selection.calls
                or not operator.regions
                or isinstance(operator.args, _ConstructArgs)
            ):
                continue
            scratch = 0
            for phase in _transfers._pairs(operator).values():
                for source, destination in phase:
                    if (source, destination) in layout.shared_transfers:
                        continue
                    required = _requirements(
                        transfer_kernel.parameter_schema,
                        _transfers._parameters(layout, source, destination),
                    )
                    scratch = max(scratch, required.scratch)
            if isinstance(operator.args, _ReduceArgs):
                for source in (*operator.args.values, *operator.args.initial):
                    copy_source = layout.buffers[source]._binding()
                    scalar = _TensorBinding((), (), copy_source.dtype)
                    sliced = _TensorBinding(
                        (),
                        (),
                        copy_source.dtype,
                        storage_shape=copy_source.storage_shape,
                    )
                    parameters = transfer_kernel.bind(
                        _KernelBindingContext({"x": sliced}, (scalar,), {})
                    )
                    scratch = max(
                        scratch,
                        _requirements(
                            transfer_kernel.parameter_schema, parameters
                        ).scratch,
                    )
            first, last = layout.positions[operator]
            layout.transfer_scratch[operator] = _Workspace(
                0,
                scratch,
                transfer_kernel.parameter_schema.scratch_alignment,
                first,
                last,
            )
    _lifetimes(layout, live)
    _inplace(layout, selection)
    _lifetimes(layout, live)
    items: list[_Buffer | _Workspace] = [
        buffer
        for buffer in layout.buffers.values()
        if buffer.storage == "arena" and buffer.last >= buffer.first
    ]
    items.extend(workspace for workspace in layout.scratch.values() if workspace.size)
    items.extend(
        workspace for workspace in layout.transfer_scratch.values() if workspace.size
    )
    layout.arena_size = _arena(items)
    for cache in graph._caches.values():
        shape = tuple(
            dimension for dimension in cache.shape.dimensions if dimension is not None
        )
        layout.cache_size = _align(layout.cache_size, 64) + _byte_size(
            cache.dtype, shape
        )
    layout.cache_size = _align(layout.cache_size, 64)
    layout.constants = bytes(constants)
    return layout
