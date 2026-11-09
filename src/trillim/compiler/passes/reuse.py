from __future__ import annotations

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import _WhileArgs
from ...tdl.core.tensor import Tensor
from ..artifacts.artifact_dataclasses import _Buffer, _Layout
from ..artifacts.parameters import _can_alias
from ..dtype import _QUANTIZED_DTYPES
from ..kernels.bindings import _contiguous_strides
from .passes_dataclasses import _Selection
from .walk import _walk


def _backing(buffer: _Buffer, by_name: dict[str, _Buffer]) -> _Buffer:
    while buffer.storage == "alias":
        assert buffer.source is not None
        buffer = by_name[buffer.source]
    return buffer


def _bind_loops(region: _Region, layout: _Layout, selection: _Selection) -> None:
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}
    for _, operator in _walk(region):
        if (
            not isinstance(operator.args, _WhileArgs)
            or operator not in layout.positions
            or operator in selection.covered
        ):
            continue
        condition, body = operator.regions
        invariant = all(
            layout.inference.tensors[result].shape
            == layout.inference.tensors[parameter].shape
            for parameter, result in zip(body.parameters, body.outputs, strict=True)
        )
        snapshots: set[Tensor] = set()
        for index, parameter in enumerate(body.parameters):
            original = layout.buffers[parameter]
            for slot, result in enumerate(body.outputs):
                result_buffer = layout.buffers[result]
                if _backing(result_buffer, by_name) is original and (
                    slot != index
                    or result_buffer.shape != original.shape
                    or result_buffer.strides != original.strides
                    or result_buffer.byte_offset is not None
                ):
                    snapshots.add(parameter)
        claimed: set[str] = set()
        for initial, output, test, parameter in zip(
            operator.args.initial,
            operator.outputs,
            condition.parameters,
            body.parameters,
            strict=True,
        ):
            target = layout.buffers[output]
            original = layout.buffers[initial]
            root = _backing(original, by_name)
            if (
                invariant
                and root.storage == "arena"
                and root.name not in claimed
                and root.last == layout.positions[operator][0]
                and root.size >= target.size
                and original.dtype == target.dtype
                and original.maximum_shape == target.maximum_shape
                and original.strides == _contiguous_strides(original.shape)
                and original.byte_offset is None
            ):
                target.storage, target.source = "alias", original.name
                target.shape, target.strides = original.shape, original.strides
                target.storage_shape = original.storage_shape
                root.last = max(root.last, target.last)
                root.alignment = max(root.alignment, target.alignment)
                claimed.add(root.name)
                layout.shared_transfers.add((initial, output))
            for value in (test, parameter):
                if value is parameter and (not invariant or value in snapshots):
                    continue
                buffer = layout.buffers[value]
                buffer.storage, buffer.source = "alias", target.name
                buffer.shape, buffer.strides = target.shape, target.strides
                buffer.storage_shape = target.storage_shape
                buffer.size = target.size
                layout.shared_transfers.add((output, value))


def _reuse_loops(region: _Region, layout: _Layout, selection: _Selection) -> None:
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}
    by_tensor = {buffer.name: tensor for tensor, buffer in layout.buffers.items()}
    for _, operator in reversed(tuple(_walk(region))):
        if (
            not isinstance(operator.args, _WhileArgs)
            or operator not in layout.positions
            or operator in selection.covered
        ):
            continue
        body = operator.regions[1]
        local: set[Tensor] = set(body.parameters)
        for scope, node in _walk(body):
            local.update(scope.parameters)
            local.update(node.outputs)
        for parameter, result, output in zip(
            body.parameters, body.outputs, operator.outputs, strict=True
        ):
            if (
                layout.inference.tensors[result].shape
                != layout.inference.tensors[parameter].shape
            ):
                continue
            value, target = layout.buffers[result], layout.buffers[output]
            root, bank = _backing(value, by_name), _backing(target, by_name)
            if (
                value.dtype != target.dtype
                or value.strides != _contiguous_strides(value.shape)
                or value.byte_offset is not None
            ):
                continue
            if root is bank:
                layout.shared_transfers.add((result, output))
                continue
            tensor = by_tensor[root.name]
            producer = tensor._producer
            call = selection.calls.get(producer) if producer is not None else None
            if (
                call is None
                or root.storage != "arena"
                or bank.storage != "arena"
                or root.dtype in _QUANTIZED_DTYPES
                or root.size > bank.size
                or layout.positions[call.anchor][0]
                < min(layout.buffers[p].first for p in body.parameters)
            ):
                continue
            start = layout.positions[call.anchor][0]
            if any(
                layout.buffers[old].last_read > start
                and _backing(layout.buffers[old], by_name) is bank
                for old in local
            ):
                continue
            slot = call.outputs.index(tensor)
            compatible = True
            for input_slot, source in enumerate(call.inputs):
                original = layout.buffers[source]
                if _backing(original, by_name) is not bank:
                    continue
                if (
                    original.dtype != root.dtype
                    or layout.inference.tensors[source].shape
                    != layout.inference.tensors[tensor].shape
                    or original.strides != _contiguous_strides(original.shape)
                    or original.byte_offset is not None
                    or not _can_alias(
                        call.kernel.parameter_schema,
                        layout.parameters[call.anchor],
                        slot,
                        input_slot,
                    )
                ):
                    compatible = False
                    break
            if not compatible:
                continue
            root.storage, root.source = "alias", target.name
            bank.alignment = max(
                bank.alignment,
                root.alignment,
                call.kernel.parameter_schema.output_alignments[slot],
            )
            layout.shared_transfers.add((result, output))


def _inplace(layout: _Layout, selection: _Selection) -> None:
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}

    for operator, call in sorted(
        selection.calls.items(), key=lambda item: layout.positions[item[0]][0]
    ):
        schema = call.kernel.parameter_schema
        if schema.can_alias_address is None:
            continue
        start = layout.positions[operator][0]
        claimed: set[str] = set()
        for output_slot, tensor in enumerate(call.outputs):
            output = layout.buffers[tensor]
            if output.storage != "arena" or output.first != start:
                continue
            for source in call.inputs:
                original = layout.buffers[source]
                root = _backing(original, by_name)
                if (
                    root.storage != "arena"
                    or root.dtype in _QUANTIZED_DTYPES
                    or root.name in claimed
                    or root.last != start
                    or root.size < output.size
                    or original.dtype != output.dtype
                    or original.shape != output.shape
                    or original.strides != _contiguous_strides(output.shape)
                    or original.byte_offset is not None
                ):
                    continue
                overlapping = [
                    slot
                    for slot, value in enumerate(call.inputs)
                    if _backing(layout.buffers[value], by_name) is root
                ]
                if not all(
                    layout.buffers[call.inputs[slot]].dtype == output.dtype
                    and layout.buffers[call.inputs[slot]].shape == output.shape
                    and layout.buffers[call.inputs[slot]].strides
                    == _contiguous_strides(output.shape)
                    and layout.buffers[call.inputs[slot]].byte_offset is None
                    and _can_alias(
                        schema, layout.parameters[operator], output_slot, slot
                    )
                    for slot in overlapping
                ):
                    continue
                output.storage, output.source = "alias", root.name
                root.last = max(root.last, output.last)
                root.alignment = max(
                    root.alignment,
                    output.alignment,
                    schema.output_alignments[output_slot],
                )
                claimed.add(root.name)
                break
