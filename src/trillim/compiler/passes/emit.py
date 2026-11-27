from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    _CacheArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _ConstantArgs,
    _ConstructArgs,
    _IfArgs,
    _InputArgs,
    _OutputArgs,
    _ReduceArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..artifacts import transfers as _transfers
from ..artifacts.artifact_dataclasses import _JSON, _Expressions, _Layout
from ..artifacts.artifact_pydantic_models import _ExecutableManifest
from ..artifacts.expressions import _align, _byte_size, _number, _references, _serialize
from ..compiler_dataclasses import _ShapeConstraint, _ShapeExpression
from ..compiler_pydantic_models import _CompileRequest
from ..kernels.kernel_dataclasses import _ParameterValue
from ..kernels.native import _CAPABILITY_FLAGS
from .cleanup import _cleanup
from .optimize import _observable
from .passes_dataclasses import _Selection


def _parameter(
    value: _ParameterValue, names: dict[Tensor, str], expressions: _Expressions
) -> _JSON:
    if isinstance(value, _ShapeExpression):
        return _serialize(value, names, expressions)
    if isinstance(value, tuple):
        return [_parameter(item, names, expressions) for item in value]
    return _number(value)


def _program(
    region: _Region,
    layout: _Layout,
    selection: _Selection,
    names: dict[Tensor, str],
    guards: Mapping[_Operator, list[_ShapeConstraint]],
    expressions: _Expressions,
) -> list[_JSON]:
    result: list[_JSON] = []
    by_name = {buffer.name: buffer for buffer in layout.buffers.values()}

    def tensors(values: tuple[Tensor, ...]) -> list[_JSON]:
        return [names[tensor] for tensor in values]

    def copies(operator: _Operator, phase: str) -> list[_JSON]:
        kernel = layout.transfer_kernel
        if kernel is None:
            raise ValueError("control flow requires a native transfer kernel")
        workspace = layout.transfer_scratch[operator]
        result: list[_JSON] = []
        for source, destination in _transfers._pairs(operator)[phase]:
            if (source, destination) in layout.shared_transfers:
                if layout.buffers[destination].storage == "alias":
                    result.append({"op": "views", "outputs": [names[destination]]})
                continue
            result.append(
                {
                    "op": "call",
                    "kernel": kernel.name,
                    "inputs": [names[source]],
                    "outputs": [names[destination]],
                    "shape_from": {names[destination]: names[source]},
                    "parameters": {
                        key: _parameter(value, names, expressions)
                        for key, value in _transfers._parameters(
                            layout, source, destination
                        ).items()
                    },
                    "scratch": {"offset": workspace.offset, "size": workspace.size},
                }
            )
        return result

    for operator in region.operators:
        call = selection.calls.get(operator)
        if operator in selection.covered and call is None:
            continue
        args = operator.args
        owned = call.covered if call else (operator,)
        before: list[_JSON] = []
        after: list[_JSON] = []
        for owner in owned:
            for constraint in guards.get(owner, ()):
                check: _JSON = {
                    "op": "guard",
                    "predicate": constraint.predicate,
                    "left": _serialize(constraint.left, names, expressions),
                    "right": _serialize(constraint.right, names, expressions),
                    "reason": constraint.reason,
                }
                dependencies = _references(constraint.left) | _references(
                    constraint.right
                )
                (after if dependencies & set(operator.outputs) else before).append(
                    check
                )
        result.extend(before)
        if call is not None:
            for length, capacity in layout.cache_guards.get(operator, ()):
                result.append(
                    {
                        "op": "guard",
                        "predicate": "le",
                        "left": _serialize(length, names, expressions),
                        "right": capacity,
                        "reason": "cache capacity",
                    }
                )
            workspace = layout.scratch[operator]
            parameters = layout.parameters[operator]
            argument_key = dict.fromkeys(
                value
                for field in parameters.values()
                for value in (field if isinstance(field, tuple) else (field,))
                if isinstance(value, _ShapeExpression) and value.op != "const"
            )
            for tensor in (*call.inputs, *call.outputs):
                buffer = layout.buffers[tensor]
                while True:
                    argument_key.update(
                        dict.fromkeys(
                            value
                            for value in (
                                *buffer.shape,
                                *buffer.strides,
                                buffer.byte_offset,
                            )
                            if value is not None and value.op != "const"
                        )
                    )
                    if buffer.storage != "alias":
                        break
                    assert buffer.source is not None
                    buffer = by_name[buffer.source]
            pending = list(argument_key)
            argument_key.clear()
            visited: set[_ShapeExpression] = set()
            while pending:
                value = pending.pop()
                if value in visited:
                    continue
                visited.add(value)
                if value.args:
                    pending.extend(value.args)
                elif value.op != "const":
                    argument_key[value] = None
            result.append(
                {
                    "op": "call",
                    "kernel": call.kernel.name,
                    "effectful": any(_observable(owner) for owner in owned),
                    "inputs": tensors(call.inputs),
                    "outputs": tensors(call.outputs),
                    "input_packings": [
                        call.packings.get(tensor, "canonical") for tensor in call.inputs
                    ],
                    "parameters": {
                        key: _parameter(value, names, expressions)
                        for key, value in parameters.items()
                    },
                    "argument_key": [
                        _serialize(value, names, expressions) for value in argument_key
                    ],
                    "scratch": {"offset": workspace.offset, "size": workspace.size},
                }
            )
        elif isinstance(args, _ConstructArgs):
            result.extend(
                _program(
                    operator.regions[0], layout, selection, names, guards, expressions
                )
            )
            result.append({"op": "views", "outputs": tensors(operator.outputs)})
        elif isinstance(args, (_InputArgs, _OutputArgs, _ConstantArgs, _CacheArgs)):
            pass
        elif isinstance(args, _IfArgs):
            result.append(
                {
                    "op": "if",
                    "condition": names[args.condition],
                    "outputs": tensors(operator.outputs),
                    "branches": [
                        {
                            "program": _program(
                                body, layout, selection, names, guards, expressions
                            ),
                            "results": tensors(body.outputs),
                            "copies": copies(operator, str(index)),
                        }
                        for index, body in enumerate(operator.regions)
                    ],
                }
            )
        elif isinstance(args, _WhileArgs):
            condition, body = operator.regions
            maximum = layout.inference.loop_iterations.get(operator)
            if maximum is None:
                raise ValueError(
                    "loop has no proved iteration bound; bound its scalar inputs"
                )
            result.append(
                {
                    "op": "while",
                    "initial": tensors(args.initial),
                    "initial_copies": copies(operator, "initial"),
                    "outputs": tensors(operator.outputs),
                    "maximum_iterations": maximum,
                    "condition": {
                        "parameters": tensors(condition.parameters),
                        "copies": copies(operator, "condition"),
                        "program": _program(
                            condition, layout, selection, names, guards, expressions
                        ),
                        "result": names[condition.outputs[0]],
                    },
                    "body": {
                        "parameters": tensors(body.parameters),
                        "copies": copies(operator, "body"),
                        "program": _program(
                            body, layout, selection, names, guards, expressions
                        ),
                        "results": tensors(body.outputs),
                        "updates": copies(operator, "update"),
                    },
                }
            )
        elif isinstance(args, _ReduceArgs):
            body = operator.regions[0]
            result.append(
                {
                    "op": "reduce",
                    "values": tensors(args.values),
                    "initial": tensors(args.initial),
                    "axes": list(args.axes),
                    "keepdims": args.keepdims,
                    "outputs": tensors(operator.outputs),
                    "parameters": tensors(body.parameters),
                    "program": _program(
                        body, layout, selection, names, guards, expressions
                    ),
                    "results": tensors(body.outputs),
                    "transfer": {
                        "kernel": layout.transfer_kernel.name
                        if layout.transfer_kernel is not None
                        else None,
                        "scratch": {
                            "offset": layout.transfer_scratch[operator].offset,
                            "size": layout.transfer_scratch[operator].size,
                        },
                    },
                }
            )
        elif isinstance(args, _CacheReadArgs) and not isinstance(
            args, _CacheAssignArgs
        ):
            result.append(
                {
                    "op": "cache_read",
                    "cache": args.cache.name,
                    "output": names[operator.outputs[0]],
                    "snapshot": layout.buffers[operator.outputs[0]].storage != "cache",
                }
            )
        elif isinstance(args, _CacheAssignArgs):
            result.append(
                {
                    "op": "cache_write",
                    "cache": args.cache.name,
                    "input": names[args.value],
                    "action": args.action,
                    "metadata_only": operator in layout.direct_writes,
                }
            )
        elif operator.outputs and all(
            layout.buffers[tensor].storage == "alias" for tensor in operator.outputs
        ):
            result.append({"op": "views", "outputs": tensors(operator.outputs)})
        else:
            raise ValueError(f"no executable kernel covers {operator.op!r}")
        result.extend(after)
    return result


def _emit(
    graph: Graph,
    layout: _Layout,
    selection: _Selection,
    request: _CompileRequest,
    destination: Path,
) -> _ExecutableManifest:
    if graph._region is None:
        raise ValueError("graph is not complete")
    expressions = _Expressions()
    names = {tensor: buffer.name for tensor, buffer in layout.buffers.items()}
    kernels: dict[str, _JSON] = {}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kernels").mkdir(exist_ok=True)
    libraries: dict[Path, str] = {}
    used_kernels = {call.kernel.name: call.kernel for call in selection.calls.values()}
    if layout.transfer_kernel is not None:
        used_kernels[layout.transfer_kernel.name] = layout.transfer_kernel
    for kernel in used_kernels.values():
        if kernel.library not in libraries:
            filename = f"kernels/{kernel.backend}.so"
            if filename in libraries.values():
                filename = f"kernels/{kernel.backend}-{len(libraries)}.so"
            shutil.copyfile(kernel.library, destination / filename)
            libraries[kernel.library] = filename
        schema = kernel.parameter_schema
        kernels[kernel.name] = {
            "library": libraries[kernel.library],
            "symbol": kernel.name,
            "parameters": {
                "size": schema.size,
                "alignment": schema.alignment,
                "fields": [
                    {
                        "name": item.name,
                        "type": item.type,
                        "offset": item.offset,
                        "length": item.length,
                    }
                    for item in schema.fields
                ],
            },
            "inputs": schema.input_count,
            "outputs": schema.output_count,
            "input_alignments": list(schema.input_alignments),
            "output_alignments": list(schema.output_alignments),
            "scratch_alignment": schema.scratch_alignment,
            "capabilities": kernel.capabilities,
            "features": [
                name
                for name, flag in _CAPABILITY_FLAGS[kernel.backend].items()
                if kernel.capabilities & flag
            ],
        }
    cache_offset = 0
    caches: dict[str, _JSON] = {}
    initial = {
        operator.args.name: operator.args.initial
        for operator in graph._region.operators
        if isinstance(operator.args, _CacheArgs)
    }
    for name, cache in graph._caches.items():
        shape = tuple(
            dimension for dimension in cache.shape.dimensions if dimension is not None
        )
        size = _byte_size(cache.dtype, shape)
        cache_offset = _align(cache_offset, 64)
        caches[name] = {
            "dtype": cache.dtype,
            "shape": list(shape),
            "mode": cache.mode,
            "axis": cache.axis,
            "initial": _number(initial[name]),
            "offset": cache_offset,
            "size": size,
        }
        cache_offset += size
    buffers: dict[str, _JSON] = {}
    for buffer in layout.buffers.values():
        if buffer.first > buffer.last:
            continue
        buffers[buffer.name] = {
            "dtype": buffer.dtype,
            "shape": [_serialize(item, names, expressions) for item in buffer.shape],
            "maximum_shape": list(buffer.maximum_shape),
            "strides": [
                _serialize(item, names, expressions) for item in buffer.strides
            ],
            "storage": buffer.storage,
            "source": buffer.source,
            "offset": buffer.offset,
            "size": buffer.size,
            "alignment": buffer.alignment,
            "first": buffer.first,
            "last": buffer.last,
            "byte_offset": _serialize(buffer.byte_offset, names, expressions)
            if buffer.byte_offset is not None
            else 0,
        }
    guards: dict[_Operator, list[_ShapeConstraint]] = {}
    for constraint in layout.inference.constraints:
        if constraint.operator is not None:
            guards.setdefault(constraint.operator, []).append(constraint)
    manifest = _ExecutableManifest.model_validate(
        {
            "backend": request.backend,
            "threads": request.threads,
            "arena_size": layout.arena_size,
            "cache_size": layout.cache_size,
            "inputs": {name: names[tensor] for name, tensor in graph._inputs.items()},
            "outputs": {name: names[tensor] for name, tensor in graph._outputs.items()},
            "buffers": buffers,
            "caches": caches,
            "kernels": kernels,
            "program": _program(
                graph._region, layout, selection, names, guards, expressions
            ),
            "expressions": expressions.nodes,
            "transfer_kernel": layout.transfer_kernel.name
            if layout.transfer_kernel is not None
            else None,
        }
    )
    manifest = _cleanup(manifest)
    (destination / "constants.bin").write_bytes(layout.constants)
    (destination / "executable.json").write_text(manifest.model_dump_json() + "\n")
    return manifest
