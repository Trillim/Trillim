from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from ...tdl.core.core_dataclasses import Region
from ...tdl.core.core_pydantic_models import (
    CacheArgs,
    CacheAssignArgs,
    CacheReadArgs,
    ConstantArgs,
    ConstructArgs,
    IfArgs,
    InputArgs,
    OutputArgs,
    ReduceArgs,
    WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import Operator
from ...tdl.core.tensor import Tensor
from ..artifacts import transfers
from ..artifacts.artifact_dataclasses import JSON, Expressions, Layout
from ..artifacts.artifact_pydantic_models import ExecutableManifest
from ..artifacts.expressions import align, byte_size, number, references, serialize
from ..compiler_dataclasses import ShapeConstraint, ShapeExpression
from ..compiler_pydantic_models import CompileRequest
from ..kernels.kernel_dataclasses import ParameterValue
from .passes_dataclasses import Selection


def _parameter(
    value: ParameterValue, names: dict[Tensor, str], expressions: Expressions
) -> JSON:
    if isinstance(value, ShapeExpression):
        return serialize(value, names, expressions)
    if isinstance(value, tuple):
        return [_parameter(item, names, expressions) for item in value]
    return number(value)


def _program(
    region: Region,
    layout: Layout,
    selection: Selection,
    names: dict[Tensor, str],
    guards: Mapping[Operator, list[ShapeConstraint]],
    expressions: Expressions,
) -> list[JSON]:
    result: list[JSON] = []

    def tensors(values: tuple[Tensor, ...]) -> list[JSON]:
        return [names[tensor] for tensor in values]

    def copies(operator: Operator, phase: str) -> list[JSON]:
        kernel = layout.transfer_kernel
        if kernel is None:
            raise ValueError("control flow requires a native transfer kernel")
        workspace = layout.transfer_scratch[operator]
        return [
            {
                "op": "call",
                "kernel": kernel.name,
                "inputs": [names[source]],
                "outputs": [names[destination]],
                "shape_from": {names[destination]: names[source]},
                "parameters": {
                    key: _parameter(value, names, expressions)
                    for key, value in transfers.parameters(
                        layout, source, destination
                    ).items()
                },
                "scratch": {"offset": workspace.offset, "size": workspace.size},
            }
            for source, destination in transfers.pairs(operator)[phase]
        ]

    for operator in region.operators:
        call = selection.calls.get(operator)
        if operator in selection.covered and call is None:
            continue
        args = operator.args
        owned = call.covered if call else (operator,)
        before: list[JSON] = []
        after: list[JSON] = []
        for owner in owned:
            for constraint in guards.get(owner, ()):
                check: JSON = {
                    "op": "guard",
                    "predicate": constraint.predicate,
                    "left": serialize(constraint.left, names, expressions),
                    "right": serialize(constraint.right, names, expressions),
                    "reason": constraint.reason,
                }
                dependencies = references(constraint.left) | references(
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
                        "left": serialize(length, names, expressions),
                        "right": capacity,
                        "reason": "cache capacity",
                    }
                )
            workspace = layout.scratch[operator]
            result.append(
                {
                    "op": "call",
                    "kernel": call.kernel.name,
                    "inputs": tensors(call.inputs),
                    "outputs": tensors(call.outputs),
                    "parameters": {
                        key: _parameter(value, names, expressions)
                        for key, value in layout.parameters[operator].items()
                    },
                    "scratch": {"offset": workspace.offset, "size": workspace.size},
                }
            )
        elif isinstance(args, ConstructArgs):
            result.extend(
                _program(
                    operator.regions[0], layout, selection, names, guards, expressions
                )
            )
            result.append({"op": "views", "outputs": tensors(operator.outputs)})
        elif isinstance(args, (InputArgs, OutputArgs, ConstantArgs, CacheArgs)):
            continue
        elif isinstance(args, IfArgs):
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
        elif isinstance(args, WhileArgs):
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
        elif isinstance(args, ReduceArgs):
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
        elif isinstance(args, CacheReadArgs) and not isinstance(args, CacheAssignArgs):
            result.append(
                {
                    "op": "cache_read",
                    "cache": args.cache.name,
                    "output": names[operator.outputs[0]],
                    "snapshot": layout.buffers[operator.outputs[0]].storage != "cache",
                }
            )
        elif isinstance(args, CacheAssignArgs):
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


def emit(
    graph: Graph,
    layout: Layout,
    selection: Selection,
    request: CompileRequest,
    destination: Path,
) -> ExecutableManifest:
    if graph.region is None:
        raise ValueError("graph is not complete")
    expressions = Expressions()
    names = {tensor: buffer.name for tensor, buffer in layout.buffers.items()}
    kernels: dict[str, JSON] = {}
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
                raise ValueError("different libraries claim the same backend")
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
            "hardware": list(kernel.capabilities.hardware),
            "software": list(kernel.capabilities.software),
        }
    cache_offset = 0
    caches: dict[str, JSON] = {}
    initial = {
        operator.args.name: operator.args.initial
        for operator in graph.region.operators
        if isinstance(operator.args, CacheArgs)
    }
    for name, cache in graph.caches.items():
        shape = tuple(
            dimension for dimension in cache.shape.dimensions if dimension is not None
        )
        size = byte_size(cache.dtype, shape)
        cache_offset = align(cache_offset, 64)
        caches[name] = {
            "dtype": cache.dtype,
            "shape": list(shape),
            "mode": cache.mode,
            "axis": cache.axis,
            "initial": number(initial[name]),
            "offset": cache_offset,
            "size": size,
        }
        cache_offset += size
    buffers: dict[str, JSON] = {}
    for buffer in layout.buffers.values():
        if buffer.first > buffer.last:
            continue
        buffers[buffer.name] = {
            "dtype": buffer.dtype,
            "shape": [serialize(item, names, expressions) for item in buffer.shape],
            "maximum_shape": list(buffer.maximum_shape),
            "strides": [serialize(item, names, expressions) for item in buffer.strides],
            "storage": buffer.storage,
            "source": buffer.source,
            "offset": buffer.offset,
            "size": buffer.size,
            "alignment": buffer.alignment,
            "first": buffer.first,
            "last": buffer.last,
            "byte_offset": serialize(buffer.byte_offset, names, expressions)
            if buffer.byte_offset is not None
            else 0,
        }
    guards: dict[Operator, list[ShapeConstraint]] = {}
    for constraint in layout.inference.constraints:
        if constraint.operator is not None:
            guards.setdefault(constraint.operator, []).append(constraint)
    manifest = ExecutableManifest.model_validate(
        {
            "backend": request.backend,
            "threads": request.threads,
            "arena_size": layout.arena_size,
            "cache_size": layout.cache_size,
            "inputs": {name: names[tensor] for name, tensor in graph.inputs.items()},
            "outputs": {name: names[tensor] for name, tensor in graph.outputs.items()},
            "buffers": buffers,
            "caches": caches,
            "kernels": kernels,
            "program": _program(
                graph.region, layout, selection, names, guards, expressions
            ),
            "expressions": expressions.nodes,
            "transfer_kernel": layout.transfer_kernel.name
            if layout.transfer_kernel is not None
            else None,
        }
    )
    (destination / "constants.bin").write_bytes(layout.constants)
    (destination / "executable.json").write_text(manifest.model_dump_json() + "\n")
    return manifest
