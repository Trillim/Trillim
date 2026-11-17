from __future__ import annotations

from collections.abc import Mapping
from graphlib import TopologicalSorter
from itertools import pairwise
from pathlib import Path

from pydantic import JsonValue

from ..artifacts.artifact_pydantic_models import _BufferRecord, _ExecutableManifest
from ..artifacts.expressions import _byte_size, _dimension_dependencies
from ..weights.bundle import _read_manifest
from ..weights.weights_pydantic_models import _WeightManifest


def _strings(value: JsonValue, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of tensor names")
    return [item for item in value if isinstance(item, str)]


def _block(value: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("invalid instruction block")
    return [item for item in value if isinstance(item, dict)]


def _references(value: JsonValue, expressions: Mapping[str, JsonValue]) -> set[str]:
    pending = [value]
    visited: set[str] = set()
    result: set[str] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            reference = current.get("tensor")
            if isinstance(reference, str):
                result.add(reference)
            name = current.get("expr")
            if isinstance(name, str):
                if name not in expressions:
                    raise ValueError(f"unknown expression: {name}")
                if name not in visited:
                    visited.add(name)
                    pending.append(expressions[name])
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return result


def _expressions(manifest: _ExecutableManifest) -> None:
    dependencies: dict[str, set[str]] = {}
    for name, expression in manifest.expressions.items():
        pending = [expression]
        referenced: set[str] = set()
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                target = current.get("expr")
                if isinstance(target, str):
                    if target not in manifest.expressions:
                        raise ValueError(f"unknown expression: {target}")
                    referenced.add(target)
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        dependencies[name] = referenced
    tuple(TopologicalSorter(dependencies).static_order())


def _program(
    program: list[dict[str, JsonValue]],
    manifest: _ExecutableManifest,
    available: set[str],
    weights: _WeightManifest,
) -> set[str]:
    resolving: set[str] = set()
    dimensions: dict[str, frozenset[str] | None] = {}

    def require(values: list[str]) -> None:
        for name in set(values) - available:
            if name not in manifest.buffers or name in resolving:
                raise ValueError(f"unavailable tensor: {name}")
            buffer = manifest.buffers[name]
            dependencies = _dimension_dependencies(
                [*buffer.shape, *buffer.strides, buffer.byte_offset],
                manifest.expressions,
                dimensions,
            )
            if (
                buffer.storage != "alias"
                or buffer.source is None
                or dependencies is None
            ):
                raise ValueError(f"unavailable tensor: {name}")
            resolving.add(name)
            require([buffer.source, *dependencies])
            resolving.remove(name)
            available.add(name)

    def define(values: list[str]) -> None:
        if not set(values) <= manifest.buffers.keys():
            raise ValueError("instruction defines an unknown tensor")
        available.update(values)

    for instruction in program:
        op = instruction.get("op")
        if op == "guard":
            require(sorted(_references(instruction, manifest.expressions)))
            if instruction.get("predicate") not in {
                "eq",
                "le",
                "ge",
                "divisible",
                "broadcast",
            }:
                raise ValueError("invalid shape guard predicate")
        elif op == "call":
            name = instruction.get("kernel")
            if not isinstance(name, str) or name not in manifest.kernels:
                raise ValueError("instruction references an unknown kernel")
            kernel = manifest.kernels[name]
            inputs, outputs = (
                _strings(instruction.get("inputs"), "inputs"),
                _strings(instruction.get("outputs"), "outputs"),
            )
            require(inputs)
            formats = _strings(
                instruction.get("input_packings", ["canonical"] * len(inputs)),
                "input packings",
            )
            if len(formats) != len(inputs):
                raise ValueError("kernel packing count mismatch")
            for tensor, packing in zip(inputs, formats, strict=True):
                buffer = manifest.buffers[tensor]
                while buffer.storage == "alias":
                    assert buffer.source is not None
                    buffer = manifest.buffers[buffer.source]
                actual = (
                    weights.entries[buffer.source].layout
                    if buffer.storage == "weight" and buffer.source is not None
                    else "canonical"
                )
                if packing != actual:
                    raise ValueError(
                        f"kernel input packing mismatch: {tensor}: {packing} != {actual}"
                    )
            if len(inputs) != kernel["inputs"] or len(outputs) != kernel["outputs"]:
                raise ValueError("kernel binding count mismatch")
            arguments = instruction.get("parameters")
            if not isinstance(arguments, dict):
                raise ValueError("missing invocation parameters")
            require(sorted(_references(arguments, manifest.expressions)))
            require(
                sorted(
                    _references(
                        instruction.get("argument_key", []), manifest.expressions
                    )
                )
            )
            schema = kernel.get("parameters")
            if not isinstance(schema, dict) or not isinstance(
                schema.get("fields"), list
            ):
                raise ValueError("invalid native parameter schema")
            fields = schema["fields"]
            if not isinstance(fields, list) or set(arguments) != {
                field.get("name") for field in fields if isinstance(field, dict)
            }:
                raise ValueError("invocation parameters disagree with native schema")
            scratch = instruction.get("scratch")
            if (
                not isinstance(scratch, dict)
                or not isinstance(scratch.get("offset"), int)
                or not isinstance(scratch.get("size"), int)
            ):
                raise ValueError("invalid scratch allocation")
            offset, size = scratch["offset"], scratch["size"]
            if (
                not isinstance(offset, int)
                or not isinstance(size, int)
                or offset < 0
                or size < 0
                or offset + size > manifest.arena_size
            ):
                raise ValueError("scratch span exceeds arena")
            for output_name in outputs:
                require(
                    sorted(
                        _references(
                            manifest.buffers[output_name].byte_offset,
                            manifest.expressions,
                        )
                    )
                )
            define(outputs)
        elif op == "views":
            outputs = _strings(instruction.get("outputs"), "outputs")
            for name in outputs:
                buffer = manifest.buffers[name]
                if buffer.storage != "alias" or buffer.source is None:
                    raise ValueError("view must reference backing storage")
                require([buffer.source])
            define(outputs)
        elif op == "if":
            condition = instruction.get("condition")
            if not isinstance(condition, str):
                raise ValueError("invalid branch condition")
            require([condition])
            if (
                manifest.buffers[condition].dtype != "bool"
                or manifest.buffers[condition].maximum_shape
            ):
                raise ValueError("branch condition must be scalar bool")
            branches = instruction.get("branches")
            outputs = _strings(instruction.get("outputs"), "outputs")
            if not isinstance(branches, list) or len(branches) != 2:
                raise ValueError("if must have two branches")
            for branch in branches:
                if not isinstance(branch, dict):
                    raise TypeError("invalid branch")
                branch_available = _program(
                    _block(branch.get("program")),
                    manifest,
                    available.copy(),
                    weights,
                )
                copied = _program(
                    _block(branch.get("copies")),
                    manifest,
                    branch_available.copy(),
                    weights,
                )
                if not set(outputs) <= copied:
                    raise ValueError("branch outputs are not transferred")
                results = _strings(branch.get("results"), "branch results")
                if len(results) != len(outputs) or not set(results) <= branch_available:
                    raise ValueError("invalid branch result bindings")
            define(outputs)
        elif op == "while":
            initial, outputs = (
                _strings(instruction.get("initial"), "initial"),
                _strings(instruction.get("outputs"), "outputs"),
            )
            require(initial)
            limit = instruction.get("maximum_iterations")
            if not isinstance(limit, int) or limit < 0 or len(initial) != len(outputs):
                raise ValueError("invalid loop bound or carried values")
            state = _program(
                _block(instruction.get("initial_copies")),
                manifest,
                available.copy(),
                weights,
            )
            if not set(outputs) <= state:
                raise ValueError("loop state is not initialized")
            for name in ("condition", "body"):
                block = instruction.get(name)
                if not isinstance(block, dict):
                    raise TypeError("missing loop region")
                parameters = _strings(block.get("parameters"), "loop parameters")
                if len(parameters) != len(initial):
                    raise ValueError("loop parameter arity mismatch")
                entered = _program(
                    _block(block.get("copies")), manifest, state.copy(), weights
                )
                if not set(parameters) <= entered:
                    raise ValueError("loop parameters are not transferred")
                local = _program(
                    _block(block.get("program")), manifest, entered, weights
                )
                if name == "body":
                    updated = _program(
                        _block(block.get("updates")),
                        manifest,
                        local.copy(),
                        weights,
                    )
                    if not set(outputs) <= updated:
                        raise ValueError("loop state is not updated")
                results = (
                    _strings(block.get("results"), "body results")
                    if name == "body"
                    else [str(block.get("result"))]
                )
                if not set(results) <= local or len(results) != (
                    len(initial) if name == "body" else 1
                ):
                    raise ValueError("invalid loop result bindings")
                if name == "condition" and (
                    manifest.buffers[results[0]].dtype != "bool"
                    or manifest.buffers[results[0]].maximum_shape
                ):
                    raise ValueError("loop condition must be scalar bool")
            define(outputs)
        elif op == "reduce":
            values, initial = (
                _strings(instruction.get("values"), "reduction values"),
                _strings(instruction.get("initial"), "reduction initial"),
            )
            require(values + initial)
            parameters = _strings(instruction.get("parameters"), "reducer parameters")
            local = _program(
                _block(instruction.get("program")),
                manifest,
                available | set(parameters),
                weights,
            )
            results = _strings(instruction.get("results"), "reducer results")
            outputs = _strings(instruction.get("outputs"), "reduction outputs")
            if (
                len(parameters) != len(values) + len(initial)
                or len(results) != len(initial)
                or len(outputs) != len(initial)
                or not set(results) <= local
            ):
                raise ValueError("invalid reduction bindings")
            define(outputs)
        elif op in {"cache_read", "cache_write"}:
            name = instruction.get("cache")
            if not isinstance(name, str) or name not in manifest.caches:
                raise ValueError("unknown cache binding")
            if op == "cache_read":
                output = instruction.get("output")
                if not isinstance(output, str) or not isinstance(
                    instruction.get("snapshot"), bool
                ):
                    raise ValueError("cache read must preserve a snapshot")
                define([output])
            else:
                source = instruction.get("input")
                if not isinstance(source, str) or instruction.get("action") not in {
                    "assign",
                    "append",
                }:
                    raise ValueError("invalid cache update")
                require([source])
        else:
            raise ValueError(f"unknown executable instruction: {op!r}")
    return available


def _verify_plan(
    manifest: _ExecutableManifest,
    directory: Path,
    weights: _WeightManifest,
    constant_size: int,
) -> None:
    _expressions(manifest)
    cache_spans: list[tuple[int, int]] = []
    for name, cache_record in manifest.caches.items():
        shape, dtype = cache_record.get("shape"), cache_record.get("dtype")
        offset, size = cache_record.get("offset"), cache_record.get("size")
        if (
            not isinstance(shape, list)
            or not all(
                isinstance(dimension, int) and dimension >= 0 for dimension in shape
            )
            or not isinstance(dtype, str)
            or not isinstance(offset, int)
            or not isinstance(size, int)
            or offset < 0
            or offset % 64
            or size < 0
            or offset + size > manifest.cache_size
        ):
            raise ValueError(f"invalid cache allocation: {name}")
        dimensions = tuple(
            dimension for dimension in shape if isinstance(dimension, int)
        )
        if size != _byte_size(dtype, dimensions):
            raise ValueError(f"cache size disagrees with its capacity: {name}")
        if size:
            cache_spans.append((offset, offset + size))
    cache_spans.sort()
    if any(left[1] > right[0] for left, right in pairwise(cache_spans)):
        raise ValueError("cache allocations overlap")
    allocations = []
    for name, buffer in manifest.buffers.items():
        metadata: list[JsonValue] = [*buffer.shape, *buffer.strides, buffer.byte_offset]
        if not _references(metadata, manifest.expressions) <= manifest.buffers.keys():
            raise ValueError(f"tensor metadata references an unknown buffer: {name}")
        if (
            len(buffer.shape) != len(buffer.maximum_shape)
            or len(buffer.strides) != len(buffer.shape)
            or any(dimension < 0 for dimension in buffer.maximum_shape)
        ):
            raise ValueError(f"invalid tensor dimensions: {name}")
        if (
            buffer.alignment & (buffer.alignment - 1)
            or buffer.offset % buffer.alignment
            or buffer.first > buffer.last
        ):
            raise ValueError(f"invalid tensor allocation: {name}")
        if buffer.storage == "arena":
            if (
                buffer.offset + buffer.size > manifest.arena_size
                or buffer.size < _byte_size(buffer.dtype, tuple(buffer.maximum_shape))
            ):
                raise ValueError(f"tensor span exceeds arena: {name}")
            allocations.append((name, buffer))
        elif (
            buffer.storage == "constant" and buffer.offset + buffer.size > constant_size
        ):
            raise ValueError("constant span exceeds its file")
        elif buffer.storage == "weight":
            record = weights.entries.get(buffer.source or "")
            if record is None or (record.offset, record.size, record.dtype) != (
                buffer.offset,
                buffer.size,
                buffer.dtype,
            ):
                raise ValueError("weight binding disagrees with weight bundle")
        elif buffer.storage == "cache":
            cache = manifest.caches.get(buffer.source or "")
            if cache is None or cache.get("dtype") != buffer.dtype:
                raise ValueError("invalid direct cache binding")
            cache_bytes = cache.get("size")
            if (
                not isinstance(cache_bytes, int)
                or buffer.offset + buffer.size > cache_bytes
            ):
                raise ValueError("direct cache span exceeds cache capacity")
        elif buffer.storage == "alias":
            seen = {name}
            source = buffer.source
            while source is not None:
                if source in seen or source not in manifest.buffers:
                    raise ValueError("invalid or cyclic view binding")
                seen.add(source)
                backing = manifest.buffers[source]
                source = backing.source if backing.storage == "alias" else None
    # A sweep over live intervals checks allocation safety without quadratic graph scans.
    active: list[tuple[str, _BufferRecord]] = []
    for name, buffer in sorted(allocations, key=lambda item: item[1].first):
        active = [(other, span) for other, span in active if span.last >= buffer.first]
        for other, span in active:
            if (
                buffer.size
                and span.size
                and max(buffer.offset, span.offset)
                < min(buffer.offset + buffer.size, span.offset + span.size)
            ):
                raise ValueError(f"live tensor allocations overlap: {other}, {name}")
        active.append((name, buffer))
    for kernel in manifest.kernels.values():
        filename = kernel.get("library")
        if (
            not isinstance(filename, str)
            or not (directory / filename).is_file()
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            raise ValueError("kernel library is not contained in the artifact")
    available = {
        name
        for name, buffer in manifest.buffers.items()
        if buffer.storage in {"constant", "weight", "input"}
    }
    available = _program(manifest.program, manifest, available, weights)
    if not set(manifest.outputs.values()) <= available:
        raise ValueError("model output is not produced by the program")


def _verify(directory: Path) -> _ExecutableManifest:
    manifest = _ExecutableManifest.model_validate_json(
        (directory / "executable.json").read_bytes()
    )
    weights = _read_manifest(directory / manifest.weights)
    constant_size = (directory / manifest.constants).stat().st_size
    _verify_plan(manifest, directory, weights, constant_size)
    previous = 0
    for plan in manifest.plans:
        if (
            plan.minimum_tokens <= previous
            or plan.minimum_tokens > plan.maximum_tokens
            or plan.arena_size > manifest.arena_size
        ):
            raise ValueError("invalid or overlapping workload plan ranges")
        selected = manifest.model_copy(
            update={
                "buffers": plan.buffers,
                "program": plan.program,
                "arena_size": plan.arena_size,
                "plans": [],
                "expressions": plan.expressions,
            }
        )
        _verify_plan(selected, directory, weights, constant_size)
        previous = plan.maximum_tokens
    if manifest.workload is not None:
        name, axis = manifest.workload.get("input"), manifest.workload.get("axis")
        if (
            not isinstance(name, str)
            or name not in manifest.inputs
            or not isinstance(axis, int)
            or not 0 <= axis < len(manifest.buffers[manifest.inputs[name]].shape)
        ):
            raise ValueError("workload does not name an input dimension")
    return manifest
