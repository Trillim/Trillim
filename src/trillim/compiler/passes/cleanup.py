from __future__ import annotations

from pydantic import JsonValue

from ..artifacts.artifact_pydantic_models import _ExecutableManifest
from ..artifacts.expressions import _dimension_dependencies


def _cleanup(manifest: _ExecutableManifest) -> _ExecutableManifest:
    """Remove work made redundant by the selected kernel cover."""
    expression_reads: dict[str, set[str]] = {}
    expression_keys: dict[str, int] = {}
    dimensions: dict[str, frozenset[str] | None] = {}
    keys: dict[tuple[object, ...], int] = {}

    def references(value: JsonValue) -> set[str]:
        if isinstance(value, str):
            return {value} if value in manifest.buffers else set()
        if isinstance(value, list):
            return set().union(*(references(item) for item in value))
        if isinstance(value, dict):
            name = value.get("expr")
            if isinstance(name, str):
                if name not in expression_reads:
                    expression_reads[name] = references(manifest.expressions[name])
                return expression_reads[name]
            return set().union(
                *(
                    references(item)
                    for key, item in value.items()
                    if key != "outputs"
                    or "op" not in value
                    or value.get("effectful")
                    or value.get("op") in {"if", "while", "reduce"}
                )
            )
        return set()

    def intern(value: JsonValue) -> int:
        if isinstance(value, dict):
            name = value.get("expr")
            if isinstance(name, str):
                if name not in expression_keys:
                    expression_keys[name] = intern(manifest.expressions[name])
                return expression_keys[name]
            key: tuple[object, ...] = tuple(
                (name, intern(item)) for name, item in sorted(value.items())
            )
        elif isinstance(value, list):
            key = ("list", *(intern(item) for item in value))
        else:
            key = (type(value).__name__, value)
        return keys.setdefault(key, len(keys))

    protected = set(manifest.inputs.values()) | set(manifest.outputs.values())

    def protect(value: JsonValue) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"parameters", "results", "initial", "result"} or (
                    key == "outputs" and value.get("op") in {"if", "while", "reduce"}
                ):
                    protected.update(references(item))
                protect(item)
        elif isinstance(value, list):
            for item in value:
                protect(item)

    protect(list(manifest.program))
    implicit: set[str] = set()
    for name, buffer in manifest.buffers.items():
        if name in protected or buffer.storage != "alias":
            continue
        shape_dependencies = _dimension_dependencies(
            [*buffer.shape, *buffer.strides, buffer.byte_offset],
            manifest.expressions,
            dimensions,
        )
        if shape_dependencies is not None and name not in shape_dependencies:
            implicit.add(name)
    needed: set[str] = set()

    def rewrite(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {name: rewrite(item) for name, item in value.items()}
        if not isinstance(value, list):
            return value
        result: list[JsonValue] = []
        seen: dict[tuple[int, int, int], frozenset[str] | None] = {}
        for item in value:
            if isinstance(item, dict):
                op = item.get("op")
                outputs = item.get("outputs")
                if isinstance(outputs, list) and op in {"call", "views"}:
                    live: list[JsonValue] = [
                        name
                        for name in outputs
                        if isinstance(name, str) and name in needed
                    ]
                    if op == "views":
                        live = [name for name in live if name not in implicit]
                        if not live:
                            continue
                        item = item | {"outputs": live}
                    elif outputs and not live and not item.get("effectful"):
                        continue
                if op == "guard":
                    a, b = intern(item["left"]), intern(item["right"])
                    predicate = item["predicate"]
                    if predicate in {"eq", "broadcast"}:
                        a, b = min(a, b), max(a, b)
                    if a == b and predicate in {"eq", "le", "ge", "broadcast"}:
                        continue
                    guard = (intern(predicate), a, b)
                    if guard in seen:
                        continue
                    operands = [item["left"], item["right"]]
                    seen[guard] = _dimension_dependencies(
                        operands, manifest.expressions, dimensions
                    )
                else:
                    if op in {"if", "while", "reduce"} or item.get("effectful"):
                        seen.clear()
                    else:
                        changed = (
                            {name for name in outputs if isinstance(name, str)}
                            if isinstance(outputs, list)
                            else set()
                        )
                        output = item.get("output")
                        if isinstance(output, str):
                            changed.add(output)
                        seen = {
                            guard: dependencies
                            for guard, dependencies in seen.items()
                            if dependencies is not None and not dependencies & changed
                        }
            result.append(rewrite(item))
        return result

    program: JsonValue = list(manifest.program)
    while True:
        needed = (
            references(program)
            | set(manifest.inputs.values())
            | set(manifest.outputs.values())
        )
        pending = list(needed)
        while pending:
            buffer = manifest.buffers[pending.pop()]
            dependencies = references(
                [*buffer.shape, *buffer.strides, buffer.byte_offset]
            )
            if buffer.storage == "alias" and buffer.source is not None:
                dependencies |= references(buffer.source)
            pending.extend(dependencies - needed)
            needed |= dependencies
        cleaned = rewrite(program)
        if cleaned == program:
            break
        program = cleaned
    return _ExecutableManifest.model_validate(
        manifest.model_dump() | {"program": program}
    )
