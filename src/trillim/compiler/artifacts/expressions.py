from __future__ import annotations

import math
from collections.abc import Mapping

from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import ShapeExpression
from ..kernels.kernel_dataclasses import ParameterValue
from .artifact_dataclasses import JSON, Expressions


def number(value: float) -> JSON:
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    return value


def references(value: ShapeExpression) -> set[Tensor]:
    pending = [value]
    visited: set[ShapeExpression] = set()
    result: set[Tensor] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current.tensor is not None:
            result.add(current.tensor)
        pending.extend(current.args)
    return result


def parameter_references(value: ParameterValue) -> set[Tensor]:
    if isinstance(value, ShapeExpression):
        return references(value)
    if isinstance(value, tuple):
        return set().union(*(parameter_references(item) for item in value))
    return set()


def bound(value: ShapeExpression, *, upper: bool = True) -> int:
    known = value.maximum if upper else value.minimum
    if known is not None:
        return known
    if value.op == "const" and value.value is not None:
        return value.value
    if value.op in {"scalar", "dimension", "cache_length"}:
        raise ValueError(f"missing finite bound for {value.op}")
    if value.op == "sub":
        return bound(value.args[0], upper=upper) - bound(value.args[1], upper=not upper)
    if value.op == "mul":
        left, right = value.args
        products = [
            bound(left, upper=x) * bound(right, upper=y)
            for x in (False, True)
            for y in (False, True)
        ]
        return max(products) if upper else min(products)
    if value.op == "broadcast_stride":
        return max(0, bound(value.args[1])) if upper else 0
    if value.op == "broadcast":
        values = [bound(item, upper=upper) for item in value.args]
        return max(values) if upper else min(0, *values)
    a, b = (bound(item, upper=upper) for item in value.args)
    if value.op == "add":
        return a + b
    if value.op == "min":
        return min(a, b)
    if value.op == "max":
        return max(a, b)
    if value.op == "mod":
        divisor = bound(value.args[1])
        if divisor <= 0:
            raise ValueError("modulo needs a positive bounded divisor")
        return divisor - 1 if upper else 0
    if value.op in {"floor_div", "ceil_div"}:
        divisor = bound(value.args[1], upper=not upper)
        if divisor <= 0:
            raise ValueError("shape division needs a positive divisor")
        return a // divisor if value.op == "floor_div" else -(-a // divisor)
    raise ValueError(f"unsupported shape expression: {value.op}")


def serialize(
    value: ShapeExpression, names: Mapping[Tensor, str], expressions: Expressions
) -> JSON:
    if value.op == "const":
        assert value.value is not None
        return value.value
    if value.tensor is not None:
        if value.tensor not in names:
            raise ValueError("shape expression references an eliminated tensor")
        result: dict[str, JSON] = {"op": value.op, "tensor": names[value.tensor]}
        if value.axis is not None:
            result["axis"] = value.axis
        return result
    if value.op == "cache_length":
        return {"op": value.op, "cache": value.cache}
    existing = expressions.names.get(value)
    if existing is not None:
        return {"expr": existing}
    name = f"e{len(expressions.names)}"
    expressions.names[value] = name
    expressions.nodes[name] = {
        "op": value.op,
        "args": [serialize(item, names, expressions) for item in value.args],
    }
    return {"expr": name}


def byte_size(dtype: str, shape: tuple[int, ...]) -> int:
    from ...dtype import payload_sections, validate_dtype

    if any(dimension < 0 for dimension in shape):
        raise ValueError("tensor capacity cannot be negative")
    return sum(size for _, size in payload_sections(validate_dtype(dtype), shape))


def align(value: int, alignment: int) -> int:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment
