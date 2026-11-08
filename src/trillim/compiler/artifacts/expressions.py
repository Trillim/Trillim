from __future__ import annotations

import math
from collections.abc import Mapping

from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import _ShapeExpression
from ..dtype import _payload_sections
from ..kernels.kernel_dataclasses import _ParameterValue
from .artifact_dataclasses import _JSON, _Expressions


def _number(value: float) -> _JSON:
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    return value


def _references(value: _ShapeExpression) -> set[Tensor]:
    pending = [value]
    visited: set[_ShapeExpression] = set()
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


def _parameter_references(value: _ParameterValue) -> set[Tensor]:
    if isinstance(value, _ShapeExpression):
        return _references(value)
    if isinstance(value, tuple):
        return set().union(*(_parameter_references(item) for item in value))
    return set()


def _bound(value: _ShapeExpression, *, upper: bool = True) -> int:
    known = value.maximum if upper else value.minimum
    if known is not None:
        return known
    if value.op == "const" and value.value is not None:
        return value.value
    if value.op in {"scalar", "dimension", "cache_length"}:
        raise ValueError(f"missing finite bound for {value.op}")
    if value.op == "sub":
        return _bound(value.args[0], upper=upper) - _bound(
            value.args[1], upper=not upper
        )
    if value.op == "mul":
        left, right = value.args
        products = [
            _bound(left, upper=x) * _bound(right, upper=y)
            for x in (False, True)
            for y in (False, True)
        ]
        return max(products) if upper else min(products)
    if value.op == "broadcast_stride":
        return max(0, _bound(value.args[1])) if upper else 0
    if value.op == "broadcast":
        values = [_bound(item, upper=upper) for item in value.args]
        return max(values) if upper else min(0, *values)
    a, b = (_bound(item, upper=upper) for item in value.args)
    if value.op == "add":
        return a + b
    if value.op == "min":
        return min(a, b)
    if value.op == "max":
        return max(a, b)
    if value.op == "mod":
        divisor = _bound(value.args[1])
        if divisor <= 0:
            raise ValueError("modulo needs a positive bounded divisor")
        return divisor - 1 if upper else 0
    if value.op in {"floor_div", "ceil_div"}:
        divisor = _bound(value.args[1], upper=not upper)
        if divisor <= 0:
            raise ValueError("shape division needs a positive divisor")
        return a // divisor if value.op == "floor_div" else -(-a // divisor)
    raise ValueError(f"unsupported shape expression: {value.op}")


def _serialize(
    value: _ShapeExpression, names: Mapping[Tensor, str], expressions: _Expressions
) -> _JSON:
    if value.op == "const":
        assert value.value is not None
        return value.value
    if value.tensor is not None:
        if value.tensor not in names:
            raise ValueError("shape expression references an eliminated tensor")
        result: dict[str, _JSON] = {"op": value.op, "tensor": names[value.tensor]}
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
        "args": [_serialize(item, names, expressions) for item in value.args],
    }
    return {"expr": name}


def _byte_size(dtype: str, shape: tuple[int, ...]) -> int:
    return sum(size for _, size in _payload_sections(dtype, shape))


def _align(value: int, alignment: int) -> int:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment
