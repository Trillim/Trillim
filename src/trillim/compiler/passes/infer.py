from __future__ import annotations

from functools import reduce
from itertools import zip_longest
from typing import cast

import numpy as np

from ...tdl.core.core_dataclasses import Shape, _Dimensions, _Region, _Scalar
from ...tdl.core.core_pydantic_models import (
    Weight,
    _BinaryArgs,
    _BroadcastArgs,
    _CacheArgs,
    _CacheAssignArgs,
    _CacheReadArgs,
    _CastArgs,
    _CompareArgs,
    _ConstantArgs,
    _ConstructArgs,
    _GatherArgs,
    _IfArgs,
    _InputArgs,
    _IotaArgs,
    _MatmulArgs,
    _OutputArgs,
    _PermuteArgs,
    _RandomArgs,
    _ReduceArgs,
    _ReshapeArgs,
    _ScatterArgs,
    _SelectArgs,
    _UnaryArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType, _validate_dtype
from ..compiler_dataclasses import (
    _Inference,
    _InferenceState,
    _ShapeConstraint,
    _ShapeExpression,
    _ShapeOperation,
    _ShapePredicate,
    _TensorInfo,
)
from ..dtype import _QUANTIZED_DTYPES
from .rewrite import _rebuild
from .walk import _ordered_operators


def _constant_expression(value: int) -> _ShapeExpression:
    return _ShapeExpression("const", value=value, minimum=value, maximum=value)


def _expression(op: _ShapeOperation, *args: _ShapeExpression) -> _ShapeExpression:
    if len(args) != 2 or op not in {
        "add",
        "sub",
        "mul",
        "floor_div",
        "mod",
        "min",
        "max",
        "ceil_div",
        "broadcast",
        "broadcast_stride",
    }:
        raise ValueError(f"{op!r} requires two arithmetic shape operands")
    left, right = args
    if op in {"min", "max"}:
        if left == right:
            return left
        for smaller, larger in ((left, right), (right, left)):
            if (
                smaller.maximum is not None
                and larger.minimum is not None
                and smaller.maximum <= larger.minimum
            ):
                return smaller if op == "min" else larger
            if (
                smaller.op == "max"
                and larger.minimum is not None
                and larger.minimum >= 0
            ):
                for difference, zero in (smaller.args, tuple(reversed(smaller.args))):
                    if (
                        zero.value == 0
                        and difference.op == "sub"
                        and difference.args[0] == larger
                        and difference.args[1].minimum is not None
                        and difference.args[1].minimum >= 0
                    ):
                        return smaller if op == "min" else larger
    if left.op == right.op == "const":
        assert left.value is not None and right.value is not None
        a, b = left.value, right.value
        if op in {"floor_div", "mod", "ceil_div"} and b == 0:
            raise ValueError("shape expression divides by zero")
        value = {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "floor_div": lambda: a // b,
            "mod": lambda: a % b,
            "ceil_div": lambda: -(-a // b),
            "min": lambda: min(a, b),
            "max": lambda: max(a, b),
            "broadcast": lambda: b if a == 1 else a,
            "broadcast_stride": lambda: 0 if a == 1 else b,
        }[op]()
        return _constant_expression(value)
    if op in {"add", "sub"} and right.op == "const" and right.value == 0:
        return left
    if op == "add" and left.op == "const" and left.value == 0:
        return right
    if op == "add":
        for number, other in ((left, right), (right, left)):
            if number.op == "const" and number.value is not None and number.value < 0:
                return _expression("sub", other, _constant_expression(-number.value))
    if op == "sub" and left == right:
        return _constant_expression(0)
    if op == "sub" and right.op in {"min", "max"}:
        for difference, zero in (right.args, tuple(reversed(right.args))):
            if (
                zero.op == "const"
                and zero.value == 0
                and difference.op == "sub"
                and difference.args[0] == left
            ):
                return _expression(
                    "min" if right.op == "max" else "max", left, difference.args[1]
                )
    if op == "mul":
        if left.value == 0 or right.value == 0:
            return _constant_expression(0)
        if left.op == "const" and left.value == 1:
            return right
        if right.op == "const" and right.value == 1:
            return left
        if left.op == "const":
            left, right = right, left
        if right.op == "const" and left.op == "mul" and left.args[1].op == "const":
            return _expression(
                "mul", left.args[0], _expression("mul", left.args[1], right)
            )
    if op in {"floor_div", "ceil_div"} and right.op == "const" and right.value == 1:
        return left
    if (
        op in {"floor_div", "ceil_div", "mod"}
        and right.op == "const"
        and right.value
        and left.op == "mul"
        and left.args[1].op == "const"
    ):
        factor = left.args[1].value
        assert factor is not None
        if factor % right.value == 0:
            return (
                _constant_expression(0)
                if op == "mod"
                else _expression(
                    "mul", left.args[0], _constant_expression(factor // right.value)
                )
            )
    if op == "broadcast":
        if left.op == "const":
            return right if left.value == 1 else left
        if right.op == "const":
            return left if right.value == 1 else right
        if left == right:
            return left
    if op == "broadcast_stride" and left.op == "const":
        return _constant_expression(0) if left.value == 1 else right
    low: int | None = None
    high: int | None = None
    if op == "add":
        low = (
            None
            if left.minimum is None or right.minimum is None
            else left.minimum + right.minimum
        )
        high = (
            None
            if left.maximum is None or right.maximum is None
            else left.maximum + right.maximum
        )
    elif op == "sub":
        low = (
            None
            if left.minimum is None or right.maximum is None
            else left.minimum - right.maximum
        )
        high = (
            None
            if left.maximum is None or right.minimum is None
            else left.maximum - right.minimum
        )
    elif op == "mul":
        if None not in (left.minimum, left.maximum, right.minimum, right.maximum):
            values = [
                a * b
                for a in (left.minimum, left.maximum)
                for b in (right.minimum, right.maximum)
                if a is not None and b is not None
            ]
            low, high = min(values), max(values)
        elif (
            left.minimum is not None
            and left.minimum >= 0
            and right.minimum is not None
            and right.minimum >= 0
        ):
            low = left.minimum * right.minimum
    elif op in {"floor_div", "ceil_div"}:
        if (
            right.minimum is not None
            and right.maximum is not None
            and (right.minimum > 0 or right.maximum < 0)
            and left.minimum is not None
            and left.maximum is not None
        ):
            values = [
                a // b if op == "floor_div" else -(-a // b)
                for a in (left.minimum, left.maximum)
                for b in (right.minimum, right.maximum)
            ]
            low, high = min(values), max(values)
    elif op == "mod":
        if right.minimum is not None and right.minimum > 0:
            low = 0
            high = None if right.maximum is None else right.maximum - 1
        elif right.maximum is not None and right.maximum < 0:
            low = None if right.minimum is None else right.minimum + 1
            high = 0
    elif op == "min":
        low = (
            None
            if left.minimum is None or right.minimum is None
            else min(left.minimum, right.minimum)
        )
        known = [value for value in (left.maximum, right.maximum) if value is not None]
        high = min(known) if known else None
    elif op == "max":
        known = [value for value in (left.minimum, right.minimum) if value is not None]
        low = max(known) if known else None
        high = (
            None
            if left.maximum is None or right.maximum is None
            else max(left.maximum, right.maximum)
        )
    elif op == "broadcast":
        low = (
            max(left.minimum, right.minimum)
            if left.minimum is not None
            and right.minimum is not None
            and left.minimum >= 1
            and right.minimum >= 1
            else 0
        )
        high = (
            None
            if left.maximum is None or right.maximum is None
            else max(left.maximum, right.maximum)
        )
    elif op == "broadcast_stride":
        low = None if right.minimum is None else min(0, right.minimum)
        high = None if right.maximum is None else max(0, right.maximum)
    return _ShapeExpression(op, (left, right), minimum=low, maximum=high)


_FLOATS = {"fp16", "bf16", "fp32"}
_INTEGERS = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}


def _integer(dtype: DType | None) -> bool:
    return dtype == "bool" or dtype in _INTEGERS


def _promote(left: DType, right: DType, *, numeric: bool = True) -> DType:
    if left in _QUANTIZED_DTYPES or right in _QUANTIZED_DTYPES:
        return "fp32"
    if left == right:
        return "int64" if numeric and left == "bool" else left
    if left == "bool":
        return right
    if right == "bool":
        return left
    if left in _FLOATS or right in _FLOATS:
        return "fp32"
    signed = left.startswith("int") or right.startswith("int")
    left_bits = int(left.removeprefix("u").removeprefix("int"))
    right_bits = int(right.removeprefix("u").removeprefix("int"))
    bits = max(left_bits, right_bits)
    if signed and (
        left.startswith("uint")
        and left_bits >= right_bits
        or right.startswith("uint")
        and right_bits >= left_bits
    ):
        bits *= 2
    if bits > 64:
        raise ValueError("signed and uint64 operands require an explicit common cast")
    return _validate_dtype(("int" if signed else "uint") + str(bits))


def _literal_dtype(value: _Scalar) -> DType:
    return (
        "bool"
        if isinstance(value, bool)
        else "int64"
        if isinstance(value, int)
        else "fp32"
    )


def _binary_dtype(
    left: DType,
    right: Tensor | _Scalar,
    state: _InferenceState,
    *,
    numeric: bool = True,
) -> DType:
    if isinstance(right, Tensor):
        return _promote(left, _dtype(state, right), numeric=numeric)
    if left in _QUANTIZED_DTYPES:
        return "fp32"
    if left in _FLOATS:
        return left
    if isinstance(right, float):
        return "fp32"
    if left == "bool":
        return "bool" if not numeric and isinstance(right, bool) else "int64"
    limits = _type_range(left)
    if limits is not None and not limits[0] <= int(right) <= limits[1]:
        return _promote(left, _literal_dtype(right), numeric=numeric)
    return left


def _dtype(state: _InferenceState, tensor: Tensor) -> DType:
    dtype = state.result.tensors[tensor].dtype
    if dtype is None:
        raise ValueError(
            "tensor dtype is unresolved; resolve checkpoint metadata first"
        )
    return dtype


def _shape(state: _InferenceState, tensor: Tensor) -> tuple[_ShapeExpression, ...]:
    shape = state.result.tensors[tensor].shape
    if shape is None:
        raise ValueError("tensor rank is unresolved; resolve checkpoint metadata first")
    return shape


def _range(state: _InferenceState, tensor: Tensor) -> tuple[int | None, int | None]:
    return state.result.value_ranges.get(tensor, (None, None))


def _type_range(dtype: DType | None) -> tuple[int, int] | None:
    if dtype == "bool":
        return 0, 1
    if dtype not in _INTEGERS or dtype is None:
        return None
    bits = int(dtype.removeprefix("u").removeprefix("int"))
    return (
        (0, 2**bits - 1)
        if dtype.startswith("uint")
        else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    )


def _store(
    state: _InferenceState,
    tensor: Tensor,
    shape: tuple[_ShapeExpression, ...] | None,
    dtype: DType | None,
    values: tuple[_ShapeExpression, ...] | None = None,
    value_range: tuple[int | None, int | None] | None = None,
) -> _TensorInfo:
    if value_range is None and values:
        lower_bounds = [value.minimum for value in values]
        upper_bounds = [value.maximum for value in values]
        value_range = (
            None
            if None in lower_bounds
            else min(value for value in lower_bounds if value is not None),
            None
            if None in upper_bounds
            else max(value for value in upper_bounds if value is not None),
        )
    if value_range is None and dtype == "bool":
        value_range = (0, 1)
    limits = _type_range(dtype)
    if limits is not None:
        low, high = value_range or (None, None)
        if low is None or high is None or low < limits[0] or high > limits[1]:
            value_range = limits
        if values is not None:
            values = tuple(
                value
                if value.minimum is not None
                and value.maximum is not None
                and limits[0] <= value.minimum <= value.maximum <= limits[1]
                else _ShapeExpression(
                    "scalar",
                    tensor=tensor,
                    axis=index,
                    minimum=limits[0],
                    maximum=limits[1],
                )
                for index, value in enumerate(values)
            )
    if value_range is not None:
        state.result.value_ranges[tensor] = value_range
    else:
        state.result.value_ranges.pop(tensor, None)
    if shape == () and values is None and _integer(dtype):
        low, high = value_range or (None, None)
        values = (
            _ShapeExpression(
                "scalar", tensor=tensor, axis=0, minimum=low, maximum=high
            ),
        )
    info = _TensorInfo(
        shape,
        None if shape is None else tuple(value.maximum for value in shape),
        dtype,
        values,
    )
    state.result.tensors[tensor] = info
    tensor.shape = (
        None
        if shape is None
        else Shape(
            tuple(value.value if value.op == "const" else None for value in shape)
        )
    )
    tensor.dtype = dtype
    return info


def _copy(state: _InferenceState, tensor: Tensor, source: Tensor) -> _TensorInfo:
    info = state.result.tensors[source]
    return _store(
        state, tensor, info.shape, info.dtype, info.values, _range(state, source)
    )


def _scalar(state: _InferenceState, value: int | Tensor) -> _ShapeExpression:
    if isinstance(value, int):
        return _constant_expression(value)
    info = state.result.tensors[value]
    if info.shape != () or not _integer(info.dtype):
        raise ValueError(
            "shape and index scalar expressions require scalar integer tensors"
        )
    if info.values is not None:
        return info.values[0]
    low, high = _range(state, value)
    return _ShapeExpression("scalar", tensor=value, axis=0, minimum=low, maximum=high)


def _constraint(
    state: _InferenceState,
    predicate: str,
    left: _ShapeExpression,
    right: _ShapeExpression,
    reason: str,
    convert: bool,
) -> None:
    if left == right and predicate in {"eq", "le", "ge", "broadcast"}:
        return
    if predicate == "broadcast" and (left.value == 1 or right.value == 1):
        return
    if predicate in {"le", "ge"}:
        smaller, larger = (left, right) if predicate == "le" else (right, left)
        if (
            smaller.maximum is not None
            and larger.minimum is not None
            and smaller.maximum <= larger.minimum
        ):
            return
    if left.op == right.op == "const":
        assert left.value is not None and right.value is not None
        a, b = left.value, right.value
        valid = {
            "eq": a == b,
            "le": a <= b,
            "ge": a >= b,
            "broadcast": a == 1 or b == 1 or a == b,
            "divisible": b != 0 and a % b == 0,
        }[predicate]
        if not valid:
            raise ValueError(reason)
        return
    if convert:
        item = _ShapeConstraint(
            predicate=cast(_ShapePredicate, predicate),
            left=left,
            right=right,
            reason=reason,
            operator=state.operators[-1] if state.operators else None,
        )
        if item not in state.constraint_keys:
            state.constraint_keys.add(item)
            state.result.constraints.append(item)


def _broadcast(
    state: _InferenceState,
    left: tuple[_ShapeExpression, ...],
    right: tuple[_ShapeExpression, ...],
    convert: bool,
) -> tuple[_ShapeExpression, ...]:
    shape = []
    for a, b in zip_longest(
        reversed(left), reversed(right), fillvalue=_constant_expression(1)
    ):
        _constraint(
            state, "broadcast", a, b, "incompatible broadcast dimensions", convert
        )
        shape.append(_expression("broadcast", a, b))
    return tuple(reversed(shape))


def _product(shape: tuple[_ShapeExpression, ...]) -> _ShapeExpression:
    return reduce(
        lambda left, right: _expression("mul", left, right),
        shape,
        _constant_expression(1),
    )


def _reshape_quotient(
    numerator: _ShapeExpression, denominator: _ShapeExpression
) -> _ShapeExpression:
    def factors(value: _ShapeExpression) -> list[_ShapeExpression]:
        if value.op == "mul":
            return [factor for argument in value.args for factor in factors(argument)]
        return [] if value.op == "const" and value.value == 1 else [value]

    remaining = factors(numerator)
    divisors = []
    for factor in factors(denominator):
        if factor in remaining:
            remaining.remove(factor)
        else:
            divisors.append(factor)
    return _expression(
        "floor_div", _product(tuple(remaining)), _product(tuple(divisors))
    )


def _dimensions(
    state: _InferenceState, dimensions: _Dimensions
) -> tuple[_ShapeExpression, ...]:
    if isinstance(dimensions, tuple):
        return tuple(_scalar(state, dimension) for dimension in dimensions)
    info = state.result.tensors[dimensions]
    if not _integer(info.dtype) or info.shape is None or len(info.shape) != 1:
        raise ValueError("shape tensors must be rank-one integers")
    length = info.shape[0]
    if length.op != "const" or length.value is None:
        raise ValueError("shape tensor length must be statically known")
    if info.values is not None:
        return info.values
    low, high = _range(state, dimensions)
    return tuple(
        _ShapeExpression(
            "scalar", tensor=dimensions, axis=index, minimum=low, maximum=high
        )
        for index in range(length.value)
    )


def _cast(
    state: _InferenceState, region: _Region, tensor: Tensor, dtype: DType, convert: bool
) -> Tensor:
    if _dtype(state, tensor) == dtype or not convert:
        return tensor
    operator = _Operator(
        "cast", _CastArgs(x=tensor, dtype=dtype), outputs=((tensor.shape, dtype),)
    )
    region.operators = (*region.operators, operator)
    source = state.result.tensors[tensor]
    values = source.values if _integer(dtype) and _integer(source.dtype) else None
    _store(
        state, operator.outputs[0], source.shape, dtype, values, _range(state, tensor)
    )
    return operator.outputs[0]


def _operand(
    state: _InferenceState,
    region: _Region,
    value: Tensor | _Scalar,
    dtype: DType,
    convert: bool,
) -> Tensor | _Scalar:
    if isinstance(value, Tensor):
        return _cast(state, region, value, dtype, convert)
    if not convert:
        return value
    constant_args = _ConstantArgs(value=value, dtype=dtype)
    operator = _Operator("constant", constant_args, outputs=((Shape(()), dtype),))
    region.operators = (*region.operators, operator)
    _constant(state, operator, constant_args)
    return operator.outputs[0]


def _condition(state: _InferenceState, tensor: Tensor, *, scalar: bool) -> None:
    if _dtype(state, tensor) != "bool" or scalar and _shape(state, tensor) != ():
        raise ValueError(
            "conditions require boolean tensors"
            + (" with scalar shape" if scalar else "")
        )


def _values(
    state: _InferenceState, value: Tensor | _Scalar
) -> tuple[_ShapeExpression, ...] | None:
    if isinstance(value, Tensor):
        return state.result.tensors[value].values
    return (
        (_constant_expression(int(value)),) if isinstance(value, (bool, int)) else None
    )


def _binary_values(
    state: _InferenceState, op: str, left: Tensor, right: Tensor | _Scalar
) -> tuple[_ShapeExpression, ...] | None:
    a, b = _values(state, left), _values(state, right)
    if a is None or b is None or op not in {"add", "sub", "mul", "floor_div", "mod"}:
        return None
    if len(a) != len(b):
        if len(a) == 1:
            a *= len(b)
        elif len(b) == 1:
            b *= len(a)
        else:
            return None
    return tuple(
        _expression(cast(_ShapeOperation, op), x, y) for x, y in zip(a, b, strict=True)
    )


def _numeric_range(
    state: _InferenceState, op: str, left: Tensor, right: Tensor | _Scalar
) -> tuple[int | None, int | None] | None:
    if op not in {"add", "sub", "mul", "floor_div", "mod"}:
        return None
    low, high = _range(state, left)
    a = _ShapeExpression("scalar", tensor=left, minimum=low, maximum=high)
    if isinstance(right, Tensor):
        low, high = _range(state, right)
        b = _ShapeExpression("scalar", tensor=right, minimum=low, maximum=high)
    elif isinstance(right, (int, bool)):
        b = _constant_expression(int(right))
    else:
        return None
    result = _expression(cast(_ShapeOperation, op), a, b)
    return result.minimum, result.maximum


def _select_values(
    state: _InferenceState, args: _SelectArgs
) -> tuple[_ShapeExpression, ...] | None:
    when_true, when_false = (
        _values(state, args.when_true),
        _values(state, args.when_false),
    )
    condition = _values(state, args.condition)
    if condition is not None and len(condition) == 1 and condition[0].op == "const":
        return when_true if condition[0].value else when_false
    producer = args.condition._producer
    if producer is not None and isinstance(producer.args, _CompareArgs):
        compare = producer.args
        a, b = _values(state, compare.a), _values(state, compare.b)
        if (
            a is not None
            and b is not None
            and when_true is not None
            and when_false is not None
            and when_true == a
            and when_false == b
            and len(a) == len(b)
            and compare.predicate in {"ge", "gt", "le", "lt"}
        ):
            op: _ShapeOperation = "max" if compare.predicate in {"ge", "gt"} else "min"
            return tuple(_expression(op, x, y) for x, y in zip(a, b, strict=True))
    if when_true is not None and when_true == when_false:
        return when_true
    return None


def _join_range(
    left: tuple[int | None, int | None], right: tuple[int | None, int | None]
) -> tuple[int | None, int | None]:
    return (
        None if left[0] is None or right[0] is None else min(left[0], right[0]),
        None if left[1] is None or right[1] is None else max(left[1], right[1]),
    )


def _input(state: _InferenceState, operator: _Operator, args: _InputArgs) -> None:
    tensor = operator.outputs[0]
    shape = tuple(
        _constant_expression(dimension)
        if dimension is not None
        else _ShapeExpression(
            "dimension",
            tensor=tensor,
            axis=axis,
            minimum=0,
            maximum=state.max_input_size,
        )
        for axis, dimension in enumerate(args.shape.dimensions)
    )
    _store(state, tensor, shape, args.dtype)


def _constant(state: _InferenceState, operator: _Operator, args: _ConstantArgs) -> None:
    value = args.value
    if isinstance(value, Weight):
        dtype = args.dtype or value.dtype
        shape = (
            None
            if value.shape is None
            else tuple(
                _constant_expression(dimension)
                if dimension is not None
                else _ShapeExpression(
                    "dimension", tensor=operator.outputs[0], axis=axis, minimum=0
                )
                for axis, dimension in enumerate(value.shape.dimensions)
            )
        )
        _store(state, operator.outputs[0], shape, dtype)
        return
    dtype = args.dtype
    if isinstance(value, np.ndarray):
        shape = tuple(_constant_expression(dimension) for dimension in value.shape)
        if dtype is None:
            dtype = (
                "fp16"
                if value.dtype.name == "float16"
                else "fp32"
                if value.dtype.kind == "f"
                else _validate_dtype(value.dtype.name)
            )
        values = (
            tuple(
                _constant_expression(int(bool(item)) if dtype == "bool" else int(item))
                for item in value.flat
            )
            if _integer(dtype) and value.size <= 4096
            else None
        )
    else:
        shape = ()
        dtype = dtype or _literal_dtype(value)
        values = (
            (_constant_expression(int(bool(value)) if dtype == "bool" else int(value)),)
            if _integer(dtype)
            else None
        )
    limits = _type_range(dtype)
    if (
        limits is not None
        and values is not None
        and any(
            item.value is not None and not limits[0] <= item.value <= limits[1]
            for item in values
        )
    ):
        raise ValueError(f"constant integer value does not fit {dtype}")
    _store(state, operator.outputs[0], shape, dtype, values)


def _ordinary(
    state: _InferenceState, region: _Region, operator: _Operator, convert: bool
) -> None:
    args, op = operator.args, operator.op
    values: tuple[_ShapeExpression, ...] | None
    shape: tuple[_ShapeExpression, ...]
    if isinstance(args, _InputArgs):
        _input(state, operator, args)
        return
    if isinstance(args, _ConstantArgs):
        _constant(state, operator, args)
        return
    if isinstance(args, (_CacheArgs, _OutputArgs)):
        return
    if isinstance(args, _CacheAssignArgs):
        actual = _shape(state, args.value)
        expected = args.cache.shape.dimensions
        if len(actual) != len(expected):
            raise ValueError(
                f"cache {args.cache.name!r} write rank differs from storage"
            )
        for axis, (dimension, capacity) in enumerate(
            zip(actual, expected, strict=True)
        ):
            assert capacity is not None
            predicate = "le" if axis == args.cache.axis else "eq"
            _constraint(
                state,
                predicate,
                dimension,
                _constant_expression(capacity),
                "cache write exceeds its declared shape",
                convert,
            )
        operator.args = args.model_copy(
            update={
                "value": _cast(state, region, args.value, args.cache.dtype, convert)
            }
        )
        return
    if isinstance(args, _CacheReadArgs):
        shape = tuple(
            _ShapeExpression(
                "dimension",
                tensor=operator.outputs[0],
                axis=axis,
                minimum=0,
                maximum=dimension,
            )
            if axis == args.cache.axis
            else _constant_expression(dimension)
            for axis, dimension in enumerate(args.cache.shape.dimensions)
            if dimension is not None
        )
        _store(
            state,
            operator.outputs[0],
            shape,
            args.cache.dtype,
        )
        return
    output = operator.outputs[0]
    if isinstance(args, _CompareArgs):
        dtype = _binary_dtype(_dtype(state, args.a), args.b, state, numeric=False)
        shape = _broadcast(
            state,
            _shape(state, args.a),
            _shape(state, args.b) if isinstance(args.b, Tensor) else (),
            convert,
        )
        operator.args = args.model_copy(
            update={
                "a": _cast(state, region, args.a, dtype, convert),
                "b": _operand(state, region, args.b, dtype, convert),
            }
        )
        a, b = _values(state, args.a), _values(state, args.b)
        values = None
        if a is not None and b is not None and len(a) == len(b) == 1:
            difference = _expression("sub", a[0], b[0])
            low, high = difference.minimum, difference.maximum
            negative = high is not None and high < 0
            positive = low is not None and low > 0
            nonpositive = high is not None and high <= 0
            nonnegative = low is not None and low >= 0
            zero = low == high == 0
            nonzero = negative or positive
            proven_true, proven_false = {
                "eq": (zero, nonzero),
                "ne": (nonzero, zero),
                "lt": (negative, nonnegative),
                "le": (nonpositive, positive),
                "gt": (positive, nonpositive),
                "ge": (nonnegative, negative),
            }[args.predicate]
            if proven_true or proven_false:
                values = (_constant_expression(int(proven_true)),)
        _store(state, output, shape, "bool", values, (0, 1))
    elif isinstance(args, _BinaryArgs):
        dtype = _binary_dtype(_dtype(state, args.a), args.b, state)
        if op == "div" and dtype not in _FLOATS:
            dtype = "fp32"
        if op in {"floor_div", "mod"} and not _integer(dtype):
            raise ValueError(f"{op} requires integer operands")
        shape = _broadcast(
            state,
            _shape(state, args.a),
            _shape(state, args.b) if isinstance(args.b, Tensor) else (),
            convert,
        )
        values = _binary_values(state, op, args.a, args.b) if _integer(dtype) else None
        value_range = (
            _numeric_range(state, op, args.a, args.b) if _integer(dtype) else None
        )
        operator.args = args.model_copy(
            update={
                "a": _cast(state, region, args.a, dtype, convert),
                "b": _operand(state, region, args.b, dtype, convert),
            }
        )
        _store(state, output, shape, dtype, values, value_range)
    elif isinstance(args, _CastArgs):
        cast_info = state.result.tensors[args.x]
        values = (
            cast_info.values
            if _integer(cast_info.dtype) and _integer(args.dtype)
            else None
        )
        _store(
            state, output, cast_info.shape, args.dtype, values, _range(state, args.x)
        )
    elif isinstance(args, _SelectArgs):
        _condition(state, args.condition, scalar=False)
        dtype_a = (
            _dtype(state, args.when_true)
            if isinstance(args.when_true, Tensor)
            else _literal_dtype(args.when_true)
        )
        dtype = _binary_dtype(dtype_a, args.when_false, state, numeric=False)
        if not isinstance(args.when_true, Tensor) and isinstance(
            args.when_false, Tensor
        ):
            dtype = _binary_dtype(
                _dtype(state, args.when_false), args.when_true, state, numeric=False
            )
        left = (
            _shape(state, args.when_true) if isinstance(args.when_true, Tensor) else ()
        )
        right = (
            _shape(state, args.when_false)
            if isinstance(args.when_false, Tensor)
            else ()
        )
        shape = _broadcast(
            state,
            _broadcast(state, left, right, convert),
            _shape(state, args.condition),
            convert,
        )
        values = _select_values(state, args) if _integer(dtype) else None
        true_range = (
            _range(state, args.when_true)
            if isinstance(args.when_true, Tensor)
            else (int(args.when_true), int(args.when_true))
            if isinstance(args.when_true, (bool, int))
            else (None, None)
        )
        false_range = (
            _range(state, args.when_false)
            if isinstance(args.when_false, Tensor)
            else (int(args.when_false), int(args.when_false))
            if isinstance(args.when_false, (bool, int))
            else (None, None)
        )
        operator.args = args.model_copy(
            update={
                "when_true": _operand(state, region, args.when_true, dtype, convert),
                "when_false": _operand(state, region, args.when_false, dtype, convert),
            }
        )
        _store(
            state,
            output,
            shape,
            dtype,
            values,
            None if values is not None else _join_range(true_range, false_range),
        )
    elif isinstance(args, _MatmulArgs):
        left, right = _shape(state, args.a), _shape(state, args.b)
        if len(left) < 2 or len(right) < 2:
            raise ValueError("matmul requires rank-two or higher tensors")
        _constraint(
            state,
            "eq",
            left[-1],
            right[-2],
            "matmul contracting dimensions differ",
            convert,
        )
        shape = (
            *_broadcast(state, left[:-2], right[:-2], convert),
            left[-2],
            right[-1],
        )
        dtype = _promote(_dtype(state, args.a), _dtype(state, args.b))
        _store(state, output, shape, dtype)
    elif isinstance(args, _IotaArgs):
        start, stop = _scalar(state, args.start), _scalar(state, args.stop)
        delta = (
            _expression("sub", stop, start)
            if args.step > 0
            else _expression("sub", start, stop)
        )
        length = _expression(
            "max",
            _constant_expression(0),
            _expression("ceil_div", delta, _constant_expression(abs(args.step))),
        )
        values = None
        if (
            start.op == stop.op == "const"
            and length.value is not None
            and length.value <= 4096
        ):
            assert start.value is not None and stop.value is not None
            values = tuple(
                _constant_expression(value)
                for value in range(start.value, stop.value, args.step)
            )
        limits = (
            (start.minimum, None if stop.maximum is None else stop.maximum - 1)
            if args.step > 0
            else (None if stop.minimum is None else stop.minimum + 1, start.maximum)
        )
        _store(state, output, (length,), args.dtype, values, limits)
    elif isinstance(args, _ReshapeArgs):
        source = _shape(state, args.x)
        reshape_dimensions = list(_dimensions(state, args.shape))
        inferred = [
            index
            for index, value in enumerate(reshape_dimensions)
            if value.op == "const" and value.value == -1
        ]
        if len(inferred) > 1:
            raise ValueError("reshape permits at most one inferred dimension")
        if inferred:
            index = inferred[0]
            denominator = _product(
                tuple(
                    value
                    for axis, value in enumerate(reshape_dimensions)
                    if axis != index
                )
            )
            _constraint(
                state,
                "ge",
                denominator,
                _constant_expression(1),
                "reshape inferred dimension has an ambiguous zero divisor",
                convert,
            )
            # Element-count equality below also rejects fractional inferred dimensions.
            reshape_dimensions[index] = _reshape_quotient(_product(source), denominator)
        for value in reshape_dimensions:
            _constraint(
                state,
                "ge",
                value,
                _constant_expression(0),
                "reshape dimensions cannot be negative",
                convert,
            )
        _constraint(
            state,
            "eq",
            _product(source),
            _product(tuple(reshape_dimensions)),
            "reshape changes the number of elements",
            convert,
        )
        info = state.result.tensors[args.x]
        _store(
            state,
            output,
            tuple(reshape_dimensions),
            info.dtype,
            info.values,
            _range(state, args.x),
        )
    elif isinstance(args, _PermuteArgs):
        source = _shape(state, args.x)
        if len(source) != len(args.axes):
            raise ValueError("permute axes must match tensor rank")
        info = state.result.tensors[args.x]
        values = info.values if len(source) <= 1 else None
        _store(
            state,
            output,
            tuple(source[axis] for axis in args.axes),
            info.dtype,
            values,
            _range(state, args.x),
        )
    elif isinstance(args, _BroadcastArgs):
        target = _dimensions(state, args.shape)
        source = _shape(state, args.x)
        if len(source) > len(target):
            raise ValueError("broadcast cannot remove tensor dimensions")
        result = _broadcast(state, source, target, convert)
        for actual_dimension, expected_dimension in zip(result, target, strict=True):
            _constraint(
                state,
                "eq",
                actual_dimension,
                expected_dimension,
                "broadcast input does not fit its target shape",
                convert,
            )
            _constraint(
                state,
                "ge",
                expected_dimension,
                _constant_expression(0),
                "broadcast dimensions cannot be negative",
                convert,
            )
        info = state.result.tensors[args.x]
        values = info.values if info.shape == target else None
        count = _product(target).value
        if (
            info.values is not None
            and len(info.values) == 1
            and count is not None
            and 0 <= count <= 4096
        ):
            values = info.values * count
        _store(state, output, target, info.dtype, values, _range(state, args.x))
    elif isinstance(args, (_GatherArgs, _ScatterArgs)):
        source = _shape(state, args.x)
        indices = _shape(state, args.indices)
        if not indices or indices[-1].op != "const" or indices[-1].value is None:
            raise ValueError("gather/scatter coordinate width must be statically known")
        coordinates = indices[-1].value
        if not 0 <= coordinates <= len(source) or not _integer(
            _dtype(state, args.indices)
        ):
            raise ValueError(
                "gather/scatter indices must be integers with valid coordinate width"
            )
        indexed = (*indices[:-1], *source[coordinates:])
        if args.mask is not None:
            _condition(state, args.mask, scalar=False)
            _broadcast(state, _shape(state, args.mask), indices[:-1], convert)
        updates: dict[str, object] = {
            "indices": _cast(state, region, args.indices, "int64", convert)
        }
        if isinstance(args, _GatherArgs):
            info = state.result.tensors[args.x]
            values = None
            coordinates_values = state.result.tensors[args.indices].values
            if (
                len(source) == 1
                and coordinates == 1
                and info.values is not None
                and coordinates_values is not None
                and all(value.op == "const" for value in coordinates_values)
            ):
                positions = [value.value for value in coordinates_values]
                if all(
                    position is not None and 0 <= position < len(info.values)
                    for position in positions
                ):
                    values = tuple(
                        info.values[position]
                        for position in positions
                        if position is not None
                    )
            _store(state, output, indexed, info.dtype, values, _range(state, args.x))
        else:
            dtype = _promote(
                _dtype(state, args.x),
                _dtype(state, args.values),
                numeric=args.mode == "add",
            )
            broadcast = _broadcast(state, _shape(state, args.values), indexed, convert)
            if len(broadcast) != len(indexed):
                raise ValueError("scatter updates cannot add tensor axes")
            for actual_dimension, expected_dimension in zip(
                broadcast, indexed, strict=True
            ):
                _constraint(
                    state,
                    "eq",
                    actual_dimension,
                    expected_dimension,
                    "scatter values do not fit indexed slices",
                    convert,
                )
            updates.update(
                x=_cast(state, region, args.x, dtype, convert),
                values=_cast(state, region, args.values, dtype, convert),
            )
            shape_values = state.result.tensors[args.x].values
            coordinates_values = state.result.tensors[args.indices].values
            update_values = state.result.tensors[args.values].values
            mask_values = (
                state.result.tensors[args.mask].values
                if args.mask is not None
                else (_constant_expression(1),)
            )
            propagated = None
            if (
                len(source) == 1
                and coordinates == 1
                and shape_values is not None
                and coordinates_values is not None
                and update_values is not None
                and mask_values is not None
                and all(
                    value.op == "const" for value in (*coordinates_values, *mask_values)
                )
                and len(update_values) in (1, len(coordinates_values))
                and len(mask_values) in (1, len(coordinates_values))
            ):
                replaced = list(shape_values)
                for index, position in enumerate(coordinates_values):
                    active = mask_values[index % len(mask_values)].value
                    coordinate = position.value
                    if active:
                        if coordinate is None or not 0 <= coordinate < len(replaced):
                            break
                        updated = update_values[index % len(update_values)]
                        replaced[coordinate] = (
                            updated
                            if args.mode == "replace"
                            else _expression("add", replaced[coordinate], updated)
                        )
                else:
                    propagated = tuple(replaced)
            _store(
                state,
                output,
                source,
                dtype,
                values=propagated,
                value_range=_join_range(
                    _range(state, args.x), _range(state, args.values)
                ),
            )
        operator.args = args.model_copy(update=updates)
    elif isinstance(args, _RandomArgs):
        shape = _dimensions(state, args.shape)
        for value in shape:
            _constraint(
                state,
                "ge",
                value,
                _constant_expression(0),
                "random dimensions cannot be negative",
                convert,
            )
        for random_scalar, name in ((args.seed, "seed"), (args.offset, "offset")):
            scalar = _scalar(state, random_scalar)
            _constraint(
                state,
                "ge",
                scalar,
                _constant_expression(0),
                f"random {name} cannot be negative",
                convert,
            )
            _constraint(
                state,
                "le",
                scalar,
                _constant_expression(2**64 - 1),
                f"random {name} exceeds uint64",
                convert,
            )
        _store(state, output, shape, args.dtype)
    elif isinstance(args, _UnaryArgs):
        shape = _shape(state, args.x)
        if op == "shape":
            _store(state, output, (_constant_expression(len(shape)),), "int64", shape)
        elif op in {"exp", "log", "sqrt", "sin", "cos", "erf"}:
            dtype = _dtype(state, args.x)
            if dtype not in _FLOATS:
                dtype = "fp32"
            operator.args = args.model_copy(
                update={"x": _cast(state, region, args.x, dtype, convert)}
            )
            _store(state, output, shape, dtype)
        else:
            raise TypeError(f"inference has no rule for core operator {op!r}")
    else:
        raise TypeError(f"inference has no rule for core operator {op!r}")


def _joined_shape(
    output: Tensor, left: _TensorInfo, right: _TensorInfo
) -> tuple[_ShapeExpression, ...]:
    if left.shape is None or right.shape is None or len(left.shape) != len(right.shape):
        raise ValueError("control-flow results must have a stable tensor rank")
    dimensions = []
    for axis, (a, b) in enumerate(zip(left.shape, right.shape, strict=True)):
        if a == b:
            dimensions.append(a)
        else:
            low = (
                None
                if a.minimum is None or b.minimum is None
                else min(a.minimum, b.minimum)
            )
            high = (
                None
                if a.maximum is None or b.maximum is None
                else max(a.maximum, b.maximum)
            )
            dimensions.append(
                _ShapeExpression(
                    "dimension", tensor=output, axis=axis, minimum=low, maximum=high
                )
            )
    return tuple(dimensions)


def _parameter(
    state: _InferenceState,
    tensor: Tensor,
    info: _TensorInfo,
    dtype: DType,
    value_range: tuple[int | None, int | None],
) -> None:
    shape = (
        None
        if info.shape is None
        else tuple(
            value
            if value.op == "const"
            else _ShapeExpression(
                "dimension",
                tensor=tensor,
                axis=axis,
                minimum=value.minimum,
                maximum=value.maximum,
            )
            for axis, value in enumerate(info.shape)
        )
    )
    _store(state, tensor, shape, dtype, value_range=value_range)


def _known_bool(state: _InferenceState, tensor: Tensor | _Scalar) -> bool | None:
    values = _values(state, tensor)
    if values is not None and len(values) == 1 and values[0].op == "const":
        return bool(values[0].value)
    return None


def _comparisons(
    state: _InferenceState, tensor: Tensor, truth: bool = True
) -> list[tuple[_CompareArgs, bool]]:
    producer = tensor._producer
    if producer is None:
        return []
    args = producer.args
    if isinstance(args, _ConstructArgs):
        return _comparisons(state, producer.regions[0].outputs[0], truth)
    if isinstance(args, _CastArgs) and _dtype(state, args.x) == "bool":
        return _comparisons(state, args.x, truth)
    if isinstance(args, _CompareArgs):
        return [(args, truth)]
    if isinstance(args, _SelectArgs):
        left, right = (
            _known_bool(state, args.when_true),
            _known_bool(state, args.when_false),
        )
        result = []
        if right is not None and right != truth and isinstance(args.when_true, Tensor):
            result.extend(_comparisons(state, args.condition, True))
            result.extend(_comparisons(state, args.when_true, truth))
        elif left is not None and left != truth and isinstance(args.when_false, Tensor):
            result.extend(_comparisons(state, args.condition, False))
            result.extend(_comparisons(state, args.when_false, truth))
        return result
    return []


def _guard_bounds(
    state: _InferenceState, condition: _Region
) -> dict[int, tuple[int | None, int | None]]:
    parameters = {tensor: index for index, tensor in enumerate(condition.parameters)}
    result: dict[int, tuple[int | None, int | None]] = {}
    for args, truth in _comparisons(state, condition.outputs[0]):
        predicate: str = args.predicate
        if not truth:
            predicate = {
                "lt": "ge",
                "le": "gt",
                "gt": "le",
                "ge": "lt",
                "eq": "ne",
                "ne": "eq",
            }[predicate]
        try:
            left = _scalar(state, args.a)
            right = (
                _scalar(state, args.b) if isinstance(args.b, (int, Tensor)) else None
            )
        except ValueError:
            continue
        if right is None:
            continue
        if right.op == "scalar" and right.tensor in parameters:
            left, right = right, left
            predicate = {
                "lt": "gt",
                "le": "ge",
                "gt": "lt",
                "ge": "le",
                "eq": "eq",
                "ne": "ne",
            }[predicate]
        if left.op != "scalar" or left.tensor not in parameters:
            continue
        assert left.tensor is not None
        index = parameters[left.tensor]
        low, high = result.get(index, (None, None))
        if predicate in {"lt", "le", "eq"} and right.maximum is not None:
            limit = right.maximum - int(predicate == "lt")
            high = limit if high is None else min(high, limit)
        if predicate in {"gt", "ge", "eq"} and right.minimum is not None:
            limit = right.minimum + int(predicate == "gt")
            low = limit if low is None else max(low, limit)
        result[index] = low, high
    return result


def _step(value: _ShapeExpression, parameter: Tensor) -> int | None:
    if value.op == "scalar" and value.tensor is parameter:
        return 0
    if value.op in {"add", "sub"}:
        left, right = value.args
        offset = _step(left, parameter)
        if offset is not None and right.op == "const" and right.value is not None:
            return offset + right.value if value.op == "add" else offset - right.value
        if value.op == "add" and left.op == "const" and left.value is not None:
            offset = _step(right, parameter)
            return None if offset is None else left.value + offset
    return None


def _iteration_bound(
    state: _InferenceState,
    operator: _Operator,
    initial: tuple[Tensor, ...],
    guards: dict[int, tuple[int | None, int | None]],
) -> int | None:
    body = operator.regions[1]
    candidates = []
    for index, (parameter, result) in enumerate(
        zip(body.parameters, body.outputs, strict=True)
    ):
        values = state.result.tensors[result].values
        if values is None or len(values) != 1:
            continue
        step = _step(values[0], parameter)
        low, high = _range(state, initial[index])
        guard_low, guard_high = guards.get(index, (None, None))
        if step is not None and step > 0 and low is not None and guard_high is not None:
            candidates.append(max(0, (guard_high - low) // step + 1))
        elif (
            step is not None and step < 0 and high is not None and guard_low is not None
        ):
            candidates.append(max(0, (high - guard_low) // -step + 1))
    # Cache capacity bounds iterations only with unconditional, monotonic growth.
    reset = {
        current.args.cache
        for _, current in _operators(body)
        if isinstance(current.args, _CacheAssignArgs)
        and current.args.action == "assign"
    }
    pending = list(body.operators)
    while pending:
        current = pending.pop()
        args = current.args
        if isinstance(args, _ConstructArgs):
            pending.extend(current.regions[0].operators)
        if (
            isinstance(args, _CacheAssignArgs)
            and args.action == "append"
            and args.cache.mode == "append"
            and args.cache not in reset
        ):
            axis = args.cache.axis
            assert axis is not None
            amount = _shape(state, args.value)[axis].minimum
            capacity = args.cache.shape.dimensions[axis]
            if amount is not None and amount > 0 and capacity is not None:
                candidates.append(capacity // amount)
    return min(candidates) if candidates else None


def _operators(region: _Region) -> list[tuple[_Region, _Operator]]:
    result = []
    for operator in region.operators:
        result.append((region, operator))
        for body in operator.regions:
            result.extend(_operators(body))
    return result


def _reduction_ranges(
    state: _InferenceState,
    operator: _Operator,
    args: _ReduceArgs,
    ranges: list[tuple[int | None, int | None]],
    unstable: set[int],
) -> list[tuple[int | None, int | None]]:
    source = _shape(state, args.values[0])
    axes = {axis % len(source) for axis in args.axes} if source else set()
    count = _product(
        tuple(value for axis, value in enumerate(source) if axis in axes)
    ).maximum
    body = operator.regions[0]
    for index, (parameter, output) in enumerate(zip(body.parameters, body.outputs)):
        if index >= len(args.initial):
            break
        values = state.result.tensors[output].values
        offset = None
        if values is not None and len(values) == 1 and values[0].op in {"add", "sub"}:
            left, right = values[0].args
            if left.op == "scalar" and left.tensor is parameter:
                offset = (
                    right
                    if values[0].op == "add"
                    else _expression("mul", _constant_expression(-1), right)
                )
            elif (
                values[0].op == "add"
                and right.op == "scalar"
                and right.tensor is parameter
            ):
                offset = left
        initial_low, initial_high = _range(state, args.initial[index])
        if count == 0:
            ranges[index] = initial_low, initial_high
        elif (
            offset is not None
            and count is not None
            and offset.minimum is not None
            and offset.maximum is not None
            and initial_low is not None
            and initial_high is not None
        ):
            ranges[index] = (
                initial_low + min(0, count * offset.minimum),
                initial_high + max(0, count * offset.maximum),
            )
        elif index in unstable:
            ranges[index] = _type_range(_dtype(state, output)) or (None, None)
    return ranges


def _control(
    state: _InferenceState, parent: _Region, operator: _Operator, convert: bool
) -> None:
    args = operator.args
    if isinstance(args, _ConstructArgs):
        body = operator.regions[0]
        _region(state, body, convert)
        for output, value in zip(operator.outputs, body.outputs, strict=True):
            _copy(state, output, value)
        return
    if isinstance(args, _IfArgs):
        _condition(state, args.condition, scalar=True)
        left, right = operator.regions
        _region(state, left, convert)
        _region(state, right, convert)
        left_outputs, right_outputs = list(left.outputs), list(right.outputs)
        for index, (output, a, b) in enumerate(
            zip(operator.outputs, left.outputs, right.outputs, strict=True)
        ):
            dtype = _promote(_dtype(state, a), _dtype(state, b), numeric=False)
            shape = _joined_shape(
                output, state.result.tensors[a], state.result.tensors[b]
            )
            left_outputs[index] = _cast(state, left, a, dtype, convert)
            right_outputs[index] = _cast(state, right, b, dtype, convert)
            values_a, values_b = (
                state.result.tensors[a].values,
                state.result.tensors[b].values,
            )
            _store(
                state,
                output,
                shape,
                dtype,
                values_a if values_a == values_b else None,
                _join_range(_range(state, a), _range(state, b)),
            )
        if convert:
            left.outputs, right.outputs = tuple(left_outputs), tuple(right_outputs)
        return
    if not isinstance(args, (_WhileArgs, _ReduceArgs)):
        raise TypeError(f"inference has no region rule for {operator.op!r}")
    initial = args.initial
    if isinstance(args, _WhileArgs):
        condition, body = operator.regions
    else:
        body = operator.regions[0]
        for value in args.initial:
            if _shape(state, value) != ():
                raise ValueError("reduction initial values must be scalar")
        for parameter, value in zip(
            body.parameters[len(initial) :], args.values, strict=True
        ):
            _store(
                state,
                parameter,
                (),
                _dtype(state, value),
                value_range=_range(state, value),
            )
    infos = [state.result.tensors[value] for value in initial]
    dtypes = [_dtype(state, value) for value in initial]
    ranges = [_range(state, value) for value in initial]
    guards: dict[int, tuple[int | None, int | None]] = {}
    known_iterations: int | None = None
    unstable_ranges: set[int] = set()
    previous: (
        tuple[
            tuple[DType, ...],
            tuple[tuple[int | None, ...] | None, ...],
            tuple[tuple[int | None, int | None], ...],
        ]
        | None
    ) = None
    for _ in range(12):
        if isinstance(args, _WhileArgs):
            for parameter, info, dtype, interval in zip(
                condition.parameters, infos, dtypes, ranges, strict=True
            ):
                _parameter(state, parameter, info, dtype, interval)
            _region(state, condition, False)
            _condition(state, condition.outputs[0], scalar=True)
            guards = _guard_bounds(state, condition)
        for index, (parameter, info, dtype, interval) in enumerate(
            zip(body.parameters[: len(initial)], infos, dtypes, ranges, strict=True)
        ):
            low, high = interval
            guard_low, guard_high = guards.get(index, (None, None))
            if guard_low is not None:
                low = guard_low
            if guard_high is not None:
                high = guard_high
            previous_values = (
                state.result.tensors[body.outputs[index]].values
                if body.outputs[index] in state.result.tensors
                else None
            )
            if (
                isinstance(args, _WhileArgs)
                and known_iterations is not None
                and previous_values is not None
                and len(previous_values) == 1
            ):
                step = _step(previous_values[0], parameter)
                initial_low, initial_high = _range(state, initial[index])
                if step is not None and step > 0 and initial_high is not None:
                    limit = initial_high + max(0, known_iterations - 1) * step
                    high = limit if high is None else min(high, limit)
                elif step is not None and step < 0 and initial_low is not None:
                    limit = initial_low + max(0, known_iterations - 1) * step
                    low = limit if low is None else max(low, limit)
            if low is not None and high is not None and low > high:
                low = high
            _parameter(state, parameter, info, dtype, (low, high))
        _region(state, body, False)
        if isinstance(args, _WhileArgs):
            known_iterations = _iteration_bound(state, operator, initial, guards)
        joined_infos = []
        joined_dtypes = []
        joined_ranges = []
        for index, (value, result) in enumerate(
            zip(initial, body.outputs, strict=True)
        ):
            dtype = _promote(_dtype(state, value), _dtype(state, result), numeric=False)
            start, end = state.result.tensors[value], state.result.tensors[result]
            if isinstance(args, _ReduceArgs):
                if end.shape != ():
                    raise ValueError("reducer must return scalar accumulator values")
                shape = ()
            else:
                shape = _joined_shape(body.parameters[index], start, end)
            joined_infos.append(
                _TensorInfo(shape, tuple(d.maximum for d in shape), dtype)
            )
            joined_dtypes.append(dtype)
            interval = _join_range(_range(state, value), _range(state, result))
            values = state.result.tensors[result].values
            if (
                isinstance(args, _WhileArgs)
                and known_iterations is not None
                and values is not None
                and len(values) == 1
            ):
                step = _step(values[0], body.parameters[index])
                initial_low, initial_high = _range(state, value)
                if (
                    step is not None
                    and initial_low is not None
                    and initial_high is not None
                ):
                    interval = (
                        initial_low + min(0, known_iterations * step),
                        initial_high + max(0, known_iterations * step),
                    )
                    guard_low, guard_high = guards.get(index, (None, None))
                    if step > 0 and guard_high is not None:
                        interval = (
                            interval[0],
                            max(initial_high, min(interval[1], guard_high + step)),
                        )
                    elif step < 0 and guard_low is not None:
                        interval = (
                            min(initial_low, max(interval[0], guard_low + step)),
                            interval[1],
                        )
            joined_ranges.append(interval)
        current = (
            tuple(joined_dtypes),
            tuple(info.maximum_shape for info in joined_infos),
            tuple(joined_ranges),
        )
        infos, dtypes, ranges = joined_infos, joined_dtypes, joined_ranges
        if (
            current == previous
            or isinstance(args, _ReduceArgs)
            and previous is not None
            and current[:2] == previous[:2]
        ):
            if isinstance(args, _ReduceArgs) and previous is not None:
                unstable_ranges = {
                    index
                    for index, (a, b) in enumerate(
                        zip(current[2], previous[2], strict=True)
                    )
                    if a != b
                }
            break
        previous = current
    else:
        raise ValueError(
            f"{operator.op} carried shapes or values have no proven finite fixed point"
        )
    if isinstance(args, _WhileArgs):
        iteration_limit = _iteration_bound(state, operator, initial, guards)
        if iteration_limit is not None:
            state.result.loop_iterations[operator] = iteration_limit
        for parameter, info, dtype, interval in zip(
            condition.parameters, infos, dtypes, ranges, strict=True
        ):
            _parameter(state, parameter, info, dtype, interval)
        _region(state, condition, convert)
        if convert:
            _region(state, body, True)
        output_shape = [info.shape for info in infos]
    else:
        if convert:
            _region(state, body, True)
        source = _shape(state, args.values[0])
        for value in args.values[1:]:
            shape = _shape(state, value)
            if len(shape) != len(source):
                raise ValueError("reduction values must have the same rank")
            for left_dimension, right_dimension in zip(source, shape, strict=True):
                _constraint(
                    state,
                    "eq",
                    left_dimension,
                    right_dimension,
                    "reduction value shapes differ",
                    convert,
                )
        rank = len(source)
        if any(not -rank <= axis < rank for axis in args.axes):
            raise ValueError("reduction axis is outside tensor rank")
        axes = {axis % rank for axis in args.axes}
        if len(axes) != len(args.axes):
            raise ValueError("reduction axes alias after normalization")
        shape = tuple(
            _constant_expression(1) if axis in axes else value
            for axis, value in enumerate(source)
            if args.keepdims or axis not in axes
        )
        output_shape = [shape] * len(initial)
        ranges = _reduction_ranges(state, operator, args, ranges, unstable_ranges)
    converted_initial = tuple(
        _cast(state, parent, value, dtype, convert)
        for value, dtype in zip(initial, dtypes, strict=True)
    )
    if convert:
        operator.args = args.model_copy(update={"initial": converted_initial})
        body.outputs = tuple(
            _cast(state, body, value, dtype, True)
            for value, dtype in zip(body.outputs, dtypes, strict=True)
        )
    for output, final_shape, dtype, interval in zip(
        operator.outputs, output_shape, dtypes, ranges, strict=True
    ):
        if isinstance(args, _WhileArgs) and final_shape is not None:
            final_shape = tuple(
                value
                if value.op == "const"
                else _ShapeExpression(
                    "dimension",
                    tensor=output,
                    axis=axis,
                    minimum=value.minimum,
                    maximum=value.maximum,
                )
                for axis, value in enumerate(final_shape)
            )
        _store(state, output, final_shape, dtype, value_range=interval)


def _region(state: _InferenceState, region: _Region, convert: bool) -> None:
    for operator in tuple(_ordered_operators(region)):
        state.operators.append(operator)
        try:
            if operator.regions:
                _control(state, region, operator, convert)
            else:
                _ordinary(state, region, operator, convert)
        except (ValueError, TypeError) as error:
            error.add_note(f"While inferring TDL operator {operator.op!r}")
            raise
        finally:
            state.operators.pop()


def _infer(
    graph: Graph,
    *,
    max_input_size: int | None = None,
) -> _Inference:
    """Infer logical tensor facts and insert promotions on the existing graph."""
    if graph._region is None:
        raise ValueError("graph must be completed before inference")
    if max_input_size is not None and max_input_size <= 0:
        raise ValueError("maximum input size must be positive")
    state = _InferenceState(graph, max_input_size)
    _region(state, graph._region, True)
    _rebuild(graph)
    return state.result
