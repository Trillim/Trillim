from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable
from dataclasses import replace
from fractions import Fraction

import numpy as np
from pydantic import BaseModel

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    Weight,
    _BinaryArgs,
    _BroadcastArgs,
    _CastArgs,
    _CompareArgs,
    _ConstantArgs,
    _ConstructArgs,
    _GatherArgs,
    _InputArgs,
    _MatmulArgs,
    _PermuteArgs,
    _ReduceArgs,
    _ReshapeArgs,
    _ScatterArgs,
    _SelectArgs,
    _UnaryArgs,
)
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import _Inference, _ShapeExpression
from .algebra import (
    _function,
    _inherit,
    _node,
    _number,
    _power,
    _product,
    _sum,
)
from .passes_dataclasses import _Canonical, _MatchingState

_POINTWISE = frozenset(
    {"add", "mul", "min", "max", "abs", "exp", "log", "sin", "cos", "erf", "power"}
)


def _pure(region: _Region, state: _MatchingState) -> bool:
    identity = id(region)
    if identity not in state.purity:
        state.purity[identity] = (
            not region.reads
            and not region.writes
            and all(
                operator.op
                not in {"cache", "cache_assign", "cache_read", "while", "if"}
                and all(_pure(body, state) for body in operator.regions)
                for operator in region.operators
            )
        )
    return state.purity[identity]


def _unwrapped(
    tensor: Tensor, state: _MatchingState, covered: set[_Operator] | None = None
) -> Tensor:
    original = tensor
    if original not in state.unwrapped:
        wrappers: list[_Operator] = []
        while (
            (producer := tensor._producer) is not None
            and isinstance(producer.args, _ConstructArgs)
            and _pure(producer.regions[0], state)
        ):
            wrappers.append(producer)
            tensor = producer.regions[0].outputs[producer.outputs.index(tensor)]
        state.unwrapped[original] = tensor, tuple(wrappers)
    result, wrappers_tuple = state.unwrapped[original]
    if covered is not None:
        covered.update(wrappers_tuple)
    return result


def _canonical(
    tensor: Tensor,
    inference: _Inference,
    state: _MatchingState,
    covered: set[_Operator] | None = None,
) -> Tensor:
    while True:
        tensor = _unwrapped(tensor, state, covered)
        operator = tensor._producer
        if operator is None:
            return tensor
        args = operator.args
        nodes = [operator]
        wrappers: set[_Operator] = set()
        if isinstance(args, _CastArgs):
            source = args.x
            if inference.tensors[source].dtype != inference.tensors[tensor].dtype:
                return tensor
        elif isinstance(args, _ReshapeArgs):
            source = _unwrapped(args.x, state, wrappers)
            while source._producer is not None and isinstance(
                source._producer.args, _ReshapeArgs
            ):
                nodes.append(source._producer)
                source = _unwrapped(source._producer.args.x, state, wrappers)
            if inference.tensors[source].shape != inference.tensors[tensor].shape:
                return tensor
        elif isinstance(args, _PermuteArgs):
            source = _unwrapped(args.x, state, wrappers)
            axes = args.axes
            while source._producer is not None and isinstance(
                source._producer.args, _PermuteArgs
            ):
                previous = source._producer.args
                axes = tuple(previous.axes[axis] for axis in axes)
                nodes.append(source._producer)
                source = _unwrapped(previous.x, state, wrappers)
            if axes != tuple(range(len(axes))):
                return tensor
        else:
            return tensor
        if covered is not None:
            covered.update(nodes)
            covered.update(wrappers)
        tensor = source


def _terms(
    tensor: Tensor,
    operation: str,
    dtype: DType,
    inference: _Inference,
    state: _MatchingState,
    leaves: dict[Tensor, str],
    covered: set[_Operator],
) -> tuple[Tensor, ...] | None:
    pending = [tensor]
    result: list[Tensor] = []
    while pending:
        value = pending.pop()
        if value not in leaves:
            value = _canonical(value, inference, state, covered)
        operator = value._producer
        if (
            value not in leaves
            and operator is not None
            and operator.op == operation
            and inference.tensors[value].dtype == dtype
            and isinstance(operator.args, _BinaryArgs)
            and isinstance(operator.args.b, Tensor)
        ):
            covered.add(operator)
            pending.extend((operator.args.b, operator.args.a))
        else:
            result.append(value)
        if len(result) + len(pending) > 16:
            return None
    return tuple(result)


def _normal(
    tensor: Tensor,
    inference: _Inference,
    state: _MatchingState,
    context: tuple[Tensor, ...] = (),
) -> _Canonical:
    cache_key = tensor, context
    if cache_key in state.normal:
        return state.normal[cache_key]
    if len(state.normal_active) >= 64:
        return replace(_node(state, "leaf", value=tensor), sources=(tensor,))
    if cache_key in state.normal_active:
        return _node(state, "leaf", value=tensor)
    state.normal_active.add(cache_key)
    try:
        covered: set[_Operator] = set()
        value = _unwrapped(tensor, state, covered)
        operator = value._producer
        if value in context:
            result = _node(state, "parameter", value=context.index(value))
        elif operator is None or isinstance(operator.args, _InputArgs):
            result = _node(state, "leaf", value=value)
            bounds = inference.value_ranges.get(value, (None, None))
            dtype = inference.tensors[value].dtype
            result = replace(
                result,
                lower=bounds[0],
                upper=bounds[1],
                finite=dtype is not None and dtype.startswith(("int", "uint", "bool")),
            )
        elif (
            operator.op in {"cache", "cache_read", "cache_assign", "while", "if"}
            or operator.regions
            and not (
                isinstance(operator.args, _ReduceArgs)
                and all(_pure(body, state) for body in operator.regions)
            )
        ):
            result = _node(state, "leaf", value=value)
        else:
            args = operator.args
            children: list[_Canonical] = []

            def encode(item: object) -> Hashable:
                if isinstance(item, Tensor):
                    children.append(_normal(item, inference, state, context))
                    return ("input", len(children) - 1)
                if isinstance(item, Weight):
                    return ("weight", item.root)
                if isinstance(item, np.ndarray):
                    if item.size <= 64 and item.dtype.kind in "biuf":
                        return (
                            "array",
                            item.shape,
                            tuple(_scalar_key(number.item()) for number in item.flat),
                        )
                    return (
                        "array",
                        item.dtype.str,
                        item.shape,
                        hashlib.sha256(item.tobytes()).digest(),
                    )
                if isinstance(item, (tuple, list)):
                    return tuple(encode(part) for part in item)
                if isinstance(item, dict):
                    return tuple((name, encode(part)) for name, part in item.items())
                if isinstance(item, BaseModel):
                    return tuple(
                        (name, encode(part)) for name, part in vars(item).items()
                    )
                if isinstance(item, (int, float, bool)):
                    return _scalar_key(item)
                if isinstance(item, (str, bytes)) or item is None:
                    return item
                return (type(item).__qualname__, repr(item))

            if isinstance(args, _ConstantArgs):
                item = args.value
                if isinstance(item, np.ndarray) and item.shape == ():
                    item = item.item()
                if isinstance(item, (int, float, bool)):
                    result = _number(state, item)
                else:
                    result = (
                        _node(
                            state,
                            "dimensions",
                            tuple(_number(state, int(number)) for number in item),
                        )
                        if isinstance(item, np.ndarray)
                        and item.ndim == 1
                        and item.dtype.kind in "iu"
                        else _node(state, "literal", value=encode(item))
                    )
                    if (
                        isinstance(item, np.ndarray)
                        and item.dtype.kind in "biuf"
                        and item.size
                    ):
                        result = replace(
                            result,
                            lower=float(np.asarray(item, dtype=np.float64).min()),
                            upper=float(np.asarray(item, dtype=np.float64).max()),
                            finite=bool(np.isfinite(item).all()),
                        )
            elif isinstance(args, _CastArgs):
                result = _normal(args.x, inference, state, context)
                covered.add(operator)
            elif isinstance(args, _BinaryArgs):
                a = _normal(args.a, inference, state, context)
                b = (
                    _normal(args.b, inference, state, context)
                    if isinstance(args.b, Tensor)
                    else _number(state, args.b)
                )
                if isinstance(args, _CompareArgs):
                    result = _comparison(state, args.predicate, a, b)
                elif operator.op == "add":
                    result = _sum(state, (a, b))
                elif operator.op == "sub":
                    result = _sum(state, (a, _product(state, (_number(state, -1), b))))
                elif operator.op == "mul":
                    result = _product(state, (a, b))
                elif operator.op == "div":
                    result = _product(state, (a, _power(state, b, Fraction(-1))))
                else:
                    result = _node(state, operator.op, (a, b))
                covered.add(operator)
            elif isinstance(args, _SelectArgs):
                condition = _normal(args.condition, inference, state, context)
                a = (
                    _normal(args.when_true, inference, state, context)
                    if isinstance(args.when_true, Tensor)
                    else _number(state, args.when_true)
                )
                b = (
                    _normal(args.when_false, inference, state, context)
                    if isinstance(args.when_false, Tensor)
                    else _number(state, args.when_false)
                )
                result = _selection(state, condition, a, b)
                covered.add(operator)
            elif isinstance(args, _UnaryArgs) and operator.op == "shape":
                result = _shape_of(args.x, inference, state, context)
                covered.add(operator)
            elif isinstance(args, _UnaryArgs) and operator.op in {
                "exp",
                "log",
                "sqrt",
                "sin",
                "cos",
                "erf",
            }:
                result = _function(
                    state, operator.op, _normal(args.x, inference, state, context)
                )
                covered.add(operator)
            elif isinstance(args, _PermuteArgs):
                result = _permutation(
                    state,
                    _normal(args.x, inference, state, context),
                    args.axes,
                    inference,
                )
                covered.add(operator)
            elif isinstance(args, (_ReshapeArgs, _BroadcastArgs)):
                shape = _shape_of(value, inference, state, context)
                original = _normal(args.x, inference, state, context)
                op = "reshape" if isinstance(args, _ReshapeArgs) else "broadcast"
                result = _view(state, op, original, shape, inference)
                covered.add(operator)
            elif isinstance(args, _MatmulArgs):
                ranks = [inference.tensors[item].shape for item in (args.a, args.b)]
                a, b = (
                    _normal(item, inference, state, context)
                    for item in (args.a, args.b)
                )
                result = (
                    _matrix(state, (a, b), inference)
                    if all(shape is not None and len(shape) >= 2 for shape in ranks)
                    else _node(state, "matmul_vector", (a, b))
                )
                covered.add(operator)
            elif isinstance(args, _GatherArgs):
                source = _normal(args.x, inference, state, context)
                indices = _normal(args.indices, inference, state, context)
                mask = (
                    _normal(args.mask, inference, state, context)
                    if args.mask is not None
                    else None
                )
                if (
                    mask is not None
                    and _uniform(mask).op == "constant"
                    and _uniform(mask).value == 0
                ):
                    result = _inherit(_number(state, args.fill), (mask,))
                elif args.mask is None and source.op == "dimensions":
                    coordinates = _integer_values(args.indices, inference)
                    if (
                        coordinates is not None
                        and len(coordinates) == 1
                        and 0 <= coordinates[0] < len(source.arguments)
                    ):
                        result = _inherit(
                            source.arguments[coordinates[0]], (source, indices)
                        )
                    else:
                        result = _node(
                            state, "gather", (source, indices), (args.fill, False)
                        )
                elif (
                    args.mask is None
                    and source.op == "scatter"
                    and source.value == "replace"
                    and source.arguments[1].key == indices.key
                    and _valid_indices(args.indices, args.x, inference)
                ):
                    result = _inherit(source.arguments[2], (source, indices))
                else:
                    result = _node(
                        state,
                        "gather",
                        (source, indices, *((mask,) if mask is not None else ())),
                        (args.fill, mask is not None),
                    )
                covered.add(operator)
            elif isinstance(args, _ScatterArgs):
                source = _normal(args.x, inference, state, context)
                indices = _normal(args.indices, inference, state, context)
                update = _normal(args.values, inference, state, context)
                mask = (
                    _normal(args.mask, inference, state, context)
                    if args.mask is not None
                    else None
                )
                if (
                    mask is not None
                    and _uniform(mask).op == "constant"
                    and _uniform(mask).value == 0
                ):
                    result = _inherit(source, (mask,))
                elif (
                    args.mode == "replace"
                    and mask is None
                    and update.op == "gather"
                    and len(update.arguments) == 2
                    and update.arguments[0].key == source.key
                    and update.arguments[1].key == indices.key
                    and _valid_indices(args.indices, args.x, inference)
                ):
                    result = _inherit(source, (update, indices))
                else:
                    result = _node(
                        state,
                        "scatter",
                        (
                            source,
                            indices,
                            update,
                            *((mask,) if mask is not None else ()),
                        ),
                        args.mode if mask is None else (args.mode, "masked"),
                    )
                covered.add(operator)
            elif isinstance(args, _ReduceArgs):
                result = _normal_reduction(
                    value, operator, args, inference, state, context
                )
                covered.add(operator)
            else:
                attributes = tuple(
                    (name, encode(item)) for name, item in vars(args).items()
                )
                result = _node(state, operator.op, tuple(children), attributes)
                covered.add(operator)
        if operator is not None and isinstance(operator.args, _ConstantArgs):
            covered.discard(operator)
        for _ in range(8):
            simplified = _rewrite(state, result)
            if simplified.key == result.key:
                break
            result = _inherit(simplified, (result,))
        sources = tuple(dict.fromkeys((tensor, *result.sources)))
        result = replace(result, sources=sources, origins=result.origins | covered)
        state.normal[cache_key] = result
        return result
    finally:
        state.normal_active.remove(cache_key)


def _scalar_key(value: float | bool) -> Hashable:
    if isinstance(value, float) and not math.isfinite(value):
        return ("nonfinite", repr(value))
    return Fraction(value)


def _comparison(
    state: _MatchingState, predicate: str, a: _Canonical, b: _Canonical
) -> _Canonical:
    if predicate in {"lt", "le"}:
        a, b = b, a
        predicate = "gt" if predicate == "lt" else "ge"
    if predicate in {"eq", "ne"} and a.key > b.key:
        a, b = b, a
    answer: bool | None = None
    if a.key == b.key:
        if predicate in {"lt", "gt"}:
            answer = False
        elif a.finite:
            answer = predicate in {"eq", "ge", "le"}
    elif a.lower is not None and b.upper is not None and a.lower > b.upper:
        answer = predicate in {"gt", "ge", "ne"}
    elif a.upper is not None and b.lower is not None and a.upper < b.lower:
        answer = predicate == "ne"
    elif (
        a.op == b.op == "constant"
        and isinstance(a.value, Fraction)
        and isinstance(b.value, Fraction)
    ):
        answer = {
            "eq": a.value == b.value,
            "ne": a.value != b.value,
            "gt": a.value > b.value,
            "ge": a.value >= b.value,
        }[predicate]
    return (
        _inherit(_number(state, int(answer)), (a, b))
        if answer is not None
        else _node(state, "compare", (a, b), predicate)
    )


def _selection(
    state: _MatchingState, condition: _Canonical, a: _Canonical, b: _Canonical
) -> _Canonical:
    if a.key == b.key:
        return _inherit(a, (condition, b))
    if condition.op == "constant" and isinstance(condition.value, Fraction):
        return _inherit(a if condition.value else b, (condition,))
    if condition.op == "compare":
        x, y = condition.arguments
        predicate = condition.value
        if {a.key, b.key} == {x.key, y.key} and predicate in {"gt", "ge"}:
            op = "max" if a.key == x.key else "min"
            return _inherit(_extremum(state, op, (x, y)), (condition,))
    return _node(state, "select", (condition, a, b))


def _extremum(
    state: _MatchingState, op: str, values: tuple[_Canonical, ...]
) -> _Canonical:
    entries: dict[int, _Canonical] = {}
    removed: list[_Canonical] = []
    for value in values:
        if value.op == op:
            entries.update((part.key, part) for part in value.arguments)
            removed.append(value)
        else:
            entries[value.key] = value
    result = tuple(sorted(entries.values(), key=lambda part: part.key))
    if len(result) == 1:
        return _inherit(result[0], removed)
    if len(result) == 2:
        a, b = result
        negative = _product(state, (_number(state, -1), a), expand=False)
        if negative.key == b.key and op == "max":
            return _inherit(_function(state, "abs", a), (*removed, b))
        negative = _product(state, (_number(state, -1), b), expand=False)
        if negative.key == a.key and op == "max":
            return _inherit(_function(state, "abs", b), (*removed, a))
        if a.lower is not None and b.upper is not None and a.lower >= b.upper:
            return _inherit(a if op == "max" else b, (*removed, a, b))
        if b.lower is not None and a.upper is not None and b.lower >= a.upper:
            return _inherit(b if op == "max" else a, (*removed, a, b))
    return _inherit(_node(state, op, result), removed)


def _shape_expression(
    value: _ShapeExpression,
    inference: _Inference,
    state: _MatchingState,
    context: tuple[Tensor, ...],
) -> _Canonical:
    key = value, id(inference), context
    cached = state.shape_normal.get(key)
    if cached is not None:
        return cached[1]
    if value.op == "const":
        assert value.value is not None
        result = _number(state, value.value)
    elif value.op == "scalar" and value.tensor is not None:
        source = _normal(value.tensor, inference, state, context)
        if inference.tensors[value.tensor].shape == ():
            result = source
        elif source.op == "dimensions" and value.axis is not None:
            result = source.arguments[value.axis]
        else:
            result = _node(state, "element", (source,), value.axis)
    elif value.op == "dimension" and value.tensor is not None and value.axis is not None:
        shape = inference.tensors[value.tensor].shape
        if shape is not None and shape[value.axis] != value:
            result = _shape_expression(shape[value.axis], inference, state, context)
        else:
            source = replace(
                _node(state, "leaf", value=value.tensor), sources=(value.tensor,)
            )
            result = _node(state, "dimension", (source,), value.axis)
    else:
        arguments = tuple(
            _shape_expression(argument, inference, state, context)
            for argument in value.args
        )
        if value.op == "add":
            result = _sum(state, arguments)
        elif value.op == "sub":
            result = _sum(
                state,
                (arguments[0], _product(state, (_number(state, -1), arguments[1]))),
            )
        elif value.op == "mul":
            result = _product(state, arguments)
        elif value.op in {"min", "max"}:
            result = _extremum(state, value.op, arguments)
        else:
            result = _node(state, value.op, arguments, value.cache)
    # Retain the inference along with its identity used in the cache key.
    state.shape_normal[key] = inference, result
    return result


def _shape_of(
    tensor: Tensor,
    inference: _Inference,
    state: _MatchingState,
    context: tuple[Tensor, ...] = (),
) -> _Canonical:
    shape = inference.tensors[tensor].shape
    assert shape is not None
    return _node(
        state,
        "dimensions",
        tuple(_shape_expression(value, inference, state, context) for value in shape),
    )


def _matrix(
    state: _MatchingState,
    values: tuple[_Canonical, ...],
    inference: _Inference | None = None,
    *,
    expand: bool = True,
) -> _Canonical:
    factors: list[_Canonical] = []
    removed: list[_Canonical] = []
    for value in values:
        if value.op == "matmul":
            factors.extend(value.arguments)
            removed.append(value)
        else:
            factors.append(value)
    coefficient = Fraction(1)
    stripped: list[_Canonical] = []
    for factor in factors:
        if (
            factor.op == "mul"
            and factor.arguments[0].op == "constant"
            and isinstance(factor.arguments[0].value, Fraction)
        ):
            coefficient *= factor.arguments[0].value
            stripped.append(_product(state, factor.arguments[1:], expand=False))
            removed.append(factor)
        else:
            stripped.append(factor)
    factors = stripped
    alternatives: list[tuple[_Canonical, ...]] = []
    for factor in factors:
        shape = (
            inference.tensors[factor.sources[0]].shape
            if inference is not None and factor.sources
            else None
        )
        distribute = (
            expand
            and factor.op == "add"
            and shape is not None
            and all(
                part.sources
                and inference is not None
                and inference.tensors[part.sources[0]].shape == shape
                for part in factor.arguments
            )
        )
        alternatives.append(factor.arguments if distribute else (factor,))
    count = math.prod(len(parts) for parts in alternatives)
    if expand and 1 < count <= 32:
        products: list[tuple[_Canonical, ...]] = [()]
        for parts in alternatives:
            products = [(*prefix, part) for prefix in products for part in parts]
        value = _sum(
            state,
            (_matrix(state, parts, inference, expand=False) for parts in products),
        )
        removed.extend(
            factor
            for factor, parts in zip(factors, alternatives, strict=True)
            if len(parts) > 1
        )
    else:
        value = _node(state, "matmul", tuple(factors))
    if coefficient != 1:
        value = _product(state, (_number(state, coefficient), value))
    return _inherit(value, removed)


def _permutation(
    state: _MatchingState,
    value: _Canonical,
    axes: tuple[int, ...],
    inference: _Inference,
) -> _Canonical:
    removed: list[_Canonical] = []
    while value.op == "permute" and isinstance(value.value, tuple):
        previous = value.value
        if not all(isinstance(axis, int) for axis in previous):
            break
        axes = tuple(int(previous[axis]) for axis in axes)
        removed.append(value)
        value = value.arguments[0]
    if axes == tuple(range(len(axes))):
        return _inherit(value, removed)
    transpose = tuple(range(len(axes) - 2)) + (len(axes) - 1, len(axes) - 2)
    if len(axes) >= 2 and axes == transpose and value.op == "matmul":
        factors: list[_Canonical] = []
        for factor in reversed(value.arguments):
            shape = (
                inference.tensors[factor.sources[0]].shape if factor.sources else None
            )
            if shape is None or len(shape) < 2:
                break
            order = tuple(range(len(shape) - 2)) + (len(shape) - 1, len(shape) - 2)
            factors.append(_permutation(state, factor, order, inference))
        else:
            return _inherit(
                _matrix(state, tuple(factors), inference), (*removed, value)
            )
    if value.op in _POINTWISE and value.sources:
        original = inference.tensors[value.sources[0]].shape
        if all(
            part.op == "constant"
            or part.sources
            and inference.tensors[part.sources[0]].shape == original
            for part in value.arguments
        ):
            parts = tuple(
                part
                if part.op == "constant"
                else _permutation(state, part, axes, inference)
                for part in value.arguments
            )
            return _inherit(
                _rewrite(state, _node(state, value.op, parts, value.value)),
                (*removed, value),
            )
    return _inherit(_node(state, "permute", (value,), axes), removed)


def _view(
    state: _MatchingState,
    op: str,
    value: _Canonical,
    shape: _Canonical,
    inference: _Inference,
) -> _Canonical:
    removed: list[_Canonical] = []
    while value.op == op:
        removed.append(value)
        value = value.arguments[0]
    if value.sources and _shape_of(value.sources[0], inference, state).key == shape.key:
        return _inherit(value, removed)
    if value.op in _POINTWISE and value.sources:
        original = inference.tensors[value.sources[0]].shape
        if all(
            part.op == "constant"
            or part.sources
            and inference.tensors[part.sources[0]].shape == original
            for part in value.arguments
        ):
            parts = tuple(
                part
                if part.op == "constant"
                else _view(state, op, part, shape, inference)
                for part in value.arguments
            )
            return _inherit(
                _rewrite(state, _node(state, value.op, parts, value.value)),
                (*removed, value),
            )
    return _inherit(_node(state, op, (value, shape)), removed)


def _uniform(value: _Canonical) -> _Canonical:
    while value.op in {"broadcast", "reshape", "permute"}:
        value = value.arguments[0]
    return value


def _normal_reduction(
    tensor: Tensor,
    operator: _Operator,
    args: _ReduceArgs,
    inference: _Inference,
    state: _MatchingState,
    context: tuple[Tensor, ...],
) -> _Canonical:
    body = operator.regions[0]
    nested = (*context, *body.parameters)
    outputs = tuple(_normal(value, inference, state, nested) for value in body.outputs)
    values = tuple(_normal(value, inference, state, context) for value in args.values)
    initial = tuple(_normal(value, inference, state, context) for value in args.initial)
    shape = inference.tensors[args.values[0]].shape
    assert shape is not None
    axes = tuple(axis % len(shape) for axis in args.axes)
    slot = operator.outputs.index(tensor)
    if (
        slot == 0
        and len(values) == len(initial) == len(outputs) == 2
        and _uniform(values[1]).op == _uniform(initial[1]).op == "constant"
        and _uniform(values[1]).value == 1
        and _uniform(initial[1]).value == 0
    ):
        a = _node(state, "parameter", value=len(context))
        b = _node(state, "parameter", value=len(context) + 2)
        flags = {
            len(context) + 1: _number(state, 1),
            len(context) + 3: _number(state, 1),
        }
        subsequent = _substitute(state, outputs[0], flags)
        subsequent_flag = _substitute(state, outputs[1], flags)
        flags[len(context) + 1] = _number(state, 0)
        first = _substitute(state, outputs[0], flags)
        first_flag = _substitute(state, outputs[1], flags)
        if (
            first_flag.op == subsequent_flag.op == "constant"
            and first_flag.value == subsequent_flag.value == 1
            and first.key == b.key
            and subsequent.op == "max"
            and {part.key for part in subsequent.arguments} == {a.key, b.key}
        ):
            value = _node(
                state,
                "reduce_max",
                (values[0], _uniform(initial[0])),
                (tuple(sorted(axes)), args.keepdims),
            )
            return replace(
                value,
                origins=_origins(outputs[0])
                | _origins(outputs[1])
                | initial[0].origins
                | initial[1].origins
                | values[1].origins,
            )
    if len(values) == len(initial) == len(outputs) == 1:
        output = outputs[0]
        neutral = _uniform(initial[0])
        parameters = {
            _node(state, "parameter", value=len(context) + index).key
            for index in range(2)
        }
        if (
            output.op in {"add", "mul", "min", "max"}
            and {value.key for value in output.arguments} == parameters
        ):
            expected = 0 if output.op == "add" else 1 if output.op == "mul" else None
            if (
                expected is not None
                and neutral.op == "constant"
                and neutral.value == expected
            ):
                kind = "reduce_sum" if output.op == "add" else "reduce_product"
                source = values[0]
                axes = tuple(sorted(axes))
                removed = [output, initial[0]]
                if source.op == kind and isinstance(source.value, tuple):
                    previous_axes, previous_keep = source.value
                    if (
                        isinstance(previous_axes, tuple)
                        and previous_keep == args.keepdims
                    ):
                        if args.keepdims:
                            combined = {int(axis) for axis in previous_axes} | set(axes)
                        else:
                            original_shape = (
                                inference.tensors[source.arguments[0].sources[0]].shape
                                if source.arguments[0].sources
                                else None
                            )
                            if original_shape is None:
                                return _node(
                                    state, kind, (source,), (axes, args.keepdims)
                                )
                            kept = [
                                axis
                                for axis in range(len(original_shape))
                                if axis not in previous_axes
                            ]
                            combined = {int(axis) for axis in previous_axes} | {
                                kept[axis] for axis in axes
                            }
                        removed.append(source)
                        source = source.arguments[0]
                        axes = tuple(sorted(combined))
                reduced = _reduce_view(
                    state, kind, source, axes, args.keepdims, inference
                )
                return replace(
                    reduced,
                    origins=reduced.origins
                    | _origins(output)
                    | frozenset().union(*(item.origins for item in removed)),
                )
    return _node(
        state,
        "reduce",
        (*values, *initial, *outputs),
        (slot, axes, args.keepdims, len(values), len(initial)),
    )


def _integer_values(tensor: Tensor, inference: _Inference) -> tuple[int, ...] | None:
    values = inference.tensors[tensor].values
    if (
        values is not None
        and len(values) <= 64
        and all(value.op == "const" for value in values)
    ):
        return tuple(value.value for value in values if value.value is not None)
    return None


def _valid_indices(indices: Tensor, source: Tensor, inference: _Inference) -> bool:
    shape = inference.tensors[indices].shape
    target = inference.tensors[source].shape
    values = _integer_values(indices, inference)
    if shape is None or not shape or target is None or values is None:
        return False
    coordinates = shape[-1].value
    if (
        coordinates is None
        or coordinates <= 0
        or coordinates > len(target)
        or len(values) % coordinates
    ):
        return False
    positions = tuple(
        tuple(values[begin : begin + coordinates])
        for begin in range(0, len(values), coordinates)
    )
    return len(set(positions)) == len(positions) and all(
        target[axis].op == "const"
        and target[axis].value is not None
        and 0 <= coordinate < (target[axis].value or 0)
        for position in positions
        for axis, coordinate in enumerate(position)
    )


def _rewrite(state: _MatchingState, value: _Canonical) -> _Canonical:
    arguments = value.arguments
    if value.op == "add":
        return _sum(state, arguments)
    if value.op == "mul":
        return _product(state, arguments)
    if value.op == "power" and isinstance(value.value, Fraction):
        return _power(state, arguments[0], value.value)
    if value.op in {"abs", "exp", "log", "sqrt", "sin", "cos", "erf"}:
        return _function(state, value.op, arguments[0])
    if value.op in {"min", "max"}:
        return _extremum(state, value.op, arguments)
    if value.op == "compare" and isinstance(value.value, str):
        return _comparison(state, value.value, arguments[0], arguments[1])
    if value.op == "select":
        return _selection(state, *arguments)
    if value.op == "floor_div" and len(arguments) == 2:
        a, b = arguments
        if b.op == "constant" and isinstance(b.value, Fraction) and b.value != 0:
            if a.op == "constant" and isinstance(a.value, Fraction):
                return _inherit(_number(state, a.value // b.value), arguments)
            if b.value == 1:
                return _inherit(a, (b,))
    if value.op == "mod" and len(arguments) == 2:
        a, b = arguments
        quotient = _rewrite(state, _node(state, "floor_div", arguments))
        return _sum(state, (a, _product(state, (_number(state, -1), quotient, b))))
    return value


def _substitute(
    state: _MatchingState, value: _Canonical, variables: dict[int, _Canonical]
) -> _Canonical:
    if (
        value.op == "parameter"
        and isinstance(value.value, int)
        and value.value in variables
    ):
        return variables[value.value]
    if not value.arguments:
        return value
    arguments = tuple(_substitute(state, part, variables) for part in value.arguments)
    return _inherit(
        _rewrite(state, _node(state, value.op, arguments, value.value)), (value,)
    )


def _origins(value: _Canonical) -> frozenset[_Operator]:
    pending = [value]
    visited: set[_Canonical] = set()
    result: set[_Operator] = set()
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        result.update(item.origins)
        pending.extend(item.arguments)
    return frozenset(result)


def _reduce_view(
    state: _MatchingState,
    kind: str,
    source: _Canonical,
    axes: tuple[int, ...],
    keepdims: bool,
    inference: _Inference,
) -> _Canonical:
    if (
        source.op == "permute"
        and isinstance(source.value, tuple)
        and all(isinstance(axis, int) for axis in source.value)
    ):
        permutation = tuple(int(axis) for axis in source.value)
        mapped = tuple(sorted(permutation[axis] for axis in axes))
        reduced = _node(state, kind, (source.arguments[0],), (mapped, keepdims))
        remaining = [axis for axis in range(len(permutation)) if axis not in mapped]
        order = (
            permutation
            if keepdims
            else tuple(
                remaining.index(permutation[axis])
                for axis in range(len(permutation))
                if axis not in axes
            )
        )
        return _inherit(_permutation(state, reduced, order, inference), (source,))
    return _node(state, kind, (source,), (axes, keepdims))
