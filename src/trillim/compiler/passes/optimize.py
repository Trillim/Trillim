from __future__ import annotations

from collections.abc import Hashable
from math import prod

import numpy as np
from numpy.typing import NDArray

from ...tdl.core.core_dataclasses import _Region
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
    _OperatorArgs,
    _OutputArgs,
    _PermuteArgs,
    _RandomArgs,
    _ReshapeArgs,
    _SelectArgs,
    _UnaryArgs,
    _WhileArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ...tdl.core.types import DType
from ..compiler_dataclasses import _Inference, _OptimizationState, _TensorInfo
from ..dtype import _QUANTIZED_DTYPES, _decode, _encode
from .rewrite import _rebuild, _replace_arguments, _replace_tensors
from .walk import _ordered_operators, _walk

_LIMIT = 65536
_UNHASHABLE = object()
_FUSED_WEIGHT_DTYPES = _QUANTIZED_DTYPES | {"bf16", "fp16"}


def _effective_dtype(tensor: Tensor, state: _OptimizationState) -> DType | None:
    tensor = _replacement(tensor, state.replacements)
    producer = tensor._producer
    if producer is not None and isinstance(
        producer.args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs)
    ):
        return _effective_dtype(producer.args.x, state)
    if producer is not None and isinstance(producer.args, _ConstructArgs):
        index = producer.outputs.index(tensor)
        return _effective_dtype(producer.regions[0].outputs[index], state)
    return tensor.dtype


def _weight_source(tensor: Tensor) -> bool:
    producer = tensor._producer
    if producer is None:
        return False
    if isinstance(producer.args, _ConstantArgs):
        return isinstance(producer.args.value, Weight)
    return isinstance(
        producer.args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs)
    ) and _weight_source(producer.args.x)


def _matrix_uses(
    tensor: Tensor, state: _OptimizationState, visited: set[Tensor]
) -> bool:
    if tensor in visited:
        return True
    visited.add(tensor)
    for consumer in tensor._consumers:
        args = consumer.args
        if isinstance(args, _ConstructArgs):
            for index, result in enumerate(consumer.regions[0].outputs):
                if result is tensor and not _matrix_uses(
                    consumer.outputs[index], state, visited
                ):
                    return False
        elif (
            isinstance(args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs))
            and args.x is tensor
            or isinstance(args, _CastArgs)
            and args.dtype == tensor.dtype
        ):
            if not _matrix_uses(consumer.outputs[0], state, visited):
                return False
        elif isinstance(args, _MatmulArgs):
            other = args.b if args.a is tensor else args.a
            other_dtype = _effective_dtype(other, state)
            if (
                consumer.outputs[0].dtype != "fp32"
                or other_dtype not in _QUANTIZED_DTYPES | {"fp32"}
            ):
                return False
        elif consumer.op != "shape":
            return False
    return True


def _weight_casts(graph: Graph, state: _OptimizationState) -> None:
    assert graph._region is not None
    for _, operator in _walk(graph._region):
        args = operator.args
        if (
            isinstance(args, _CastArgs)
            and args.dtype == "fp32"
            and args.x.dtype in _FUSED_WEIGHT_DTYPES
            and _weight_source(args.x)
            and _matrix_uses(operator.outputs[0], state, set())
        ):
            state.replacements[operator.outputs[0]] = args.x
            state.changed = True


def _convert(
    value: NDArray[np.generic], dtype: DType | None
) -> NDArray[np.generic] | None:
    if dtype is None or dtype in _QUANTIZED_DTYPES:
        return None
    return _decode(_encode(value, dtype), value.shape)


def _shape(tensor: Tensor, state: _OptimizationState) -> tuple[int, ...] | None:
    if state.inference is not None and tensor in state.inference.tensors:
        shape = state.inference.tensors[tensor].shape
        if shape is not None and all(
            value.op == "const" and value.value is not None for value in shape
        ):
            return tuple(value.value for value in shape if value.value is not None)
    if tensor.shape is not None and all(
        value is not None for value in tensor.shape.dimensions
    ):
        return tuple(value for value in tensor.shape.dimensions if value is not None)
    return None


def _literal(
    value: object, state: _OptimizationState, dtype: DType | None
) -> NDArray[np.generic] | None:
    if isinstance(value, Tensor):
        return state.constants.get(value)
    if isinstance(value, (bool, int, float)):
        return _convert(np.asarray(value), dtype)
    return None


def _observable(operator: _Operator) -> bool:
    if isinstance(
        operator.args,
        (
            _InputArgs,
            _OutputArgs,
            _CacheArgs,
            _CacheAssignArgs,
            _RandomArgs,
            _WhileArgs,
        ),
    ):
        return True
    return any(
        _observable(child) for body in operator.regions for child in body.operators
    )


def _metadata_constant(
    tensor: Tensor, state: _OptimizationState
) -> NDArray[np.generic] | None:
    if state.inference is None:
        return None
    info = state.inference.tensors.get(tensor)
    shape = _shape(tensor, state)
    if (
        info is None
        or info.values is None
        or shape is None
        or prod(shape) > _LIMIT
        or any(value.op != "const" for value in info.values)
    ):
        return None
    data = np.asarray([value.value for value in info.values])
    return _convert(data.reshape(shape), tensor.dtype)


def _gather(
    args: _GatherArgs, state: _OptimizationState, output: Tensor
) -> NDArray[np.generic] | None:
    source = state.constants.get(args.x)
    indices = state.constants.get(args.indices)
    mask = state.constants.get(args.mask) if args.mask is not None else np.asarray(True)
    shape = _shape(output, state)
    if shape is None or prod(shape) > _LIMIT or mask is None:
        return None
    if not np.any(mask):
        return _convert(np.full(shape, args.fill), output.dtype)
    if source is None or indices is None:
        return None
    width = indices.shape[-1]
    count = prod(indices.shape[:-1])
    coordinates = indices.reshape((count, width))
    visible = np.broadcast_to(mask, indices.shape[:-1]).reshape(count)
    result = np.full(shape, args.fill, dtype=source.dtype)
    rows = result.reshape((count, *source.shape[width:]))
    for index, coordinate in enumerate(coordinates):
        if visible[index]:
            position = tuple(int(value) for value in coordinate)
            if any(
                not 0 <= value < source.shape[axis]
                for axis, value in enumerate(position)
            ):
                raise ValueError("constant gather coordinate is out of bounds")
            rows[index] = source[position]
    return result


def _compute(
    operator: _Operator, state: _OptimizationState
) -> tuple[NDArray[np.generic], ...] | None:
    if _observable(operator) or isinstance(operator.args, _CacheReadArgs):
        return None
    known_values = [_metadata_constant(output, state) for output in operator.outputs]
    if known_values and all(item is not None for item in known_values):
        return tuple(item for item in known_values if item is not None)
    args, op = operator.args, operator.op
    if isinstance(args, _ConstantArgs):
        if isinstance(args.value, Weight):
            return None
        if isinstance(args.value, np.ndarray) and args.value.size > _LIMIT:
            return None
        constant_value = _convert(np.asarray(args.value), operator.outputs[0].dtype)
        return None if constant_value is None else (constant_value,)
    if isinstance(args, _ConstructArgs):
        outputs = [state.constants.get(value) for value in operator.regions[0].outputs]
        return (
            tuple(value for value in outputs if value is not None)
            if all(value is not None for value in outputs)
            else None
        )
    if len(operator.outputs) != 1:
        return None
    output = operator.outputs[0]
    shape = _shape(output, state)
    if shape is not None and prod(shape) > _LIMIT:
        return None
    value: NDArray[np.generic] | None = None
    if isinstance(args, _CastArgs):
        source = state.constants.get(args.x)
        if source is not None:
            value = _convert(source, args.dtype)
    elif isinstance(args, _CompareArgs):
        left = state.constants.get(args.a)
        right = _literal(args.b, state, args.a.dtype)
        if left is not None and right is not None:
            value = {
                "eq": np.equal,
                "ne": np.not_equal,
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
            }[args.predicate](left, right)
    elif isinstance(args, _BinaryArgs):
        left = state.constants.get(args.a)
        right = _literal(args.b, state, args.a.dtype)
        if left is not None and right is not None:
            if op in {"floor_div", "mod"} and np.any(right == 0):
                raise ValueError("constant integer division by zero")
            function = {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "div": np.divide,
                "floor_div": np.floor_divide,
                "mod": np.mod,
            }.get(op)
            if function is not None:
                value = function(left, right)
    elif isinstance(args, _SelectArgs):
        condition = state.constants.get(args.condition)
        left = _literal(args.when_true, state, output.dtype)
        right = _literal(args.when_false, state, output.dtype)
        if condition is not None and left is not None and right is not None:
            value = np.where(condition, left, right)
    elif isinstance(args, _IotaArgs):
        start = _literal(args.start, state, "int64")
        stop = _literal(args.stop, state, "int64")
        if start is not None and stop is not None and start.size == stop.size == 1:
            count = max(
                0,
                (
                    int(stop.item())
                    - int(start.item())
                    + args.step
                    - (1 if args.step > 0 else -1)
                )
                // args.step,
            )
            if count <= _LIMIT:
                value = np.arange(int(start.item()), int(stop.item()), args.step)
    elif isinstance(args, _ReshapeArgs):
        source = state.constants.get(args.x)
        if source is not None and shape is not None:
            value = source.reshape(shape)
    elif isinstance(args, _PermuteArgs):
        source = state.constants.get(args.x)
        if source is not None:
            value = source.transpose(args.axes)
    elif isinstance(args, _BroadcastArgs):
        source = state.constants.get(args.x)
        if source is not None and shape is not None:
            value = np.broadcast_to(source, shape).copy()
    elif isinstance(args, _GatherArgs):
        value = _gather(args, state, output)
    elif isinstance(args, _UnaryArgs):
        source = state.constants.get(args.x)
        if op == "shape":
            dimensions = _shape(args.x, state)
            if dimensions is not None:
                value = np.asarray(dimensions, dtype=np.int64)
        elif source is not None:
            unary_function = {
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "sin": np.sin,
                "cos": np.cos,
            }.get(op)
            if unary_function is not None:
                value = unary_function(source)
    if value is None or value.size > _LIMIT:
        return None
    converted = _convert(value, output.dtype)
    return None if converted is None else (converted,)


def _key(value: object) -> Hashable:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Weight):
        return "weight", value.root, value.dtype
    if isinstance(value, _OperatorArgs):
        values = tuple((name, _key(item)) for name, item in vars(value).items())
        return _UNHASHABLE if any(item is _UNHASHABLE for _, item in values) else values
    if isinstance(value, np.ndarray):
        return (
            (value.dtype.str, value.shape, value.tobytes())
            if value.size <= _LIMIT
            else _UNHASHABLE
        )
    if isinstance(value, (tuple, list)):
        sequence_values = tuple(_key(item) for item in value)
        return (
            _UNHASHABLE
            if any(item is _UNHASHABLE for item in sequence_values)
            else sequence_values
        )
    if isinstance(value, dict):
        values = tuple((key, _key(item)) for key, item in value.items())
        return _UNHASHABLE if any(item is _UNHASHABLE for _, item in values) else values
    return value if isinstance(value, Hashable) else _UNHASHABLE


def _constant_nodes(
    operator: _Operator,
    values: tuple[NDArray[np.generic], ...],
    state: _OptimizationState,
) -> tuple[_Operator, ...]:
    nodes = []
    for output, value in zip(operator.outputs, values, strict=True):
        value = value.copy()
        value.setflags(write=False)
        node = _Operator(
            "constant",
            _ConstantArgs(value=value, dtype=output.dtype),
            outputs=((output.shape, output.dtype),),
        )
        state.constants[node.outputs[0]] = value
        state.replacements[output] = node.outputs[0]
        if state.inference is not None and output in state.inference.tensors:
            state.inference.tensors[node.outputs[0]] = state.inference.tensors[output]
        nodes.append(node)
    return tuple(nodes)


def _weight_view(
    tensor: Tensor, state: _OptimizationState
) -> tuple[Tensor, tuple[_Operator, ...]]:
    producer = tensor._producer
    if producer is None:
        return tensor, ()
    args = producer.args
    if isinstance(args, _CastArgs) and args.dtype == "fp32":
        weight_producer = args.x._producer
        if (
            weight_producer is not None
            and isinstance(weight_producer.args, _ConstantArgs)
            and isinstance(weight_producer.args.value, Weight)
            and args.x.dtype in _FUSED_WEIGHT_DTYPES
        ):
            return args.x, ()
    if isinstance(args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs)):
        source, nodes = _weight_view(args.x, state)
        if source is not args.x:
            clone = _Operator(
                producer.op,
                args.model_copy(update={"x": source}),
                outputs=((tensor.shape, source.dtype),),
            )
            if state.inference is not None and tensor in state.inference.tensors:
                info = state.inference.tensors[tensor]
                state.inference.tensors[clone.outputs[0]] = _TensorInfo(
                    info.shape, info.maximum_shape, source.dtype, info.values
                )
            return clone.outputs[0], (*nodes, clone)
    return tensor, ()


def _simplify(region: _Region, state: _OptimizationState) -> None:
    operators: list[_Operator] = []
    common: dict[Hashable, _Operator] = {}
    for operator in tuple(_ordered_operators(region)):
        operator.args = _replace_arguments(operator.args, state.replacements)
        for body in operator.regions:
            _simplify(body, state)
        args = operator.args
        if isinstance(args, _IfArgs) and args.condition in state.constants:
            condition = state.constants[args.condition]
            if condition.shape == ():
                selected = operator.regions[0 if bool(condition.item()) else 1]
                for child in selected.operators:
                    child.dependencies = tuple(
                        dict.fromkeys((*operator.dependencies, *child.dependencies))
                    )
                operators.extend(selected.operators)
                state.replacements.update(
                    zip(operator.outputs, selected.outputs, strict=True)
                )
                state.changed = True
                continue
        if isinstance(args, _CastArgs) and args.x.dtype == args.dtype:
            state.replacements[operator.outputs[0]] = args.x
            state.changed = True
            continue
        if isinstance(args, _MatmulArgs) and operator.outputs[0].dtype == "fp32":
            if _effective_dtype(args.a, state) == "fp32":
                weight, nodes = _weight_view(args.b, state)
                if weight is not args.b:
                    operators.extend(nodes)
                    operator.args = args.model_copy(update={"b": weight})
                    state.changed = True
            elif _effective_dtype(args.b, state) == "fp32":
                weight, nodes = _weight_view(args.a, state)
                if weight is not args.a:
                    operators.extend(nodes)
                    operator.args = args.model_copy(update={"a": weight})
                    state.changed = True
        with np.errstate(all="ignore"):
            values = _compute(operator, state)
        if values is not None:
            for output, value in zip(operator.outputs, values, strict=True):
                state.constants[output] = value
            if not isinstance(args, _ConstantArgs):
                operators.extend(_constant_nodes(operator, values, state))
                state.changed = True
                continue
        if (
            not operator.regions
            and not _observable(operator)
            and not isinstance(args, _CacheReadArgs)
        ):
            arguments = _key(operator.args)
            if arguments is not _UNHASHABLE:
                key = (
                    operator.op,
                    arguments,
                    tuple(output.dtype for output in operator.outputs),
                )
                previous = common.get(key)
                if previous is not None:
                    state.replacements.update(
                        zip(operator.outputs, previous.outputs, strict=True)
                    )
                    state.changed = True
                    continue
                common[key] = operator
        operators.append(operator)
    region.operators = tuple(operators)
    region.outputs = tuple(
        _replacement(output, state.replacements) for output in region.outputs
    )


def _replacement(tensor: Tensor, replacements: dict[Tensor, Tensor]) -> Tensor:
    while tensor in replacements:
        tensor = replacements[tensor]
    return tensor


def _dead(region: _Region) -> bool:
    changed = False
    for operator in region.operators:
        for body in operator.regions:
            changed = _dead(body) or changed
    local = set(region.operators)
    needed = {operator for operator in local if _observable(operator)}
    needed.update(
        tensor._producer for tensor in region.outputs if tensor._producer in local
    )
    pending = list(needed)
    while pending:
        operator = pending.pop()
        for tensor in operator.inputs:
            if tensor._producer in local and tensor._producer not in needed:
                assert tensor._producer is not None
                needed.add(tensor._producer)
                pending.append(tensor._producer)
    kept = tuple(operator for operator in region.operators if operator in needed)
    changed = changed or len(kept) != len(region.operators)
    region.operators = kept
    return changed


def _total(operator: _Operator, state: _OptimizationState) -> bool:
    if (
        _observable(operator)
        or isinstance(operator.args, _CacheReadArgs)
        or operator.regions
    ):
        return False
    if state.inference is not None and any(
        constraint.operator is operator for constraint in state.inference.constraints
    ):
        return False
    if isinstance(operator.args, _ConstantArgs) or operator.op == "shape":
        return True
    return operator.op in {
        "add",
        "sub",
        "mul",
        "compare",
        "select",
        "permute",
        "reshape",
        "broadcast",
    }


def _hoist(region: _Region, state: _OptimizationState) -> None:
    operators: list[_Operator] = []
    available = set(region.parameters) | set(region.captures)
    for operator in region.operators:
        for body in operator.regions:
            _hoist(body, state)
        if isinstance(operator.args, _WhileArgs):
            for body in operator.regions:
                kept = []
                for child in body.operators:
                    if (
                        set(child.inputs) <= available
                        and not set(child.dependencies).intersection(body.operators)
                        and _total(child, state)
                    ):
                        child.dependencies = tuple(
                            dict.fromkeys((*operator.dependencies, *child.dependencies))
                        )
                        operators.append(child)
                        available.update(child.outputs)
                        state.changed = True
                    else:
                        kept.append(child)
                body.operators = tuple(kept)
        operators.append(operator)
        available.update(operator.outputs)
    region.operators = tuple(operators)


def _optimize(graph: Graph, inference: _Inference | None = None) -> bool:
    """Fold bounded constants and eliminate redundant pure work in the existing IR."""
    if graph._region is None:
        raise ValueError("graph must be completed before optimization")
    state = _OptimizationState(inference)
    _weight_casts(graph, state)
    _simplify(graph._region, state)
    _replace_tensors(graph, state.replacements)
    state.changed = _dead(graph._region) or state.changed
    _rebuild(graph)
    _hoist(graph._region, state)
    _rebuild(graph)
    return state.changed
