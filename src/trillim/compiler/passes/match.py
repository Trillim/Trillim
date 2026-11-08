from __future__ import annotations

import ctypes
import ctypes.util
import inspect
from pathlib import Path
from typing import get_args

import numpy as np
from pydantic import BaseModel

from ...tdl.core import Graph, Input, Output, TensorShape
from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    Weight,
    _BroadcastArgs,
    _ConstantArgs,
    _ConstructArgs,
    _InputArgs,
    _PermuteArgs,
    _ReduceArgs,
    _ReshapeArgs,
)
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..artifacts.expressions import _bound, _byte_size, _parameter_references
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import _Inference, _ShapeConstraint, _ShapeExpression
from ..kernels.bindings import (
    _broadcast_strides,
    _contiguous_strides,
    _reshape_strides,
    _storage_shape,
)
from ..kernels.kernel_dataclasses import _Kernel, _KernelBindingContext, _TensorBinding
from .infer import _infer
from .optimize import _optimize
from .passes_dataclasses import _KernelMatch, _KernelPattern, _MatchingState, _MatchSet
from .walk import _walk


def _fields(operator: _Operator) -> dict[str, object]:
    if isinstance(operator.args, _ConstructArgs):
        return operator.args.arguments
    return dict(vars(operator.args))


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


def _literal_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, BaseModel) and isinstance(right, BaseModel):
        return type(left) is type(right) and _literal_equal(vars(left), vars(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _literal_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _literal_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return left is right
    return type(left) is type(right) and bool(left == right)


def _equivalent(
    pattern: Tensor,
    target: Tensor,
    template: _KernelPattern,
    state: _MatchingState,
    bindings: dict[str, Tensor],
    parameters: dict[Tensor, Tensor],
    covered: set[_Operator],
    pairs: set[tuple[Tensor, Tensor]],
) -> bool:
    if pattern in template.inputs:
        if (
            template.inference.tensors[pattern].dtype
            != state.inference.tensors[target].dtype
        ):
            return False
        name = template.inputs[pattern]
        previous = bindings.setdefault(name, target)
        return _unwrapped(previous, state) is _unwrapped(target, state)
    if pattern in parameters:
        return parameters[pattern] is target
    pattern, target = _unwrapped(pattern, state), _unwrapped(target, state, covered)
    pair = pattern, target
    if pair in pairs:
        return True
    if (
        template.inference.tensors[pattern].dtype
        != state.inference.tensors[target].dtype
    ):
        return False
    p, t = pattern._producer, target._producer
    if p is None or t is None or p.op != t.op:
        return False
    if p.outputs.index(pattern) != t.outputs.index(target) or len(p.regions) != len(
        t.regions
    ):
        return False
    if t.op in {"cache", "cache_assign", "cache_read", "while", "if"}:
        return False
    if p.regions and not (
        isinstance(p.args, _ReduceArgs)
        and all(_pure(body, state) for body in t.regions)
    ):
        return False
    p_fields, t_fields = _fields(p), _fields(t)

    def equal_value(a: object, b: object) -> bool:
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            return _equivalent(
                a, b, template, state, bindings, parameters, covered, pairs
            )
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(
                equal_value(x, y) for x, y in zip(a, b, strict=True)
            )
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(
                equal_value(value, b[key]) for key, value in a.items()
            )
        return _literal_equal(a, b)

    saved = (
        (dict(bindings), set(covered), set(pairs)) if p.op in {"add", "mul"} else None
    )
    matched = p_fields.keys() == t_fields.keys() and all(
        equal_value(value, t_fields[name]) for name, value in p_fields.items()
    )
    if not matched and saved is not None and p_fields.keys() == t_fields.keys():
        saved_bindings, saved_covered, saved_pairs = saved
        bindings.clear()
        bindings.update(saved_bindings)
        covered.clear()
        covered.update(saved_covered)
        pairs.clear()
        pairs.update(saved_pairs)
        swapped = {**t_fields, "a": t_fields["b"], "b": t_fields["a"]}
        matched = all(
            equal_value(value, swapped[name]) for name, value in p_fields.items()
        )
    if not matched:
        return False
    for p_body, t_body in zip(p.regions, t.regions, strict=True):
        if len(p_body.parameters) != len(t_body.parameters) or len(
            p_body.outputs
        ) != len(t_body.outputs):
            return False
        nested = {
            **parameters,
            **dict(zip(p_body.parameters, t_body.parameters, strict=True)),
        }
        if not all(
            _equivalent(a, b, template, state, bindings, nested, covered, pairs)
            for a, b in zip(p_body.outputs, t_body.outputs, strict=True)
        ):
            return False
    if not isinstance(t.args, _ConstantArgs):
        covered.add(t)
    pairs.add(pair)
    return True


def _available(kernel: _Kernel) -> bool:
    if kernel.capabilities.abi_version != 1:
        return False
    features: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith(("flags", "Features")):
                features.update(line.partition(":")[2].split())
    if any(
        feature.replace("-", "_") not in features
        for feature in kernel.capabilities.hardware
    ):
        return False
    for software in kernel.capabilities.software:
        library = ctypes.util.find_library(software)
        if library is None:
            return False
        try:
            ctypes.CDLL(library)
        except OSError:
            return False
    return True


def _signature(kernel: _Kernel) -> tuple[tuple[str, inspect.Parameter, object], ...]:
    annotations = inspect.get_annotations(kernel.pattern, eval_str=True)
    return tuple(
        (name, parameter, annotations.get(name))
        for name, parameter in inspect.signature(kernel.pattern).parameters.items()
    )


def _arguments(
    target: _Operator,
    signature: tuple[tuple[str, inspect.Parameter, object], ...],
) -> dict[str, object] | None:
    actual = _fields(target)
    arguments: dict[str, object] = {}
    used = {value for value in actual.values() if isinstance(value, Tensor)}
    candidates = iter(value for value in target.inputs if value not in used)
    for name, parameter, annotation in signature:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if name in actual:
            value = actual[name]
            if isinstance(value, Weight):
                value = next(
                    (
                        output
                        for body in target.regions
                        for _, node in _walk(body)
                        if isinstance(node.args, _ConstantArgs)
                        and isinstance(node.args.value, Weight)
                        and node.args.value.root == value.root
                        for output in node.outputs
                    ),
                    value,
                )
            arguments[name] = value
        elif annotation is Tensor or Tensor in get_args(annotation):
            source = next(candidates, None)
            if source is None:
                return None
            arguments[name] = source
        elif parameter.default is not inspect.Parameter.empty:
            arguments[name] = parameter.default
        else:
            return None
    return arguments


def _key(value: object, inference: _Inference) -> object:
    if isinstance(value, Tensor):
        info = inference.tensors[value]
        shape = (
            tuple(item.value if item.op == "const" else None for item in info.shape)
            if info.shape is not None
            else None
        )
        return ("tensor", info.dtype, shape)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return tuple(_key(item, inference) for item in value)
    if isinstance(value, dict):
        return tuple((key, _key(item, inference)) for key, item in value.items())
    return (type(value).__name__, repr(value))


def _pattern(
    kernel: _Kernel, arguments: dict[str, object], inference: _Inference
) -> _KernelPattern | None:
    leaves: dict[Tensor, str] = {}
    graph = Graph()

    def placeholder(value: object, name: str) -> object:
        if isinstance(value, Tensor):
            info = inference.tensors[value]
            if info.shape is None or info.dtype is None:
                raise ValueError("pattern input needs known rank and dtype")
            shape = tuple(
                item.value if item.op == "const" else None for item in info.shape
            )
            tensor = Input(name=name, shape=TensorShape(shape), dtype=info.dtype)
            leaves[tensor] = name
            return tensor
        if isinstance(value, tuple):
            return tuple(
                placeholder(item, f"{name}.{index}") for index, item in enumerate(value)
            )
        if isinstance(value, list):
            return [
                placeholder(item, f"{name}.{index}") for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {
                key: placeholder(item, f"{name}.{key}") for key, item in value.items()
            }
        return value

    try:
        with graph:
            outputs = kernel.pattern(
                **{name: placeholder(value, name) for name, value in arguments.items()}
            )
            for index, output in enumerate(outputs):
                Output(output, name=f"output{index}")
        facts = _infer(graph)
        if _optimize(graph, facts):
            facts = _infer(graph)
        return _KernelPattern(graph, leaves, tuple(graph._outputs.values()), facts)
    except (TypeError, ValueError, IndexError, NotImplementedError):
        return None


def _binding(tensor: Tensor, state: _MatchingState) -> _TensorBinding:
    if tensor in state.tensor_bindings:
        return state.tensor_bindings[tensor]
    result = _tensor_binding(tensor, state)
    state.tensor_bindings[tensor] = result
    return result


def _tensor_binding(tensor: Tensor, state: _MatchingState) -> _TensorBinding:
    inference = state.inference
    info = inference.tensors[tensor]
    if info.shape is None or info.maximum_shape is None or info.dtype is None:
        raise ValueError("kernel input has unresolved rank or dtype")
    values = info.values
    if (
        values is None
        and info.shape == ()
        and info.dtype
        in {
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        }
    ):
        low, high = inference.value_ranges.get(tensor, (None, None))
        values = (
            _ShapeExpression(
                "scalar", tensor=tensor, axis=0, minimum=low, maximum=high
            ),
        )
    result = _TensorBinding(
        info.shape,
        info.maximum_shape,
        info.dtype,
        _contiguous_strides(info.shape),
        values=values,
    )
    producer = tensor._producer
    if producer is None:
        return result
    args = producer.args
    if isinstance(args, _ConstructArgs) and _pure(producer.regions[0], state):
        return _binding(
            producer.regions[0].outputs[producer.outputs.index(tensor)], state
        )
    if isinstance(args, (_PermuteArgs, _BroadcastArgs, _ReshapeArgs)):
        source = _binding(args.x, state)
        strides = source.strides or _contiguous_strides(source.shape)
        if isinstance(args, _PermuteArgs):
            strides = tuple(strides[axis] for axis in args.axes)
        elif isinstance(args, _BroadcastArgs):
            strides = _broadcast_strides(source.shape, info.shape, strides)
        else:
            reshaped = _reshape_strides(source.shape, strides, info.shape)
            if reshaped is None:
                return result
            strides = reshaped
        return _TensorBinding(
            info.shape,
            info.maximum_shape,
            info.dtype,
            strides,
            _storage_shape(source),
            info.values,
        )
    return result


def _tensor_arguments(value: object, name: str) -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        return {name: value}
    if isinstance(value, (tuple, list)):
        return {
            path: tensor
            for index, item in enumerate(value)
            for path, tensor in _tensor_arguments(item, f"{name}.{index}").items()
        }
    if isinstance(value, dict):
        return {
            path: tensor
            for key, item in value.items()
            for path, tensor in _tensor_arguments(item, f"{name}.{key}").items()
        }
    return {}


def _safe_boundary(tensor: Tensor, internals: set[_Operator]) -> bool:
    producer = tensor._producer
    if producer not in internals:
        return True
    if isinstance(producer.args, _ConstantArgs):
        return True
    if isinstance(producer.args, (_PermuteArgs, _ReshapeArgs, _BroadcastArgs)):
        return _safe_boundary(producer.args.x, internals)
    return False


def _capacity(binding: _TensorBinding) -> int:
    dimensions = _storage_shape(binding)
    return _byte_size(binding.dtype, tuple(_bound(value) for value in dimensions))


def _match(
    graph: Graph, kernels: tuple[_Kernel, ...], inference: _Inference
) -> _MatchSet:
    if graph._region is None:
        raise ValueError("cannot match an incomplete graph")
    result = _MatchSet()
    state = _MatchingState(inference)
    eligible = tuple(kernel for kernel in kernels if _available(kernel))
    signatures = {id(kernel): _signature(kernel) for kernel in eligible}
    constraints: dict[_Operator | None, list[_ShapeConstraint]] = {}
    for constraint in inference.constraints:
        constraints.setdefault(constraint.operator, []).append(constraint)
    cache: dict[tuple[int, object], _KernelPattern | None] = {}
    for _, operator in _walk(graph._region):
        if isinstance(operator.args, (_PermuteArgs, _BroadcastArgs)):
            result.views.add(operator)
        elif isinstance(operator.args, _ReshapeArgs):
            source = _binding(operator.args.x, state)
            if (
                _reshape_strides(
                    source.shape,
                    source.strides or _contiguous_strides(source.shape),
                    _binding(operator.outputs[0], state).shape,
                )
                is not None
            ):
                result.views.add(operator)
        if not operator.outputs or isinstance(
            operator.args, (_InputArgs, _ConstantArgs)
        ):
            continue
        matches: list[_KernelMatch] = []
        internals = {node for body in operator.regions for _, node in _walk(body)}
        for kernel in eligible:
            if kernel.parameter_schema.output_count != len(operator.outputs):
                continue
            arguments = _arguments(operator, signatures[id(kernel)])
            if arguments is None:
                continue
            key = (id(kernel), _key(arguments, inference))
            if key not in cache:
                cache[key] = _pattern(kernel, arguments, inference)
            pattern = cache[key]
            if pattern is None or len(pattern.outputs) != len(operator.outputs):
                continue
            if all(output in pattern.inputs for output in pattern.outputs):
                continue
            if isinstance(operator.args, _ConstructArgs) and not all(
                output._producer is not None
                and isinstance(output._producer.args, _ConstructArgs)
                for output in pattern.outputs
            ):
                continue
            bindings: dict[str, Tensor] = {}
            covered: set[_Operator] = set()
            pairs: set[tuple[Tensor, Tensor]] = set()
            if not all(
                _equivalent(p, t, pattern, state, bindings, {}, covered, pairs)
                for p, t in zip(pattern.outputs, operator.outputs, strict=True)
            ):
                continue
            # Metadata-only inputs can disappear when their shape is statically known.
            for name, value in arguments.items():
                for path, tensor in _tensor_arguments(value, name).items():
                    if path in pattern.inputs.values():
                        bindings.setdefault(path, tensor)
            physical = (
                kernel.physical_inputs
                if kernel.physical_inputs is not None
                else tuple(pattern.inputs.values())
            )
            if len(physical) != kernel.parameter_schema.input_count or any(
                name not in bindings for name in physical
            ):
                result.rejections.setdefault(operator, {})[kernel.name] = (
                    f"physical inputs {physical} do not match native input count "
                    f"{kernel.parameter_schema.input_count} or named tensor bindings"
                )
                continue
            if any(not _safe_boundary(value, internals) for value in bindings.values()):
                continue
            outputs = set(operator.outputs) | {
                _unwrapped(output, state) for output in operator.outputs
            }
            covered.add(operator)
            if any(value._producer in covered for value in bindings.values()):
                continue
            if any(
                output not in outputs
                and any(consumer not in covered for consumer in output._consumers)
                for internal in covered
                for output in internal.outputs
            ):
                continue
            try:
                context = _KernelBindingContext(
                    {
                        name: _binding(tensor, state)
                        for name, tensor in bindings.items()
                    },
                    tuple(_binding(tensor, state) for tensor in operator.outputs),
                    _fields(operator),
                )
                parameters = kernel.bind(context)
                references = set().union(
                    *(_parameter_references(value) for value in parameters.values())
                )
                for owner in (*covered, None):
                    for constraint in constraints.get(owner, ()):
                        references.update(
                            _parameter_references(constraint.left)
                            | _parameter_references(constraint.right)
                        )
                eliminated = {
                    value
                    for node in covered
                    for body in node.regions
                    for value in body.parameters
                } | {value for node in covered for value in node.outputs}
                if any(
                    value in eliminated
                    and value not in bindings.values()
                    and value not in outputs
                    for value in references
                ):
                    continue
                spans = _requirements(kernel.parameter_schema, parameters)
                for name, size in zip(physical, spans.inputs, strict=True):
                    capacity = _capacity(context.inputs[name])
                    if size > capacity:
                        raise ValueError(
                            f"input {name} requires {size} bytes, storage has {capacity}"
                        )
                for index, (binding, size) in enumerate(
                    zip(context.outputs, spans.outputs, strict=True)
                ):
                    capacity = _capacity(binding)
                    if size > capacity:
                        raise ValueError(
                            f"output {index} requires {size} bytes, storage has {capacity}"
                        )
            except (
                KeyError,
                TypeError,
                ValueError,
                IndexError,
                OverflowError,
            ) as error:
                result.rejections.setdefault(operator, {})[kernel.name] = str(error)
                continue
            matches.append(
                _KernelMatch(
                    kernel,
                    operator,
                    frozenset(covered),
                    bindings,
                    tuple(bindings[name] for name in physical),
                    operator.outputs,
                    parameters,
                )
            )
        result.candidates[operator] = tuple(
            sorted(matches, key=lambda item: (-len(item.covered), item.kernel.name))
        )
    return result
