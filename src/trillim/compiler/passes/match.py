from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import cache
from itertools import chain
from typing import get_args

import numpy as np
from pydantic import BaseModel

from ...tdl.core import Graph, Input, Output, TensorShape
from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    Weight,
    _BroadcastArgs,
    _CastArgs,
    _ConstantArgs,
    _ConstructArgs,
    _GatherArgs,
    _InputArgs,
    _PermuteArgs,
    _ReduceArgs,
    _ReshapeArgs,
    _WhileArgs,
)
from ...tdl.core.graph import _build_region
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..artifacts.expressions import _bound, _byte_size, _parameter_references
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import (
    _Inference,
    _ShapeConstraint,
    _ShapeExpression,
    _TensorInfo,
)
from ..kernels.bindings import (
    _broadcast_strides,
    _contiguous_strides,
    _reshape_strides,
    _storage_shape,
)
from ..kernels.catalog import _available_capabilities
from ..kernels.kernel_dataclasses import _Kernel, _KernelBindingContext, _TensorBinding
from .canonicalize import _canonical, _normal, _pure, _terms, _unwrapped
from .infer import _expression, _infer
from .optimize import _observable, _optimize
from .passes_dataclasses import (
    _Canonical,
    _KernelMatch,
    _KernelPattern,
    _MatchingState,
    _MatchSet,
    _PatternBindings,
    _Selection,
    _SymbolicPattern,
)
from .walk import _walk


def _fields(operator: _Operator) -> dict[str, object]:
    if isinstance(operator.args, _ConstructArgs):
        return operator.args.arguments
    if isinstance(operator.args, _ConstantArgs):
        return {"value": operator.args.value}
    return dict(vars(operator.args))


def _literal_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and left.shape == ():
        left = left.item()
    if isinstance(right, np.ndarray) and right.shape == ():
        right = right.item()
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


def _unwrap_loop_construct(
    tensor: Tensor,
    inference: _Inference,
    state: _MatchingState,
    covered: set[_Operator] | None = None,
) -> Tensor:
    while (
        (producer := tensor._producer) is not None
        and isinstance(producer.args, _ConstructArgs)
        and _structural_loop(producer, inference, state)
    ):
        if covered is not None:
            covered.add(producer)
        output = producer.regions[0].outputs[producer.outputs.index(tensor)]
        tensor = _unwrapped(output, state, covered)
    return tensor


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
    if state.canonical:
        pattern = _canonical(pattern, template.inference, state)
        target = _canonical(target, state.inference, state, covered)
    if pattern in template.inputs:
        if (
            template.inference.tensors[pattern].dtype
            != state.inference.tensors[target].dtype
        ):
            return False
        if state.canonical:
            expected = template.inference.tensors[pattern].shape
            actual = state.inference.tensors[target].shape
            if (
                expected is None
                or actual is None
                or len(expected) != len(actual)
                or any(
                    dimension.op == "const" and dimension.value != value.value
                    for dimension, value in zip(expected, actual, strict=True)
                )
            ):
                return False
        name = template.inputs[pattern]
        previous = bindings.setdefault(name, target)
        if state.canonical:
            return _canonical(previous, state.inference, state) is target
        return _unwrapped(previous, state) is _unwrapped(target, state)
    if pattern in parameters:
        return parameters[pattern] is target
    pattern = _unwrapped(pattern, state)
    target = _unwrapped(target, state, covered)
    if not state.canonical:
        loop_pattern = _unwrap_loop_construct(pattern, template.inference, state)
        if loop_pattern is not pattern:
            pattern = loop_pattern
            target = _unwrap_loop_construct(target, state.inference, state, covered)
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
    if t.op in {"cache", "cache_assign", "cache_read", "if"}:
        return False
    if p.regions and not (
        isinstance(p.args, _ReduceArgs)
        and all(_pure(body, state) for body in t.regions)
        or not state.canonical
        and _structural_loop(p, template.inference, state)
        and _structural_loop(t, state.inference, state)
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
    if not matched and saved is not None and state.canonical:
        saved_bindings, saved_covered, saved_pairs = saved
        bindings.clear()
        bindings.update(saved_bindings)
        covered.clear()
        covered.update(saved_covered)
        pairs.clear()
        pairs.update(saved_pairs)
        dtype = template.inference.tensors[pattern].dtype
        if dtype is not None and (dtype == "fp32" or dtype.startswith(("int", "uint"))):
            nodes: set[_Operator] = set()
            left = _terms(
                pattern, p.op, dtype, template.inference, state, template.inputs, set()
            )
            right = _terms(target, t.op, dtype, state.inference, state, {}, nodes)
            if (
                left is not None
                and right is not None
                and len(left) > 2
                and len(left) == len(right)
            ):
                # Bind constrained expressions before wildcard inputs.
                ordered = sorted(left, key=lambda value: value in template.inputs)

                def assign(index: int, pending: tuple[Tensor, ...]) -> bool:
                    if index == len(ordered):
                        return True
                    for slot, value in enumerate(pending):
                        if state.remaining == 0:
                            return False
                        state.remaining -= 1
                        before = dict(bindings), set(covered), set(pairs)
                        if _equivalent(
                            ordered[index],
                            value,
                            template,
                            state,
                            bindings,
                            parameters,
                            covered,
                            pairs,
                        ) and assign(index + 1, pending[:slot] + pending[slot + 1 :]):
                            return True
                        bindings.clear()
                        bindings.update(before[0])
                        covered.clear()
                        covered.update(before[1])
                        pairs.clear()
                        pairs.update(before[2])
                    return False

                matched = assign(0, right)
                if matched:
                    covered.update(nodes)
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


def _shape_matches(
    pattern: Tensor, target: Tensor, template: _KernelPattern, state: _MatchingState
) -> bool:
    expected = template.inference.tensors[pattern].shape
    actual = state.inference.tensors[target].shape
    return (
        expected is not None
        and actual is not None
        and len(expected) == len(actual)
        and all(
            dimension.op != "const" or dimension.value == value.value
            for dimension, value in zip(expected, actual, strict=True)
        )
    )


def _mathematical(
    pattern: _Canonical,
    target: _Canonical,
    template: _KernelPattern,
    state: _MatchingState,
    bindings: dict[str, Tensor],
    covered: set[_Operator],
) -> bool:
    if state.remaining <= 0:
        return False
    state.remaining -= 1
    if (
        pattern.op == "leaf"
        and isinstance(pattern.value, Tensor)
        and pattern.value in template.inputs
    ):
        source = pattern.value
        candidates = [
            value
            for value in target.sources
            if _shape_matches(source, value, template, state)
        ]
        dtype = template.inference.tensors[source].dtype
        typed = [
            value
            for value in candidates
            if state.inference.tensors[value].dtype == dtype
        ]
        choices = typed or candidates
        if not choices:
            return False
        value = choices[-1]
        name = template.inputs[source]
        previous = bindings.get(name)
        if previous is not None and (
            state.inference.tensors[previous].shape
            != state.inference.tensors[value].shape
            or previous is not value
            and _normal(previous, state.inference, state).key
            != _normal(value, state.inference, state).key
        ):
            return False
        bindings.setdefault(name, value)
        covered.update(target.origins - _normal(value, state.inference, state).origins)
        return True
    if (
        pattern.op != target.op
        or pattern.value != target.value
        or len(pattern.arguments) != len(target.arguments)
    ):
        return False
    covered.update(target.origins)
    commutative = (
        pattern.op in {"add", "mul", "min", "max"}
        or pattern.op == "compare"
        and pattern.value in {"eq", "ne"}
    )
    if not commutative:
        return all(
            _mathematical(a, b, template, state, bindings, covered)
            for a, b in zip(pattern.arguments, target.arguments, strict=True)
        )
    ordered = sorted(
        pattern.arguments,
        key=lambda value: value.op == "leaf" and value.value in template.inputs,
    )

    def assign(index: int, remaining: tuple[_Canonical, ...]) -> bool:
        if index == len(ordered):
            return True
        for position, value in enumerate(remaining):
            before = dict(bindings), set(covered)
            if _mathematical(
                ordered[index], value, template, state, bindings, covered
            ) and assign(index + 1, remaining[:position] + remaining[position + 1 :]):
                return True
            bindings.clear()
            bindings.update(before[0])
            covered.clear()
            covered.update(before[1])
        return False

    return assign(0, target.arguments)


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
    for name, parameter, annotation in signature:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        tensor = annotation is Tensor or Tensor in get_args(annotation)
        if name in actual:
            value = actual[name]
            arguments[name] = (
                Tensor(_parameter=True)
                if tensor and isinstance(value, Weight)
                else value
            )
        elif tensor:
            arguments[name] = (
                None if parameter.default is None else Tensor(_parameter=True)
            )
        elif parameter.default is not inspect.Parameter.empty:
            arguments[name] = parameter.default
        else:
            return None
    return arguments


def _attribute_source(
    root: _Operator,
    required: frozenset[str],
    state: _MatchingState,
    cached: dict[tuple[_Operator, frozenset[str]], tuple[_Operator, int] | None],
) -> _Operator | None:
    pending = [(root, False)]
    while pending:
        operator, expanded = pending.pop()
        key = operator, required
        if key in cached:
            continue
        if required <= _fields(operator).keys():
            cached[key] = operator, 0
            continue
        parents = tuple(
            value._producer
            for value in operator.inputs
            if value._producer is not None
            and value._producer.op
            not in {"cache", "cache_assign", "cache_read", "while", "if"}
            and all(_pure(body, state) for body in value._producer.regions)
        )
        if not expanded:
            pending.append((operator, True))
            pending.extend(
                (parent, False)
                for parent in parents
                if (parent, required) not in cached
            )
            continue
        choices = [
            (source[0], source[1] + 1)
            for parent in parents
            if (source := cached[parent, required]) is not None
        ]
        cached[key] = min(choices, key=lambda item: item[1]) if choices else None
    result = cached[root, required]
    return None if result is None else result[0]


def _template(
    kernel: _Kernel, arguments: dict[str, object], state: _MatchingState | None = None
) -> _SymbolicPattern | None:
    if state is None:
        state = _MatchingState(_Inference())
    leaves: dict[Tensor, str] = {}

    def placeholder(value: object, name: str) -> object:
        if isinstance(value, Tensor):
            tensor = Tensor(_parameter=True)
            leaves[tensor] = name
            return tensor
        if isinstance(value, tuple):
            return tuple(
                placeholder(item, f"{name}.{i}") for i, item in enumerate(value)
            )
        if isinstance(value, list):
            return [placeholder(item, f"{name}.{i}") for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {
                key: placeholder(item, f"{name}.{key}") for key, item in value.items()
            }
        return value

    kwargs = {name: placeholder(value, name) for name, value in arguments.items()}
    try:
        with Graph():
            region = _build_region(
                tuple(leaves), lambda unused: kernel.pattern(**kwargs)
            )
        return _SymbolicPattern(
            leaves,
            region.outputs,
            region,
            tuple(_root_ops(value, state) for value in region.outputs),
        )
    except (TypeError, ValueError, IndexError, AttributeError, NotImplementedError):
        return None


def _pattern_pure(region: _Region, state: _MatchingState) -> bool:
    key = id(region)
    if key not in state.pattern_purity:
        state.pattern_purity[key] = (
            not region.reads
            and not region.writes
            and all(
                operator.op
                not in {"cache", "cache_assign", "cache_read", "if", "random"}
                and all(_pattern_pure(body, state) for body in operator.regions)
                for operator in region.operators
            )
        )
    return state.pattern_purity[key]


def _template_value(value: Tensor, state: _MatchingState) -> Tensor:
    original = value
    if original in state.pattern_values:
        return state.pattern_values[original]
    while (
        (operator := value._producer) is not None
        and isinstance(operator.args, _ConstructArgs)
        and _pattern_pure(operator.regions[0], state)
    ):
        value = operator.regions[0].outputs[operator.outputs.index(value)]
    state.pattern_values[original] = value
    return value


def _root_ops(value: Tensor, state: _MatchingState) -> frozenset[str]:
    if value in state.pattern_roots:
        return state.pattern_roots[value]
    roots: set[str] = set()
    pending = [value]
    visited: set[Tensor] = set()
    while pending:
        tensor = _template_value(pending.pop(), state)
        if tensor in visited:
            continue
        visited.add(tensor)
        operator = tensor._producer
        if operator is None:
            continue
        roots.add(operator.op)
        if isinstance(
            operator.args, (_CastArgs, _ReshapeArgs, _PermuteArgs, _BroadcastArgs)
        ):
            pending.append(operator.args.x)
    result = frozenset(roots)
    state.pattern_roots[value] = result
    return result


def _captures(
    template: _SymbolicPattern,
    outputs: tuple[Tensor, ...],
    state: _MatchingState,
) -> Iterator[dict[str, Tensor]]:
    @cache
    def metadata(tensor: Tensor) -> bool:
        tensor = _template_value(tensor, state)
        if tensor in template.inputs or tensor._producer is None:
            return False
        operator = tensor._producer
        if operator.op == "shape" or isinstance(operator.args, _ConstantArgs):
            return True
        if isinstance(operator.args, _GatherArgs) and operator.args.mask is not None:
            mask = _template_value(operator.args.mask, state)._producer
            if (
                mask is not None
                and isinstance(mask.args, _ConstantArgs)
                and _literal_equal(mask.args.value, False)
            ):
                return True
        if operator.regions or operator.op not in {
            "add",
            "sub",
            "mul",
            "floor_div",
            "div",
            "sqrt",
            "mod",
            "min",
            "max",
            "cast",
            "gather",
            "reshape",
            "iota",
            "broadcast",
            "permute",
            "select",
            "compare",
            "scatter",
        }:
            return False
        return all(metadata(item) for item in operator.inputs)

    def value(
        pattern: object,
        target: object,
        bindings: dict[str, Tensor],
        parameters: dict[Tensor, Tensor],
        pairs: frozenset[tuple[Tensor, Tensor]],
    ) -> Iterator[_PatternBindings]:
        if isinstance(pattern, Tensor) and isinstance(target, Tensor):
            if pattern in template.inputs:
                name = template.inputs[pattern]
                previous = bindings.get(name)
                if previous is None:
                    yield _PatternBindings({**bindings, name: target}, pairs)
                elif _canonical(previous, state.inference, state) is _canonical(
                    target, state.inference, state
                ):
                    yield _PatternBindings(bindings, pairs)
                producer = target._producer
                if producer is not None and isinstance(producer.args, _CastArgs):
                    # Typed replay must recreate any cast crossed during discovery.
                    yield from value(
                        pattern, producer.args.x, bindings, parameters, pairs
                    )
                return
            if pattern in parameters:
                if _canonical(
                    parameters[pattern], state.inference, state
                ) is _canonical(target, state.inference, state):
                    yield _PatternBindings(bindings, pairs)
                return
            pattern = _template_value(pattern, state)
            original_target = target
            target = _unwrapped(target, state)
            pair = pattern, target
            if pair in pairs:
                yield _PatternBindings(bindings, pairs)
                return
            p, t = pattern._producer, target._producer
            if p is None or t is None:
                return
            if (
                p.op != t.op
                and metadata(pattern)
                and not isinstance(p.args, _ConstantArgs)
            ):
                yield _PatternBindings(bindings, pairs)
                return
            # A typed replay proves whether these candidate views are identities.
            if p.op != t.op:
                if isinstance(
                    p.args, (_CastArgs, _ReshapeArgs, _PermuteArgs, _BroadcastArgs)
                ):
                    yield from value(
                        p.args.x, original_target, bindings, parameters, pairs | {pair}
                    )
                    if original_target is not target:
                        yield from value(
                            p.args.x, target, bindings, parameters, pairs | {pair}
                        )
                if isinstance(
                    t.args, (_CastArgs, _ReshapeArgs, _PermuteArgs, _BroadcastArgs)
                ):
                    yield from value(
                        pattern, t.args.x, bindings, parameters, pairs | {pair}
                    )
            if isinstance(t.args, _ConstructArgs) and _structural_loop(
                t, state.inference, state
            ):
                target = t.regions[0].outputs[t.outputs.index(target)]
                yield from value(pattern, target, bindings, parameters, pairs)
                return
            if p.op != t.op or p.outputs.index(pattern) != t.outputs.index(target):
                return
            if len(p.regions) != len(t.regions) or t.op in {
                "cache",
                "cache_assign",
                "cache_read",
                "if",
                "random",
            }:
                return
            if isinstance(t.args, _WhileArgs) and not _bounded_while(
                t, state.inference, state
            ):
                return
            fields, actual = _fields(p), _fields(t)
            if fields.keys() != actual.keys():
                return
            alternatives = [actual]
            if p.op in {"add", "mul", "min", "max"}:
                alternatives.append({**actual, "a": actual["b"], "b": actual["a"]})
            nested = dict(parameters)
            for left, right in zip(p.regions, t.regions, strict=True):
                if len(left.parameters) != len(right.parameters) or len(
                    left.outputs
                ) != len(right.outputs):
                    return
                nested.update(zip(left.parameters, right.parameters, strict=True))
            for alternative in alternatives:
                comparisons = [
                    (item, alternative[name]) for name, item in fields.items()
                ]
                comparisons.extend(
                    (a, b)
                    for left, right in zip(p.regions, t.regions, strict=True)
                    for a, b in zip(left.outputs, right.outputs, strict=True)
                )
                yield from sequence(comparisons, 0, bindings, nested, pairs | {pair})
            return
        if isinstance(target, Tensor) and isinstance(pattern, (bool, int, float)):
            producer = _unwrapped(target, state)._producer
            if producer is not None and isinstance(producer.args, _ConstantArgs):
                literal = producer.args.value
                if isinstance(literal, np.ndarray) and literal.shape == ():
                    literal = literal.item()
                if isinstance(literal, (bool, int, float)) and pattern == literal:
                    yield _PatternBindings(bindings, pairs)
            return
        if isinstance(pattern, (tuple, list)) and isinstance(target, (tuple, list)):
            if len(pattern) == len(target):
                yield from sequence(
                    list(zip(pattern, target, strict=True)),
                    0,
                    bindings,
                    parameters,
                    pairs,
                )
        elif isinstance(pattern, dict) and isinstance(target, dict):
            if pattern.keys() == target.keys():
                yield from sequence(
                    [(item, target[name]) for name, item in pattern.items()],
                    0,
                    bindings,
                    parameters,
                    pairs,
                )
        elif _literal_equal(pattern, target):
            yield _PatternBindings(bindings, pairs)

    def sequence(
        comparisons: list[tuple[object, object]],
        index: int,
        bindings: dict[str, Tensor],
        parameters: dict[Tensor, Tensor],
        pairs: frozenset[tuple[Tensor, Tensor]],
    ) -> Iterator[_PatternBindings]:
        if index == len(comparisons):
            yield _PatternBindings(bindings, pairs)
            return
        left, right = comparisons[index]
        for bound in value(left, right, bindings, parameters, pairs):
            yield from sequence(
                comparisons, index + 1, bound.values, parameters, bound.pairs
            )

    for captured in sequence(
        list(zip(template.outputs, outputs, strict=True)), 0, {}, {}, frozenset()
    ):
        yield captured.values


def _algebraic_captures(
    template: _SymbolicPattern, outputs: tuple[Tensor, ...], state: _MatchingState
) -> Iterator[dict[str, Tensor]]:
    # Formal expressions discover variables; typed replay checks shapes and semantics.
    facts = _Inference()
    for parameter in template.inputs:
        facts.tensors[parameter] = _TensorInfo(None, None, None)
    for _, operator in _walk(template.region):
        for tensor in operator.outputs:
            facts.tensors[tensor] = _TensorInfo(None, None, tensor.dtype)

    def compare(
        p: _Canonical, t: _Canonical, bindings: dict[str, Tensor]
    ) -> Iterator[dict[str, Tensor]]:
        if state.remaining <= 0:
            return
        state.remaining -= 1
        if (
            p.op == "leaf"
            and isinstance(p.value, Tensor)
            and p.value in template.inputs
        ):
            name = template.inputs[p.value]
            for source in reversed(t.sources):
                previous = bindings.get(name)
                if previous is None or _unwrapped(previous, state) is _unwrapped(
                    source, state
                ):
                    yield {**bindings, name: source}
            return
        if p.op != t.op or p.value != t.value or len(p.arguments) != len(t.arguments):
            return
        commutative = p.op in {"add", "mul", "min", "max"}
        ordered = (
            sorted(
                p.arguments, key=lambda v: v.op == "leaf" and v.value in template.inputs
            )
            if commutative
            else p.arguments
        )

        def assign(
            index: int, remaining: tuple[_Canonical, ...], bound: dict[str, Tensor]
        ) -> Iterator[dict[str, Tensor]]:
            if index == len(ordered):
                yield bound
                return
            for slot in (
                range(len(remaining)) if commutative else range(min(1, len(remaining)))
            ):
                for candidate in compare(ordered[index], remaining[slot], bound):
                    yield from assign(
                        index + 1, remaining[:slot] + remaining[slot + 1 :], candidate
                    )

        yield from assign(0, t.arguments, bindings)

    try:
        state.normal_budget = 4096
        state.remaining = 4096
        patterns = tuple(_normal(p, facts, state) for p in template.outputs)
        targets = tuple(_normal(t, state.inference, state) for t in outputs)

        def match(
            index: int, bindings: dict[str, Tensor]
        ) -> Iterator[dict[str, Tensor]]:
            if index == len(patterns):
                yield bindings
                return
            for candidate in compare(patterns[index], targets[index], bindings):
                yield from match(index + 1, candidate)

        yield from match(0, {})
    except (
        AssertionError,
        KeyError,
        ValueError,
        OverflowError,
        RecursionError,
        ZeroDivisionError,
    ):
        return


def _bound_arguments(
    arguments: dict[str, object], bindings: dict[str, Tensor], inference: _Inference
) -> dict[str, object]:
    def bound(value: object, name: str) -> object:
        if isinstance(value, Tensor):
            if name in bindings:
                return bindings[name]
            if value in inference.tensors:
                return value
            raise ValueError(f"unbound pattern operand: {name}")
        if isinstance(value, tuple):
            return tuple(bound(item, f"{name}.{i}") for i, item in enumerate(value))
        if isinstance(value, list):
            return [bound(item, f"{name}.{i}") for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: bound(item, f"{name}.{key}") for key, item in value.items()}
        return value

    return {name: bound(value, name) for name, value in arguments.items()}


def _key(value: object, inference: _Inference | None = None) -> object:
    if isinstance(value, Tensor):
        if inference is None:
            return ("tensor",)
        info = inference.tensors[value]
        shape = (
            tuple(item.value if item.op == "const" else None for item in info.shape)
            if info.shape is not None
            else None
        )
        return ("tensor", info.dtype, shape, info.maximum_shape)
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
            maximum = info.maximum_shape
            tensor = Input(
                name=name,
                shape=TensorShape(shape),
                dtype=info.dtype,
                maximum_shape=tuple(x for x in maximum if x is not None)
                if maximum is not None and all(x is not None for x in maximum)
                else None,
            )
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
        return _KernelPattern(leaves, tuple(graph._outputs.values()), facts)
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
    graph: Graph,
    kernels: tuple[_Kernel, ...],
    inference: _Inference,
    *,
    algebraic: bool = True,
) -> _MatchSet:
    if graph._region is None:
        raise ValueError("cannot match an incomplete graph")
    result = _MatchSet()
    state = _MatchingState(inference)
    available = {
        backend: _available_capabilities(backend)
        for backend in {kernel.backend for kernel in kernels}
    }
    eligible = tuple(
        kernel
        for kernel in kernels
        if kernel.capabilities & available[kernel.backend] == kernel.capabilities
    )
    signatures = {id(kernel): _signature(kernel) for kernel in eligible}
    constraints: dict[_Operator | None, list[_ShapeConstraint]] = {}
    for constraint in inference.constraints:
        constraints.setdefault(constraint.operator, []).append(constraint)
    attribute_sources: dict[
        tuple[_Operator, frozenset[str]], tuple[_Operator, int] | None
    ] = {}
    required_attributes = {
        id(kernel): frozenset(
            name
            for name, parameter, annotation in signatures[id(kernel)]
            if parameter.default is inspect.Parameter.empty
            and parameter.kind != inspect.Parameter.VAR_KEYWORD
            and not (annotation is Tensor or Tensor in get_args(annotation))
        )
        for kernel in eligible
    }
    templates: dict[tuple[int, object], _SymbolicPattern | None] = {}
    typed: dict[tuple[int, object], _KernelPattern | None] = {}
    static: dict[int, tuple[dict[str, object], _SymbolicPattern]] = {}
    root_index: dict[tuple[int, int, str], set[int]] = {}
    dynamic: set[int] = set()
    for kernel in eligible:
        signature = signatures[id(kernel)]
        if any(
            not (annotation is Tensor or Tensor in get_args(annotation))
            or parameter.default is not inspect.Parameter.empty
            for _, parameter, annotation in signature
        ):
            dynamic.add(id(kernel))
            continue
        default_arguments: dict[str, object] = {
            name: Tensor(_parameter=True) for name, _, _ in signature
        }
        symbolic = _template(kernel, default_arguments, state)
        if symbolic is None:
            continue
        static[id(kernel)] = default_arguments, symbolic
        for index, operations in enumerate(symbolic.roots):
            for operation in operations:
                root_index.setdefault(
                    (len(symbolic.outputs), index, operation), set()
                ).add(id(kernel))
    by_id = {id(kernel): kernel for kernel in eligible}
    order = {id(kernel): index for index, kernel in enumerate(eligible)}
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
        roots = tuple(_root_ops(value, state) for value in operator.outputs)
        internals = {node for body in operator.regions for _, node in _walk(body)}
        indexed: set[int] | None = None
        for index, operations in enumerate(roots):
            candidates = set().union(
                *(
                    root_index.get((len(roots), index, operation), set())
                    for operation in operations
                )
            )
            indexed = candidates if indexed is None else indexed & candidates
        candidate_ids = set(static) if algebraic else (indexed or set())
        candidate_ids.update(dynamic)
        for kernel_id in sorted(candidate_ids, key=order.__getitem__):
            kernel = by_id[kernel_id]
            if kernel.parameter_schema.output_count != len(operator.outputs):
                continue
            known = static.get(id(kernel))
            arguments = (
                known[0]
                if known is not None
                else _arguments(operator, signatures[id(kernel)])
            )
            if arguments is None:
                attribute_source = _attribute_source(
                    operator, required_attributes[id(kernel)], state, attribute_sources
                )
                if attribute_source is not None:
                    arguments = _arguments(attribute_source, signatures[id(kernel)])
                    assert arguments is not None
                    outer = _fields(operator)
                    for name in arguments:
                        if name in outer and not isinstance(outer[name], Tensor):
                            arguments[name] = outer[name]
            if arguments is None:
                continue
            key = id(kernel), _key(arguments)
            if key not in templates:
                templates[key] = (
                    known[1]
                    if known is not None
                    else _template(kernel, arguments, state)
                )
            symbolic = templates[key]
            if symbolic is None or len(symbolic.outputs) != len(operator.outputs):
                continue
            if all(output in symbolic.inputs for output in symbolic.outputs):
                continue
            structural = all(a & b for a, b in zip(symbolic.roots, roots, strict=True))
            if not structural and not algebraic:
                continue
            captures = (
                _captures(symbolic, operator.outputs, state) if structural else iter(())
            )
            if algebraic:
                captures = chain(
                    captures, _algebraic_captures(symbolic, operator.outputs, state)
                )
            attempted: set[tuple[tuple[str, Tensor], ...]] = set()
            for captured in captures:
                identity = tuple(sorted(captured.items()))
                if identity in attempted:
                    continue
                attempted.add(identity)
                if any(
                    not _safe_boundary(value, internals) for value in captured.values()
                ):
                    continue
                try:
                    actual_arguments = _bound_arguments(arguments, captured, inference)
                except ValueError:
                    continue
                typed_key = id(kernel), _key(actual_arguments, inference)
                if typed_key not in typed:
                    typed[typed_key] = _pattern(kernel, actual_arguments, inference)
                pattern = typed[typed_key]
                if pattern is None or all(
                    output in pattern.inputs for output in pattern.outputs
                ):
                    continue
                bindings = dict(captured)
                covered: set[_Operator] = set()
                pairs: set[tuple[Tensor, Tensor]] = set()
                state.canonical = False
                matched = all(
                    _equivalent(p, t, pattern, state, bindings, {}, covered, pairs)
                    for p, t in zip(pattern.outputs, operator.outputs, strict=True)
                )
                if not matched:
                    bindings.clear()
                    bindings.update(captured)
                    covered.clear()
                    pairs.clear()
                    state.canonical = True
                    state.remaining = 256
                    matched = all(
                        _equivalent(p, t, pattern, state, bindings, {}, covered, pairs)
                        for p, t in zip(pattern.outputs, operator.outputs, strict=True)
                    )
                mathematical = algebraic and not matched
                if mathematical:
                    bindings.clear()
                    bindings.update(captured)
                    covered.clear()
                    state.normal_budget = 4096
                    state.remaining = 4096
                    try:
                        matched = all(
                            _shape_matches(p, t, pattern, state)
                            and _mathematical(
                                _normal(p, pattern.inference, state),
                                _normal(t, inference, state),
                                pattern,
                                state,
                                bindings,
                                covered,
                            )
                            for p, t in zip(
                                pattern.outputs, operator.outputs, strict=True
                            )
                        )
                    except (
                        OverflowError,
                        RecursionError,
                        ValueError,
                        ZeroDivisionError,
                    ):
                        matched = False
                if not matched:
                    continue
                # Replacing a region also removes its unused pure computations.
                removed_nodes = {
                    node
                    for parent in covered
                    for body in parent.regions
                    for _, node in _walk(body)
                }
                boundary = internals | removed_nodes
                covered.update(
                    node
                    for node in removed_nodes
                    if not isinstance(node.args, (_ConstantArgs, _InputArgs))
                )
                # Metadata-only inputs can disappear when their shape is statically known.
                for name, value in actual_arguments.items():
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
                if any(
                    not _safe_boundary(value, boundary) for value in bindings.values()
                ):
                    continue
                if any(
                    name not in physical
                    or (producer := bindings[name]._producer) is None
                    or not isinstance(producer.args, _ConstantArgs)
                    or not isinstance(producer.args.value, Weight)
                    for name in kernel.packings
                ):
                    continue
                storage = {
                    bindings[name]: packing
                    for name, packing in kernel.packings.items()
                    if packing != "canonical"
                }
                if any(
                    storage.get(bindings[name], "canonical")
                    != kernel.packings.get(name, "canonical")
                    for name in physical
                ):
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
                    loop_dimensions = _loop_dimensions(covered, inference)
                    if loop_dimensions is None:
                        continue
                    rewrite = (
                        _shape_rewriter(loop_dimensions) if loop_dimensions else None
                    )
                    attributes = {
                        **_fields(operator),
                        **{
                            name: bindings.get(name, value)
                            if isinstance(value, Tensor)
                            else value
                            for name, value in actual_arguments.items()
                        },
                    }
                    context = _KernelBindingContext(
                        {
                            name: _binding(tensor, state)
                            for name, tensor in bindings.items()
                        },
                        tuple(_binding(tensor, state) for tensor in operator.outputs),
                        attributes,
                    )
                    parameters = kernel.bind(context)
                    references = set().union(
                        *(_parameter_references(value) for value in parameters.values())
                    )
                    if rewrite is not None:
                        for output in context.outputs:
                            for dimension in output.shape:
                                references.update(
                                    _parameter_references(rewrite(dimension))
                                )
                    for owner in (*covered, None):
                        for constraint in constraints.get(owner, ()):
                            if rewrite is not None:
                                adjusted = _rewrite_constraint(constraint, rewrite)
                                if adjusted is None:
                                    continue
                                constraint = adjusted
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
                        attributes,
                        storage,
                        loop_dimensions,
                    )
                )
                break
        result.candidates[operator] = tuple(
            sorted(matches, key=lambda item: (-len(item.covered), item.kernel.name))
        )
    return result


def _bounded_while(
    operator: _Operator, inference: _Inference, state: _MatchingState
) -> bool:
    return operator in inference.loop_iterations and all(
        _pure(body, state) and not any(_observable(node) for node in body.operators)
        for body in operator.regions
    )


def _structural_loop(
    operator: _Operator, inference: _Inference, state: _MatchingState
) -> bool:
    if isinstance(operator.args, _WhileArgs):
        return _bounded_while(operator, inference, state)
    if not isinstance(operator.args, _ConstructArgs):
        return False
    region = operator.regions[0]
    found = False
    for scope, node in _walk(region):
        if scope.reads or scope.writes:
            return False
        if isinstance(node.args, _WhileArgs):
            if not _bounded_while(node, inference, state):
                return False
            found = True
        elif not isinstance(node.args, _ConstructArgs) and (
            node.op in {"cache_read", "if"} or _observable(node)
        ):
            return False
    return found


def _shape_rewriter(
    dimensions: dict[Tensor, tuple[_ShapeExpression, ...]],
) -> Callable[[_ShapeExpression], _ShapeExpression]:
    @cache
    def rewrite(value: _ShapeExpression) -> _ShapeExpression:
        tensor = value.tensor
        if value.op == "dimension" and tensor is not None and tensor in dimensions:
            assert value.axis is not None
            return rewrite(dimensions[tensor][value.axis])
        args = tuple(rewrite(arg) for arg in value.args)
        return value if args == value.args else _expression(value.op, *args)

    return rewrite


def _rewrite_constraint(
    constraint: _ShapeConstraint,
    rewrite: Callable[[_ShapeExpression], _ShapeExpression],
) -> _ShapeConstraint | None:
    left, right = rewrite(constraint.left), rewrite(constraint.right)
    if left == right and constraint.predicate in {"eq", "le", "ge", "broadcast"}:
        return None
    return replace(constraint, left=left, right=right)


def _loop_dimensions(
    covered: set[_Operator], inference: _Inference
) -> dict[Tensor, tuple[_ShapeExpression, ...]] | None:
    dimensions: dict[Tensor, tuple[_ShapeExpression, ...]] = {}
    returns: list[tuple[Tensor, tuple[_ShapeExpression, ...]]] = []
    nested: set[_Operator] = set()
    for node in covered:
        if not isinstance(node.args, _WhileArgs):
            continue
        condition, body = node.regions
        for initial, outcome, test, parameter, result in zip(
            node.args.initial,
            node.outputs,
            condition.parameters,
            body.parameters,
            body.outputs,
            strict=True,
        ):
            shape = inference.tensors[initial].shape
            if shape is None:
                return None
            dimensions[outcome] = dimensions[test] = dimensions[parameter] = shape
            returns.append((result, shape))
        nested.update(child for region in node.regions for _, child in _walk(region))
    if not dimensions:
        return dimensions
    rewrite = _shape_rewriter(dimensions)
    # Substitute the entry dimensions, then prove that each update preserves them.
    for result, initial_shape in returns:
        result_shape = inference.tensors[result].shape
        if result_shape is None or tuple(map(rewrite, result_shape)) != tuple(
            map(rewrite, initial_shape)
        ):
            return None
    # A body guard cannot run before the call when the original loop may skip it.
    if any(
        guard.operator in nested and _rewrite_constraint(guard, rewrite) is not None
        for guard in inference.constraints
    ):
        return None
    return dimensions


def _selected_loop_inference(
    inference: _Inference, selection: _Selection
) -> _Inference:
    dimensions = {
        tensor: shape
        for call in selection.calls.values()
        for tensor, shape in call.loop_dimensions.items()
    }
    if not dimensions:
        return inference
    rewrite = _shape_rewriter(dimensions)
    tensors = {
        tensor: replace(
            info,
            shape=None if info.shape is None else tuple(map(rewrite, info.shape)),
            values=None if info.values is None else tuple(map(rewrite, info.values)),
        )
        for tensor, info in inference.tensors.items()
    }
    constraints = [
        rewritten
        for guard in inference.constraints
        if (rewritten := _rewrite_constraint(guard, rewrite)) is not None
    ]
    return replace(inference, tensors=tensors, constraints=constraints)
