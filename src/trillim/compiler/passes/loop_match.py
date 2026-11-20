from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import cache

from ...tdl.core.core_pydantic_models import _ConstructArgs, _WhileArgs
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..compiler_dataclasses import _Inference, _ShapeConstraint, _ShapeExpression
from .canonicalize import _pure
from .infer import _expression
from .optimize import _observable
from .passes_dataclasses import _MatchingState, _Selection
from .walk import _walk


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
