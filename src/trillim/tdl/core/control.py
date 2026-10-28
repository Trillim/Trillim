from __future__ import annotations

from collections.abc import Callable

from .core_dataclasses import Shape, TensorTuple
from .core_pydantic_models import IfArgs, ReduceArgs, WhileArgs
from .graph import _build_region
from .operator import Operator
from .tensor import Tensor


def _validate_condition(condition: Tensor) -> None:
    if condition.dtype is not None and condition.dtype != "bool":
        raise ValueError("control-flow conditions must have bool dtype")
    if condition.shape is not None and condition.shape.dimensions != ():
        raise ValueError("control-flow conditions must be scalar")


def If(
    condition: Tensor,
    *,
    then_branch: Callable[[], TensorTuple],
    else_branch: Callable[[], TensorTuple],
) -> TensorTuple:
    """Build both branches; execute only the selected branch at runtime."""
    args = IfArgs(condition=condition)
    _validate_condition(args.condition)

    def build_then(parameters: TensorTuple) -> TensorTuple:
        return then_branch()

    def build_else(parameters: TensorTuple) -> TensorTuple:
        return else_branch()

    then_region = _build_region((), build_then)
    else_region = _build_region((), build_else)
    if len(then_region.outputs) != len(else_region.outputs):
        raise ValueError("If branches must return the same number of tensors")
    return Operator(
        "if",
        args,
        outputs=tuple((None, None) for _ in then_region.outputs),
        regions=(then_region, else_region),
    ).outputs


def While(
    initial: TensorTuple,
    *,
    condition: Callable[[TensorTuple], Tensor],
    body: Callable[[TensorTuple], TensorTuple],
) -> TensorTuple:
    """Carry tensors through a loop; condition and body are traced once.

    The condition executes before every iteration, including the final check.
    It may read caches but cannot update them. The compiler infers a stable
    carried dtype from initial values and body results, inserting entry casts
    for numeric promotion. The body preserves carried arity, dtype and rank;
    dimension sizes may change within compiler-supported bounds.
    """
    args = WhileArgs(initial=initial)
    condition_parameters = tuple(
        Tensor(shape=value.shape, _parameter=True) for value in args.initial
    )
    body_parameters = tuple(
        Tensor(shape=value.shape, _parameter=True) for value in args.initial
    )

    def build_condition(parameters: TensorTuple) -> TensorTuple:
        return (condition(parameters),)

    condition_region = _build_region(condition_parameters, build_condition)
    if condition_region.writes:
        raise ValueError("While conditions cannot update caches")
    _validate_condition(condition_region.outputs[0])
    body_region = _build_region(body_parameters, body)
    if len(body_region.outputs) != len(args.initial):
        raise ValueError("While body must preserve the number of carried tensors")
    return Operator(
        "while",
        args,
        outputs=tuple((None, None) for _ in args.initial),
        regions=(condition_region, body_region),
    ).outputs


def Reduce(
    values: TensorTuple,
    *,
    initial: TensorTuple,
    axes: tuple[int, ...],
    reducer: Callable[[TensorTuple, TensorTuple], TensorTuple],
    keepdims: bool = False,
) -> TensorTuple:
    """Reduce corresponding elements using a pure, associative scalar graph.

    The reducer receives accumulator and element tuples and returns the next
    accumulator tuple. The compiler infers stable accumulator dtypes from the
    initial values and reducer, inserting promotion casts where required.
    Initial values are scalars and are returned in the accumulator dtype for
    empty domains. Tuple accumulators support value/index tie-breaking.
    """
    args = ReduceArgs(values=values, initial=initial, axes=axes, keepdims=keepdims)
    for value in args.initial:
        if value.shape is not None and value.shape.dimensions != ():
            raise ValueError("reduction initial values must be scalar")
    accumulators = tuple(Tensor(shape=Shape(()), _parameter=True) for _ in args.initial)
    elements = tuple(
        Tensor(shape=Shape(()), dtype=value.dtype, _parameter=True)
        for value in args.values
    )
    count = len(accumulators)

    def build(parameters: TensorTuple) -> TensorTuple:
        return reducer(parameters[:count], parameters[count:])

    region = _build_region((*accumulators, *elements), build)
    if region.reads or region.writes:
        raise ValueError(
            "reducers cannot access caches; capture invariant tensors instead"
        )
    if len(region.outputs) != count:
        raise ValueError("reducer must preserve accumulator arity")
    return Operator(
        "reduce",
        args,
        outputs=tuple((None, None) for _ in args.initial),
        regions=(region,),
    ).outputs
