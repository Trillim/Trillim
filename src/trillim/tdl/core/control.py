from __future__ import annotations

from collections.abc import Callable

from .core_dataclasses import Shape, _TensorTuple
from .core_pydantic_models import _IfArgs, _ReduceArgs, _WhileArgs
from .graph import _build_region
from .operator import _Operator
from .tensor import Tensor


def _validate_condition(condition: Tensor) -> None:
    if condition.dtype is not None and condition.dtype != "bool":
        raise ValueError("control-flow conditions must have bool dtype")
    if condition.shape is not None and condition.shape.dimensions != ():
        raise ValueError("control-flow conditions must be scalar")


def If(
    condition: Tensor,
    *,
    then_branch: Callable[[], _TensorTuple],
    else_branch: Callable[[], _TensorTuple],
) -> _TensorTuple:
    """Build both branches; execute only the selected branch at runtime."""
    args = _IfArgs(condition=condition)
    _validate_condition(args.condition)

    def build_then(parameters: _TensorTuple) -> _TensorTuple:
        return then_branch()

    def build_else(parameters: _TensorTuple) -> _TensorTuple:
        return else_branch()

    then_region = _build_region((), build_then)
    else_region = _build_region((), build_else)
    if len(then_region.outputs) != len(else_region.outputs):
        raise ValueError("If branches must return the same number of tensors")
    return _Operator(
        "if",
        args,
        outputs=tuple((None, None) for _ in then_region.outputs),
        regions=(then_region, else_region),
    ).outputs


def While(
    initial: _TensorTuple,
    *,
    condition: Callable[[_TensorTuple], Tensor],
    body: Callable[[_TensorTuple], _TensorTuple],
) -> _TensorTuple:
    """Carry tensors through a loop; condition and body are traced once.

    The condition executes before every iteration, including the final check.
    It may read caches but cannot update them. The compiler infers a stable
    carried dtype from initial values and body results, inserting entry casts
    for numeric promotion. The body preserves carried arity, dtype and rank;
    dimension sizes may change within compiler-supported bounds.
    """
    args = _WhileArgs(initial=initial)
    condition_parameters = tuple(
        Tensor(shape=value.shape, _parameter=True) for value in args.initial
    )
    body_parameters = tuple(
        Tensor(shape=value.shape, _parameter=True) for value in args.initial
    )

    def build_condition(parameters: _TensorTuple) -> _TensorTuple:
        return (condition(parameters),)

    condition_region = _build_region(condition_parameters, build_condition)
    if condition_region.writes:
        raise ValueError("While conditions cannot update caches")
    _validate_condition(condition_region.outputs[0])
    body_region = _build_region(body_parameters, body)
    if len(body_region.outputs) != len(args.initial):
        raise ValueError("While body must preserve the number of carried tensors")
    return _Operator(
        "while",
        args,
        outputs=tuple((None, None) for _ in args.initial),
        regions=(condition_region, body_region),
    ).outputs


def Reduce(
    values: _TensorTuple,
    *,
    initial: _TensorTuple,
    axes: tuple[int, ...],
    reducer: Callable[[_TensorTuple, _TensorTuple], _TensorTuple],
    keepdims: bool = False,
) -> _TensorTuple:
    """Reduce corresponding elements using a pure, associative scalar graph.

    The reducer receives accumulator and element tuples and returns the next
    accumulator tuple. The compiler infers stable accumulator dtypes from the
    initial values and reducer, inserting promotion casts where required.
    Initial values are scalars and are returned in the accumulator dtype for
    empty domains. Tuple accumulators support value/index tie-breaking.
    """
    args = _ReduceArgs(values=values, initial=initial, axes=axes, keepdims=keepdims)
    for value in args.initial:
        if value.shape is not None and value.shape.dimensions != ():
            raise ValueError("reduction initial values must be scalar")
    accumulators = tuple(Tensor(shape=Shape(()), _parameter=True) for _ in args.initial)
    elements = tuple(
        Tensor(shape=Shape(()), dtype=value.dtype, _parameter=True)
        for value in args.values
    )
    count = len(accumulators)

    def build(parameters: _TensorTuple) -> _TensorTuple:
        return reducer(parameters[:count], parameters[count:])

    region = _build_region((*accumulators, *elements), build)
    if region.reads or region.writes:
        raise ValueError(
            "reducers cannot access caches; capture invariant tensors instead"
        )
    if len(region.outputs) != count:
        raise ValueError("reducer must preserve accumulator arity")
    return _Operator(
        "reduce",
        args,
        outputs=tuple((None, None) for _ in args.initial),
        regions=(region,),
    ).outputs
