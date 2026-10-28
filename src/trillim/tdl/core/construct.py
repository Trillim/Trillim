from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import cast

from pydantic import ConfigDict, validate_call

from .core_pydantic_models import ConstructArgs
from .graph import _build_region
from .operator import Operator
from .tensor import Tensor


def Construct[**P, R: (Tensor, tuple[Tensor, ...])](
    function: Callable[P, R],
) -> Callable[P, R]:
    """Record a named operation with a complete, recursively lowerable body.

    The compiler executes the sole region and binds its results to the call's
    outputs, or substitutes an equivalent kernel. User-defined constructs use
    this same decorator. Callbacks run at graph construction time only.
    """
    checked = validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, strict=True)
    )(function)
    parameters = signature(function)

    @wraps(function)
    def build(*args: P.args, **kwargs: P.kwargs) -> R:
        multiple = False

        def body(unused: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            nonlocal multiple
            result = checked(*args, **kwargs)
            if isinstance(result, tuple):
                multiple = True
                return result
            return (result,)

        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        region = _build_region((), body)
        operator = Operator(
            f"{function.__module__}.{function.__qualname__}",
            ConstructArgs(arguments=dict(bound.arguments)),
            outputs=tuple((value.shape, value.dtype) for value in region.outputs),
            regions=(region,),
        )
        return cast(R, operator.outputs if multiple else operator.outputs[0])

    return build
