from __future__ import annotations

from collections.abc import Mapping

from .tdl_pydantic_models import op_args
from .tensor import Tensor


class Operator:
    def __init__(
        self,
        *,
        family: str,
        op: str,
        args: op_args,
        inputs: tuple[Tensor, ...],
    ) -> None:
        self.family = family
        self.op = op
        self.args = args
        self.inputs = inputs
        self.output: Tensor | None = None


def _validate_variant(
    *,
    family: str,
    op: str,
    args: op_args,
    registry: Mapping[str, type[op_args]],
) -> None:
    expected = registry.get(op)

    if expected is None:
        raise ValueError(f'Unknown {family} op "{op}"')

    if type(args) is not expected:
        raise TypeError(
            f'{family}(op="{op}") expects '
            f"{expected.__name__}, got {type(args).__name__}"
        )


def _create_operator(
    *,
    family: str,
    op: str,
    args: op_args,
) -> Tensor:
    inputs: list[Tensor] = []

    for value in vars(args).values():
        if isinstance(value, Tensor):
            inputs.append(value)
        elif isinstance(value, list):
            inputs.extend(item for item in value if isinstance(item, Tensor))

    operator = Operator(family=family, op=op, args=args, inputs=tuple(inputs))
    output = Tensor(producer=operator)
    operator.output = output

    for tensor in inputs:
        tensor.consumers.append(operator)

    return output
