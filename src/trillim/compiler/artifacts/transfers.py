from __future__ import annotations

from ...tdl.core.core_pydantic_models import IfArgs, WhileArgs
from ...tdl.core.operator import Operator
from ...tdl.core.tensor import Tensor
from ..kernels.kernel_dataclasses import (
    KernelBindingContext,
    ParameterValue,
    TensorBinding,
)
from .artifact_dataclasses import Layout


def pairs(operator: Operator) -> dict[str, tuple[tuple[Tensor, Tensor], ...]]:
    args = operator.args
    if isinstance(args, IfArgs):
        return {
            str(index): tuple(zip(body.outputs, operator.outputs, strict=True))
            for index, body in enumerate(operator.regions)
        }
    if isinstance(args, WhileArgs):
        condition, body = operator.regions
        return {
            "initial": tuple(zip(args.initial, operator.outputs, strict=True)),
            "condition": tuple(
                zip(operator.outputs, condition.parameters, strict=True)
            ),
            "body": tuple(zip(operator.outputs, body.parameters, strict=True)),
            "update": tuple(zip(body.outputs, operator.outputs, strict=True)),
        }
    return {}


def parameters(
    layout: Layout, source: Tensor, destination: Tensor
) -> dict[str, ParameterValue]:
    if layout.transfer_kernel is None:
        raise ValueError("control flow requires a native transfer kernel")
    original = layout.buffers[source].binding()
    target = TensorBinding(
        original.shape, original.maximum_shape, layout.buffers[destination].dtype
    )
    return layout.transfer_kernel.bind(
        KernelBindingContext({"x": original}, (target,), {})
    )
