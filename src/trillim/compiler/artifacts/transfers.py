from __future__ import annotations

from ...tdl.core.core_pydantic_models import _IfArgs, _WhileArgs
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..kernels.kernel_dataclasses import (
    _KernelBindingContext,
    _ParameterValue,
    _TensorBinding,
)
from .artifact_dataclasses import _Layout


def _pairs(operator: _Operator) -> dict[str, tuple[tuple[Tensor, Tensor], ...]]:
    args = operator.args
    if isinstance(args, _IfArgs):
        return {
            str(index): tuple(zip(body.outputs, operator.outputs, strict=True))
            for index, body in enumerate(operator.regions)
        }
    if isinstance(args, _WhileArgs):
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


def _parameters(
    layout: _Layout, source: Tensor, destination: Tensor
) -> dict[str, _ParameterValue]:
    if layout.transfer_kernel is None:
        raise ValueError("control flow requires a native transfer kernel")
    original = layout.buffers[source]._binding()
    target = _TensorBinding(
        original.shape, original.maximum_shape, layout.buffers[destination].dtype
    )
    return layout.transfer_kernel.bind(
        _KernelBindingContext({"x": original}, (target,), {})
    )
