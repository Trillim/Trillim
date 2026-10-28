from __future__ import annotations

from pydantic import BaseModel

from .core_dataclasses import Region, Shape
from .core_pydantic_models import OperatorArgs
from .graph import _record_operator
from .tensor import Tensor
from .types import DType


class Operator:
    """Create a node and its tensor links; record it when tracing is active."""

    def __init__(
        self,
        op: str,
        args: OperatorArgs,
        *,
        outputs: tuple[tuple[Shape | None, DType | None], ...] = ((None, None),),
        regions: tuple[Region, ...] = (),
    ) -> None:
        self.op = op
        self.args = args
        self.regions = regions
        self.dependencies: tuple[Operator, ...] = ()
        inputs: dict[Tensor, None] = {}

        def collect(value: object) -> None:
            if isinstance(value, Tensor):
                inputs[value] = None
            elif isinstance(value, BaseModel):
                for item in vars(value).values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(args)
        for region in regions:
            collect(region.captures)
        self.inputs = tuple(inputs)
        self.outputs = tuple(
            Tensor(producer=self, shape=shape, dtype=dtype) for shape, dtype in outputs
        )
        _record_operator(self)
        for tensor in inputs:
            tensor.consumers.append(self)
