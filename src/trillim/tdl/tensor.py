from __future__ import annotations

from typing import TYPE_CHECKING

from .tdl_dataclasses import Shape

if TYPE_CHECKING:
    from .operator import Operator


class Tensor:
    def __init__(
        self,
        producer: Operator | None = None,
        shape: Shape | None = None,
    ) -> None:
        self.producer = producer
        self.shape = shape
        self.consumers: list[Operator] = []
