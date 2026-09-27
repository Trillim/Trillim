from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .operator import Operator


class Tensor:
    def __init__(self, producer: Operator | None = None) -> None:
        self.producer = producer
        self.consumers: list[Operator] = []
