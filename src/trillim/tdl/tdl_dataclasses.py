from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Shape:
    dimensions: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if sum(dimension is None for dimension in self.dimensions) > 1:
            raise ValueError("a shape can have at most one variable dimension")

        if any(
            dimension is not None and dimension < 0 for dimension in self.dimensions
        ):
            raise ValueError("fixed dimensions cannot be negative")
