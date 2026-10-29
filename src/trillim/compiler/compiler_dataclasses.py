from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tdl.core.core_dataclasses import CacheSpec
    from ..tdl.core.graph import Graph
    from ..tdl.core.operator import Operator
    from ..tdl.core.tensor import Tensor


@dataclass(slots=True)
class CompileArguments(Namespace):
    tdl_path: Path = field(init=False)
    checkpoints: list[tuple[str, Path]] = field(default_factory=list)


@dataclass(slots=True)
class ValidationState:
    graph: Graph
    regions: set[int] = field(default_factory=set)
    operators: set[Operator] = field(default_factory=set)
    consumers: dict[Tensor, set[Operator]] = field(default_factory=dict)
    declarations: dict[CacheSpec, Operator] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationScope:
    tensors: set[Tensor] = field(default_factory=set)
    operators: set[Operator] = field(default_factory=set)
    writers: dict[CacheSpec, Operator] = field(default_factory=dict)
    readers: dict[CacheSpec, list[Operator]] = field(default_factory=dict)
