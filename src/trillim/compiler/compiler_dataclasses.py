from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CompileArguments(Namespace):
    tdl_path: Path = field(init=False)
    checkpoints: list[tuple[str, Path]] = field(default_factory=list)
