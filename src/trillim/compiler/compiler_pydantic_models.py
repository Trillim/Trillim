from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CompileRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tdl_path: Path
    checkpoints: dict[str, Path] = Field(default_factory=dict)
