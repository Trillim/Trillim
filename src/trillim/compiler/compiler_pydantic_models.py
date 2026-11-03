from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..tdl.core.types import DType


class CompileRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tdl_path: Path
    checkpoints: dict[str, Path] = Field(default_factory=dict)
    output: Path = Path("build/model")
    backend: Literal["scalar", "avx2"] = "scalar"
    bounds: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    value_bounds: dict[str, tuple[int, int]] = Field(default_factory=dict)
    dtypes: dict[str, DType] = Field(default_factory=dict)
    depth: Literal["none", "low", "medium", "high"] = "none"
    threads: int = Field(default=1, ge=1)
    memory_budget: int = Field(default=8 * 1024**3, gt=0)
    beam_width: int = Field(default=4, ge=1)
    search_seconds: float = Field(default=10.0, gt=0)

    @field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, value: dict[str, tuple[int, ...]]
    ) -> dict[str, tuple[int, ...]]:
        if any(dimension < 0 for shape in value.values() for dimension in shape):
            raise ValueError("input shape bounds cannot be negative")
        return value

    @field_validator("value_bounds")
    @classmethod
    def validate_value_bounds(
        cls, value: dict[str, tuple[int, int]]
    ) -> dict[str, tuple[int, int]]:
        if any(minimum > maximum for minimum, maximum in value.values()):
            raise ValueError("scalar bounds require minimum <= maximum")
        return value

    @field_validator("checkpoints")
    @classmethod
    def validate_checkpoints(cls, value: dict[str, Path]) -> dict[str, Path]:
        for name, path in value.items():
            try:
                mode = path.stat().st_mode
            except OSError as error:
                raise ValueError(f"checkpoint {name!r}: {error}") from error
            if stat.S_ISDIR(mode):
                permissions = os.R_OK | os.X_OK
            elif stat.S_ISREG(mode):
                permissions = os.R_OK
            else:
                raise ValueError(
                    f"checkpoint {name!r} must be a file or directory: {path}"
                )
            if not os.access(path, permissions):
                raise ValueError(f"checkpoint {name!r} is not accessible: {path}")
        return value
