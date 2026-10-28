from __future__ import annotations

import os
import stat
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompileRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tdl_path: Path
    checkpoints: dict[str, Path] = Field(default_factory=dict)

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
