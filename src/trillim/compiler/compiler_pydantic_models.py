from __future__ import annotations

import os
import stat
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bench.bench_dataclasses import BenchmarkLevel
from .kernels.build import host_backend
from .kernels.kernel_dataclasses import Backend


class CompileRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tdl_path: Path
    checkpoints: dict[str, Path] = Field(default_factory=dict)
    checkpoint_path: Path | None = None
    output: Path | None = None
    backend: Backend = Field(default_factory=host_backend)
    max_input_size: int = Field(default=4096, gt=0)
    depth: BenchmarkLevel = "none"
    threads: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> CompileRequest:
        paths = tuple(self.checkpoints.items())
        if self.checkpoint_path is not None:
            paths += (("positional", self.checkpoint_path),)
        for name, path in paths:
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
        return self
