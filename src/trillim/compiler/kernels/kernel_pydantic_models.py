from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .kernel_dataclasses import _Backend


class _KernelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sources: tuple[Path, ...] = ()
    backend: _Backend | None = None

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        sources = tuple(path.expanduser().resolve() for path in value)
        for path in sources:
            if path.suffix != ".c" or not path.is_file():
                raise ValueError(f"kernel source must be an existing .c file: {path}")
        if len(set(sources)) != len(sources):
            raise ValueError("kernel source paths must be unique")
        return sources


class _KernelIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1] = 1
    symbols: tuple[str, ...] = Field(min_length=1)

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(
            not symbol.isascii() or not symbol.isidentifier() for symbol in value
        ):
            raise ValueError("kernel symbols must be unique C identifiers")
        return value
