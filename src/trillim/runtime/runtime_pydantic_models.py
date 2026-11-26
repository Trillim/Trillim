from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..tdl.core.types import DType


class _RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    directory: Path
    threads: int | None = Field(default=None, ge=1, le=65536)


class _Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    dtype: DType
    shape: list[int]
    nbytes: int


class _Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    ok: bool
    ready: bool = False
    error: str = ""
    cancelled: bool = False
    event: Literal["token", "result"] | None = None
    output: str | None = None
    index: int | None = None
    token: int | None = None
    outputs: list[_Output] | None = None
    stats: dict[str, JsonValue] = Field(default_factory=dict)
