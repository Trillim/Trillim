from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    directory: Path
    threads: int | None = Field(default=None, ge=1, le=65536)
