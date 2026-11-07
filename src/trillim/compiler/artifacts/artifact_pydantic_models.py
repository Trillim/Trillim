from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ...tdl.core.types import DType
from ..kernels.kernel_dataclasses import Backend


class BufferRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dtype: DType
    shape: list[JsonValue]
    maximum_shape: list[int]
    strides: list[JsonValue]
    storage: Literal["arena", "input", "weight", "constant", "alias", "cache"]
    source: str | None = None
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    alignment: int = Field(gt=0)
    first: int = Field(ge=0)
    last: int = Field(ge=0)
    byte_offset: JsonValue = 0


class WorkloadPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_tokens: int = Field(ge=1)
    maximum_tokens: int = Field(ge=1)
    arena_size: int = Field(ge=0)
    buffers: dict[str, BufferRecord]
    program: list[dict[str, JsonValue]]
    expressions: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutableManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["trillim.executable"] = "trillim.executable"
    version: Literal[1] = 1
    abi_version: Literal[1] = 1
    endian: Literal["little"] = "little"
    backend: Backend
    threads: int = Field(ge=1)
    arena_size: int = Field(ge=0)
    cache_size: int = Field(ge=0)
    weights: str = "weights.bin"
    constants: str = "constants.bin"
    inputs: dict[str, str]
    outputs: dict[str, str]
    buffers: dict[str, BufferRecord]
    caches: dict[str, dict[str, JsonValue]]
    kernels: dict[str, dict[str, JsonValue]]
    program: list[dict[str, JsonValue]]
    workloads: list[dict[str, JsonValue]] = Field(default_factory=list)
    transfer_kernel: str | None = None
    workload: dict[str, JsonValue] | None = None
    plans: list[WorkloadPlan] = Field(default_factory=list)
    expressions: dict[str, JsonValue] = Field(default_factory=dict)
