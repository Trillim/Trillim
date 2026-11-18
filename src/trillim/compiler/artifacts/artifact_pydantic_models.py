from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ...processing.processing_pydantic_models import _ProcessingSpec
from ...tdl.core.types import DType
from ..kernels.kernel_dataclasses import _Backend


class _BufferRecord(BaseModel):
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


class _WorkloadPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum_tokens: int = Field(ge=1)
    maximum_tokens: int = Field(ge=1)
    arena_size: int = Field(ge=0)
    buffers: dict[str, _BufferRecord]
    program: list[dict[str, JsonValue]]
    expressions: dict[str, JsonValue] = Field(default_factory=dict)


class _ExecutableManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["trillim.executable"] = "trillim.executable"
    version: Literal[1] = 1
    abi_version: Literal[1] = 1
    endian: Literal["little"] = "little"
    backend: _Backend
    threads: int = Field(ge=1)
    arena_size: int = Field(ge=0)
    cache_size: int = Field(ge=0)
    weights: str = "weights.bin"
    constants: str = "constants.bin"
    processing: _ProcessingSpec | None = None
    inputs: dict[str, str]
    outputs: dict[str, str]
    buffers: dict[str, _BufferRecord]
    caches: dict[str, dict[str, JsonValue]]
    kernels: dict[str, dict[str, JsonValue]]
    program: list[dict[str, JsonValue]]
    workloads: list[dict[str, JsonValue]] = Field(default_factory=list)
    transfer_kernel: str | None = None
    workload: dict[str, JsonValue] | None = None
    plans: list[_WorkloadPlan] = Field(default_factory=list)
    expressions: dict[str, JsonValue] = Field(default_factory=dict)
