from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...dtype import DType
from .weights_dataclasses import SourceDType

Nonnegative = Annotated[int, Field(ge=0, strict=True)]


class TensorHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    dtype: SourceDType
    shape: list[Nonnegative]
    data_offsets: list[Nonnegative] = Field(min_length=2, max_length=2)


class SafetensorsIndex(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    weight_map: dict[str, str] = Field(min_length=1)


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    quant_method: str = ""
    quantization_mode: str = ""
    linear_class: str = ""


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    model_type: str = ""
    architectures: list[str] = Field(default_factory=list)
    tie_word_embeddings: bool = False
    quantization_config: QuantizationConfig | None = None


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint: str
    tensor: str
    file: str
    dtype: SourceDType
    shape: list[int]
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class WeightRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    logical: str = Field(min_length=1)
    dtype: DType
    shape: list[Nonnegative]
    layout: str = "canonical"
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    aliases: list[str] = Field(default_factory=list)
    import_method: str
    provenance: list[SourceProvenance]


class WeightManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["trillim.weights"] = "trillim.weights"
    version: Literal[1] = 1
    endian: Literal["little"] = "little"
    data_start: int = Field(ge=8)
    entries: dict[str, WeightRecord]
