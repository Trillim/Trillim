from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...tdl.core.types import DType
from .weights_dataclasses import _SourceDType

_Nonnegative = Annotated[int, Field(ge=0, strict=True)]


class _TensorHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    dtype: _SourceDType
    shape: list[_Nonnegative]
    data_offsets: list[_Nonnegative] = Field(min_length=2, max_length=2)


class _SafetensorsIndex(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    weight_map: dict[str, str] = Field(min_length=1)


class _QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    quant_method: str = ""
    quantization_mode: str = ""
    linear_class: str = ""
    bits: _Nonnegative = 0
    group_size: _Nonnegative = 0
    mode: str = ""
    weight_block_size: list[_Nonnegative] | None = Field(
        default=None, min_length=2, max_length=2
    )


class _CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    model_type: str = ""
    architectures: list[str] = Field(default_factory=list)
    tie_word_embeddings: bool = False
    quantization_config: _QuantizationConfig | None = None
    quantization: _QuantizationConfig | None = None


class _SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint: str
    tensor: str
    file: str
    dtype: _SourceDType
    shape: list[int]
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class _WeightRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    logical: str = Field(min_length=1)
    dtype: DType
    shape: list[_Nonnegative]
    layout: str = "canonical"
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    aliases: list[str] = Field(default_factory=list)
    import_method: str
    provenance: list[_SourceProvenance]


class _WeightManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["trillim.weights"] = "trillim.weights"
    version: Literal[1] = 1
    endian: Literal["little"] = "little"
    data_start: int = Field(ge=8)
    entries: dict[str, _WeightRecord]
