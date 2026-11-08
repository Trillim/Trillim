from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, JsonValue, TypeAdapter

from ...tdl.core.types import DType
from .weights_dataclasses import _Checkpoint, _SourceDType, _WeightSource
from .weights_pydantic_models import _CheckpointConfig, _SafetensorsIndex, _TensorHeader

_SOURCE_DTYPES: dict[_SourceDType, str] = {
    "BOOL": "?",
    "I8": "i1",
    "U8": "u1",
    "I16": "<i2",
    "U16": "<u2",
    "I32": "<i4",
    "U32": "<u4",
    "I64": "<i8",
    "U64": "<u8",
    "F16": "<f2",
    "BF16": "<u2",
    "F32": "<f4",
}
_LOGICAL_DTYPES: dict[_SourceDType, DType] = {
    "BOOL": "bool",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "fp16",
    "BF16": "bf16",
    "F32": "fp32",
}
_OBJECT = TypeAdapter(dict[str, JsonValue], config=ConfigDict(strict=True))
_METADATA = TypeAdapter(dict[str, str], config=ConfigDict(strict=True))
_CHUNK_BYTES = 8 * 1024**2


def _unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _json_object(data: bytes) -> dict[str, JsonValue]:
    parsed = json.loads(data, object_pairs_hook=_unique)
    return _OBJECT.validate_python(parsed)


def _load_checkpoint(name: str, location: Path) -> _Checkpoint:
    location = location.absolute()
    directory = location if location.is_dir() else location.parent
    config_path = directory / "config.json"
    config = (
        _CheckpointConfig.model_validate(_json_object(config_path.read_bytes()))
        if config_path.is_file()
        else _CheckpointConfig()
    )
    index_path = (
        location
        if location.name.endswith(".index.json")
        else directory / "model.safetensors.index.json"
    )
    weight_map: dict[str, str] = {}
    if location.is_file() and location.suffix == ".safetensors":
        paths = [location]
    elif index_path.is_file():
        weight_map = _SafetensorsIndex.model_validate(
            _json_object(index_path.read_bytes())
        ).weight_map
        shard_names = sorted(set(weight_map.values()))
        if any(Path(shard).name != shard for shard in shard_names):
            raise ValueError("safetensors shard names must be local basenames")
        paths = [directory / shard for shard in shard_names]
    elif (directory / "model.safetensors").is_file():
        paths = [directory / "model.safetensors"]
    else:
        paths = sorted(directory.glob("*.safetensors"))
        if len(paths) != 1:
            raise ValueError(
                f"checkpoint needs one safetensors file or an explicit shard index: {location}"
            )
    tensors: dict[str, _WeightSource] = {}
    metadata: dict[str, str] = {}
    for path in paths:
        state = path.stat()
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > min(100_000_000, state.st_size - 8):
                raise ValueError(f"invalid safetensors header length: {path}")
            header = _json_object(source.read(header_size))
        start = 8 + header_size
        ranges = []
        for tensor_name, value in header.items():
            if tensor_name == "__metadata__":
                for key, item in _METADATA.validate_python(value).items():
                    if key in metadata and metadata[key] != item:
                        raise ValueError(f"conflicting checkpoint metadata: {key}")
                    metadata[key] = item
                continue
            info = _TensorHeader.model_validate(value)
            begin, end = info.data_offsets
            expected = (
                math.prod(info.shape) * np.dtype(_SOURCE_DTYPES[info.dtype]).itemsize
            )
            if begin > end or end - begin != expected or start + end > state.st_size:
                raise ValueError(f"invalid safetensors extent: {tensor_name}")
            if tensor_name in tensors:
                raise ValueError(f"duplicate checkpoint tensor: {tensor_name}")
            if weight_map and weight_map.get(tensor_name) != path.name:
                raise ValueError(f"shard index disagrees with tensor: {tensor_name}")
            ranges.append((begin, end))
            tensors[tensor_name] = _WeightSource(
                name,
                tensor_name,
                path.resolve(),
                info.dtype,
                tuple(info.shape),
                start + begin,
                expected,
                state.st_size,
                state.st_mtime_ns,
            )
        position = 0
        for begin, end in sorted(ranges):
            if begin != position:
                raise ValueError(
                    f"noncontiguous or overlapping safetensors extents: {path}"
                )
            position = end
        if start + position != state.st_size:
            raise ValueError(f"unreferenced safetensors bytes: {path}")
    if not tensors:
        raise ValueError(f"checkpoint contains no tensors: {location}")
    if weight_map and set(tensors) != set(weight_map):
        raise ValueError("safetensors shard index contains missing tensors")
    aliases = {
        key: value
        for key, value in metadata.items()
        if value in tensors and key != value
    }
    if config.tie_word_embeddings:
        candidates = {
            "lm_head.weight": (
                "model.embed_tokens.weight",
                "transformer.wte.weight",
                "model.decoder.embed_tokens.weight",
            ),
            "proj_out.weight": (
                "model.decoder.embed_tokens.weight",
                "decoder.embed_tokens.weight",
            ),
        }
        for head, embeddings in candidates.items():
            for embedding in embeddings:
                if embedding in tensors:
                    aliases[head] = embedding
                    break
    return _Checkpoint(name, config, tensors, aliases)


def _unchanged(source: _WeightSource) -> None:
    state = source.path.stat()
    if state.st_size != source.file_size or state.st_mtime_ns != source.modified_ns:
        raise ValueError(f"checkpoint changed while compiling: {source.path}")


def _read_values(source: _WeightSource, start: int, count: int) -> NDArray[np.generic]:
    length = math.prod(source.shape)
    if start < 0 or count < 0 or start + count > length:
        raise ValueError(f"checkpoint slice exceeds tensor: {source.tensor}")
    _unchanged(source)
    dtype = np.dtype(_SOURCE_DTYPES[source.dtype])
    with source.path.open("rb") as file:
        file.seek(source.offset + start * dtype.itemsize)
        data = file.read(count * dtype.itemsize)
    if len(data) != count * dtype.itemsize:
        raise ValueError(f"checkpoint tensor truncated: {source.tensor}")
    values = np.frombuffer(data, dtype=dtype)
    if source.dtype == "BF16":
        return (values.astype(np.uint32) << 16).view(np.float32)
    return cast(NDArray[np.generic], values)
