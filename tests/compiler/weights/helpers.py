from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from trillim.compiler.weights.reader import _load_checkpoint, _read_bytes
from trillim.compiler.weights.resolve import _entry
from trillim.compiler.weights.weights_dataclasses import _WeightEntry
from trillim.compiler.weights.weights_pydantic_models import _CheckpointConfig

FIXTURES = Path(__file__).parents[1] / "fixtures" / "weights"
PROJECTION = "model.layers.0.self_attn.q_proj.weight"


def source(model: str = "Qwen3-0.6B") -> _WeightEntry:
    checkpoint = _load_checkpoint("Model", FIXTURES / f"{model}.safetensors")
    name = PROJECTION if model == "Qwen3-0.6B" else next(iter(checkpoint.tensors))
    config = _CheckpointConfig(model_type="bitnet" if model == "BitNet" else "")
    return _entry(checkpoint, name, config)


def checkpoint(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
) -> Path:
    header: dict[str, JsonValue] = {}
    payload = bytearray()
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def raw(entry: _WeightEntry) -> bytes:
    return _read_bytes(entry.source, 0, entry.source.size)


def payload(entry: _WeightEntry, dtype: str = "fp32") -> bytes:
    from typing import cast

    from trillim.compiler.weights.resolve import _encoded_chunks
    from trillim.tdl.core.types import DType

    chunks = list(_encoded_chunks(entry, cast(DType, dtype), {}))
    result = bytearray(max((offset + len(data) for offset, data in chunks), default=0))
    for offset, data in chunks:
        result[offset : offset + len(data)] = data
    return bytes(result)


def golden_values(model: str = "Qwen3-0.6B") -> NDArray[np.float32]:
    expected = json.loads((FIXTURES / "expected.json").read_text())
    return np.asarray(expected[model], dtype=np.float32)


def bitnet_reference() -> tuple[NDArray[np.float32], np.float32]:
    values = golden_values("BitNet")
    mean = np.float32(sum(abs(float(value)) for value in values) / len(values))
    inverse = np.float32(1) / mean
    bits = struct.unpack("<I", struct.pack("<f", np.float32(1) / inverse))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    gamma = np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])
    return np.clip(np.rint(values * inverse), -1, 1) * gamma, gamma
