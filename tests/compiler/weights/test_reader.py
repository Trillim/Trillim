from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from trillim.compiler.weights.reader import _json_object, _load_checkpoint, _read_bytes

from .helpers import FIXTURES, checkpoint, raw, source


def test_shard_index_and_metadata(tmp_path: Path) -> None:
    entry = source()
    checkpoint(
        tmp_path / "first.safetensors",
        {"first": ("BF16", entry.shape, raw(entry))},
        {"first_alias": "first"},
    )
    checkpoint(
        tmp_path / "second.safetensors",
        {"second": ("BF16", entry.shape, raw(entry))},
        {"format": "pt"},
    )
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": entry.source.size * 2},
                "weight_map": {
                    "first": "first.safetensors",
                    "second": "second.safetensors",
                },
            }
        )
    )
    loaded = _load_checkpoint("Model", index)
    assert set(loaded.tensors) == {"first", "second"}
    assert loaded.metadata == {"first_alias": "first", "format": "pt"}
    index.write_text(
        index.read_text().replace("second.safetensors", "first.safetensors")
    )
    with pytest.raises(ValueError, match="missing tensors"):
        _load_checkpoint("Model", index)


@pytest.mark.parametrize(
    "damage", ["short", "length", "extent", "overlap", "trailing", "duplicate", "empty"]
)
def test_reject_corrupted_real_checkpoint(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "model.safetensors"
    original = (FIXTURES / "Qwen3-0.6B.safetensors").read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8 : 8 + length])
    data = original[8 + length :]
    if damage == "short":
        path.write_bytes(original[:4])
    elif damage == "length":
        path.write_bytes(struct.pack("<Q", len(original)) + original[8:])
    elif damage == "trailing":
        path.write_bytes(original + data[:1])
    elif damage == "duplicate":
        key, value = next(iter(header.items()))
        encoded = (
            "{"
            + json.dumps(key)
            + ":"
            + json.dumps(value)
            + ","
            + json.dumps(key)
            + ":"
            + json.dumps(value)
            + "}"
        ).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    else:
        if damage == "empty":
            header = {}
            data = b""
        elif damage == "extent":
            next(iter(header.values()))["data_offsets"][1] -= 1
        else:
            entries = list(header.values())
            entries[1]["data_offsets"] = [0, 64]
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    with pytest.raises(ValueError):
        _load_checkpoint("Model", path)


def test_json_duplicate_and_source_mutation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _json_object(b'{"tensor":{},"tensor":{}}')
    path = tmp_path / "model.safetensors"
    shutil.copyfile(FIXTURES / "Qwen3-0.6B.safetensors", path)
    tensor = next(iter(_load_checkpoint("Model", path).tensors.values()))
    with pytest.raises(ValueError, match="slice exceeds"):
        _read_bytes(tensor, tensor.size, 1)
    with path.open("ab") as file:
        file.write(raw(source())[:1])
    with pytest.raises(ValueError, match="changed"):
        _read_bytes(tensor, 0, 1)
