from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from trillim.compiler.weights.bundle import _read_manifest, _write_weights
from trillim.compiler.weights.reader import _load_checkpoint, _read_bytes
from trillim.compiler.weights.resolve import _entry, _resolve_payloads
from trillim.compiler.weights.weights_dataclasses import _WeightIndex, _WeightRequest
from trillim.compiler.weights.weights_pydantic_models import _CheckpointConfig

from .helpers import FIXTURES, payload


@pytest.mark.parametrize("model", ("bitnet", "gpt-oss"))
def test_original_packed_checkpoint_bytes(model: str, tmp_path: Path) -> None:
    filename = (
        "BitNet-packed.safetensors"
        if model == "bitnet"
        else "GPT-oss-MXFP4.safetensors"
    )
    checkpoint = _load_checkpoint("checkpoint", FIXTURES / filename)
    expected_hashes = json.loads((FIXTURES / "expected_hashes.json").read_text())[model]
    for name, source in checkpoint.tensors.items():
        data = _read_bytes(source, 0, source.size)
        assert hashlib.sha256(data).hexdigest() == expected_hashes[name]
    if model == "bitnet":
        name = "model.layers.0.self_attn.q_proj.weight"
        config = _CheckpointConfig.model_validate(
            {
                "model_type": "bitnet",
                "quantization_config": {
                    "quant_method": "bitnet",
                    "linear_class": "autobitlinear",
                    "quantization_mode": "offline",
                },
            }
        )
        raw = _read_bytes(checkpoint.tensors[name], 0, 64)
        codes = np.frombuffer(raw, np.uint8).reshape(2, 32)
        decoded = np.concatenate(
            [((codes >> shift) & 3).astype(np.float32) - 1 for shift in (0, 2, 4, 6)]
        )
        scale = _read_bytes(
            checkpoint.tensors[name.removesuffix("weight") + "weight_scale"], 0, 2
        )
        gamma = struct.unpack("<f", b"\0\0" + scale)[0]
        expected = decoded * gamma
        target = "mat_ternary"
    else:
        name = "model.layers.0.mlp.experts.down_proj.weight"
        config = _CheckpointConfig()
        stem = name.removesuffix(".weight")
        blocks = _read_bytes(checkpoint.tensors[stem + "_blocks"], 0, 32)
        scales = _read_bytes(checkpoint.tensors[stem + "_scales"], 0, 2)
        code = np.frombuffer(blocks, np.uint8).reshape(2, 16)
        indices = np.stack((code & 15, code >> 4), axis=-1).reshape(2, 32)
        palette = np.array(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6), np.float32
        )
        exponents = np.frombuffer(scales, np.uint8).astype(np.int32) - 127
        expected = (palette[indices] * np.exp2(exponents[:, None])).astype(np.float32)
        target = "mxfp4"
    entry = _entry(checkpoint, name, config)
    actual = np.frombuffer(payload(entry), "<f4").reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    index = _WeightIndex(
        sources={entry.identity: entry}, checkpoints={"checkpoint": checkpoint}
    )
    requests = [
        _WeightRequest(entry.identity, entry.dtype, name=entry.identity + "@native"),
        _WeightRequest(entry.identity, "fp32", name=entry.identity + "@fp32"),
    ]
    destination = tmp_path / "weights.bin"
    manifest = _write_weights(_resolve_payloads(index, requests), destination)
    assert _read_manifest(destination) == manifest
    converted = next(r for r in manifest.entries.values() if r.dtype == "fp32")
    with destination.open("rb") as file:
        file.seek(manifest.data_start + converted.offset)
        np.testing.assert_array_equal(
            np.frombuffer(file.read(converted.size), "<f4"), expected.reshape(-1)
        )
    native = next(r for r in manifest.entries.values() if r.dtype == target)
    assert native.import_method == (
        "bitnet_packed:autobitlinear" if model == "bitnet" else "mxfp4"
    )
