from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _entry
from trillim.compiler.weights.weights_pydantic_models import _CheckpointConfig

from .helpers import (
    FIXTURES,
    PROJECTION,
    bitnet_reference,
    checkpoint,
    golden_values,
    payload,
    raw,
    source,
)


@pytest.mark.parametrize(
    "model", ["Qwen3-0.6B", "Qwen3.5", "PocketTTS", "Phi-MoE", "Ternary-Bonsai-1.7B"]
)
def test_native_copy_and_conversion_against_source_values(model: str) -> None:
    entry = source(model)
    assert payload(entry, entry.dtype) == raw(entry)
    expected = golden_values(model).astype("<f4").tobytes()
    assert payload(entry) == expected
    assert (
        payload(entry, "fp16") == np.frombuffer(expected, "<f4").astype("<f2").tobytes()
    )


def test_bitnet_absmean_preserves_pretrained_quantizer() -> None:
    entry = source("BitNet")
    values = golden_values("BitNet")
    mean = np.float32(sum(abs(float(value)) for value in values) / len(values))
    inverse = np.float32(1) / mean
    gamma_bits = struct.unpack("<I", struct.pack("<f", np.float32(1) / inverse))[0]
    gamma_bits = (gamma_bits + 0x7FFF + ((gamma_bits >> 16) & 1)) & 0xFFFF0000
    gamma = struct.unpack("<f", struct.pack("<I", gamma_bits))[0]
    codes = np.clip(np.rint(values * inverse), -1, 1).astype(np.int8)
    assert (
        payload(entry)
        == (np.clip(np.rint(values * inverse), -1, 1) * gamma).astype("<f4").tobytes()
    )
    expected_codes = bytes((int(code) + 1) << 6 | 0x15 for code in codes)
    assert payload(entry, "mat_ternary") == expected_codes + struct.pack("<f", gamma)


@pytest.mark.parametrize("linear_class", ["bitlinear", "autobitlinear"])
def test_packed_bitnet_from_real_latent_codes(
    tmp_path: Path, linear_class: str
) -> None:
    latent = source("BitNet")
    effective, gamma = bitnet_reference()
    effective = effective.reshape(32, 32)
    codes = (np.sign(effective).astype(np.int16) + 1).astype(np.uint8)
    packed = codes[:8] | codes[8:16] << 2 | codes[16:24] << 4 | codes[24:] << 6
    assert payload(latent) == effective.astype("<f4").tobytes()
    scale = (
        np.float32(1) / np.float32(gamma)
        if linear_class == "bitlinear"
        else np.float32(gamma)
    )
    path = checkpoint(
        tmp_path / "packed.safetensors",
        {
            PROJECTION: ("U8", (8, 32), packed.tobytes()),
            PROJECTION.removesuffix("weight") + "weight_scale": (
                "F32",
                (),
                struct.pack("<f", scale),
            ),
        },
    )
    cp = _load_checkpoint("Model", path)
    config = _CheckpointConfig.model_validate(
        {
            "model_type": "bitnet",
            "quantization_config": {
                "quantization_mode": "offline",
                "linear_class": linear_class,
            },
        }
    )
    entry = _entry(cp, PROJECTION, config)
    reconstructed_gamma = (
        np.float32(1) / scale if linear_class == "bitlinear" else scale
    )
    assert (
        payload(entry)
        == (np.sign(effective) * reconstructed_gamma).astype("<f4").tobytes()
    )
    damaged = bytearray(packed.tobytes())
    damaged[0] |= 3
    checkpoint(
        path,
        {
            PROJECTION: ("U8", (8, 32), bytes(damaged)),
            PROJECTION.removesuffix("weight") + "weight_scale": (
                "F32",
                (),
                struct.pack("<f", scale),
            ),
        },
    )
    with pytest.raises(ValueError, match="code 3"):
        payload(_entry(_load_checkpoint("Model", path), PROJECTION, config))


def test_aliases_and_declared_ties(tmp_path: Path) -> None:
    original = source()
    path = checkpoint(
        tmp_path / "tied.safetensors",
        {
            "embedding": ("BF16", original.shape, raw(original)),
            "head": ("BF16", original.shape, raw(original)),
        },
    )
    cp = _load_checkpoint("Model", path)
    cp.aliases = {"head": "alias", "alias": "embedding"}
    entry = _entry(cp, "head", _CheckpointConfig())
    assert _entry(cp, "embedding", _CheckpointConfig()) == entry
    assert entry.identity == "Model.embedding"
    assert len(entry.tied_sources) == 1
    assert entry.aliases == ("Model.head", "Model.alias")
    cp.aliases = {"head": "alias", "alias": "head"}
    with pytest.raises(ValueError, match="cyclic"):
        _entry(cp, "head", _CheckpointConfig())
    with pytest.raises(ValueError, match="has no tensor"):
        _entry(cp, "absent", _CheckpointConfig())


def test_mxfp4_import_with_source_derived_official_codes(tmp_path: Path) -> None:
    values = golden_values().reshape(32, 32)
    magnitudes = np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    exponents = np.ceil(np.log2(np.max(np.abs(values), axis=1) / 6)).astype(np.int32)
    scales = np.exp2(exponents.astype(np.float32))
    indices = np.argmin(
        np.abs(np.abs(values)[:, :, None] / scales[:, None, None] - magnitudes), axis=2
    ).astype(np.uint8)
    codes = indices | (np.signbit(values).astype(np.uint8) << 3)
    packed = codes[:, ::2] | codes[:, 1::2] << 4
    encoded_scales = (exponents + 127).astype(np.uint8)
    path = checkpoint(
        tmp_path / "mxfp4.safetensors",
        {
            "projection_blocks": ("U8", (32, 1, 16), packed.tobytes()),
            "projection_scales": ("U8", (32, 1), encoded_scales.tobytes()),
        },
    )
    entry = _entry(
        _load_checkpoint("Model", path), "projection.weight", _CheckpointConfig()
    )
    assert entry.shape == (32, 32)
    assert payload(entry, "mxfp4") == packed.tobytes() + encoded_scales.tobytes()
    expected = (
        magnitudes[indices] * np.where(np.signbit(values), -1, 1) * scales[:, None]
    )
    np.testing.assert_array_equal(
        np.frombuffer(payload(entry), "<f4").reshape(32, 32), expected
    )


def test_fp8_import_preserves_partial_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trillim.compiler.weights import resolve

    selection = np.indices((129, 259)).sum(axis=0) % 3
    codes = np.array([56, 184, 64], dtype=np.uint8)[selection]
    scales = np.array([[0.5, 2, 4], [8, 16, 32]], dtype="<f4")
    path = checkpoint(
        tmp_path / "fp8.safetensors",
        {
            "projection.weight": ("F8_E4M3", codes.shape, codes.tobytes()),
            "projection.weight_scale_inv": (
                "BF16",
                scales.shape,
                (scales.view("<u4") >> 16).astype("<u2").tobytes(),
            ),
        },
    )
    config = _CheckpointConfig.model_validate(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        }
    )
    entry = _entry(_load_checkpoint("Model", path), "projection.weight", config)
    monkeypatch.setattr(resolve, "_CHUNK_BYTES", 1000)
    assert payload(entry, "fp8_e4m3") == codes.tobytes() + scales.tobytes()
    expected = (
        np.array([1, -1, 2], dtype="<f4")[selection]
        * scales.repeat(128, axis=0).repeat(128, axis=1)[:129, :259]
    )
    assert payload(entry) == expected.tobytes()


@pytest.mark.parametrize(
    ("dtype", "target", "size"),
    [
        ("BOOL", "bool", 1),
        ("I8", "int8", 1),
        ("U8", "uint8", 1),
        ("I16", "int16", 2),
        ("U16", "uint16", 2),
        ("I32", "int32", 4),
        ("U32", "uint32", 4),
        ("I64", "int64", 8),
        ("U64", "uint64", 8),
    ],
)
def test_native_integer_bit_preservation(
    tmp_path: Path, dtype: str, target: str, size: int
) -> None:
    values = golden_values()
    if dtype == "BOOL":
        converted = values > 0
    else:
        integers = np.rint(np.abs(values) * (2**56 if size == 8 else 128))
        numpy_dtype = np.dtype(("i" if dtype.startswith("I") else "u") + str(size))
        converted = integers.astype(numpy_dtype)
    data = converted.tobytes()
    path = checkpoint(
        tmp_path / "native.safetensors", {"tensor": (dtype, (len(data) // size,), data)}
    )
    entry = _entry(_load_checkpoint("Model", path), "tensor", _CheckpointConfig())
    assert payload(entry, target) == data


def test_bounded_row_int8_and_block_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trillim.compiler.weights import resolve

    entry = source()
    values = golden_values().reshape(32, 32)
    maximum = np.maximum(np.max(np.abs(values), axis=1), np.float32(1e-5))
    codes = np.clip(
        np.rint(values * (np.float32(127) / maximum[:, None])), -127, 127
    ).astype(np.int8)
    bits = (maximum / np.float32(127)).view(np.uint32)
    bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
    expected = codes.tobytes() + bf16
    monkeypatch.setattr(resolve, "_CHUNK_BYTES", 128)
    assert payload(entry, "row_int8") == expected
    wider = checkpoint(
        tmp_path / "wide.safetensors",
        {"projection": ("BF16", (32, 64), raw(entry) + raw(entry))},
    )
    wide_entry = _entry(
        _load_checkpoint("Model", wider), "projection", _CheckpointConfig()
    )
    monkeypatch.setattr(resolve, "_CHUNK_BYTES", 8 * 1024**2)
    chunks = list(
        resolve._encoded_chunks(wide_entry, "row_int8", {}, "row_int8_block32")
    )
    combined = bytearray(32 * 64 + 64)
    for offset, data in chunks:
        combined[offset : offset + len(data)] = data
    wide_values = np.concatenate((values.reshape(-1), values.reshape(-1))).reshape(
        32, 64
    )
    wide_maximum = np.maximum(np.max(np.abs(wide_values), axis=1), np.float32(1e-5))
    wide_codes = np.clip(
        np.rint(wide_values * (np.float32(127) / wide_maximum[:, None])), -127, 127
    ).astype(np.int8)
    expected_codes = b"".join(
        wide_codes[row, column : column + 32].tobytes()
        for column in (0, 32)
        for row in range(32)
    )
    assert combined[:2048] == expected_codes
    for dtype, shape, packing in [
        ("fp32", (32, 64), "row_int8_block32"),
        ("row_int8", (31, 64), "row_int8_block32"),
        ("row_int8", (32, 64), "unknown"),
    ]:
        from typing import cast

        from trillim.compiler.weights.packing import _validate_packing
        from trillim.tdl.core.types import DType

        with pytest.raises(ValueError, match="unsupported weight packing"):
            _validate_packing(cast(DType, dtype), shape, packing)


@pytest.mark.parametrize(
    "damage",
    [
        "rank",
        "integer",
        "missing_offline",
        "nonscalar_scale",
        "zero_scale",
        "nonfinite",
    ],
)
def test_bitnet_rejects_malformed_source_derivatives(
    tmp_path: Path, damage: str
) -> None:
    original = source("BitNet")
    data = raw(original)
    dtype = "BF16"
    shape = original.shape
    config_data: dict[str, object] = {"model_type": "bitnet"}
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    if damage == "rank":
        shape = (1024,)
    elif damage == "integer":
        dtype = "I16"
    elif damage in ("missing_offline", "nonscalar_scale", "zero_scale"):
        dtype = "U8"
        shape = (32, 64)
        if damage != "missing_offline":
            config_data["quantization_config"] = {
                "quantization_mode": "offline",
                "linear_class": "autobitlinear",
            }
            scale_data = (
                data[:8]
                if damage == "nonscalar_scale"
                else (np.frombuffer(data[:4], "<f4") * 0).tobytes()
            )
            tensors[PROJECTION.removesuffix("weight") + "weight_scale"] = (
                "F32",
                (2,) if damage == "nonscalar_scale" else (),
                scale_data,
            )
    elif damage == "nonfinite":
        words = np.frombuffer(data, "<u2").copy()
        words[0] |= 0x7F80
        data = words.tobytes()
    tensors[PROJECTION] = (dtype, shape, data)
    path = checkpoint(tmp_path / "bad.safetensors", tensors)
    config = _CheckpointConfig.model_validate(config_data)
    with np.errstate(divide="ignore", invalid="ignore"), pytest.raises(ValueError):
        payload(_entry(_load_checkpoint("Model", path), PROJECTION, config))


@pytest.mark.parametrize("damage", ["dtype", "rank", "group", "scales", "missing"])
def test_mxfp4_rejects_malformed_components(tmp_path: Path, damage: str) -> None:
    data = raw(source())[:16]
    block_dtype = "I8" if damage == "dtype" else "U8"
    block_shape = (
        (16,) if damage == "rank" else (1, 2, 8) if damage == "group" else (1, 1, 16)
    )
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "projection_blocks": (block_dtype, block_shape, data)
    }
    if damage != "missing":
        tensors["projection_scales"] = (
            "U8",
            (2,) if damage == "scales" else (1, 1),
            raw(source())[: 2 if damage == "scales" else 1],
        )
    path = checkpoint(tmp_path / "bad.safetensors", tensors)
    with pytest.raises(ValueError):
        _entry(
            _load_checkpoint("Model", path), "projection.weight", _CheckpointConfig()
        )


def test_native_matrix_ternary_uses_global_absmax() -> None:
    entry = source()
    values = golden_values().reshape(32, 32)
    gamma = np.max(np.abs(values))
    quantized = np.clip(np.rint(values / gamma), -1, 1).astype(np.int8)
    expected = bytearray()
    for row in quantized:
        for column in range(32):
            expected.append((int(row[column]) + 1) << 6 | 1 << 4 | 1 << 2 | 1)
    expected.extend(struct.pack("<f", gamma))
    assert payload(entry, "mat_ternary") == bytes(expected)


def test_tied_shape_dtype_and_empty_bitnet(tmp_path: Path) -> None:
    from dataclasses import replace

    original = source("BitNet")
    cp = _load_checkpoint("Model", FIXTURES / "BitNet.safetensors")
    cp.aliases = {"head": PROJECTION}
    cp.tensors["head"] = replace(original.source, shape=(1024,))
    with pytest.raises(ValueError, match="shape or dtype"):
        _entry(cp, "head", _CheckpointConfig())
    path = checkpoint(
        tmp_path / "empty.safetensors", {PROJECTION: ("BF16", (0, 32), b"")}
    )
    entry = _entry(
        _load_checkpoint("Model", path),
        PROJECTION,
        _CheckpointConfig(model_type="bitnet"),
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        payload(entry)


@pytest.mark.parametrize(
    "configuration",
    [
        {"architectures": ["BitNetForCausalLM"]},
        {"quantization_config": {"quant_method": "bitnet"}},
    ],
)
def test_bitnet_detection_from_checkpoint_metadata(
    configuration: dict[str, object],
) -> None:
    cp = _load_checkpoint("Model", FIXTURES / "BitNet.safetensors")
    entry = _entry(cp, PROJECTION, _CheckpointConfig.model_validate(configuration))
    assert entry.method == "bitnet_absmean"
    assert payload(entry) == payload(source("BitNet"))


@pytest.mark.parametrize("target", ["mat_ternary", "row_int8"])
def test_nonfinite_native_quantization_fails_before_emitting_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from trillim.compiler.weights import resolve

    data = np.frombuffer(raw(source()), "<u2").copy()
    data[0] |= 0x7F80
    path = checkpoint(
        tmp_path / "nonfinite.safetensors",
        {"projection": ("BF16", (32, 32), data.tobytes())},
    )
    entry = _entry(_load_checkpoint("Model", path), "projection", _CheckpointConfig())
    monkeypatch.setattr(resolve, "_CHUNK_BYTES", 128)
    with pytest.raises(ValueError, match="finite FP32"):
        payload(entry, target)


def test_empty_native_and_zero_derived_ternary(tmp_path: Path) -> None:
    path = checkpoint(
        tmp_path / "empty.safetensors", {"projection": ("BF16", (0, 32), b"")}
    )
    entry = _entry(_load_checkpoint("Model", path), "projection", _CheckpointConfig())
    assert payload(entry) == b""
    assert payload(entry, "bf16") == b""
    zeros = (golden_values().reshape(32, 32) * 0).astype("<f4")
    path = checkpoint(
        tmp_path / "zero.safetensors",
        {"projection": ("F32", (32, 32), zeros.tobytes())},
    )
    entry = _entry(_load_checkpoint("Model", path), "projection", _CheckpointConfig())
    assert payload(entry, "mat_ternary") == bytes([0x55]) * 1024 + struct.pack("<f", 0)


def test_bitnet_fp16_gamma_rounds_to_source_precision(tmp_path: Path) -> None:
    values = golden_values("Ternary-Bonsai-1.7B")
    path = checkpoint(
        tmp_path / "fp16.safetensors",
        {PROJECTION: ("F16", (32, 32), values.astype("<f2").tobytes())},
    )
    entry = _entry(
        _load_checkpoint("Model", path),
        PROJECTION,
        _CheckpointConfig(model_type="bitnet"),
    )
    mean = np.float32(sum(abs(float(value)) for value in values) / len(values))
    inverse = np.float32(1) / mean
    gamma = np.float32(np.float16(np.float32(1) / inverse))
    expected = np.clip(np.rint(values * inverse), -1, 1) * gamma
    assert payload(entry) == expected.astype("<f4").tobytes()
