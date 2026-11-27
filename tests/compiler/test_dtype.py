"""Canonical codecs checked against scalar packing and quantization formulae."""

import math
import struct
from typing import get_args

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from trillim.compiler.dtype import _decode, _encode, _import_mxfp4
from trillim.compiler.weights.packing import _repack_row_int8, _validate_packing
from trillim.tdl.core.types import DType

NATIVE = {
    "bool": "bool",
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "int64": "<i8",
    "uint64": "<u8",
    "fp16": "<f2",
    "fp32": "<f4",
}
PACKED = tuple(
    name for name in get_args(DType) if name not in NATIVE and name != "bf16"
)
GROUP = {
    "grp_int8": 32,
    "grp_int4": 32,
    "grp_int5": 32,
    "dubgrp_int6": 256,
    "grp_binary": 128,
    "grp_ternary": 128,
    "mxfp4": 32,
    "fp8_e4m3": 128,
}


def bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 32767 + ((bits >> 16) & 1)) >> 16
    return float(struct.unpack("<f", struct.pack("<I", rounded << 16))[0])


def reference_row(values: list[float], dtype: str) -> list[float]:
    """Quantize with scalar equations, including physical zero padding."""
    if dtype == "row_int8":
        maximum = max(max(map(abs, values)), 1e-5)
        scale = bf16(maximum / 127)
        multiplier = float(np.float32(127 / maximum))
        return [
            max(-127, min(127, round(float(np.float32(x * multiplier))))) * scale
            for x in values
        ]
    if dtype == "mat_ternary":
        maximum = max(map(abs, values))
        return [round(x / maximum) * maximum if maximum else 0 for x in values]
    size = GROUP[dtype]
    result: list[float] = []
    for start in range(0, len(values), size):
        block = values[start : start + size]
        valid = len(block)
        block += [0.0] * (size - len(block))
        maximum = max(map(abs, block))
        if dtype == "fp8_e4m3":
            scale = float(np.float32(maximum / 448)) if maximum else 1.0
            levels = [
                math.ldexp(code & 7, -9)
                if code < 8
                else math.ldexp(1 + (code & 7) / 8, (code >> 3) - 7)
                for code in range(127)
            ]
            for value in block:
                magnitude = float(np.float32(abs(value) / scale))
                index = min(
                    range(127), key=lambda i: (abs(magnitude - levels[i]), i % 2)
                )
                result.append(math.copysign(levels[index] * scale, value))
        elif dtype == "mxfp4":
            exponent = (
                max(-127, min(127, math.floor(math.log2(maximum)) - 2))
                if maximum
                else 0
            )
            levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            for value in block:
                index = min(
                    range(8),
                    key=lambda i: (abs(abs(value) / 2**exponent - levels[i]), i % 2),
                )
                result.append(math.copysign(levels[index] * 2**exponent, value))
        elif dtype == "grp_binary":
            scale = bf16(sum(map(abs, block)) / valid)
            result.extend(scale if x >= 0 else -scale for x in block)
        elif dtype == "grp_ternary":
            scale = bf16(maximum)
            result.extend(
                scale
                if x >= maximum / 2 and maximum
                else -scale
                if x <= -maximum / 2 and maximum
                else 0
                for x in block
            )
        elif dtype == "dubgrp_int6":
            ideals = [max(map(abs, block[i : i + 16])) / 31 for i in range(0, size, 16)]
            master = bf16(float(np.float32(max(ideals) / 127)))
            anchor = ideals.index(max(ideals))
            for i, ideal in enumerate(ideals):
                chunk = block[i * 16 : (i + 1) * 16]
                initial = (
                    min(127, round(float(np.float32(ideal)) / master)) if master else 0
                )
                effective = float(np.float32(master * initial))
                codes = [
                    max(-32, min(31, round(float(np.float32(x / effective)))))
                    if effective
                    else 0
                    for x in chunk
                ]
                decoded = [float(np.float32(q * initial * master)) for q in codes]
                products = [
                    round(float(np.float32(x / master))) if master else 0 for x in chunk
                ]
                lattice = all(
                    abs(float(np.float32(product * master)) - x) <= 1e-5
                    for product, x in zip(products, chunk, strict=True)
                )
                if (
                    i != anchor
                    and lattice
                    and any(
                        abs(x - y) > 1e-5 for x, y in zip(chunk, decoded, strict=True)
                    )
                ):
                    common = math.gcd(*products)
                    legal = [
                        sub
                        for sub in range(1, 128)
                        if common % sub == 0
                        and min(products) >= -32 * sub
                        and max(products) <= 31 * sub
                    ]
                    if legal:
                        selected = min(legal, key=lambda sub: (abs(sub - initial), sub))
                        decoded = [
                            float(np.float32((product // selected) * selected * master))
                            for product in products
                        ]
                result.extend(decoded)
        else:
            bound = {"grp_int8": 127, "grp_int4": 8, "grp_int5": 16}[dtype]
            if dtype == "grp_int8":
                ideal = float(np.float32(maximum / bound))
            else:
                ideal = bf16(max(max(block) / (bound - 1), -min(block) / bound))
            upper = 127 if dtype == "grp_int8" else bound - 1
            result.extend(
                max(-bound, min(upper, round(float(np.float32(x / ideal)))))
                * bf16(ideal)
                if ideal
                else 0
                for x in block
            )
    return result[: len(values)]


@pytest.mark.parametrize("dtype", PACKED)
@settings(max_examples=35, deadline=None)
@given(values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=290))
@example(values=[240, 480])
def test_quantization_matches_math(dtype: str, values: list[int]) -> None:
    source = np.array(values, dtype=np.float32) / 8
    actual = _decode(_encode(source, dtype), source.shape).astype(np.float32)
    expected = reference_row([float(x) for x in source], dtype)
    np.testing.assert_allclose(
        actual, np.array(expected, dtype=np.float32), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", tuple(NATIVE))
@given(values=st.lists(st.integers(0, 100), min_size=1, max_size=30))
def test_native_preserves_values_and_little_endian(
    dtype: str, values: list[int]
) -> None:
    source = np.array(values, dtype=NATIVE[dtype])
    payload, name = _encode(source, dtype)
    assert name == dtype
    assert payload.tobytes() == source.tobytes()
    decoded = _decode((payload, name), source.shape)
    assert decoded.dtype == source.dtype
    np.testing.assert_array_equal(decoded, source)


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("int64", [-(2**63), -(2**53 + 1), 2**63 - 1]),
        ("uint64", [0, 2**53 + 1, 2**64 - 1]),
    ],
)
def test_wide_integers_are_exact(dtype: str, values: list[int]) -> None:
    source = np.array(values, dtype=dtype)
    np.testing.assert_array_equal(_decode(_encode(source, dtype), source.shape), source)


@given(bits=st.integers(0, 0x7F7F))
def test_bf16_representable_round_trip(bits: int) -> None:
    source = np.array([bits << 16], dtype=np.uint32).view(np.float32)
    np.testing.assert_array_equal(_decode(_encode(source, "bf16"), (1,)), source)


@pytest.mark.parametrize(
    "bits, expected",
    [(0x3F808000, 0x3F800000), (0x3F818000, 0x3F820000), (0xBF808000, 0xBF800000)],
)
def test_bf16_ties_even(bits: int, expected: int) -> None:
    source = np.array([bits], dtype=np.uint32).view(np.float32)
    decoded = _decode(_encode(source, "bf16"), (1,)).astype(np.float32)
    assert int(decoded.view(np.uint32)[0]) == expected


@pytest.mark.parametrize("dtype", PACKED)
@pytest.mark.parametrize("shape", [(), (3,), (2, 33)])
def test_zero_and_padded_rows(dtype: str, shape: tuple[int, ...]) -> None:
    source = np.zeros(shape, dtype=np.float32)
    np.testing.assert_array_equal(_decode(_encode(source, dtype), shape), source)


@pytest.mark.parametrize("dtype", tuple(get_args(DType)))
def test_transposed_group_order(dtype: str) -> None:
    source = np.arange(4 * 3 * 17, dtype=np.float32).reshape(4, 3, 17) % 11
    payload, _ = _encode(source, dtype, transpose=True, groups=2)
    rows = np.array(
        [
            [source[g * 2 + i, o, k] for i in range(2) for k in range(17)]
            for g in range(2)
            for o in range(3)
        ]
    )
    expected, _ = _encode(rows, dtype)
    np.testing.assert_array_equal(payload, expected)
    logical = _decode((payload, dtype), source.shape, transpose=True, groups=2)
    canonical = (
        _decode((expected, dtype), rows.shape)
        .reshape(2, 3, 2, 17)
        .transpose(0, 2, 1, 3)
        .reshape(source.shape)
    )
    np.testing.assert_array_equal(logical, canonical)


@given(rows=st.integers(1, 3), columns=st.integers(1, 3), seed=st.integers(0, 1000))
def test_repacking_preserves_codes_scales_and_values(
    rows: int, columns: int, seed: int
) -> None:
    shape = (32 * rows, 32 * columns)
    source = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    payload, dtype = _encode(source, "row_int8")
    blocked = _repack_row_int8(payload, shape, "canonical", "row_int8_block32")
    for row in range(shape[0]):
        for column in range(shape[1]):
            offset = (
                ((row // 32) * columns + column // 32) * 32 + row % 32
            ) * 32 + column % 32
            assert blocked[offset] == payload[row * shape[1] + column]
    np.testing.assert_array_equal(blocked[source.size :], payload[source.size :])
    restored = _repack_row_int8(blocked, shape, "row_int8_block32", "canonical")
    np.testing.assert_array_equal(
        _decode((restored, dtype), shape), _decode((payload, dtype), shape)
    )


@pytest.mark.parametrize("dtype", tuple(get_args(DType)))
def test_reject_wrong_payload_size(dtype: str) -> None:
    payload, _ = _encode(np.ones((2, 33), dtype=np.float32), dtype)
    with pytest.raises(ValueError, match="payload"):
        _decode((payload[:-1].copy(), dtype), (2, 33))


@pytest.mark.parametrize("dtype", ["mat_ternary", "grp_ternary"])
def test_reject_invalid_ternary_code(dtype: str) -> None:
    payload, _ = _encode(np.zeros(1, dtype=np.float32), dtype)
    payload[0] = 0xD5
    with pytest.raises(ValueError, match="code 3"):
        _decode((payload, dtype), (1,))


def test_mxfp4_import_preserves_blocks_and_nan_scale() -> None:
    blocks = np.arange(32, dtype=np.uint8).reshape(2, 1, 16)
    scales = np.array([[127], [255]], dtype=np.uint8)
    payload, dtype = _import_mxfp4(blocks, scales)
    assert payload.tobytes() == blocks.tobytes() + scales.tobytes()
    decoded = _decode((payload, dtype), (2, 32)).astype(np.float32)
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected = [
        (-1 if code & 8 else 1) * levels[code & 7]
        for byte in blocks[0].flat
        for code in (int(byte) & 15, int(byte) >> 4)
    ]
    np.testing.assert_array_equal(decoded[0], expected)
    assert np.isnan(decoded[1]).all()


@pytest.mark.parametrize("shape", [(1, 32), (32, 31), (0, 32), (32,)])
def test_invalid_block_packing_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _validate_packing("row_int8", shape, "row_int8_block32")


@pytest.mark.parametrize("dtype", PACKED)
@given(seed=st.integers(0, 2**32 - 1))
def test_representable_packed_values_round_trip(dtype: str, seed: int) -> None:
    """Independent wire payloads with scale anchors retained by the quantizer."""
    rng = np.random.default_rng(seed)
    size = GROUP.get(dtype, 128)
    if dtype == "row_int8" or dtype == "grp_int8":
        codes = [int(x) for x in rng.integers(-127, 128, size=size)]
        codes[0] = -127
        wire = bytes(x % 256 for x in codes) + struct.pack("<H", 0x3F80)
        expected = [float(x) for x in codes]
    elif dtype in ("grp_int4", "grp_int5"):
        bound = 8 if dtype == "grp_int4" else 16
        codes = [int(x) for x in rng.integers(0, 2 * bound, size=size)]
        codes[0] = 0
        wire = bytes((codes[i] & 15) | ((codes[i + 16] & 15) << 4) for i in range(16))
        wire += struct.pack("<H", 0x3F80)
        if dtype == "grp_int5":
            wire += bytes(
                sum(((codes[i + bit] >> 4) & 1) << bit for bit in range(8))
                for i in range(0, 32, 8)
            )
        expected = [float(x - bound) for x in codes]
    elif dtype == "dubgrp_int6":
        codes = [int(x) for x in rng.integers(1, 64, size=size)]
        for i in range(0, size, 16):
            codes[i] = 63
        low = bytearray()
        high = bytearray()
        for half in (0, 128):
            low.extend(
                (codes[half + i] & 15) | ((codes[half + i + 32] & 15) << 4)
                for i in range(32)
            )
            low.extend(
                (codes[half + i + 64] & 15) | ((codes[half + i + 96] & 15) << 4)
                for i in range(32)
            )
            high.extend(
                sum(
                    (codes[half + i + field * 32] >> 4) << (field * 2)
                    for field in range(4)
                )
                for i in range(32)
            )
        wire = bytes(low) + struct.pack("<H", 0x3F80) + bytes(high) + bytes([127] * 16)
        expected = [float((x - 32) * 127) for x in codes]
    elif dtype == "grp_binary":
        codes = [int(x) for x in rng.integers(0, 2, size=size)]
        wire = bytes(
            sum(codes[i + bit] << (7 - bit) for bit in range(8))
            for i in range(0, size, 8)
        ) + struct.pack("<H", 0x3F80)
        expected = [float(2 * x - 1) for x in codes]
    elif dtype in ("mat_ternary", "grp_ternary"):
        codes = [int(x) for x in rng.integers(0, 3, size=size)]
        codes[0] = 2
        wire = bytes(
            sum(codes[i + field * 32] << (6 - field * 2) for field in range(4))
            for i in range(32)
        )
        wire += (
            struct.pack("<f", 1.0)
            if dtype == "mat_ternary"
            else struct.pack("<H", 0x3F80)
        )
        expected = [float(x - 1) for x in codes]
    elif dtype == "fp8_e4m3":
        codes = [int(x) for x in rng.integers(0, 254, size=size)]
        codes = [code + 1 if code >= 127 else code for code in codes]
        codes[0] = 126
        wire = bytes(codes) + struct.pack("<f", 1.0)
        expected = [
            math.copysign(
                math.ldexp(code & 7, -9)
                if (code & 127) < 8
                else math.ldexp(1 + (code & 7) / 8, ((code & 127) >> 3) - 7),
                -1 if code & 128 else 1,
            )
            for code in codes
        ]
    else:
        codes = [int(x) for x in rng.integers(0, 16, size=size)]
        codes[0] = 6
        wire = bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, size, 2)) + bytes(
            [127]
        )
        levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        expected = [(-1 if x & 8 else 1) * levels[x & 7] for x in codes]
    payload = np.frombuffer(wire, dtype=np.uint8)
    decoded = _decode((payload, dtype), (size,)).astype(np.float32)
    np.testing.assert_array_equal(decoded, expected)
    widened = _decode(_encode(decoded, "fp32"), (size,))
    np.testing.assert_array_equal(widened, decoded)
    restored = _decode(_encode(decoded, dtype), (size,)).astype(np.float32)
    np.testing.assert_allclose(restored, decoded, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@given(
    values=st.lists(
        st.floats(width=16, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_native_floating_values_round_trip(dtype: str, values: list[float]) -> None:
    source = np.array(values, dtype=NATIVE[dtype])
    decoded = _decode(_encode(source, dtype), source.shape)
    np.testing.assert_array_equal(decoded, source)


@pytest.mark.parametrize("dtype", PACKED)
def test_packed_inputs_require_finite_nonempty_values(dtype: str) -> None:
    for source in (
        np.array([], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            _encode(source, dtype)


@pytest.mark.parametrize("dtype", ["grp_int4", "grp_int5", "grp_binary"])
def test_encoder_output_survives_reencoding(dtype: str) -> None:
    source = np.array([8.0], dtype=np.float32)
    decoded = _decode(_encode(source, dtype), source.shape).astype(np.float32)
    restored = _decode(_encode(decoded, dtype), source.shape).astype(np.float32)
    np.testing.assert_allclose(restored, decoded, rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", PACKED)
@given(
    values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=290),
    sign=st.sampled_from([-1, 0, 1]),
)
def test_asymmetric_and_padded_round_trip(
    dtype: str, values: list[int], sign: int
) -> None:
    source = np.array(values, dtype=np.float32) / 8
    if sign:
        source = np.abs(source) * sign
    decoded = _decode(_encode(source, dtype), source.shape).astype(np.float32)
    restored = _decode(_encode(decoded, dtype), source.shape).astype(np.float32)
    np.testing.assert_allclose(restored, decoded, rtol=0, atol=1e-5)


def test_int6_subscale_survives_reencoding() -> None:
    source = np.zeros(256, dtype=np.float32)
    source[:16] = 3937
    source[16:32] = 46.81
    decoded = _decode(_encode(source, "dubgrp_int6"), source.shape).astype(np.float32)
    restored = _decode(_encode(decoded, "dubgrp_int6"), source.shape).astype(np.float32)
    np.testing.assert_allclose(restored, decoded, rtol=0, atol=1e-5)


@settings(max_examples=35, deadline=None)
@given(scales=st.lists(st.integers(0, 10000), min_size=16, max_size=16))
def test_int6_disparate_subgroups_preserve_values_and_quality(
    scales: list[int],
) -> None:
    source = (
        np.array(scales, dtype=np.float32)[:, None]
        * np.linspace(-1, 1, 16, dtype=np.float32)
    ).reshape(256)
    decoded = _decode(_encode(source, "dubgrp_int6"), source.shape).astype(np.float32)
    restored = _decode(_encode(decoded, "dubgrp_int6"), source.shape).astype(np.float32)
    np.testing.assert_allclose(restored, decoded, rtol=0, atol=1e-5)
    master = bf16(float(np.float32(float(np.float32(max(scales) / 31)) / 127)))
    baseline: list[float] = []
    for start in range(0, 256, 16):
        chunk = [float(x) for x in source[start : start + 16]]
        ideal = float(np.float32(max(map(abs, chunk)) / 31))
        subscale = min(127, round(float(np.float32(ideal / master)))) if master else 0
        effective = float(np.float32(master * subscale))
        baseline.extend(
            float(
                np.float32(
                    max(-32, min(31, round(float(np.float32(x / effective)))))
                    * subscale
                    * master
                )
            )
            if effective
            else 0
            for x in chunk
        )
    actual_error = np.sum((decoded.astype(np.float64) - source) ** 2)
    baseline_error = np.sum((np.array(baseline) - source) ** 2)
    assert actual_error <= baseline_error + 1e-5
