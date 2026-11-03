"""NumPy weight codecs.

``encode`` converts a tensor to one contiguous byte array and its dtype.
``decode`` retains native NumPy dtypes, including exact INT64/UINT64 values;
quantized formats decode to FP32. Native conversions use NumPy casts; BF16 uses
round-to-nearest-even. Multi-byte values are little-endian.

Packed weights use one row per output channel. The first axis is the output
axis, except for transposed convolution, whose second axis is the output axis
within each group. The codec reorders transposed-convolution weights within
each group. A vector is one row, and a scalar is a one-value row. Packing
pads each row independently. The caller retains the original logical shape.

Payload layout
--------------
Each payload stores all value bytes first, then all BF16 scales. INT5 and
INT6 append high-bit bytes; INT6 then appends signed INT8 subscales.
``mat_ternary`` stores one FP32 gamma after its value bytes instead of BF16
scales. MXFP4 uses one E8M0 byte per scale. Each section follows output-row order.

Packed formats
--------------
``row_int8``: One BF16 scale per row. Let S=max(absmax(row),1e-5), d=S/127.
Store q=clamp(round(x*127/S),-127,127) as signed bytes. Decode with d*q.

``mat_ternary``: One FP32 gamma=absmax(tensor). Quantize each value to
code=clamp(round(x/gamma),-1,1)+1; a zero gamma gives code 1. Within each
row, pad to 128-value segments. Byte j holds codes j,j+32,j+64,j+96 in
bits 7:6,5:4,3:2,1:0. Code 3 is invalid. Decode with gamma*(code-1).

``grp_int8``: Groups of 32 signed bytes and one BF16 scale d=absmax/127.
The quantizer emits [-127,127]. A zero group has d=0 and zero bytes.

``grp_int4``: Groups of 32 signed values in [-8,7], d=absmax/8, code=q+8.
Byte j holds code j in its low nibble and code j+16 in its high nibble.
Two encoded zeros form 0x88.

``grp_int5``: Groups of 32 signed values in [-16,15], d=absmax/16,
code=q+16. Low nibbles pair like INT4. Four extra bytes hold the high bit
of each code, LSB-first within each byte.

``dubgrp_int6``: Groups of 256 values have one BF16 master scale d and
16 signed INT8 subscales, one per consecutive 16 values. Each ideal
subscale is absmax/31; d=max(ideal_subscales)/127. Store q+32 as a 6-bit
code and decode with d*subscale*q. Each 128-value half pairs low nibbles
at offsets 0/32 and 64/96. The high two bits of each of those four values
occupy consecutive two-bit fields in one byte.

``grp_binary``: Groups of 128 values, d=mean(abs(values)) stored as BF16.
Pack signs MSB-first, with bit 1 for +d and bit 0 for -d.

``grp_ternary``: Groups of 128 values, d=absmax stored as BF16. Divide
by d and use code 2 for values >=0.5, code 0 for values <=-0.5, and
code 1 otherwise. A zero group uses code 1. Pack codes like mat_ternary.

``mxfp4``: Groups of 32 E2M1 values share one E8M0 scale. Adjacent
values occupy the low and high nibbles of each byte. The scale is 2**(byte-127);
255 denotes NaN. Encoding uses floor(log2(absmax))-2, clamped to [-127,127],
and round-to-nearest-even E2M1 with saturation. Zero groups use scale byte 127.
``import_mxfp4`` preserves checkpoint blocks and scales without requantization.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import prod
from typing import Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray

DType = Literal[
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "fp16",
    "bf16",
    "fp32",
    "row_int8",
    "mat_ternary",
    "grp_int8",
    "grp_int4",
    "grp_int5",
    "dubgrp_int6",
    "grp_binary",
    "grp_ternary",
    "mxfp4",
]


def validate_dtype(value: str) -> DType:
    if value not in get_args(DType):
        raise ValueError(f'unsupported dtype "{value}"')
    return cast(DType, value)


Encoded = tuple[NDArray[np.uint8], DType]
Encoder = Callable[[NDArray[np.generic]], Encoded]
Decoder = Callable[[NDArray[np.uint8], int, int], NDArray[np.generic]]

_NATIVE: dict[DType, np.dtype[np.generic]] = {
    "bool": np.dtype("bool"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "int16": np.dtype("int16"),
    "uint16": np.dtype("uint16"),
    "int32": np.dtype("int32"),
    "uint32": np.dtype("uint32"),
    "int64": np.dtype("int64"),
    "uint64": np.dtype("uint64"),
    "fp16": np.dtype("float16"),
    "fp32": np.dtype("float32"),
}

_GROUP_SIZE: dict[DType, int] = {
    "grp_int8": 32,
    "grp_int4": 32,
    "grp_int5": 32,
    "dubgrp_int6": 256,
    "grp_binary": 128,
    "grp_ternary": 128,
    "mxfp4": 32,
}

_MXFP4_MAGNITUDES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_MXFP4_VALUES = np.concatenate((_MXFP4_MAGNITUDES, -_MXFP4_MAGNITUDES))
_MXFP4_MIDPOINTS = (_MXFP4_MAGNITUDES[:-1] + _MXFP4_MAGNITUDES[1:]) / 2


def _bf16_bits(values: NDArray[np.float32]) -> NDArray[np.uint16]:
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    encoded = np.where(np.isnan(values), (bits >> 16) | 0x40, rounded >> 16)
    return encoded.astype(np.uint16)


def _bf16_values(bits: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _grouped(matrix: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    rows, columns = matrix.shape
    count = (columns + size - 1) // size
    padded = np.zeros((rows, count * size), dtype=np.float32)
    padded[:, :columns] = matrix
    return padded.reshape(rows, count, size)


def _quantize(
    values: NDArray[np.float32], scale: NDArray[np.float32], bound: int
) -> NDArray[np.int32]:
    denominator = np.where(scale == 0, 1, scale)
    return np.clip(np.rint(values / denominator), -bound, bound - 1).astype(np.int32)


def _pack_ternary(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    packed = (
        (codes[..., :32] << 6)
        | (codes[..., 32:64] << 4)
        | (codes[..., 64:96] << 2)
        | codes[..., 96:128]
    )
    return packed.astype(np.uint8)


def _unpack_ternary(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    codes = np.empty((*packed.shape[:-1], 128), dtype=np.uint8)
    codes[..., :32] = packed >> 6
    codes[..., 32:64] = (packed >> 4) & 3
    codes[..., 64:96] = (packed >> 2) & 3
    codes[..., 96:128] = packed & 3
    if np.any(codes == 3):
        raise ValueError("ternary code 3 is invalid")
    return codes


def _pack(dtype: DType, *parts: NDArray[np.generic]) -> tuple[NDArray[np.uint8], DType]:
    sections = (
        np.ascontiguousarray(part.astype(part.dtype.newbyteorder("<"), copy=False))
        .view(np.uint8)
        .reshape(-1)
        for part in parts
    )
    return np.concatenate(tuple(sections)), dtype


def import_mxfp4(
    blocks: NDArray[np.uint8], scales: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], DType]:
    """Pack checkpoint blocks [...,G,16] and scales [...,G] without decoding.

    The caller retains logical shape (*blocks.shape[:-2], G*32). GPT-oss
    logical stems gate_up_proj/down_proj map to _blocks and _scales keys.
    """
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8:
        raise TypeError("MXFP4 blocks and scales must be uint8 arrays")
    if blocks.ndim < 2 or blocks.shape[-1] != 16 or blocks.size == 0:
        raise ValueError("MXFP4 blocks must have nonempty shape [...,groups,16]")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError("MXFP4 scales must match the block dimensions")
    return _pack("mxfp4", blocks, scales)


def import_mat_ternary(values: NDArray[np.int8], scale: np.float32) -> Encoded:
    """Preserve already-quantized ternary values and an explicit global scale."""
    if values.ndim != 2 or values.size == 0 or np.any((values < -1) | (values > 1)):
        raise ValueError("ternary import requires a nonempty matrix of -1, 0, 1")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("ternary scale must be finite and nonnegative")
    blocks = _grouped(values.astype(np.float32), 128)
    return _pack(
        "mat_ternary",
        _pack_ternary((blocks + 1).astype(np.uint8)),
        np.array([scale], dtype=np.float32),
    )


def _split(
    payload: NDArray[np.uint8], sizes: tuple[int, ...]
) -> tuple[NDArray[np.uint8], ...]:
    if payload.ndim != 1 or payload.dtype != np.uint8 or not payload.flags.c_contiguous:
        raise TypeError("decode expects one contiguous uint8 array")
    if payload.size != sum(sizes):
        raise ValueError(f"payload has {payload.size} bytes; expected {sum(sizes)}")
    offsets = np.cumsum(sizes[:-1])
    return tuple(np.split(payload, offsets))


def _view(
    section: NDArray[np.uint8], dtype: np.dtype[np.generic], shape: tuple[int, ...]
) -> NDArray[np.generic]:
    return np.frombuffer(section, dtype=dtype.newbyteorder("<")).reshape(shape)


def _matrix_shape(
    shape: tuple[int, ...], transpose: bool, groups: int
) -> tuple[int, int]:
    if any(dimension < 0 for dimension in shape):
        raise ValueError("codec requires nonnegative tensor dimensions")
    if groups <= 0:
        raise ValueError("groups must be positive")

    if transpose:  # Currently only used for convolution
        if len(shape) < 3:
            raise ValueError(
                "transposed convolution weights require at least three axes"
            )
        if shape[0] % groups:
            raise ValueError("input channels must be divisible by groups")
        return groups * shape[1], (shape[0] // groups) * prod(shape[2:])

    if len(shape) <= 1:
        return 1, prod(shape)
    return shape[0], prod(shape[1:])


def _to_rows(
    source: NDArray[np.generic], transpose: bool, groups: int
) -> NDArray[np.generic]:
    rows, columns = _matrix_shape(source.shape, transpose, groups)
    if not transpose:
        return np.ascontiguousarray(source).reshape(rows, columns)

    input_per_group = source.shape[0] // groups
    output_per_group = source.shape[1]
    grouped = source.reshape(
        groups, input_per_group, output_per_group, *source.shape[2:]
    )
    return np.ascontiguousarray(grouped.swapaxes(1, 2)).reshape(rows, columns)


def _from_rows(
    matrix: NDArray[np.generic], shape: tuple[int, ...], transpose: bool, groups: int
) -> NDArray[np.generic]:
    if not transpose:
        return matrix.reshape(shape)

    grouped = matrix.reshape(groups, shape[1], shape[0] // groups, *shape[2:])
    return np.ascontiguousarray(grouped.swapaxes(1, 2)).reshape(shape)


def encode_native(target: DType, tensor: NDArray[np.generic]) -> Encoded:
    return _pack(target, tensor.astype(_NATIVE[target]))


def encode_bf16(tensor: NDArray[np.generic]) -> Encoded:
    return _pack("bf16", _bf16_bits(tensor.astype(np.float32, copy=False)))


def encode_row_int8(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    maximum = np.maximum(np.max(np.abs(source), axis=1), np.float32(1e-05))
    values = np.clip(np.rint(source * (127 / maximum[:, None])), -127, 127)
    scales = _bf16_bits((maximum / 127).astype(np.float32))
    return _pack("row_int8", values.astype(np.int8), scales)


def encode_mat_ternary(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    rows = source.shape[0]
    packing_chunks = _grouped(source, 128)
    count = packing_chunks.shape[1]
    gamma = np.max(np.abs(source))
    normalized = packing_chunks / gamma if gamma != 0 else packing_chunks
    codes = (np.clip(np.rint(normalized), -1, 1) + 1).astype(np.uint8)
    packed = _pack_ternary(codes).reshape(rows, count * 32)
    return _pack("mat_ternary", packed, np.array([gamma], dtype=np.float32))


def encode_mxfp4(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    blocks = _grouped(source, 32)
    maximum = np.max(np.abs(blocks), axis=-1)
    _, exponents = np.frexp(maximum)
    exponents = np.clip(np.where(maximum == 0, 0, exponents - 3), -127, 127)
    magnitude = np.abs(np.ldexp(blocks, -exponents[..., None]))
    codes = np.searchsorted(_MXFP4_MIDPOINTS, magnitude).astype(np.uint8)
    tied = magnitude == _MXFP4_MIDPOINTS[np.minimum(codes, 6)]
    codes += tied & (codes & 1 != 0)
    codes = (codes | np.signbit(blocks).astype(np.uint8) << 3).astype(np.uint8)
    packed = (codes[..., 0::2] | codes[..., 1::2] << 4).astype(np.uint8)
    return _pack("mxfp4", packed, (exponents + 127).astype(np.uint8))


def encode_grp_binary(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    blocks = _grouped(source, 128)
    scale = np.mean(np.abs(blocks), axis=2, dtype=np.float32)
    packed = np.packbits(blocks >= 0, axis=2, bitorder="big")
    return _pack("grp_binary", packed, _bf16_bits(scale))


def encode_grp_ternary(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    blocks = _grouped(source, 128)
    maximum = np.max(np.abs(blocks), axis=2)
    normalized = blocks / np.where(maximum == 0, 1, maximum)[..., None]
    codes = np.where(normalized >= 0.5, 2, np.where(normalized <= -0.5, 0, 1))
    packed = _pack_ternary(codes.astype(np.uint8))
    return _pack("grp_ternary", packed, _bf16_bits(maximum.astype(np.float32)))


def encode_dubgrp_int6(tensor: NDArray[np.generic]) -> Encoded:
    source = tensor.astype(np.float32, copy=False)
    rows = source.shape[0]
    blocks = _grouped(source, 256)
    count = blocks.shape[1]
    chunks = blocks.reshape(rows, count, 16, 16)
    ideal = np.max(np.abs(chunks), axis=3) / 31
    master = (np.max(ideal, axis=2) / 127).astype(np.float32)
    subscales = np.rint(ideal / np.where(master == 0, 1, master)[..., None])
    subscales = np.clip(subscales, 0, 127).astype(np.int8)
    effective = master[..., None] * subscales.astype(np.float32)
    quantized = _quantize(chunks, effective[..., None], 32)
    codes = (quantized + 32).astype(np.uint8).reshape(rows, count, 2, 128)
    g0, g1, g2, g3 = (
        codes[..., :32],
        codes[..., 32:64],
        codes[..., 64:96],
        codes[..., 96:128],
    )
    low = np.concatenate((g0 & 15 | (g1 & 15) << 4, g2 & 15 | (g3 & 15) << 4), axis=-1)
    high = (g0 >> 4 | g1 >> 4 << 2 | g2 >> 4 << 4 | g3 >> 4 << 6).astype(np.uint8)
    return _pack("dubgrp_int6", low, _bf16_bits(master), high, subscales)


def _integer_groups(
    tensor: NDArray[np.generic], bound: int, divisor: int
) -> tuple[NDArray[np.int32], NDArray[np.uint16]]:
    blocks = _grouped(tensor.astype(np.float32, copy=False), 32)
    ideal = (np.max(np.abs(blocks), axis=2) / np.float32(divisor)).astype(np.float32)
    return _quantize(blocks, ideal[..., None], bound), _bf16_bits(
        ideal.astype(np.float32)
    )


def encode_grp_int8(tensor: NDArray[np.generic]) -> Encoded:
    quantized, scales = _integer_groups(tensor, 128, 127)
    return _pack("grp_int8", np.clip(quantized, -127, 127).astype(np.int8), scales)


def encode_grp_int4(tensor: NDArray[np.generic]) -> Encoded:
    quantized, scales = _integer_groups(tensor, 8, 8)
    codes = (quantized + 8).astype(np.uint8)
    return _pack("grp_int4", codes[..., :16] | (codes[..., 16:] << 4), scales)


def encode_grp_int5(tensor: NDArray[np.generic]) -> Encoded:
    quantized, scales = _integer_groups(tensor, 16, 16)
    codes = (quantized + 16).astype(np.uint8)
    values = (codes[..., :16] & 15) | ((codes[..., 16:] & 15) << 4)
    high = np.packbits(codes >> 4, axis=-1, bitorder="little")
    return _pack("grp_int5", values, scales, high)


def payload_sections(
    dtype: str, shape: tuple[int, ...], *, transpose: bool = False, groups: int = 1
) -> tuple[tuple[str, int], ...]:
    target = validate_dtype(dtype)
    rows, columns = _matrix_shape(shape, transpose, groups)
    if target in _NATIVE:
        return (("values", rows * columns * _NATIVE[target].itemsize),)
    if target == "bf16":
        return (("values", rows * columns * 2),)
    if rows == 0 or columns == 0:
        raise ValueError("packed dtypes require nonempty tensor dimensions")
    if target == "row_int8":
        return (("values", rows * columns), ("scales", rows * 2))
    if target == "mat_ternary":
        return (("values", rows * ((columns + 127) // 128) * 32), ("scale", 4))
    size = _GROUP_SIZE[target]
    count = rows * ((columns + size - 1) // size)
    widths = {
        "mxfp4": 16,
        "grp_binary": 16,
        "grp_ternary": 32,
        "grp_int8": 32,
        "grp_int4": 16,
        "grp_int5": 16,
        "dubgrp_int6": 128,
    }
    result = (
        ("values", count * widths[target]),
        ("scales", count * (1 if target == "mxfp4" else 2)),
    )
    if target == "grp_int5":
        return (*result, ("high", count * 4))
    if target == "dubgrp_int6":
        return (*result, ("high", count * 64), ("subscales", count * 16))
    return result


def decode_native(
    target: DType, payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    native = _NATIVE[target]
    (values,) = _split(payload, (rows * columns * native.itemsize,))
    return _view(values, native, (rows, columns))


def decode_bf16(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    matrix_shape = (rows, columns)
    (values,) = _split(payload, (rows * columns * 2,))
    bits = _view(values, np.dtype("uint16"), matrix_shape).astype(np.uint16)
    return _bf16_values(bits)


def decode_row_int8(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    matrix_shape = (rows, columns)
    values, scales = _split(payload, (rows * columns, rows * 2))
    signed = _view(values, np.dtype("int8"), matrix_shape).astype(np.float32)
    bits = _view(scales, np.dtype("uint16"), (rows,)).astype(np.uint16)
    return signed * _bf16_values(bits)[:, None]


def decode_mat_ternary(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    count = (columns + 127) // 128
    values, gamma_bytes = _split(payload, (rows * count * 32, 4))
    gamma = _view(gamma_bytes, np.dtype("float32"), (1,)).astype(np.float32)
    codes = _unpack_ternary(values.reshape(rows, count, 32))
    return ((codes.astype(np.float32) - 1) * gamma).reshape(rows, -1)[:, :columns]


def decode_mxfp4(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    count = (columns + 31) // 32
    values, scales = _split(payload, (rows * count * 16, rows * count))
    packed = values.reshape(rows, count, 16)
    codes = np.empty((rows, count, 32), dtype=np.uint8)
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    scales = scales.reshape(rows, count)
    exponents = np.where(scales == 255, 0, scales.astype(np.int32) - 127)
    decoded = np.ldexp(_MXFP4_VALUES[codes], exponents[..., None])
    decoded = np.where((scales == 255)[..., None], np.float32(np.nan), decoded)
    return decoded.reshape(rows, -1)[:, :columns].astype(np.float32)


def _group_parts(
    payload: NDArray[np.uint8], dtype: DType, rows: int, columns: int
) -> tuple[tuple[NDArray[np.uint8], ...], NDArray[np.float32], int]:
    count = (columns + _GROUP_SIZE[dtype] - 1) // _GROUP_SIZE[dtype]
    parts = _split(
        payload, tuple(length for _, length in payload_sections(dtype, (rows, columns)))
    )
    bits = _view(parts[1], np.dtype("uint16"), (rows, count)).astype(np.uint16)
    return parts, _bf16_values(bits), count


def decode_grp_binary(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "grp_binary", rows, columns)
    values = parts[0]
    signs = (
        np.unpackbits(values.reshape(rows, count, 16), axis=-1, bitorder="big").astype(
            np.int8
        )
        * 2
        - 1
    )
    return (signs * scale[..., None]).reshape(rows, -1)[:, :columns]


def decode_grp_ternary(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "grp_ternary", rows, columns)
    values = parts[0]
    codes = _unpack_ternary(values.reshape(rows, count, 32))
    result = (codes.astype(np.float32) - 1) * scale[..., None]
    return result.reshape(rows, -1)[:, :columns]


def decode_dubgrp_int6(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "dubgrp_int6", rows, columns)
    values = parts[0]
    low_int6 = values.reshape(rows, count, 2, 64)
    high_int6 = parts[2].reshape(rows, count, 2, 32)
    subscales = _view(parts[3], np.dtype("int8"), (rows, count, 16)).astype(np.int8)
    codes_int6 = np.empty((rows, count, 2, 128), dtype=np.uint8)
    codes_int6[..., :32] = low_int6[..., :32] & 15 | (high_int6 & 3) << 4
    codes_int6[..., 32:64] = low_int6[..., :32] >> 4 | (high_int6 >> 2 & 3) << 4
    codes_int6[..., 64:96] = low_int6[..., 32:] & 15 | (high_int6 >> 4 & 3) << 4
    codes_int6[..., 96:128] = low_int6[..., 32:] >> 4 | (high_int6 >> 6 & 3) << 4
    signed_int6 = codes_int6.reshape(rows, count, 16, 16).astype(np.float32) - 32
    result = signed_int6 * subscales[..., None] * scale[..., None, None]
    return result.reshape(rows, -1)[:, :columns]


def decode_grp_int8(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "grp_int8", rows, columns)
    values = parts[0]
    signed_int8 = _view(values, np.dtype("int8"), (rows, count, 32))
    return (signed_int8.astype(np.float32) * scale[..., None]).reshape(rows, -1)[
        :, :columns
    ]


def decode_grp_int4(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "grp_int4", rows, columns)
    packed = parts[0].reshape(rows, count, 16)
    codes = np.concatenate((packed & 15, packed >> 4), axis=-1).astype(np.float32)
    return ((codes - 8) * scale[..., None]).reshape(rows, -1)[:, :columns]


def decode_grp_int5(
    payload: NDArray[np.uint8], rows: int, columns: int
) -> NDArray[np.generic]:
    parts, scale, count = _group_parts(payload, "grp_int5", rows, columns)
    packed = parts[0].reshape(rows, count, 16)
    codes = np.concatenate((packed & 15, packed >> 4), axis=-1)
    codes |= (
        np.unpackbits(parts[2].reshape(rows, count, 4), axis=-1, bitorder="little") << 4
    )
    return ((codes.astype(np.float32) - 16) * scale[..., None]).reshape(rows, -1)[
        :, :columns
    ]


ENCODERS: dict[DType, Encoder] = {
    name: partial(encode_native, name) for name in _NATIVE
}
ENCODERS.update(
    {
        "bf16": encode_bf16,
        "row_int8": encode_row_int8,
        "mat_ternary": encode_mat_ternary,
        "grp_int8": encode_grp_int8,
        "grp_int4": encode_grp_int4,
        "grp_int5": encode_grp_int5,
        "dubgrp_int6": encode_dubgrp_int6,
        "grp_binary": encode_grp_binary,
        "grp_ternary": encode_grp_ternary,
        "mxfp4": encode_mxfp4,
    }
)
DECODERS: dict[DType, Decoder] = {
    name: partial(decode_native, name) for name in _NATIVE
}
DECODERS.update(
    {
        "bf16": decode_bf16,
        "row_int8": decode_row_int8,
        "mat_ternary": decode_mat_ternary,
        "grp_int8": decode_grp_int8,
        "grp_int4": decode_grp_int4,
        "grp_int5": decode_grp_int5,
        "dubgrp_int6": decode_dubgrp_int6,
        "grp_binary": decode_grp_binary,
        "grp_ternary": decode_grp_ternary,
        "mxfp4": decode_mxfp4,
    }
)


def encode(
    tensor: NDArray[np.generic], dtype: str, *, transpose: bool = False, groups: int = 1
) -> Encoded:
    """Encode canonical little-endian bytes, retaining native integer precision."""
    target = validate_dtype(dtype)
    source = _to_rows(np.asarray(tensor), transpose, groups)
    if source.dtype.kind not in "biuf":
        raise TypeError("encode requires real numeric or boolean values")
    if target not in _NATIVE and target != "bf16":
        if source.size == 0:
            raise ValueError("packed dtypes require nonempty tensor dimensions")
        with np.errstate(over="ignore"):
            source = source.astype(np.float32, copy=False)
        if not np.isfinite(source).all():
            raise ValueError("packed dtypes require finite FP32 values")
    return ENCODERS[target](source)


def decode(
    encoded: tuple[NDArray[np.uint8], str],
    shape: tuple[int, ...],
    *,
    transpose: bool = False,
    groups: int = 1,
) -> NDArray[np.generic]:
    """Decode native values exactly; packed formats and BF16 return FP32."""
    payload, name = encoded
    target = validate_dtype(name)
    rows, columns = _matrix_shape(shape, transpose, groups)
    return _from_rows(
        DECODERS[target](payload, rows, columns), shape, transpose, groups
    )
