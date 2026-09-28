"""TDL dtypes and NumPy weight codecs.

``encode`` converts an FP32 tensor to one contiguous byte array and its dtype.
``decode`` returns FP32 values in the original logical shape. Native dtypes
use NumPy casts; BF16 uses round-to-nearest-even. Multi-byte values are
little-endian.

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
scales. Each section follows output-row order.

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
"""

from __future__ import annotations

from math import prod
from typing import Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray

DType = Literal[
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
]

_NATIVE: dict[DType, np.dtype[np.generic]] = {
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
}


def validate_dtype(value: str) -> DType:
    if value not in get_args(DType):
        raise ValueError(f'unsupported dtype "{value}"')
    return cast(DType, value)


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
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("codec requires nonempty tensor dimensions")
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
    source: NDArray[np.float32], transpose: bool, groups: int
) -> NDArray[np.float32]:
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
    matrix: NDArray[np.float32], shape: tuple[int, ...], transpose: bool, groups: int
) -> NDArray[np.float32]:
    if not transpose:
        return matrix.reshape(shape)

    grouped = matrix.reshape(groups, shape[1], shape[0] // groups, *shape[2:])
    return np.ascontiguousarray(grouped.swapaxes(1, 2)).reshape(shape)


def encode(
    tensor: NDArray[np.float32],
    dtype: str,
    *,
    transpose: bool = False,
    groups: int = 1,
) -> tuple[NDArray[np.uint8], DType]:
    """Pack an FP32 tensor into output-channel rows and return bytes with dtype."""
    target = validate_dtype(dtype)
    source = np.asarray(tensor)
    if source.dtype != np.float32:
        raise TypeError("encode expects an FP32 NumPy array")
    source = _to_rows(source, transpose, groups)
    rows = source.shape[0]

    match target:
        case (
            "int8"
            | "uint8"
            | "int16"
            | "uint16"
            | "int32"
            | "uint32"
            | "int64"
            | "uint64"
            | "fp16"
            | "fp32"
        ):
            return _pack(target, source.astype(_NATIVE[target]))
        case "bf16":
            return _pack(target, _bf16_bits(source))

    if not np.all(np.isfinite(source)):
        raise ValueError("packed dtypes require finite FP32 values")

    match target:
        case "row_int8":
            maximum = np.maximum(np.max(np.abs(source), axis=1), np.float32(1e-5))
            values = np.clip(np.rint(source * (127 / maximum[:, None])), -127, 127)
            scales = _bf16_bits((maximum / 127).astype(np.float32))
            return _pack(target, values.astype(np.int8), scales)
        case "mat_ternary":
            packing_chunks = _grouped(source, 128)
            count = packing_chunks.shape[1]
            gamma = np.max(np.abs(source))
            normalized = packing_chunks / gamma if gamma != 0 else packing_chunks
            codes = (np.clip(np.rint(normalized), -1, 1) + 1).astype(np.uint8)
            packed = _pack_ternary(codes).reshape(rows, count * 32)
            return _pack(target, packed, np.array([gamma], dtype=np.float32))

    size = _GROUP_SIZE[target]
    blocks = _grouped(source, size)
    count = blocks.shape[1]

    match target:
        case "grp_binary":
            scale = np.mean(np.abs(blocks), axis=2, dtype=np.float32)
            packed = np.packbits(blocks >= 0, axis=2, bitorder="big")
            return _pack(target, packed, _bf16_bits(scale))
        case "grp_ternary":
            maximum = np.max(np.abs(blocks), axis=2)
            normalized = blocks / np.where(maximum == 0, 1, maximum)[..., None]
            codes = np.where(normalized >= 0.5, 2, np.where(normalized <= -0.5, 0, 1))
            packed = _pack_ternary(codes.astype(np.uint8))
            return _pack(target, packed, _bf16_bits(maximum.astype(np.float32)))
        case "dubgrp_int6":
            chunks = blocks.reshape(rows, count, 16, 16)
            ideal = np.max(np.abs(chunks), axis=3) / 31
            master = (np.max(ideal, axis=2) / 127).astype(np.float32)
            subscales = np.rint(ideal / np.where(master == 0, 1, master)[..., None])
            subscales = np.clip(subscales, 0, 127).astype(np.int8)
            effective = master[..., None] * subscales
            quantized = _quantize(chunks, effective[..., None], 32)
            codes = (quantized + 32).astype(np.uint8).reshape(rows, count, 2, 128)
            g0, g1, g2, g3 = (
                codes[..., :32],
                codes[..., 32:64],
                codes[..., 64:96],
                codes[..., 96:128],
            )
            low = np.concatenate(
                ((g0 & 15) | ((g1 & 15) << 4), (g2 & 15) | ((g3 & 15) << 4)),
                axis=-1,
            )
            high = (
                (g0 >> 4) | ((g1 >> 4) << 2) | ((g2 >> 4) << 4) | ((g3 >> 4) << 6)
            ).astype(np.uint8)
            return _pack(target, low, _bf16_bits(master), high, subscales)
        case "grp_int8" | "grp_int4" | "grp_int5":
            maximum = np.max(np.abs(blocks), axis=2)
            bound = {"grp_int8": 128, "grp_int4": 8, "grp_int5": 16}[target]
            divisor = 127 if target == "grp_int8" else bound
            ideal_group = (maximum / divisor).astype(np.float32)
            quantized = _quantize(blocks, ideal_group[..., None], bound)
            scales = _bf16_bits(ideal_group)

            if target == "grp_int8":
                quantized = np.clip(quantized, -127, 127)
                return _pack(target, quantized.astype(np.int8), scales)

            codes = (quantized + bound).astype(np.uint8)
            low_nibbles = (codes[..., :16] & 15) | ((codes[..., 16:] & 15) << 4)
            if target == "grp_int4":
                return _pack(target, low_nibbles, scales)

            high_bits = np.packbits(codes >> 4, axis=-1, bitorder="little")
            return _pack(target, low_nibbles, scales, high_bits)

    raise ValueError(f'unsupported dtype "{target}"')


def _decode_rows(
    payload: NDArray[np.uint8], dtype: DType, rows: int, columns: int
) -> NDArray[np.float32]:
    matrix_shape = (rows, columns)
    match dtype:
        case (
            "int8"
            | "uint8"
            | "int16"
            | "uint16"
            | "int32"
            | "uint32"
            | "int64"
            | "uint64"
            | "fp16"
            | "fp32"
        ):
            native = _NATIVE[dtype]
            (values,) = _split(payload, (rows * columns * native.itemsize,))
            return _view(values, native, matrix_shape).astype(np.float32)
        case "bf16":
            (values,) = _split(payload, (rows * columns * 2,))
            bits = _view(values, np.dtype("uint16"), matrix_shape).astype(np.uint16)
            return _bf16_values(bits)
        case "row_int8":
            values, scales = _split(payload, (rows * columns, rows * 2))
            signed = _view(values, np.dtype("int8"), matrix_shape).astype(np.float32)
            bits = _view(scales, np.dtype("uint16"), (rows,)).astype(np.uint16)
            return signed * _bf16_values(bits)[:, None]
        case "mat_ternary":
            count = (columns + 127) // 128
            values, gamma_bytes = _split(payload, (rows * count * 32, 4))
            gamma = _view(gamma_bytes, np.dtype("float32"), (1,))[0]
            codes = _unpack_ternary(values.reshape(rows, count, 32))
            return ((codes.astype(np.float32) - 1) * gamma).reshape(rows, -1)[
                :, :columns
            ]

    size = _GROUP_SIZE[dtype]
    count = (columns + size - 1) // size
    scale_size = rows * count * 2
    match dtype:
        case "dubgrp_int6":
            parts = _split(
                payload,
                (rows * count * 128, scale_size, rows * count * 64, rows * count * 16),
            )
        case "grp_int5":
            parts = _split(payload, (rows * count * 16, scale_size, rows * count * 4))
        case "grp_binary" | "grp_ternary" | "grp_int8" | "grp_int4":
            value_width = {
                "grp_binary": 16,
                "grp_ternary": 32,
                "grp_int8": 32,
                "grp_int4": 16,
            }[dtype]
            parts = _split(payload, (rows * count * value_width, scale_size))
        case _:
            raise ValueError(f'unsupported dtype "{dtype}"')

    values, scale_bytes = parts[:2]
    bits = _view(scale_bytes, np.dtype("uint16"), (rows, count)).astype(np.uint16)
    scale = _bf16_values(bits)

    match dtype:
        case "grp_binary":
            signs = (
                np.unpackbits(
                    values.reshape(rows, count, 16), axis=-1, bitorder="big"
                ).astype(np.int8)
                * 2
                - 1
            )
            return (signs * scale[..., None]).reshape(rows, -1)[:, :columns]
        case "grp_ternary":
            codes = _unpack_ternary(values.reshape(rows, count, 32))
            result = (codes.astype(np.float32) - 1) * scale[..., None]
            return result.reshape(rows, -1)[:, :columns]
        case "dubgrp_int6":
            low_int6 = values.reshape(rows, count, 2, 64)
            high_int6 = parts[2].reshape(rows, count, 2, 32)
            subscales = _view(parts[3], np.dtype("int8"), (rows, count, 16)).astype(
                np.int8
            )
            codes_int6 = np.empty((rows, count, 2, 128), dtype=np.uint8)
            codes_int6[..., :32] = (low_int6[..., :32] & 15) | ((high_int6 & 3) << 4)
            codes_int6[..., 32:64] = (low_int6[..., :32] >> 4) | (
                ((high_int6 >> 2) & 3) << 4
            )
            codes_int6[..., 64:96] = (low_int6[..., 32:] & 15) | (
                ((high_int6 >> 4) & 3) << 4
            )
            codes_int6[..., 96:128] = (low_int6[..., 32:] >> 4) | (
                ((high_int6 >> 6) & 3) << 4
            )
            signed_int6 = (
                codes_int6.reshape(rows, count, 16, 16).astype(np.float32) - 32
            )
            result = signed_int6 * subscales[..., None] * scale[..., None, None]
            return result.reshape(rows, -1)[:, :columns]
        case "grp_int8":
            signed_int8 = _view(values, np.dtype("int8"), (rows, count, 32))
            return (signed_int8.astype(np.float32) * scale[..., None]).reshape(
                rows, -1
            )[:, :columns]
        case "grp_int4" | "grp_int5":
            low_nibbles = values.reshape(rows, count, 16)
            codes_nibbles = np.empty((rows, count, 32), dtype=np.uint8)
            codes_nibbles[..., :16] = low_nibbles & 15
            codes_nibbles[..., 16:] = low_nibbles >> 4
            if dtype == "grp_int5":
                high_int5 = parts[2].reshape(rows, count, 4)
                codes_nibbles |= (
                    np.unpackbits(high_int5, axis=-1, bitorder="little") << 4
                )
            signed_nibbles = codes_nibbles.astype(np.float32) - (
                8 if dtype == "grp_int4" else 16
            )
            return (signed_nibbles * scale[..., None]).reshape(rows, -1)[:, :columns]


def decode(
    encoded: tuple[NDArray[np.uint8], str],
    shape: tuple[int, ...],
    *,
    transpose: bool = False,
    groups: int = 1,
) -> NDArray[np.float32]:
    """Decode bytes into FP32 values in the original logical shape."""
    payload, name = encoded
    dtype = validate_dtype(name)
    rows, columns = _matrix_shape(shape, transpose, groups)
    matrix = _decode_rows(payload, dtype, rows, columns)
    return _from_rows(matrix, shape, transpose, groups)
