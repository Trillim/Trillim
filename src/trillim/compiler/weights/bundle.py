from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from ...tdl.core.types import DType
from .. import dtype as codec
from .reader import CHUNK_BYTES, read_values, unchanged
from .weights_dataclasses import WeightEntry, WeightIndex, WeightRequest, WeightSource
from .weights_pydantic_models import SourceProvenance, WeightManifest, WeightRecord


def _matrix(shape: tuple[int, ...]) -> tuple[int, int]:
    return (shape[0], math.prod(shape[1:])) if len(shape) > 1 else (1, math.prod(shape))


def _provenance(source: WeightSource) -> SourceProvenance:
    unchanged(source)
    digest = hashlib.sha256()
    with source.path.open("rb") as file:
        file.seek(source.offset)
        remaining = source.size
        while remaining:
            data = file.read(min(remaining, CHUNK_BYTES))
            if not data:
                raise ValueError(f"truncated checkpoint tensor: {source.tensor}")
            digest.update(data)
            remaining -= len(data)
    unchanged(source)
    return SourceProvenance(
        checkpoint=source.checkpoint,
        tensor=source.tensor,
        file=str(source.path),
        dtype=source.dtype,
        shape=list(source.shape),
        offset=source.offset,
        size=source.size,
        sha256=digest.hexdigest(),
    )


def _bitnet_scale(entry: WeightEntry) -> tuple[np.float32, np.float32]:
    if entry.method == "bitnet_packed":
        scale_source = entry.components[0]
        if math.prod(scale_source.shape) != 1:
            raise ValueError(f"BitNet weight_scale must be scalar: {entry.identity}")
        scale = np.float32(read_values(scale_source, 0, 1)[0])
        gamma = np.float32(1) / scale if entry.linear_class == "bitlinear" else scale
        inverse = np.float32(1) / gamma
    else:
        count = math.prod(entry.shape)
        total = np.float64(0)
        for start in range(0, count, CHUNK_BYTES // 4):
            values = read_values(
                entry.source, start, min(CHUNK_BYTES // 4, count - start)
            ).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite BitNet latent weights: {entry.identity}")
            total += np.abs(values).sum(dtype=np.float64)
        if count == 0:
            raise ValueError("BitNet projections cannot be empty")
        mean = max(np.float32(total / count), np.float32(1e-5))
        inverse = np.float32(1) / mean
        gamma = np.float32(1) / inverse
        if entry.source.dtype == "BF16":
            gamma = np.float32(
                codec.decode(
                    codec.encode(np.array([gamma], dtype=np.float32), "bf16"), (1,)
                )[0]
            )
        elif entry.source.dtype == "F16":
            gamma = np.float32(np.float16(gamma))
    if not np.isfinite(inverse) or inverse <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"BitNet scale must be finite and positive: {entry.identity}")
    return inverse, gamma


def _effective(
    entry: WeightEntry,
    start: int,
    count: int,
    scale: tuple[np.float32, np.float32] | None,
) -> NDArray[np.generic]:
    if entry.method == "native":
        return read_values(entry.source, start, count)
    if entry.method == "bitnet_absmean":
        assert scale is not None
        values = read_values(entry.source, start, count).astype(np.float32)
        return np.clip(np.rint(values * scale[0]), -1, 1).astype(np.float32) * scale[1]
    if entry.method == "bitnet_packed":
        assert scale is not None
        rows, columns = _matrix(entry.shape)
        result = np.empty(count, dtype=np.float32)
        quarter = rows // 4
        offset = 0
        while offset < count:
            row, column = divmod(start + offset, columns)
            length = min(count - offset, columns - column)
            packed = read_values(
                entry.source, (row % quarter) * columns + column, length
            ).astype(np.uint8)
            codes = (packed >> ((row // quarter) * 2)) & 3
            if np.any(codes == 3):
                raise ValueError(f"invalid BitNet packed code 3: {entry.identity}")
            result[offset : offset + length] = (codes.astype(np.float32) - 1) * scale[1]
            offset += length
        return result
    if start % 32 or count % 32:
        raise ValueError("MXFP4 import slices must align to groups of 32")
    blocks, scales = entry.components
    group_start, groups = start // 32, count // 32
    packed = (
        read_values(blocks, group_start * 16, groups * 16)
        .astype(np.uint8)
        .reshape(1, groups, 16)
    )
    exponents = (
        read_values(scales, group_start, groups).astype(np.uint8).reshape(1, groups)
    )
    return codec.decode(codec.import_mxfp4(packed, exponents), (1, count)).reshape(-1)


def _chunks(entry: WeightEntry, target: DType) -> Iterator[tuple[int, int, int, int]]:
    rows, columns = _matrix(entry.shape)
    if rows == 0 or columns == 0:
        return
    limit = CHUNK_BYTES // 8
    alignment = math.lcm(
        codec.GROUP_SIZES.get(target, 1), 32 if entry.method == "mxfp4" else 1
    )
    if columns <= limit:
        batch = max(1, limit // columns)
        for row in range(0, rows, batch):
            yield row, min(batch, rows - row), 0, columns
    else:
        step = max(alignment, limit // alignment * alignment)
        for row in range(rows):
            for column in range(0, columns, step):
                yield row, 1, column, min(step, columns - column)


def _raw_copy(output: BinaryIO, offset: int, source: WeightSource) -> None:
    unchanged(source)
    output.seek(offset)
    with source.path.open("rb") as file:
        file.seek(source.offset)
        remaining = source.size
        while remaining:
            data = file.read(min(remaining, CHUNK_BYTES))
            if not data:
                raise ValueError(f"truncated checkpoint tensor: {source.tensor}")
            output.write(data)
            remaining -= len(data)
    unchanged(source)


def _write_entry(
    output: BinaryIO,
    start: int,
    entry: WeightEntry,
    target: DType,
    scale: tuple[np.float32, np.float32] | None,
) -> None:
    sections = codec.payload_sections(target, entry.shape)
    positions = []
    cursor = start
    for _, size in sections:
        positions.append(cursor)
        cursor += size
    ends = [
        position + size for position, (_, size) in zip(positions, sections, strict=True)
    ]
    if entry.method == "native" and target == entry.dtype:
        _raw_copy(output, start, entry.source)
        return
    if entry.method == "mxfp4" and target == "mxfp4":
        for position, source in zip(positions, entry.components, strict=True):
            _raw_copy(output, position, source)
        return
    _, columns = _matrix(entry.shape)
    gamma = scale[1] if target == "mat_ternary" and scale is not None else np.float32(0)
    if target == "mat_ternary" and scale is None:
        for row, count_rows, column, count in _chunks(entry, target):
            scan = _effective(
                entry, row * columns + column, count_rows * count, scale
            ).astype(np.float32)
            if not np.isfinite(scan).all():
                raise ValueError("ternary quantization requires finite FP32 values")
            gamma = max(gamma, np.abs(scan).max())
    row_maximum = np.float32(1e-5)
    parts: tuple[NDArray[np.uint8], ...]
    for row, count_rows, column, count in _chunks(entry, target):
        values = _effective(
            entry, row * columns + column, count_rows * count, scale
        ).reshape(count_rows, count)
        if target == "mat_ternary":
            normalized = (
                values.astype(np.float32) / gamma
                if gamma
                else np.zeros(values.shape, dtype=np.float32)
            )
            quantized = np.clip(np.rint(normalized), -1, 1).astype(np.int8)
            encoded = codec.import_mat_ternary(quantized, gamma)[0]
            parts = (encoded[:-4],)
        elif target == "row_int8" and columns > CHUNK_BYTES // 8:
            if column == 0:
                row_maximum = np.float32(1e-5)
                for begin in range(0, columns, CHUNK_BYTES // 8):
                    scan = _effective(
                        entry,
                        row * columns + begin,
                        min(CHUNK_BYTES // 8, columns - begin),
                        scale,
                    ).astype(np.float32)
                    if not np.isfinite(scan).all():
                        raise ValueError(
                            "row_int8 quantization requires finite FP32 values"
                        )
                    row_maximum = max(row_maximum, np.abs(scan).max())
            multiplier = np.float32(127) / row_maximum
            quantized = np.clip(
                np.rint(values.astype(np.float32) * multiplier), -127, 127
            ).astype(np.int8)
            parts = (quantized.view(np.uint8).reshape(-1),)
            if column + count == columns:
                parts = (
                    *parts,
                    codec.encode(
                        np.array([row_maximum / np.float32(127)], dtype=np.float32),
                        "bf16",
                    )[0],
                )
        else:
            encoded = codec.encode(values, target)[0]
            sizes = codec.payload_sections(target, values.shape)
            splits = np.cumsum([size for _, size in sizes[:-1]])
            parts = tuple(np.split(encoded, splits))
        for index, part in enumerate(parts):
            output.seek(positions[index])
            output.write(part.tobytes())
            positions[index] += part.nbytes
    if target == "mat_ternary":
        output.seek(positions[1])
        output.write(np.array([gamma], dtype="<f4").tobytes())
        positions[1] += 4
    if positions != ends:
        raise RuntimeError(f"canonical section lengths disagree: {entry.identity}")


def write_weights(
    index: WeightIndex,
    destination: Path,
    required_representations: Sequence[WeightRequest] | None = None,
) -> WeightManifest:
    """Stream canonical weights into one version1 JSON-header binary bundle."""
    destination = destination.resolve()
    if required_representations is None:
        targets: dict[str, set[DType]] = {}
        for binding in index.bindings.values():
            targets.setdefault(binding.logical, set()).add(binding.dtype)
        required_representations = [
            WeightRequest(
                logical,
                dtype,
                name=logical if len(dtypes) == 1 else f"{logical}@{dtype}",
            )
            for logical, dtypes in targets.items()
            for dtype in sorted(dtypes)
        ]
    records: dict[str, WeightRecord] = {}
    work: dict[tuple[str, DType, str], tuple[WeightEntry, WeightRequest, int]] = {}
    fingerprints: dict[WeightSource, SourceProvenance] = {}
    offsets: dict[tuple[str, DType, str], int] = {}
    names_by_key: dict[tuple[str, DType, str], list[str]] = {}
    total = 0
    for request in required_representations:
        if request.packing != "canonical":
            raise ValueError(f"unsupported backend weight packing: {request.packing}")
        entry = index.sources[request.logical]
        key = entry.identity, request.dtype, request.packing
        name = request.name or request.logical
        if name in records:
            raise ValueError(f"duplicate weight representation name: {name}")
        names_by_key.setdefault(key, []).append(name)
        size = sum(
            length for _, length in codec.payload_sections(request.dtype, entry.shape)
        )
        if key not in offsets:
            total = (total + 63) // 64 * 64
            offsets[key] = total
            work[key] = entry, request, total
            total += size
        sources = tuple(
            dict.fromkeys((entry.source, *entry.components, *entry.tied_sources))
        )
        for source in sources:
            if source not in fingerprints:
                fingerprints[source] = _provenance(source)
        for tied in entry.tied_sources:
            if fingerprints[tied].sha256 != fingerprints[entry.source].sha256:
                raise ValueError(
                    f"tied checkpoint weights have different values: {entry.identity}"
                )
        records[name] = WeightRecord(
            name=name,
            logical=request.logical,
            dtype=request.dtype,
            shape=list(entry.shape),
            layout=request.packing,
            offset=offsets[key],
            size=size,
            import_method=(
                f"bitnet_packed:{entry.linear_class}"
                if entry.method == "bitnet_packed"
                else entry.method
            ),
            provenance=[fingerprints[source] for source in sources],
        )
    for names in names_by_key.values():
        for name in names:
            records[name] = records[name].model_copy(
                update={"aliases": [alias for alias in names if alias != name]}
            )
    manifest = WeightManifest(data_start=8, entries=records)
    while True:
        header = manifest.model_dump_json().encode()
        start = (8 + len(header) + 63) // 64 * 64
        if start == manifest.data_start:
            break
        manifest = manifest.model_copy(update={"data_start": start})
    source_paths = {source.path for source in fingerprints}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination in source_paths or temporary in source_paths:
        raise ValueError("weight bundle destination overlaps a checkpoint source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w+b", buffering=CHUNK_BYTES) as output:
            output.write(struct.pack("<Q", manifest.data_start - 8))
            output.write(header)
            output.write(b" " * (manifest.data_start - 8 - len(header)))
            output.truncate(manifest.data_start + total)
            scales: dict[str, tuple[np.float32, np.float32]] = {}
            for entry, request, offset in work.values():
                if (
                    entry.method in ("bitnet_absmean", "bitnet_packed")
                    and entry.identity not in scales
                ):
                    scales[entry.identity] = _bitnet_scale(entry)
                _write_entry(
                    output,
                    manifest.data_start + offset,
                    entry,
                    request.dtype,
                    scales.get(entry.identity),
                )
            for source in fingerprints:
                unchanged(source)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def read_manifest(bundle: Path) -> WeightManifest:
    with bundle.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated weight bundle header")
        length = struct.unpack("<Q", prefix)[0]
        if length > min(100_000_000, bundle.stat().st_size - 8):
            raise ValueError("invalid weight bundle header length")
        manifest = WeightManifest.model_validate_json(file.read(length))
    if manifest.data_start != 8 + length or manifest.data_start % 64:
        raise ValueError("weight bundle data_start disagrees with header")
    size = bundle.stat().st_size - manifest.data_start
    for name, entry in manifest.entries.items():
        if name != entry.name or entry.offset % 64 or entry.offset + entry.size > size:
            raise ValueError(f"invalid weight bundle entry: {name}")
        expected = sum(
            length
            for _, length in codec.payload_sections(entry.dtype, tuple(entry.shape))
        )
        if entry.layout != "canonical" or entry.size != expected:
            raise ValueError(f"invalid weight bundle representation: {name}")
        for alias in entry.aliases:
            other = manifest.entries.get(alias)
            if other is None or (
                other.offset,
                other.size,
                other.dtype,
                other.shape,
                other.layout,
            ) != (entry.offset, entry.size, entry.dtype, entry.shape, entry.layout):
                raise ValueError(f"invalid weight bundle alias: {name} -> {alias}")
    spans: dict[tuple[int, int], WeightRecord] = {}
    for entry in manifest.entries.values():
        if entry.size == 0:
            continue
        span = entry.offset, entry.offset + entry.size
        previous = spans.setdefault(span, entry)
        if (previous.dtype, previous.shape) != (entry.dtype, entry.shape):
            raise ValueError("aliased weight bundle entries disagree")
    end = 0
    for begin, stop in sorted(spans):
        if begin < end:
            raise ValueError("overlapping weight bundle entries")
        end = stop
    if (
        max(
            (entry.offset + entry.size for entry in manifest.entries.values()),
            default=0,
        )
        != size
    ):
        raise ValueError("unreferenced weight bundle bytes")
    return manifest
