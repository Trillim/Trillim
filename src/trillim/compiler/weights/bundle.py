from __future__ import annotations

import struct
from collections.abc import Iterable
from pathlib import Path

from ...tdl.core.types import DType
from .. import dtype as _codec
from .reader import _CHUNK_BYTES, _unchanged
from .weights_dataclasses import _ResolvedWeight, _WeightSource
from .weights_pydantic_models import _WeightManifest, _WeightRecord


def _write_weights(
    weights: Iterable[_ResolvedWeight],
    destination: Path,
) -> _WeightManifest:
    """Write resolved tensor payloads into a version 1 weight bundle."""
    destination = destination.resolve()
    records: dict[str, _WeightRecord] = {}
    work: dict[tuple[str, DType, str], tuple[_ResolvedWeight, int]] = {}
    offsets: dict[tuple[str, DType, str], int] = {}
    names_by_key: dict[tuple[str, DType, str], list[str]] = {}
    sources: dict[_WeightSource, None] = {}
    total = 0
    for weight in weights:
        record = weight.record
        key = weight.identity, record.dtype, record.layout
        if record.name in records:
            raise ValueError(f"duplicate weight representation name: {record.name}")
        names_by_key.setdefault(key, []).append(record.name)
        if key not in offsets:
            total = (total + 63) // 64 * 64
            offsets[key] = total
            work[key] = weight, total
            total += record.size
        records[record.name] = record.model_copy(update={"offset": offsets[key]})
        sources.update(dict.fromkeys(weight.sources))
    for names in names_by_key.values():
        for name in names:
            records[name] = records[name].model_copy(
                update={"aliases": [alias for alias in names if alias != name]}
            )
    manifest = _WeightManifest(data_start=8, entries=records)
    while True:
        header = manifest.model_dump_json().encode()
        start = (8 + len(header) + 63) // 64 * 64
        if start == manifest.data_start:
            break
        manifest = manifest.model_copy(update={"data_start": start})
    source_paths = {source.path for source in sources}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination in source_paths or temporary in source_paths:
        raise ValueError("weight bundle destination overlaps a checkpoint source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w+b", buffering=_CHUNK_BYTES) as output:
            output.write(struct.pack("<Q", manifest.data_start - 8))
            output.write(header)
            output.write(b" " * (manifest.data_start - 8 - len(header)))
            output.truncate(manifest.data_start + total)
            for weight, offset in work.values():
                for position, data in weight.chunks:
                    if position < 0 or position + len(data) > weight.record.size:
                        raise RuntimeError(
                            f"resolved payload exceeds weight extent: {weight.record.name}"
                        )
                    output.seek(manifest.data_start + offset + position)
                    output.write(data)
            for source in sources:
                _unchanged(source)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _read_manifest(bundle: Path) -> _WeightManifest:
    with bundle.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated weight bundle header")
        length = struct.unpack("<Q", prefix)[0]
        if length > min(100_000_000, bundle.stat().st_size - 8):
            raise ValueError("invalid weight bundle header length")
        manifest = _WeightManifest.model_validate_json(file.read(length))
    if manifest.data_start != 8 + length or manifest.data_start % 64:
        raise ValueError("weight bundle data_start disagrees with header")
    size = bundle.stat().st_size - manifest.data_start
    for name, entry in manifest.entries.items():
        if name != entry.name or entry.offset % 64 or entry.offset + entry.size > size:
            raise ValueError(f"invalid weight bundle entry: {name}")
        expected = sum(
            length
            for _, length in _codec._payload_sections(entry.dtype, tuple(entry.shape))
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
    spans: dict[tuple[int, int], _WeightRecord] = {}
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
