from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from trillim.compiler.weights.bundle import _read_manifest, _write_weights
from trillim.compiler.weights.resolve import _resolve_payloads
from trillim.compiler.weights.weights_dataclasses import _WeightIndex, _WeightRequest

from .helpers import payload, raw, source


def test_bundle_alignment_aliases_conversion_and_provenance(tmp_path: Path) -> None:
    entry = source()
    index = _WeightIndex(sources={"projection": entry, "alias": entry})
    requests = [
        _WeightRequest("projection", "bf16"),
        _WeightRequest("alias", "bf16"),
        _WeightRequest("projection", "fp32", name="float"),
    ]
    path = tmp_path / "weights.bin"
    manifest = _write_weights(_resolve_payloads(index, requests), path)
    assert _read_manifest(path) == manifest
    data = path.read_bytes()
    assert struct.unpack("<Q", data[:8])[0] == manifest.data_start - 8
    assert manifest.data_start % 64 == 0
    assert all(record.offset % 64 == 0 for record in manifest.entries.values())
    first = manifest.entries["projection"]
    assert first.aliases == ["alias"]
    assert first.offset == manifest.entries["alias"].offset
    assert manifest.entries["float"].offset >= first.offset + first.size
    for name, expected in [("projection", raw(entry)), ("float", payload(entry))]:
        record = manifest.entries[name]
        assert (
            data[
                manifest.data_start + record.offset : manifest.data_start
                + record.offset
                + record.size
            ]
            == expected
        )
        provenance = record.provenance[0]
        assert provenance.sha256 == hashlib.sha256(raw(entry)).hexdigest()
        assert provenance.tensor == entry.source.tensor
        assert provenance.offset == entry.source.offset


def test_failed_write_preserves_existing_destination(tmp_path: Path) -> None:
    entry = source()
    index = _WeightIndex(sources={"projection": entry})
    requests = [_WeightRequest("projection", "bf16")]
    destination = tmp_path / "weights.bin"
    original = raw(entry)
    destination.write_bytes(original)
    resolved = list(_resolve_payloads(index, requests))
    from dataclasses import replace

    broken = replace(resolved[0], chunks=iter([(resolved[0].record.size, original)]))
    with pytest.raises(RuntimeError, match="exceeds"):
        _write_weights([broken], destination)
    assert destination.read_bytes() == original
    assert not destination.with_suffix(".bin.partial").exists()
    with pytest.raises(ValueError, match="duplicate"):
        _write_weights(_resolve_payloads(index, requests * 2), destination)


@pytest.mark.parametrize("damage", ["short", "length", "trailing", "truncated_payload"])
def test_bundle_rejects_corruption(tmp_path: Path, damage: str) -> None:
    entry = source()
    destination = tmp_path / "weights.bin"
    _write_weights(
        _resolve_payloads(
            _WeightIndex(sources={"projection": entry}),
            [_WeightRequest("projection", "bf16")],
        ),
        destination,
    )
    data = destination.read_bytes()
    if damage == "short":
        data = data[:4]
    elif damage == "length":
        data = struct.pack("<Q", len(data)) + data[8:]
    elif damage == "trailing":
        data += raw(entry)[:1]
    else:
        data = data[:-1]
    destination.write_bytes(data)
    with pytest.raises(ValueError):
        _read_manifest(destination)
