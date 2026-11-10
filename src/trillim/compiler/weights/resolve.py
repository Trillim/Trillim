from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...tdl.core.core_dataclasses import Shape
from ...tdl.core.core_pydantic_models import Weight, _ConstantArgs
from ...tdl.core.graph import Graph
from ...tdl.core.types import DType
from .. import dtype as _codec
from ..passes.walk import _walk
from .packing import _repack_row_int8, _validate_packing
from .reader import (
    _CHUNK_BYTES,
    _SOURCE_DTYPES,
    _load_checkpoint,
    _read_bytes,
    _unchanged,
)
from .weights_dataclasses import (
    _Checkpoint,
    _ResolvedWeight,
    _SourceDType,
    _WeightBinding,
    _WeightEntry,
    _WeightIndex,
    _WeightRequest,
    _WeightSource,
)
from .weights_pydantic_models import _CheckpointConfig, _SourceProvenance, _WeightRecord

_LOGICAL_DTYPES: dict[_SourceDType, DType] = {
    "BOOL": "bool",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "fp16",
    "BF16": "bf16",
    "F32": "fp32",
}


def _entry(
    checkpoint: _Checkpoint, name: str, config: _CheckpointConfig
) -> _WeightEntry:
    canonical = name
    visited = set()
    while canonical in checkpoint.aliases:
        if canonical in visited:
            raise ValueError(f"cyclic checkpoint weight aliases: {name}")
        visited.add(canonical)
        canonical = checkpoint.aliases[canonical]
    aliases = tuple(
        f"{checkpoint.name}.{alias}"
        for alias, target in checkpoint.aliases.items()
        if target == canonical
    )
    source = checkpoint.tensors.get(canonical)
    if source is None:
        stem = canonical.removesuffix(".weight")
        blocks = checkpoint.tensors.get(stem + "_blocks")
        scales = checkpoint.tensors.get(stem + "_scales")
        if blocks is None or scales is None:
            raise ValueError(f"checkpoint {checkpoint.name!r} has no tensor {name!r}")
        if (
            blocks.dtype != "U8"
            or scales.dtype != "U8"
            or len(blocks.shape) < 2
            or blocks.shape[-1] != 16
            or blocks.shape[:-1] != scales.shape
        ):
            raise ValueError(f"invalid packed MXFP4 checkpoint tensors: {name}")
        shape = (*blocks.shape[:-2], blocks.shape[-2] * 32)
        return _WeightEntry(
            f"{checkpoint.name}.{canonical}",
            shape,
            "mxfp4",
            blocks,
            "mxfp4",
            (blocks, scales),
            aliases,
        )
    tied = tuple(
        checkpoint.tensors[alias]
        for alias, target in checkpoint.aliases.items()
        if target == canonical and alias in checkpoint.tensors
    )
    if any(
        other.shape != source.shape or other.dtype != source.dtype for other in tied
    ):
        raise ValueError(f"tied checkpoint tensors disagree in shape or dtype: {name}")
    quantization = config.quantization_config
    bitnet = (
        config.model_type == "bitnet"
        or any(
            "bitnet" in architecture.lower() for architecture in config.architectures
        )
        or quantization is not None
        and quantization.quant_method == "bitnet"
    )
    if bitnet and canonical.endswith("_proj.weight"):
        if len(source.shape) != 2:
            raise ValueError(f"BitNet projection must be a matrix: {name}")
        if source.dtype == "U8":
            scale_name = canonical.removesuffix("weight") + "weight_scale"
            scale = checkpoint.tensors.get(scale_name)
            if (
                quantization is None
                or quantization.quantization_mode != "offline"
                or quantization.linear_class not in ("bitlinear", "autobitlinear")
                or scale is None
            ):
                raise ValueError(
                    f"packed BitNet projection needs offline configuration and scalar weight_scale: {name}"
                )
            return _WeightEntry(
                f"{checkpoint.name}.{canonical}",
                (source.shape[0] * 4, source.shape[1]),
                "mat_ternary",
                source,
                "bitnet_packed",
                (scale,),
                aliases,
                tied,
                quantization.linear_class,
            )
        if source.dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"unsupported BitNet latent weight dtype: {source.dtype}")
        return _WeightEntry(
            f"{checkpoint.name}.{canonical}",
            source.shape,
            "mat_ternary",
            source,
            "bitnet_absmean",
            (),
            aliases,
            tied,
        )
    return _WeightEntry(
        f"{checkpoint.name}.{canonical}",
        source.shape,
        _LOGICAL_DTYPES[source.dtype],
        source,
        aliases=aliases,
        tied_sources=tied,
    )


def _resolve_weights(
    graph: Graph,
    checkpoints: Mapping[str, Path],
) -> _WeightIndex:
    """Bind checkpoint-qualified logical identities before shape inference."""
    if graph._region is None:
        raise ValueError("weight resolution requires a completed graph")
    index = _WeightIndex()
    identities: dict[str, _WeightEntry] = {}
    configurations: dict[str, _CheckpointConfig] = {}
    for _, operator in _walk(graph._region):
        args = operator.args
        if not isinstance(args, _ConstantArgs) or not isinstance(args.value, Weight):
            continue
        weight = args.value
        checkpoint = index.checkpoints.get(weight._model_name)
        if checkpoint is None:
            location = checkpoints.get(weight._model_name)
            if location is None:
                raise ValueError(f"missing checkpoint binding: {weight._model_name}")
            checkpoint = _load_checkpoint(weight._model_name, location)
            config = _CheckpointConfig.model_validate(checkpoint.config)
            configurations[weight._model_name] = config
            checkpoint.aliases = {
                key: value
                for key, value in checkpoint.metadata.items()
                if value in checkpoint.tensors and key != value
            }
            if config.tie_word_embeddings:
                candidates = {
                    "lm_head.weight": (
                        "model.embed_tokens.weight",
                        "transformer.wte.weight",
                        "model.decoder.embed_tokens.weight",
                    ),
                    "proj_out.weight": (
                        "model.decoder.embed_tokens.weight",
                        "decoder.embed_tokens.weight",
                    ),
                }
                for head, embeddings in candidates.items():
                    for embedding in embeddings:
                        if embedding in checkpoint.tensors:
                            checkpoint.aliases[head] = embedding
                            break
            index.checkpoints[weight._model_name] = checkpoint
        source = _entry(
            checkpoint, weight._tensor_name, configurations[weight._model_name]
        )
        source = identities.setdefault(source.identity, source)
        if weight.shape is not None and (
            len(weight.shape.dimensions) != len(source.shape)
            or any(
                declared is not None and declared != actual
                for declared, actual in zip(weight.shape.dimensions, source.shape)
            )
        ):
            raise ValueError(
                f"declared weight shape disagrees with checkpoint: {weight.root}"
            )
        target = weight.dtype or args.dtype or source.dtype
        weight.shape = Shape(source.shape)
        operator.args = args.model_copy(update={"dtype": target})
        for tensor in operator.outputs:
            tensor.shape, tensor.dtype = weight.shape, target
            index.bindings[tensor] = _WeightBinding(weight.root, source, target)
        index.sources[weight.root] = source
        index.sources.setdefault(source.identity, source)
        for alias in source.aliases:
            index.sources.setdefault(alias, source)
    return index


def _read_values(source: _WeightSource, start: int, count: int) -> NDArray[np.generic]:
    dtype = np.dtype(_SOURCE_DTYPES[source.dtype])
    values = np.frombuffer(
        _read_bytes(source, start * dtype.itemsize, count * dtype.itemsize), dtype=dtype
    )
    if source.dtype == "BF16":
        return (values.astype(np.uint32) << 16).view(np.float32)
    return cast(NDArray[np.generic], values)


def _matrix(shape: tuple[int, ...]) -> tuple[int, int]:
    return (shape[0], math.prod(shape[1:])) if len(shape) > 1 else (1, math.prod(shape))


def _provenance(source: _WeightSource) -> _SourceProvenance:
    _unchanged(source)
    digest = hashlib.sha256()
    for start in range(0, source.size, _CHUNK_BYTES):
        digest.update(
            _read_bytes(source, start, min(_CHUNK_BYTES, source.size - start))
        )
    _unchanged(source)
    return _SourceProvenance(
        checkpoint=source.checkpoint,
        tensor=source.tensor,
        file=str(source.path),
        dtype=source.dtype,
        shape=list(source.shape),
        offset=source.offset,
        size=source.size,
        sha256=digest.hexdigest(),
    )


def _bitnet_scale(entry: _WeightEntry) -> tuple[np.float32, np.float32]:
    if entry.method == "bitnet_packed":
        scale_source = entry.components[0]
        if math.prod(scale_source.shape) != 1:
            raise ValueError(f"BitNet weight_scale must be scalar: {entry.identity}")
        scale = np.float32(_read_values(scale_source, 0, 1)[0])
        gamma = np.float32(1) / scale if entry.linear_class == "bitlinear" else scale
        inverse = np.float32(1) / gamma
    else:
        count = math.prod(entry.shape)
        total = np.float64(0)
        for start in range(0, count, _CHUNK_BYTES // 4):
            values = _read_values(
                entry.source, start, min(_CHUNK_BYTES // 4, count - start)
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
                _codec._decode(
                    _codec._encode(np.array([gamma], dtype=np.float32), "bf16"), (1,)
                )[0]
            )
        elif entry.source.dtype == "F16":
            gamma = np.float32(np.float16(gamma))
    if not np.isfinite(inverse) or inverse <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"BitNet scale must be finite and positive: {entry.identity}")
    return inverse, gamma


def _effective(
    entry: _WeightEntry,
    start: int,
    count: int,
    scale: tuple[np.float32, np.float32] | None,
) -> NDArray[np.generic]:
    if entry.method == "native":
        return _read_values(entry.source, start, count)
    if entry.method == "bitnet_absmean":
        assert scale is not None
        values = _read_values(entry.source, start, count).astype(np.float32)
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
            packed = _read_values(
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
        _read_values(blocks, group_start * 16, groups * 16)
        .astype(np.uint8)
        .reshape(1, groups, 16)
    )
    exponents = (
        _read_values(scales, group_start, groups).astype(np.uint8).reshape(1, groups)
    )
    return _codec._decode(_codec._import_mxfp4(packed, exponents), (1, count)).reshape(
        -1
    )


def _chunks(entry: _WeightEntry, target: DType) -> Iterator[tuple[int, int, int, int]]:
    rows, columns = _matrix(entry.shape)
    if rows == 0 or columns == 0:
        return
    limit = _CHUNK_BYTES // 8
    alignment = math.lcm(
        _codec._GROUP_SIZES.get(target, 1), 32 if entry.method == "mxfp4" else 1
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


def _raw_chunks(source: _WeightSource, offset: int = 0) -> Iterator[tuple[int, bytes]]:
    for start in range(0, source.size, _CHUNK_BYTES):
        yield (
            offset + start,
            _read_bytes(source, start, min(_CHUNK_BYTES, source.size - start)),
        )


def _encoded_chunks(
    entry: _WeightEntry,
    target: DType,
    scales: dict[str, tuple[np.float32, np.float32]],
    packing: str = "canonical",
) -> Iterator[tuple[int, bytes]]:
    scale = None
    if entry.method in ("bitnet_absmean", "bitnet_packed"):
        if entry.identity not in scales:
            scales[entry.identity] = _bitnet_scale(entry)
        scale = scales[entry.identity]
    if packing != "canonical":
        _validate_packing(target, entry.shape, packing)
        rows, columns = entry.shape
        for row in range(0, rows, 32):
            values = _effective(entry, row * columns, 32 * columns, scale)
            encoded = _codec._encode(values.reshape(32, columns), target)[0]
            packed = _repack_row_int8(encoded, (32, columns), "canonical", packing)
            yield row * columns, packed[: 32 * columns].tobytes()
            yield rows * columns + row * 2, packed[32 * columns :].tobytes()
        return
    sections = _codec._payload_sections(target, entry.shape)
    positions = []
    cursor = 0
    for _, size in sections:
        positions.append(cursor)
        cursor += size
    ends = [
        position + size for position, (_, size) in zip(positions, sections, strict=True)
    ]
    if entry.method == "native" and target == entry.dtype:
        yield from _raw_chunks(entry.source)
        return
    if entry.method == "mxfp4" and target == "mxfp4":
        for position, source in zip(positions, entry.components, strict=True):
            yield from _raw_chunks(source, position)
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
            encoded = _codec._import_mat_ternary(quantized, gamma)[0]
            parts = (encoded[:-4],)
        elif target == "row_int8" and columns > _CHUNK_BYTES // 8:
            if column == 0:
                row_maximum = np.float32(1e-5)
                for begin in range(0, columns, _CHUNK_BYTES // 8):
                    scan = _effective(
                        entry,
                        row * columns + begin,
                        min(_CHUNK_BYTES // 8, columns - begin),
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
                    _codec._encode(
                        np.array([row_maximum / np.float32(127)], dtype=np.float32),
                        "bf16",
                    )[0],
                )
        else:
            encoded = _codec._encode(values, target)[0]
            sizes = _codec._payload_sections(target, values.shape)
            splits = np.cumsum([size for _, size in sizes[:-1]])
            parts = tuple(np.split(encoded, splits))
        for index, part in enumerate(parts):
            yield positions[index], part.tobytes()
            positions[index] += part.nbytes
    if target == "mat_ternary":
        yield positions[1], np.array([gamma], dtype="<f4").tobytes()
        positions[1] += 4
    if positions != ends:
        raise RuntimeError(f"canonical section lengths disagree: {entry.identity}")


def _resolve_payloads(
    index: _WeightIndex,
    requests: Sequence[_WeightRequest] | None = None,
) -> Iterator[_ResolvedWeight]:
    if requests is None:
        targets: dict[str, set[tuple[DType, str]]] = {}
        for binding in index.bindings.values():
            targets.setdefault(binding.logical, set()).add(
                (binding.dtype, binding.packing)
            )
        requests = [
            _WeightRequest(
                logical,
                dtype,
                packing,
                name=logical if len(dtypes) == 1 else f"{logical}@{dtype}@{packing}",
            )
            for logical, dtypes in targets.items()
            for dtype, packing in sorted(dtypes)
        ]
    fingerprints: dict[_WeightSource, _SourceProvenance] = {}
    scales: dict[str, tuple[np.float32, np.float32]] = {}
    for request in requests:
        entry = index.sources[request.logical]
        _validate_packing(request.dtype, entry.shape, request.packing)
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
        record = _WeightRecord(
            name=request.name or request.logical,
            logical=request.logical,
            dtype=request.dtype,
            shape=list(entry.shape),
            layout=request.packing,
            offset=0,
            size=sum(
                length
                for _, length in _codec._payload_sections(request.dtype, entry.shape)
            ),
            import_method=f"bitnet_packed:{entry.linear_class}"
            if entry.method == "bitnet_packed"
            else entry.method,
            provenance=[fingerprints[source] for source in sources],
        )
        yield _ResolvedWeight(
            entry.identity,
            record,
            sources,
            _encoded_chunks(entry, request.dtype, scales, request.packing),
        )
