"""Replay GPT-oss MXFP4 expert projections without decoding or requantizing weights.

Original E2M1 nibbles and E8M0 exponent bytes are copied into the canonical
Trillim layout. Activations are deterministic synthetic FP32; outputs are FP32.
This measures individual expert Linear projections, not full MoE/model latency.
Kernel timing excludes checkpoint loading, packing, allocation and pool creation.
"""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from trillim.compiler.bench.bench_dataclasses import (
    _BenchmarkInput,
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
)
from trillim.compiler.dtype import _decode, _import_mxfp4
from trillim.compiler.kernels.build import _build_cpu
from trillim.compiler.kernels.native import _read_schema
from trillim.compiler.weights.reader import _load_checkpoint, _unchanged
from trillim.compiler.weights.weights_dataclasses import _Checkpoint, _WeightSource
from trillim.kernels.neon.benchmark_mxfp4_dataclasses import _Mxfp4Expert
from trillim.kernels.neon.benchmark_weights import _compare, _measure, _sample_reference
from trillim.tdl.core.types import DType

_ROOT = Path.home() / ".trillim/kernel-perf-20261112"
_IDS = {name: index for index, name in enumerate(get_args(DType))}
_PALETTE = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)


class _Arguments(argparse.Namespace):
    checkpoint: Path
    baseline_library: Path | None
    scalar_library: Path
    neon_library: Path
    output: Path
    layer: int
    experts: list[int]
    projections: list[str]
    tokens: list[int]
    threads: int
    rounds: int
    prepare_only: bool


def _sources(
    checkpoint: _Checkpoint, layer: int, projection: str
) -> tuple[_WeightSource, _WeightSource]:
    stem = f"model.layers.{layer}.mlp.experts.{projection}_proj"
    blocks = checkpoint.tensors[f"{stem}_blocks"]
    scales = checkpoint.tensors[f"{stem}_scales"]
    if (
        blocks.dtype != "U8"
        or scales.dtype != "U8"
        or len(blocks.shape) != 4
        or blocks.shape[-1] != 16
        or scales.shape != blocks.shape[:-1]
    ):
        raise ValueError(f"invalid GPT-oss MXFP4 source tensors: {stem}")
    return blocks, scales


def _prepare(
    checkpoint_path: Path,
    checkpoint: _Checkpoint,
    layer: int,
    projection: str,
    expert: int,
) -> _Mxfp4Expert:
    blocks, scales = _sources(checkpoint, layer, projection)
    experts, rows, groups, _ = blocks.shape
    if not 0 <= expert < experts:
        raise ValueError(f"expert {expert} is outside [0,{experts})")
    _unchanged(blocks)
    _unchanged(scales)
    block_bytes, scale_bytes = rows * groups * 16, rows * groups
    block_offset = blocks.offset + expert * block_bytes
    scale_offset = scales.offset + expert * scale_bytes
    codes = np.memmap(
        blocks.path,
        dtype=np.uint8,
        mode="r",
        offset=block_offset,
        shape=(rows, groups, 16),
    )
    exponents = np.memmap(
        scales.path, dtype=np.uint8, mode="r", offset=scale_offset, shape=(rows, groups)
    )
    payload, dtype = _import_mxfp4(codes, exponents)
    if (
        dtype != "mxfp4"
        or not np.array_equal(payload[:block_bytes], codes.reshape(-1))
        or not np.array_equal(payload[block_bytes:], exponents.reshape(-1))
    ):
        raise ValueError("MXFP4 import changed source code or scale bytes")
    block_sha256 = hashlib.sha256(codes).hexdigest()
    scale_sha256 = hashlib.sha256(exponents).hexdigest()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    destination = (
        _ROOT
        / "mxfp4-payloads"
        / f"layer{layer}-{projection}-expert{expert}-{payload_sha256[:16]}.bin"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != payload_sha256:
            raise ValueError(f"existing canonical artifact has changed: {destination}")
    else:
        destination.write_bytes(payload.tobytes())
    _unchanged(blocks)
    _unchanged(scales)
    return _Mxfp4Expert(
        checkpoint_path,
        layer,
        projection,
        expert,
        rows,
        groups * 32,
        blocks,
        scales,
        block_offset,
        scale_offset,
        block_bytes,
        scale_bytes,
        block_sha256,
        scale_sha256,
        destination,
        payload_sha256,
    )


def _decode_independent(expert: _Mxfp4Expert) -> NDArray[np.float32]:
    payload = np.memmap(expert.payload, dtype=np.uint8, mode="r")
    groups = expert.width // 32
    packed = np.asarray(payload[: expert.block_bytes]).reshape(expert.rows, groups, 16)
    exponent_bytes = np.asarray(payload[expert.block_bytes :]).reshape(
        expert.rows, groups
    )
    nibbles = np.empty((expert.rows, groups, 32), dtype=np.uint8)
    nibbles[:, :, 0::2] = packed & 15
    nibbles[:, :, 1::2] = packed >> 4
    exponents = exponent_bytes.astype(np.int32) - 127
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.ldexp(_PALETTE[nibbles], exponents[:, :, None])
    values = np.where(exponent_bytes[:, :, None] == 255, np.float32(np.nan), values)
    return np.asarray(values, dtype=np.float32).reshape(expert.rows, expert.width)


def _codec_comparison(
    expert: _Mxfp4Expert, independent: NDArray[np.float32]
) -> dict[str, object]:
    payload = np.memmap(expert.payload, dtype=np.uint8, mode="r")
    decoded = _decode((payload, "mxfp4"), (expert.rows, expert.width)).astype(
        np.float32
    )
    equal = bool(np.array_equal(decoded.view(np.uint32), independent.view(np.uint32)))
    if not equal:
        raise ValueError(
            "independent E2M1/E8M0 decoder disagrees bitwise with existing codec"
        )
    return {
        "bitwise_equal": equal,
        "finite": bool(np.isfinite(independent).all()),
        "negative_zero_count": int(
            np.count_nonzero((independent == 0) & np.signbit(independent))
        ),
    }


def _metadata(expert: _Mxfp4Expert) -> dict[str, object]:
    return {
        "checkpoint": str(expert.checkpoint),
        "layer": expert.layer,
        "projection": expert.projection,
        "expert": expert.expert,
        "logical_shape": (expert.rows, expert.width),
        "canonical_dtype": "mxfp4",
        "blocks_tensor": expert.blocks.tensor,
        "scales_tensor": expert.scales.tensor,
        "blocks_file": str(expert.blocks.path),
        "scales_file": str(expert.scales.path),
        "expert_blocks_offset": expert.block_offset,
        "expert_scales_offset": expert.scale_offset,
        "expert_blocks_bytes": expert.block_bytes,
        "expert_scales_bytes": expert.scale_bytes,
        "expert_blocks_sha256": expert.block_sha256,
        "expert_scales_sha256": expert.scale_sha256,
        "canonical_payload": str(expert.payload),
        "canonical_payload_sha256": expert.payload_sha256,
        "byte_preservation_verified": True,
    }


def _invocations(
    expert: _Mxfp4Expert, tokens: int, threads: int
) -> tuple[_Invocation, _Invocation, NDArray[np.float32]]:
    rng = np.random.default_rng(int(expert.payload_sha256[:16], 16) ^ tokens)
    values = rng.normal(0, 0.3, (tokens, expert.width)).astype(np.float32)
    parameters: dict[str, _ConcreteParameter] = {
        "batches": 1,
        "m": tokens,
        "n": expert.rows,
        "k": expert.width,
        "rank": 0,
        "batch_shape": (),
        "a_batch_strides": (),
        "b_batch_strides": (),
        "a_row_stride": expert.width,
        "a_col_stride": 1,
        "b_row_stride": 1,
        "b_col_stride": expert.width,
        "a_dtype": _IDS["fp32"],
        "b_dtype": _IDS["mxfp4"],
        "out_dtype": _IDS["fp32"],
        "a_storage_rows": tokens,
        "a_storage_columns": expert.width,
        "b_storage_rows": expert.rows,
        "b_storage_columns": expert.width,
    }
    generic = _Invocation(
        Path(),
        "trillim_scalar_linear",
        parameters,
        (
            values.tobytes(),
            _BenchmarkInput(expert.payload, 0, expert.block_bytes + expert.scale_bytes),
        ),
        (_BenchmarkOutput("fp32", (tokens, expert.rows)),),
        threads,
    )
    packed = replace(
        generic,
        kernel="",
        parameters={
            "tokens": tokens,
            "width": expert.width,
            "outputs": expert.rows,
            "input_dtype": _IDS["fp32"],
            "weight_dtype": _IDS["mxfp4"],
            "output_dtype": _IDS["fp32"],
        },
    )
    return generic, packed, values


def _available(library: Path, kernel: str) -> bool:
    return library.is_file() and hasattr(c.CDLL(str(library)), kernel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "Trillim/models/arm-matrix/openai/gpt-oss-20b",
    )
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument(
        "--scalar-library",
        type=Path,
        default=Path.home() / ".trillim/1.0.0/kernels/scalar/kernels.so",
    )
    parser.add_argument(
        "--neon-library",
        type=Path,
        default=Path.home() / ".trillim/1.0.0/kernels/neon/kernels.so",
    )
    parser.add_argument(
        "--output", type=Path, default=_ROOT / "gptoss-mxfp4-results.json"
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 1, 31])
    parser.add_argument(
        "--projections",
        choices=("gate_up", "down"),
        nargs="+",
        default=["gate_up", "down"],
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args(namespace=_Arguments())
    if (
        min(options.threads, options.rounds, *options.tokens) < 1
        or min(options.layer, *options.experts) < 0
    ):
        parser.error(
            "threads, rounds and tokens must be positive; layer and experts must be nonnegative"
        )
    checkpoint_path = options.checkpoint.expanduser().resolve()
    baseline_library = (
        options.baseline_library
        or Path((_ROOT / "baseline-library-path.txt").read_text().strip())
    ).resolve()
    scalar_library = options.scalar_library.resolve()
    neon_library = options.neon_library.resolve()
    checkpoint = _load_checkpoint("model", checkpoint_path)
    size = sum(
        {
            source.path: source.file_size for source in checkpoint.tensors.values()
        }.values()
    )
    if size > 32_000_000_000:
        parser.error("checkpoint weights exceed the 32 GB size ceiling")
    experts = [
        _prepare(checkpoint_path, checkpoint, options.layer, projection, expert)
        for projection in options.projections
        for expert in options.experts
    ]
    libraries = {
        "original_scalar": baseline_library,
        "current_scalar": scalar_library,
        "current_neon": neon_library,
    }
    report: dict[str, object] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timing_scope": "native individual expert Linear, including pool dispatch; excludes Python setup, mapping, packing, allocation and pool creation",
        "interpretation": "projection microbenchmarks; not complete MoE, GPT-oss latency or token rate",
        "weights": "original checkpoint E2M1 nibbles/E8M0 exponents; layout copy only; no requantization",
        "original_scalar_kernel": "trillim_scalar_linear",
        "activations": "synthetic deterministic FP32 normal(mean=0,std=0.3), seeded by canonical payload SHA256 and token count",
        "outputs": "FP32",
        "cache": "warm repeated weights",
        "sample_policy": "Discard the first call of each three-call batch; retain two samples per round",
        "machine_conditions": "Shared machine; no CPU affinity, power-mode or background-service changes",
        "threads": options.threads,
        "rounds": options.rounds,
        "libraries": {name: str(path) for name, path in libraries.items()},
        "library_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
            for name, path in libraries.items()
        },
        "experts": [_metadata(expert) for expert in experts],
    }
    records: list[dict[str, object]] = []
    report["records"] = records
    options.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        options.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if options.prepare_only:
        print(f"Preserved {len(experts)} raw MXFP4 expert payloads. {options.output}")
        return
    candidates = [
        (scalar_library, "trillim_scalar_packed_linear"),
        (neon_library, "trillim_neon_packed_linear"),
    ]
    candidates = [
        (library, name) for library, name in candidates if _available(library, name)
    ]
    if not candidates:
        parser.error(
            "no installed packed Linear candidates; build kernels or select compiled libraries"
        )
    harness = _build_cpu()
    for expert in experts:
        weights = _decode_independent(expert)
        codec = _codec_comparison(expert, weights)
        for tokens in options.tokens:
            generic, packed, inputs = _invocations(expert, tokens, options.threads)
            samples, expected = _measure(
                replace(generic, library=baseline_library), harness, options.rounds
            )
            reference_median = median(samples)
            baseline_reference = _sample_reference(inputs, weights, expected)
            for library, name in candidates:
                candidate = replace(packed, library=library, kernel=name)
                record = {
                    **_metadata(expert),
                    "tokens": tokens,
                    "kernel": name,
                    "library": str(library),
                    "parameters": candidate.parameters,
                    "codec_comparison": codec,
                    "original_scalar_samples_seconds": samples,
                    "original_scalar_median_seconds": reference_median,
                    "original_scalar_fp64_sample_comparison": baseline_reference,
                }
                try:
                    schema = _read_schema(c.CDLL(str(library)), name)
                    if {field.name for field in schema.fields} != set(
                        candidate.parameters
                    ):
                        raise ValueError(
                            "packed candidate does not expose the six-field Linear schema"
                        )
                    measured, actual = _measure(candidate, harness, options.rounds)
                    timing = median(measured)
                    record.update(
                        {
                            "samples_seconds": measured,
                            "median_seconds": timing,
                            "speedup_vs_original_scalar": reference_median / timing,
                            "gflops": 2
                            * tokens
                            * expert.rows
                            * expert.width
                            / timing
                            / 1e9,
                            "scalar_comparison": _compare(actual, expected),
                            "fp64_sample_comparison": _sample_reference(
                                inputs, weights, actual
                            ),
                        }
                    )
                    print(
                        f"GPT-oss L{expert.layer} {expert.projection} E{expert.expert} T{tokens} {name}: {timing * 1e3:.4f}ms {reference_median / timing:.2f}x",
                        flush=True,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    record["rejected"] = str(error)
                    print(
                        f"REJECT GPT-oss {expert.projection} E{expert.expert} T{tokens} {name}: {error}",
                        flush=True,
                    )
                records.append(record)
                save()
        _unchanged(expert.blocks)
        _unchanged(expert.scales)


if __name__ == "__main__":
    main()
