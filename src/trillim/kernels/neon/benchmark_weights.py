"""Measure native Linear kernels with genuine checkpoint projection weights.

Activations are reproducible synthetic FP32 values. Weights retain their source
FP32/FP16/BF16 bytes, shape and strides; outputs are FP32. These are projection
timings, not model latency or token rates. Python setup, file mapping, parameter
packing, allocation and thread-pool creation are outside the timed interval.

Example from a source checkout:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
        src/trillim/kernels/neon/benchmark_weights.py --threads 20 --rounds 3
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
from trillim.compiler.bench.benchmark import _invoke
from trillim.compiler.kernels.build import _build_cpu
from trillim.compiler.kernels.kernel_pydantic_models import _KernelIndex
from trillim.compiler.kernels.native import _read_schema
from trillim.compiler.weights.reader import _load_checkpoint, _unchanged
from trillim.compiler.weights.weights_dataclasses import _WeightSource
from trillim.kernels.neon.benchmark_weights_dataclasses import (
    _Projection,
    _ReplayCheckpoint,
)
from trillim.tdl.core.types import DType

_ROOT = Path.home() / ".trillim/kernel-perf-20261112"
_IDS = {name: index for index, name in enumerate(get_args(DType))}
_NATIVE: dict[str, DType] = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}
_CHECKPOINTS = (
    _ReplayCheckpoint(
        "Qwen3",
        "trillim/base/Qwen3-0.6B",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Qwen3.5",
        "arm-matrix/Qwen/Qwen3.5-0.8B",
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Qwen3.8-vision",
        "arm-matrix/Qwen/Qwen3.8-27B-FP8/outside.safetensors",
        (
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.mlp.linear_fc1.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Bonsai",
        "trillim/base/Bonsai-1.7B",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "BitNet",
        "trillim/base/BitNet",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Zeta",
        "arm-matrix/zed-industries/zeta",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Sweep",
        "arm-matrix/sweepai/sweep-next-edit-v2-7B",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Gemma",
        "arm-matrix/google/gemma-4-E2B-it",
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
        ),
    ),
    _ReplayCheckpoint(
        "GPT-oss",
        "arm-matrix/openai/gpt-oss-20b",
        ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.router.weight"),
    ),
    _ReplayCheckpoint(
        "Whisper",
        "arm-matrix/openai/whisper-large-v3-turbo",
        (
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.fc1.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Parakeet",
        "arm-matrix/nvidia/parakeet-tdt-0.6b-v3",
        (
            "encoder.layers.0.self_attn.q_proj.weight",
            "encoder.layers.0.feed_forward1.linear1.weight",
        ),
    ),
    _ReplayCheckpoint(
        "PocketTTS",
        "arm-matrix/kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors",
        (
            "flow_lm.transformer.layers.0.self_attn.in_proj.weight",
            "flow_lm.transformer.layers.0.linear1.weight",
        ),
    ),
    _ReplayCheckpoint(
        "Bonsai-Image",
        "trillim/base/Bonsai-Image-4B/transformer/diffusion_pytorch_model.safetensors",
        ("context_embedder.weight", "norm_out.linear.weight"),
    ),
    _ReplayCheckpoint(
        "FLUX.2",
        "arm-matrix/black-forest-labs/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors",
        ("context_embedder.weight", "norm_out.linear.weight"),
    ),
)
_UNSUPPORTED = (
    (
        "Qwen3.8-language",
        "arm-matrix/Qwen/Qwen3.8-27B-FP8",
        "Language FP8 source tensors are outside the native checkpoint dtype schema. Vision BF16 projections are inventoried separately.",
    ),
    (
        "Bonsai-2",
        "arm-matrix/prism-ml/Ternary-Bonsai-2-2B-gguf",
        "Downloaded PQ2_0 GGUF is outside this native safetensors projection replay. Existing Bonsai-1.7B weights are inventoried separately.",
    ),
    (
        "Llama-bitnet",
        "trillim/hf_models/Llama3-TRNQ",
        "Existing qmodel.tensors container is outside this native safetensors projection replay.",
    ),
)


class _Arguments(argparse.Namespace):
    models_root: Path
    baseline_library: Path | None
    scalar_library: Path
    neon_library: Path
    threads: int
    rounds: int
    tokens: list[int]
    filter: str
    output: Path
    inventory_only: bool
    max_weight_mib: int


def _sha256(source: _WeightSource) -> str:
    _unchanged(source)
    digest = hashlib.sha256()
    with source.path.open("rb") as stream:
        stream.seek(source.offset)
        remaining = source.size
        while remaining:
            chunk = stream.read(min(remaining, 4 * 1024**2))
            if not chunk:
                raise ValueError(f"truncated tensor: {source.tensor}")
            digest.update(chunk)
            remaining -= len(chunk)
    _unchanged(source)
    return digest.hexdigest()


def _projections(
    options: _Arguments,
) -> tuple[list[_Projection], list[dict[str, object]]]:
    projections: list[_Projection] = []
    skipped: list[dict[str, object]] = []
    for specification in _CHECKPOINTS:
        if (
            options.filter
            and options.filter.casefold() not in specification.family.casefold()
        ):
            continue
        path = (
            (options.models_root / specification.relative_path).expanduser().resolve()
        )
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"requested checkpoint path is not downloaded: {path}"
                )
            checkpoint = _load_checkpoint("model", path)
        except (OSError, ValueError) as error:
            skipped.append(
                {
                    "family": specification.family,
                    "checkpoint": str(path),
                    "reason": str(error),
                }
            )
            continue
        checkpoint_bytes = sum(
            {
                source.path: source.file_size for source in checkpoint.tensors.values()
            }.values()
        )
        if checkpoint_bytes > 32_000_000_000:
            skipped.append(
                {
                    "family": specification.family,
                    "checkpoint": str(path),
                    "reason": "checkpoint weights exceed 32 GB size ceiling",
                }
            )
            continue
        for name in specification.tensors:
            source = checkpoint.tensors.get(name)
            reason = ""
            if source is None:
                reason = "named projection is absent from checkpoint"
            elif len(source.shape) != 2 or source.dtype not in _NATIVE:
                reason = "projection is not a native FP32/FP16/BF16 matrix"
            elif source.size > options.max_weight_mib * 1024**2:
                reason = "projection exceeds --max-weight-mib"
            if reason or source is None:
                skipped.append(
                    {
                        "family": specification.family,
                        "checkpoint": str(path),
                        "tensor": name,
                        "reason": reason,
                    }
                )
                continue
            projections.append(
                _Projection(
                    specification.family,
                    path,
                    source,
                    _NATIVE[source.dtype],
                    _sha256(source),
                )
            )
    for family, relative_path, reason in _UNSUPPORTED:
        if options.filter and options.filter.casefold() not in family.casefold():
            continue
        skipped.append(
            {
                "family": family,
                "checkpoint": str(
                    (options.models_root / relative_path).expanduser().resolve()
                ),
                "reason": reason,
            }
        )
    return projections, skipped


def _metadata(projection: _Projection) -> dict[str, object]:
    source = projection.source
    return {
        "family": projection.family,
        "checkpoint": str(projection.checkpoint),
        "tensor": source.tensor,
        "source_file": str(source.path),
        "source_offset": source.offset,
        "source_bytes": source.size,
        "source_file_size": source.file_size,
        "source_mtime_ns": source.modified_ns,
        "source_dtype": source.dtype,
        "kernel_weight_dtype": projection.dtype,
        "shape": source.shape,
        "weight_sha256": projection.sha256,
    }


def _weights(source: _WeightSource) -> NDArray[np.float32]:
    dtype = (
        np.dtype("<f4")
        if source.dtype == "F32"
        else np.dtype("<f2")
        if source.dtype == "F16"
        else np.dtype("<u2")
    )
    mapping = np.memmap(
        source.path, dtype=dtype, mode="r", offset=source.offset, shape=source.shape
    )
    if source.dtype == "BF16":
        return (np.asarray(mapping, dtype=np.uint32) << 16).view(np.float32)
    return np.asarray(mapping, dtype=np.float32)


def _invocation(
    projection: _Projection, tokens: int, threads: int
) -> tuple[_Invocation, NDArray[np.float32]]:
    rows, width = projection.source.shape
    seed = int(projection.sha256[:16], 16) ^ tokens
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 0.3, (tokens, width)).astype(np.float32)
    parameters: dict[str, _ConcreteParameter] = {
        "batches": 1,
        "m": tokens,
        "n": rows,
        "k": width,
        "rank": 0,
        "batch_shape": (),
        "a_batch_strides": (),
        "b_batch_strides": (),
        "a_row_stride": width,
        "a_col_stride": 1,
        "b_row_stride": 1,
        "b_col_stride": width,
        "a_dtype": _IDS["fp32"],
        "b_dtype": _IDS[projection.dtype],
        "out_dtype": _IDS["fp32"],
        "a_storage_rows": tokens,
        "a_storage_columns": width,
        "b_storage_rows": rows,
        "b_storage_columns": width,
    }
    invocation = _Invocation(
        Path(),
        "trillim_scalar_linear",
        parameters,
        (
            inputs.tobytes(),
            _BenchmarkInput(
                projection.source.path, projection.source.offset, projection.source.size
            ),
        ),
        (_BenchmarkOutput("fp32", (tokens, rows)),),
        threads,
    )
    return invocation, inputs


def _candidate_names(library: Path, names: tuple[str, ...]) -> list[str]:
    installed = _KernelIndex.model_validate_json(
        library.with_name("index.json").read_bytes()
    ).symbols
    return [name for name in names if name in installed]


def _measure(
    invocation: _Invocation, harness: Path, rounds: int
) -> tuple[list[float], NDArray[np.float32]]:
    samples: list[float] = []
    output = np.empty((0,), dtype=np.float32)
    for _ in range(rounds):
        measured, outputs, _ = _invoke(invocation, harness, 1024**3)
        samples.extend(measured[1:])
        output = np.frombuffer(outputs[0], dtype="<f4").reshape(
            invocation.outputs[0].shape
        )
    return samples, output


def _compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    differences = a - b
    scale = max(1.0, float(np.sqrt(np.mean(b * b))))
    return {
        "correct": bool(
            np.allclose(a, b, rtol=2e-4, atol=2e-5 * scale, equal_nan=True)
        ),
        "maximum_absolute_error": float(np.max(np.abs(differences))),
        "rms_error": float(np.sqrt(np.mean(differences * differences))),
        "reference_rms": float(np.sqrt(np.mean(b * b))),
        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def _sample_reference(
    inputs: NDArray[np.float32],
    weights: NDArray[np.float32],
    output: NDArray[np.float32],
) -> dict[str, object]:
    columns = np.unique(
        np.linspace(0, weights.shape[0] - 1, min(weights.shape[0], 8), dtype=np.int64)
    )
    reference = inputs.astype(np.float64) @ weights[columns].astype(np.float64).T
    difference = output[:, columns].astype(np.float64) - reference
    return {
        "sampled_output_columns": columns.tolist(),
        "maximum_absolute_error_vs_fp64_sample": float(np.max(np.abs(difference))),
        "rms_error_vs_fp64_sample": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root", type=Path, default=Path.home() / "Trillim/models"
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
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--filter", default="")
    parser.add_argument(
        "--output", type=Path, default=_ROOT / "real-weight-results.json"
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-weight-mib", type=int, default=256)
    options = parser.parse_args(namespace=_Arguments())
    if (
        min(options.threads, options.rounds, options.max_weight_mib, *options.tokens)
        < 1
    ):
        parser.error(
            "threads, rounds, token counts and max-weight-mib must be positive"
        )
    baseline_library = (
        options.baseline_library
        or Path((_ROOT / "baseline-library-path.txt").read_text().strip())
    ).resolve()
    scalar_library = options.scalar_library.resolve()
    neon_library = options.neon_library.resolve()
    libraries = {
        "baseline": baseline_library,
        "scalar": scalar_library,
        "neon": neon_library,
    }
    projections, skipped = _projections(options)
    report: dict[str, object] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timing_scope": "native Linear invocation including thread-pool dispatch; excludes Python setup, mapping, allocation and pool creation",
        "cache": "warm repeated projection weights",
        "sample_policy": "Discard the first call of each three-call native harness batch; retain two samples per round.",
        "machine_conditions": "Shared machine; no CPU affinity, power-mode or background-service changes.",
        "activations": "synthetic deterministic FP32 normal(mean=0,std=0.3), seeded from tensor SHA256 and token count",
        "outputs": "FP32",
        "weight_conversion": "none; original checkpoint tensor bytes",
        "interpretation": "projection microbenchmarks only; no full model latency, generation or token-rate claim",
        "baseline_library": str(baseline_library),
        "scalar_library": str(scalar_library),
        "neon_library": str(neon_library),
        "library_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else None
            for name, path in libraries.items()
        },
        "threads": options.threads,
        "rounds": options.rounds,
        "projections": [_metadata(projection) for projection in projections],
        "skipped": skipped,
    }
    records: list[dict[str, object]] = []
    report["records"] = records
    options.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        options.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if options.inventory_only:
        print(
            f"Inventoried {len(projections)} projections; {len(skipped)} skipped. {options.output}"
        )
        return
    harness = _build_cpu()
    candidates = [
        (scalar_library, name)
        for name in _candidate_names(
            scalar_library,
            ("trillim_scalar_linear", "trillim_scalar_blas_linear"),
        )
    ]
    candidates.extend(
        (neon_library, name)
        for name in _candidate_names(neon_library, ("trillim_neon_dense",))
    )
    for projection in projections:
        _unchanged(projection.source)
        weights = _weights(projection.source)
        for tokens in options.tokens:
            invocation, inputs = _invocation(projection, tokens, options.threads)
            baseline = replace(invocation, library=baseline_library)
            baseline_samples, expected = _measure(baseline, harness, options.rounds)
            baseline_median = median(baseline_samples)
            reference = _sample_reference(inputs, weights, expected)
            for library, name in candidates:
                candidate = replace(invocation, library=library, kernel=name)
                record = {
                    **_metadata(projection),
                    "tokens": tokens,
                    "kernel": name,
                    "library": str(library.resolve()),
                    "parameters": candidate.parameters,
                    "baseline_samples_seconds": baseline_samples,
                    "baseline_median_seconds": baseline_median,
                    "baseline_fp64_sample_comparison": reference,
                }
                try:
                    schema = _read_schema(c.CDLL(str(library)), name)
                    if {field.name for field in schema.fields} != set(
                        invocation.parameters
                    ):
                        raise ValueError(
                            "candidate parameter schema differs from Linear schema"
                        )
                    samples, actual = _measure(candidate, harness, options.rounds)
                    timing = median(samples)
                    rows, width = projection.source.shape
                    record.update(
                        {
                            "samples_seconds": samples,
                            "median_seconds": timing,
                            "speedup_vs_original_scalar": baseline_median / timing,
                            "gflops": 2 * tokens * rows * width / timing / 1e9,
                            "effective_weight_bandwidth_gb_s": projection.source.size
                            / timing
                            / 1e9,
                            "scalar_comparison": _compare(actual, expected),
                            "fp64_sample_comparison": _sample_reference(
                                inputs, weights, actual
                            ),
                        }
                    )
                    print(
                        f"{projection.family}/{projection.source.tensor} T={tokens} {name}: {timing * 1e3:.4f} ms {baseline_median / timing:.2f}x",
                        flush=True,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    record["rejected"] = str(error)
                    print(
                        f"REJECT {projection.family}/{projection.source.tensor} {name}: {error}",
                        flush=True,
                    )
                records.append(record)
                save()
        _unchanged(projection.source)


if __name__ == "__main__":
    main()
