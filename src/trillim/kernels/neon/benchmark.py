"""Replay identical native workloads against frozen scalar and current kernels.

Run from a source checkout with PYTHONDONTWRITEBYTECODE=1. Output defaults to
~/.trillim/kernel-perf-20261112/native-results.json. Timings exclude Python,
allocation, packing and thread-pool creation. They include native dispatch.
"""

from __future__ import annotations

import argparse
import ctypes as c
import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from trillim.compiler.bench.bench_dataclasses import (
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
)
from trillim.compiler.bench.benchmark import _invoke
from trillim.compiler.dtype import _decode, _encode
from trillim.compiler.kernels.build import _build_cpu
from trillim.compiler.kernels.kernel_pydantic_models import _KernelIndex
from trillim.compiler.kernels.native import _read_schema
from trillim.tdl.core.types import DType

_ROOT = Path.home() / ".trillim/kernel-perf-20261112"
_IDS = {name: index for index, name in enumerate(get_args(DType))}
_Case = tuple[str, str, _Invocation]


class _Arguments(argparse.Namespace):
    threads: int
    rounds: int
    filter: str
    output: Path


def _payload(values: NDArray[np.generic], dtype: DType) -> bytes:
    return _encode(values, dtype)[0].tobytes()


def _linear_cases(rng: np.random.Generator, threads: int) -> list[_Case]:
    result: list[_Case] = []
    dtypes: tuple[DType, ...] = ("fp32", "fp16", "bf16")
    for dtype in dtypes:
        for tokens, width, outputs in (
            (1, 1024, 3072),
            (8, 1024, 1024),
            (64, 512, 512),
        ):
            x = rng.normal(0, 0.3, (tokens, width)).astype(np.float32)
            w = rng.normal(0, 0.3, (outputs, width)).astype(np.float32)
            parameters: dict[str, _ConcreteParameter] = {
                "batches": 1,
                "m": tokens,
                "n": outputs,
                "k": width,
                "rank": 0,
                "batch_shape": (),
                "a_batch_strides": (),
                "b_batch_strides": (),
                "a_row_stride": width,
                "a_col_stride": 1,
                "b_row_stride": 1,
                "b_col_stride": width,
                "a_dtype": _IDS[dtype],
                "b_dtype": _IDS[dtype],
                "out_dtype": _IDS[dtype],
                "a_storage_rows": tokens,
                "a_storage_columns": width,
                "b_storage_rows": outputs,
                "b_storage_columns": width,
            }
            invocation = _Invocation(
                Path(),
                "trillim_scalar_linear",
                parameters,
                (_payload(x, dtype), _payload(w, dtype)),
                (_BenchmarkOutput(dtype, (tokens, outputs)),),
                threads,
            )
            result.append(
                (f"linear/{dtype}/{tokens}x{outputs}x{width}", "linear", invocation)
            )
            if tokens != 1:
                parameters = {
                    **parameters,
                    "b_row_stride": outputs,
                    "b_col_stride": 1,
                    "b_storage_rows": width,
                    "b_storage_columns": outputs,
                }
                invocation = replace(
                    invocation,
                    kernel="trillim_scalar_matmul",
                    parameters=parameters,
                    inputs=(invocation.inputs[0], _payload(w.T.copy(), dtype)),
                )
                result.append(
                    (f"matmul/{dtype}/{tokens}x{outputs}x{width}", "matmul", invocation)
                )
    return result


def _attention_cases(rng: np.random.Generator, threads: int) -> list[_Case]:
    result: list[_Case] = []
    dtypes: tuple[DType, ...] = ("fp32", "fp16", "bf16")
    for dtype in dtypes:
        for queries, length, heads, kv_heads, width in (
            (1, 2048, 32, 8, 128),
            (64, 64, 16, 8, 64),
        ):
            q = rng.normal(0, 0.3, (queries, heads, width)).astype(np.float32)
            k = rng.normal(0, 0.3, (length, kv_heads, width)).astype(np.float32)
            v = rng.normal(0, 0.3, (length, kv_heads, width)).astype(np.float32)
            parameters: dict[str, _ConcreteParameter] = {
                "heads": heads,
                "kv_heads": kv_heads,
                "queries": queries,
                "length": length,
                "key_dim": width,
                "value_dim": width,
                "rank": 3,
                "q_strides": (width, heads * width, 1),
                "k_strides": (width, kv_heads * width, 1),
                "v_strides": (width, kv_heads * width, 1),
                "q_rows": queries,
                "q_columns": heads * width,
                "k_rows": length,
                "k_columns": kv_heads * width,
                "v_rows": length,
                "v_columns": kv_heads * width,
                "window": 0,
                "output_dtype": _IDS[dtype],
                "q_dtype": _IDS[dtype],
                "k_dtype": _IDS[dtype],
                "v_dtype": _IDS[dtype],
                "causal": 1,
                "windowed": 0,
                "default_scale": 1,
                "scale": 0.0,
            }
            invocation = _Invocation(
                Path(),
                "trillim_scalar_attention",
                parameters,
                tuple(_payload(a, dtype) for a in (q, k, v)),
                (_BenchmarkOutput(dtype, (queries, heads, width)),),
                threads,
            )
            result.append(
                (
                    f"attention/{dtype}/q{queries}-k{length}-h{heads}-kv{kv_heads}-d{width}",
                    "attention",
                    invocation,
                )
            )
    return result


def _quantized_cases(rng: np.random.Generator, threads: int) -> list[_Case]:
    result: list[_Case] = []
    for tokens, width, outputs in ((1, 2048, 2048), (64, 1024, 1024), (3, 129, 67)):
        x = rng.integers(-127, 128, (tokens, width), dtype=np.int8)
        w = rng.normal(0, 0.3, (outputs, width)).astype(np.float32)
        scales = rng.uniform(4, 32, (tokens,)).astype(np.float32)
        formats: tuple[tuple[DType, str, str], ...] = (
            ("row_int8", "trillim_scalar_row_int8_linear", "row_int8"),
            ("mat_ternary", "trillim_scalar_quantized_linear", "quantized_linear"),
        )
        for dtype, kernel, family in formats:
            parameters: dict[str, _ConcreteParameter] = {
                "tokens": tokens,
                "width": width,
                "outputs": outputs,
                "output_dtype": _IDS["fp32"],
            }
            invocation = _Invocation(
                Path(),
                kernel,
                parameters,
                (x.tobytes(), _payload(w, dtype), scales.tobytes()),
                (_BenchmarkOutput("fp32", (tokens, outputs)),),
                threads,
            )
            result.append(
                (f"{family}/fp32/{tokens}x{outputs}x{width}", family, invocation)
            )
    return result


def _candidates(library: Path, family: str) -> list[str]:
    names = _KernelIndex.model_validate_json(
        library.with_name("index.json").read_bytes()
    ).symbols
    patterns = {
        "linear": ("neon_dense", "dense_neon", "linear_blas", "blas_linear"),
        "matmul": ("neon_matmul", "matmul_neon", "matmul_blas", "blas_matmul"),
        "attention": ("attention",),
        "row_int8": ("row_int8",),
        "quantized_linear": ("quantized_linear",),
    }
    selected: list[str] = []
    for name in names:
        if name.startswith("trillim_neon_scalar_"):
            continue
        if family == "row_int8" and "blocked" in name:
            continue
        if any(part in name for part in patterns.get(family, ())):
            selected.append(name)
    return selected


def _compare(
    actual: tuple[bytes, ...], expected: tuple[bytes, ...], invocation: _Invocation
) -> tuple[bool, float]:
    correct = True
    maximum = 0.0
    for output, reference, metadata in zip(
        actual, expected, invocation.outputs, strict=True
    ):
        a = _decode(
            (np.frombuffer(output, dtype=np.uint8), metadata.dtype), metadata.shape
        ).astype(np.float64)
        b = _decode(
            (np.frombuffer(reference, dtype=np.uint8), metadata.dtype), metadata.shape
        ).astype(np.float64)
        scale = max(1.0, float(np.sqrt(np.mean(b * b))))
        correct = correct and bool(
            np.allclose(a, b, rtol=2e-4, atol=2e-5 * scale, equal_nan=True)
        )
        if a.size:
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            with np.errstate(invalid="ignore"):
                difference = np.where(same, 0.0, np.abs(a - b))
            maximum = max(
                maximum,
                float(np.max(np.nan_to_num(difference, nan=np.inf, posinf=np.inf))),
            )
    return correct, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--filter", default="")
    parser.add_argument("--output", type=Path, default=_ROOT / "native-results.json")
    options = parser.parse_args(namespace=_Arguments())
    if options.threads < 1 or options.rounds < 1:
        parser.error("threads and rounds must be positive")
    baseline = Path((_ROOT / "baseline-library-path.txt").read_text().strip())
    current = (Path.home() / ".trillim/1.0.0/kernels/neon/kernels.so").resolve()
    scalar = (Path.home() / ".trillim/1.0.0/kernels/scalar/kernels.so").resolve()
    harness = _build_cpu()
    rng = np.random.default_rng(20261112)
    cases = (
        _linear_cases(rng, options.threads)
        + _attention_cases(rng, options.threads)
        + _quantized_cases(rng, options.threads)
    )
    records: list[dict[str, object]] = []
    for label, family, invocation in cases:
        if options.filter and options.filter not in label:
            continue
        try:
            golden_samples: list[float] = []
            expected: tuple[bytes, ...] = ()
            for _ in range(options.rounds):
                measured, expected, _ = _invoke(
                    replace(invocation, library=baseline), harness, 1024**3
                )
                golden_samples.extend(measured[1:])
        except (ValueError, RuntimeError) as error:
            print(f"SKIP {label}: {error}", flush=True)
            continue
        candidates = [(current, name) for name in _candidates(current, family)]
        candidates.extend(
            (scalar, name) for name in _candidates(scalar, family) if "blas" in name
        )
        candidates.append((scalar, invocation.kernel))
        for library, name in candidates:
            candidate = replace(invocation, library=library, kernel=name)
            try:
                schema = _read_schema(c.CDLL(str(library)), name)
                if set(candidate.parameters) != {field.name for field in schema.fields}:
                    continue
                samples: list[float] = []
                correct, maximum_error = True, 0.0
                for _ in range(options.rounds):
                    measured, actual, _ = _invoke(candidate, harness, 1024**3)
                    samples.extend(measured[1:])
                    match, difference = _compare(actual, expected, invocation)
                    correct = correct and match
                    maximum_error = max(maximum_error, difference)
                timing, reference = median(samples), median(golden_samples)
                record: dict[str, object] = {
                    "case": label,
                    "kernel": name,
                    "library": str(library),
                    "threads": options.threads,
                    "parameters": candidate.parameters,
                    "samples_seconds": samples,
                    "median_seconds": timing,
                    "original_scalar_samples_seconds": golden_samples,
                    "original_scalar_median_seconds": reference,
                    "speedup_vs_original_scalar": reference / timing,
                    "correct": correct,
                    "maximum_absolute_error": maximum_error,
                }
                records.append(record)
                print(
                    f"{label} {name}: {timing * 1e3:.4f} ms {reference / timing:.2f}x correct={correct} error={maximum_error:.6g}",
                    flush=True,
                )
            except (ValueError, RuntimeError) as error:
                records.append({"case": label, "kernel": name, "rejected": str(error)})
                print(f"REJECT {label} {name}: {error}", flush=True)
        report = {
            "platform": platform.platform(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if platform.system() == "Darwin"
            else platform.processor(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "seed": 20261112,
            "cache": "warm",
            "warmup": "discard first invocation of each three-call batch",
            "conditions": "shared machine; no affinity or global thread settings changed",
            "timing_scope": "native invocation including thread-pool dispatch",
            "original_scalar_library": str(baseline),
            "records": records,
        }
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
