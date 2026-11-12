"""Compare native quantized projections with scalar and ISA-isolated variants.

Only compiled copies force an ISA path; production kernels retain runtime CPU
feature checks. Weights/activations are deterministic synthetic values. Timings
exclude encoding, allocation and pool creation, and include native dispatch.
Run this alone, after other performance work has stopped, for useful timings.
"""

from __future__ import annotations

import argparse
import json
import os
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
from trillim.tdl.core.types import DType

_ROOT = Path.home() / ".trillim/kernel-perf-20261112/quantized"
_SOURCE = Path(__file__).resolve().parent
_IDS = {name: index for index, name in enumerate(get_args(DType))}
_FORMATS: tuple[DType, ...] = (
    "row_int8",
    "mat_ternary",
    "grp_int8",
    "grp_int4",
    "grp_int5",
    "dubgrp_int6",
    "grp_binary",
    "grp_ternary",
    "mxfp4",
)
_BATCHED: dict[tuple[Path, str, int], Path] = {}
_OPERATIONS = (
    "row_int8",
    "row_int8_blocked",
    "group_int8",
    "quantized_linear",
    "bitlinear",
    "packed_linear",
)


class _Arguments(argparse.Namespace):
    compiler: str
    threads: int
    rounds: int
    filter: str
    output: Path
    baseline: Path | None
    small: bool
    output_dtype: DType


def _compile(source: str, stem: str, compiler: str) -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    temporary = _ROOT / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(temporary)
    environment["ZIG_GLOBAL_CACHE_DIR"] = str(_ROOT / "zig-global")
    environment["ZIG_LOCAL_CACHE_DIR"] = str(_ROOT / "zig-local")
    command = (
        ["zig", "cc", "-mcpu=baseline", "-Wno-unused-command-line-argument"]
        if compiler == "zig"
        else ["cc", "-march=armv8-a"]
    )
    flags = [
        "-O3",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-fPIC",
        "-fvisibility=hidden",
        "-I",
        str(_SOURCE.parent),
    ]
    assembly = _ROOT / f"{stem}-{compiler}.s"
    object_file = assembly.with_suffix(".o")
    library = assembly.with_suffix(".so")
    subprocess.run(
        [*command, *flags, "-S", "-x", "c", "-", "-o", str(assembly)],
        input=source.encode(),
        check=True,
        env=environment,
        capture_output=True,
    )
    subprocess.run(
        [*command, "-c", str(assembly), "-o", str(object_file)],
        check=True,
        env=environment,
        capture_output=True,
    )
    subprocess.run(
        [
            *command,
            "-dynamiclib" if platform.system() == "Darwin" else "-shared",
            str(object_file),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
        env=environment,
        capture_output=True,
    )
    return library


def _payload(values: NDArray[np.generic], dtype: DType) -> bytes:
    return _encode(values, dtype)[0].tobytes()


def _case(
    operation: str,
    dtype: DType,
    tokens: int,
    width: int,
    outputs: int,
    threads: int,
    baseline: Path,
    output_dtype: DType,
) -> tuple[_Invocation, _Invocation]:
    rng = np.random.default_rng(1794476737)
    source = rng.normal(0, 0.3, (tokens, width)).astype(np.float32)
    weights = rng.normal(0, 0.03, (outputs, width)).astype(np.float32)
    parameters: dict[str, _ConcreteParameter] = {
        "tokens": tokens,
        "width": width,
        "outputs": outputs,
    }
    if operation == "bitlinear":
        output_dtype = "fp32"
    output = (_BenchmarkOutput(output_dtype, (tokens, outputs)),)
    inputs: tuple[bytes, ...]
    if operation == "packed_linear":
        inputs = (_payload(source, "fp32"), _payload(weights, dtype))
        parameters |= {
            "input_dtype": _IDS["fp32"],
            "weight_dtype": _IDS[dtype],
            "output_dtype": _IDS[output_dtype],
        }
        reference_parameters: dict[str, _ConcreteParameter] = {
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
            "a_dtype": _IDS["fp32"],
            "b_dtype": _IDS[dtype],
            "out_dtype": _IDS[output_dtype],
            "a_storage_rows": tokens,
            "a_storage_columns": width,
            "b_storage_rows": outputs,
            "b_storage_columns": width,
        }
        reference = _Invocation(
            baseline,
            "trillim_scalar_linear",
            reference_parameters,
            inputs,
            output,
            threads,
        )
    elif operation == "bitlinear":
        inputs = (_payload(source, "fp32"), _payload(weights, "mat_ternary"))
        reference = _Invocation(
            baseline,
            "trillim_scalar_bitlinear",
            parameters,
            inputs,
            output,
            threads,
        )
    else:
        weight_dtype: DType = (
            "row_int8"
            if operation.startswith("row_int8")
            else "grp_int8"
            if operation == "group_int8"
            else "mat_ternary"
        )
        scales = (
            np.float32(127) / np.maximum(np.max(abs(source), axis=1), np.float32(1e-5))
        ).astype(np.float32)
        codes: NDArray[np.int8] = np.rint(source * scales[:, None]).astype(np.int8)
        encoded = _encode(weights, weight_dtype)[0]
        inputs = (codes.tobytes(), encoded.tobytes(), scales.tobytes())
        parameters["output_dtype"] = _IDS[output_dtype]
        scalar_name = (
            "row_int8_linear"
            if operation.startswith("row_int8")
            else "group_int8_linear"
            if operation == "group_int8"
            else "quantized_linear"
        )
        reference = _Invocation(
            baseline,
            f"trillim_scalar_{scalar_name}",
            parameters,
            inputs,
            output,
            threads,
        )
        if operation == "row_int8_blocked":
            values = encoded[: outputs * width].view(np.int8).reshape(outputs, width)
            blocked = values.reshape(outputs // 32, 32, width // 32, 32).transpose(
                0, 2, 1, 3
            )
            inputs = (
                inputs[0],
                blocked.tobytes() + encoded[outputs * width :].tobytes(),
                inputs[2],
            )
    invocation = _Invocation(
        Path(),
        f"trillim_neon_{operation}",
        parameters,
        inputs,
        output,
        threads,
    )
    return reference, invocation


def _batched(invocation: _Invocation, count: int, compiler: str) -> _Invocation:
    if count == 1:
        return invocation
    key = (invocation.library, invocation.kernel, count)
    library = _BATCHED.get(key)
    if library is None:
        name = invocation.kernel
        source = f"""#include "common.h"
#include <dlfcn.h>
static TrillimKernel original;
TRILLIM_KERNEL const TrillimParameterSchema *{name}_parameters(void) {{
    void *library = dlopen({json.dumps(str(invocation.library))}, RTLD_NOW | RTLD_LOCAL);
    if (!library) return NULL;
    void *symbol = dlsym(library, "{name}");
    memcpy(&original, &symbol, sizeof(original));
    const TrillimParameterSchema *(*parameters)(void);
    symbol = dlsym(library, "{name}_parameters");
    memcpy(&parameters, &symbol, sizeof(parameters));
    return parameters ? parameters() : NULL;
}}
TRILLIM_KERNEL TrillimStatus {name}(
    void *const *outputs, const void *const *inputs, void *scratch,
    const void *parameters, const TrillimKernelContext *context
) {{
    if (!original) return TRILLIM_ERROR;
    for (size_t i = 0; i < {count}; ++i) {{
        TrillimStatus status = original(outputs, inputs, scratch, parameters, context);
        if (status != TRILLIM_OK) return status;
    }}
    return TRILLIM_OK;
}}
"""
        library = _compile(
            source, f"batch-{name}-{invocation.library.stem}-{count}", compiler
        )
        _BATCHED[key] = library
    return replace(invocation, library=library)


def _measure(
    invocation: _Invocation, harness: Path, rounds: int, count: int
) -> tuple[float, bytes]:
    samples: list[float] = []
    output = b""
    for _ in range(rounds):
        times, outputs, _ = _invoke(invocation, harness, 512 * 1024**2)
        samples.extend(times[1:])
        output = outputs[0]
    return median(samples) / count, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", choices=("cc", "zig"), default="zig")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--filter", default="")
    parser.add_argument("--output", type=Path, default=_ROOT / "results.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--output-dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    options = parser.parse_args(namespace=_Arguments())
    if options.threads < 1 or options.rounds < 1:
        parser.error("threads and rounds must be positive")
    baseline = options.baseline
    baseline_note = "explicit reference library"
    frozen = _ROOT.parent / "baseline-library-path.txt"
    if baseline is None and frozen.is_file():
        baseline = Path(frozen.read_text().strip())
        baseline_note = "frozen original scalar library"
    if baseline is None:
        parser.error("--baseline is required when no frozen scalar baseline exists")
    harness = _build_cpu()
    shapes = ((1, 2048, 2048), (32, 2048, 2048), (128, 2048, 2048))
    if options.small:
        shapes = ((1, 32, 32), (3, 67, 129), (8, 256, 256))
    records: list[dict[str, object]] = []
    for operation in _OPERATIONS:
        if options.filter and options.filter not in operation:
            continue
        source = (_SOURCE / f"{operation}_neon.c").read_text()
        variants: dict[str, Path] = {}
        masks = {"neon": 1, "dotprod": 3, "auto": 7}
        if operation == "packed_linear":
            masks = {"neon": 1}
        for label, mask in masks.items():
            compiled = source.replace("features()", f"(features() & {mask})")
            variants[label] = _compile(
                compiled, f"{operation}-{label}", options.compiler
            )
        formats: tuple[DType, ...]
        if operation == "packed_linear":
            formats = _FORMATS
        elif operation.startswith("row_int8"):
            formats = ("row_int8",)
        elif operation == "group_int8":
            formats = ("grp_int8",)
        else:
            formats = ("mat_ternary",)
        for dtype in formats:
            for tokens, width, outputs in shapes:
                if operation == "row_int8_blocked" and (width % 32 or outputs % 32):
                    continue
                if operation == "group_int8" and width % 32:
                    continue
                reference, invocation = _case(
                    operation,
                    dtype,
                    tokens,
                    width,
                    outputs,
                    options.threads,
                    baseline,
                    options.output_dtype,
                )
                count = max(1, min(1000, 1000000 // (tokens * width * outputs)))
                scalar_seconds, expected = _measure(
                    _batched(reference, count, options.compiler),
                    harness,
                    options.rounds,
                    count,
                )
                metadata = reference.outputs[0]
                reference_values = _decode(
                    (np.frombuffer(expected, np.uint8), metadata.dtype), metadata.shape
                ).astype(np.float64)
                scale = max(
                    1.0,
                    float(np.sqrt(np.mean(reference_values.astype(np.float64) ** 2))),
                )
                for label, library in variants.items():
                    seconds, actual = _measure(
                        _batched(
                            replace(invocation, library=library),
                            count,
                            options.compiler,
                        ),
                        harness,
                        options.rounds,
                        count,
                    )
                    values = _decode(
                        (np.frombuffer(actual, np.uint8), metadata.dtype),
                        metadata.shape,
                    ).astype(np.float64)
                    correct = bool(
                        np.allclose(
                            values,
                            reference_values,
                            rtol=2e-4,
                            atol=2e-5 * scale,
                            equal_nan=True,
                        )
                    )
                    record: dict[str, object] = {
                        "operation": operation,
                        "weight_dtype": dtype,
                        "output_dtype": metadata.dtype,
                        "shape_tokens_width_outputs": [tokens, width, outputs],
                        "variant": label,
                        "threads": options.threads,
                        "median_ms": seconds * 1000,
                        "scalar_ms": scalar_seconds * 1000,
                        "speedup": scalar_seconds / seconds if seconds else None,
                        "native_calls_per_sample": count,
                        "correct": correct,
                        "maximum_error": float(np.max(abs(values - reference_values))),
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
    report: dict[str, object] = {
        "machine": platform.platform(),
        "compiler": options.compiler,
        "reference": str(baseline),
        "reference_note": baseline_note,
        "samples_per_variant": options.rounds * 2,
        "warmup": "first native sample of each round discarded",
        "tiny_operations": "batched native calls amortize clock resolution and wrapper dispatch",
        "timing": "native invocation only; deterministic synthetic activations and weights",
        "records": records,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(record["correct"] is not True for record in records):
        raise SystemExit("a kernel result failed scalar comparison")


if __name__ == "__main__":
    main()
