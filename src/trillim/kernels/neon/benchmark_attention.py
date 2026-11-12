"""Compile and replay attention workloads against scalar and ARM implementations.

All libraries and JSON results stay under ~/.trillim. Use --baseline-library to
select a frozen scalar library. Timings include native kernel dispatch and pool
work, and exclude allocation, encoding, loading and thread-pool creation.
"""

from __future__ import annotations

import argparse
import ctypes as c
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import time_ns
from typing import cast, get_args

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
from trillim.compiler.kernels.kernel_dataclasses import (
    _Kernel,
    _KernelBindingContext,
    _ParameterValue,
)
from trillim.compiler.kernels.native import _read_capabilities, _read_schema
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.match import _match
from trillim.compiler.passes.optimize import _optimize
from trillim.tdl import Attention, Graph, Input, Output, Tensor
from trillim.tdl.core.types import DType

_ROOT = Path.home() / ".trillim/kernel-perf-20261112/attention"
_IDS = {name: index for index, name in enumerate(get_args(DType))}
_ROLES = (
    "standard",
    "tiled",
    "extended",
    "sinks",
    "mask",
    "mask_sinks",
)


def _compile_libraries(roles: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    sources = Path(__file__).resolve().parent
    build = _ROOT / f"build-{time_ns()}"
    temporary = build / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    libraries: dict[tuple[str, str], Path] = {}
    for role in roles:
        for backend in (
            ("neon",) if role in {"standard", "tiled"} else ("scalar", "neon")
        ):
            suffix = "" if role == "standard" else f"_{role}"
            source = sources.parent / backend / f"attention{suffix}.c"
            output = build / f"{backend}-attention{suffix}.so"
            assembly = output.with_suffix(".s")
            native_object = output.with_suffix(".o")
            flags = [
                "cc",
                "-march=armv8-a",
                "-O3",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wpedantic",
                "-fPIC",
                "-fvisibility=hidden",
                "-I",
                str(sources.parent),
            ]
            subprocess.run([*flags, "-S", str(source), "-o", str(assembly)], check=True)
            subprocess.run(
                ["cc", "-march=armv8-a", "-c", str(assembly), "-o", str(native_object)],
                check=True,
            )
            subprocess.run(
                [
                    "cc",
                    "-march=armv8-a",
                    "-dynamiclib" if platform.system() == "Darwin" else "-shared",
                    str(native_object),
                    "-lm",
                    "-o",
                    str(output),
                ],
                check=True,
            )
            libraries[backend, role] = output
    return libraries


def _graph_matches(
    libraries: dict[tuple[str, str], Path], roles: tuple[str, ...]
) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for role in roles:
        suffix = "" if role == "standard" else f"_{role}"
        name = f"trillim_neon_attention{suffix}"
        path = libraries["neon", role]
        library = c.CDLL(str(path))
        function = getattr(library, name + "_tdl")
        function.restype = c.c_char_p
        namespace: dict[str, object] = {}
        exec(function().decode(), namespace)  # noqa: S102 - trusted built-in kernel metadata
        kernel = _Kernel(
            name,
            "neon",
            cast(Callable[..., tuple[Tensor, ...]], namespace["pattern"]),
            _read_schema(library, name),
            _read_capabilities(library, name),
            cast(
                Callable[[_KernelBindingContext], dict[str, _ParameterValue]],
                namespace["bind"],
            ),
            path,
            cast(tuple[str, ...] | None, namespace.get("physical_inputs")),
        )
        graph = Graph()
        with graph:
            q = Input(name="q", shape=(7, 12, 64), dtype="fp16")
            k = Input(name="k", shape=(19, 3, 64), dtype="fp16")
            v = Input(name="v", shape=(19, 3, 48), dtype="fp16")
            mask = (
                Input(name="mask", shape=(1, 7, 19), dtype="fp32")
                if "mask" in role
                else None
            )
            sinks = (
                Input(name="sinks", shape=(12,), dtype="fp32")
                if "sinks" in role
                else None
            )
            Output(
                Attention(
                    q,
                    k,
                    v,
                    mask=mask,
                    sinks=sinks,
                    softcap=3.0 if role not in {"standard", "tiled"} else None,
                    window=12,
                ),
                name="y",
            )
        inference = _infer(graph, max_input_size=128)
        if _optimize(graph, inference):
            inference = _infer(graph, max_input_size=128)
        result = _match(graph, (kernel,), inference)
        matches[role] = any(result.candidates.values())
        if not matches[role]:
            raise RuntimeError(
                f"{name} did not match its Attention graph: {result.rejections}"
            )
    return matches


def _stored(
    values: NDArray[np.float32], dtype: DType
) -> tuple[bytes, NDArray[np.float32]]:
    storage = _encode(values, dtype)
    return storage[0].tobytes(), _decode(storage, values.shape).astype(np.float32)


def _reference(
    query: NDArray[np.float32],
    keys: NDArray[np.float32],
    values: NDArray[np.float32],
    causal: bool,
    window: int | None,
    softcap: float | None,
    mask: NDArray[np.float32] | None,
    sinks: NDArray[np.float32] | None,
    round_qk: DType | None,
) -> NDArray[np.float32]:
    queries, heads, width = query.shape
    length, kv_heads, _ = keys.shape
    result = np.zeros((queries, heads, values.shape[-1]), np.float32)
    for head in range(heads):
        scores = query[:, head, :] @ keys[:, head // (heads // kv_heads), :].T
        if round_qk is not None:
            scores = np.zeros((queries, length), np.float32)
            for channel in range(width):
                scores += (
                    query[:, head, channel, None]
                    * keys[None, :, head // (heads // kv_heads), channel]
                )
            scores = _stored(scores, round_qk)[1]
        scores *= np.float32(1 / np.sqrt(width))
        if softcap is not None:
            scores = np.tanh(scores / np.float32(softcap)) * np.float32(softcap)
        distance = (
            np.arange(length - queries, length)[:, None] - np.arange(length)[None, :]
        )
        visible = distance >= 0 if causal else np.ones((queries, length), np.bool_)
        if window is not None:
            visible &= distance < window
        scores = np.where(visible, scores, -np.inf)
        if mask is not None:
            scores += np.broadcast_to(mask, (heads, queries, length))[head]
        peak = np.max(scores, axis=-1, keepdims=True, initial=-np.inf)
        if sinks is not None:
            peak = np.maximum(peak, sinks[head])
        peak = np.where(peak == -np.inf, 0, peak)
        numerator = np.exp(scores - peak)
        denominator = numerator.sum(axis=-1, keepdims=True)
        if sinks is not None:
            denominator += np.exp(sinks[head] - peak)
        probability = numerator / np.where(denominator > 0, denominator, 1)
        result[:, head, :] = probability @ values[:, head // (heads // kv_heads), :]
    return result


def _bf16_edges(
    libraries: dict[tuple[str, str], Path], baseline: Path, harness: Path
) -> list[dict[str, object]]:
    if ("neon", "standard") not in libraries:
        return []
    parameters: dict[str, _ConcreteParameter] = {
        "heads": 1,
        "kv_heads": 1,
        "queries": 1,
        "length": 256,
        "key_dim": 8,
        "value_dim": 8,
        "rank": 3,
        "q_strides": (8, 8, 1),
        "k_strides": (8, 8, 1),
        "v_strides": (8, 8, 1),
        "q_rows": 1,
        "q_columns": 8,
        "k_rows": 256,
        "k_columns": 8,
        "v_rows": 256,
        "v_columns": 8,
        "window": 0,
        "output_dtype": 11,
        "q_dtype": 10,
        "k_dtype": 10,
        "v_dtype": 10,
        "causal": 0,
        "windowed": 0,
        "default_scale": 0,
        "scale": 1.0,
    }
    output = (_BenchmarkOutput("fp32", (1, 1, 8)),)
    patterns = (
        (1, 0x7F7F),
        (0x7F, 0x7F7F),
        (0x80, 0x7F7F),
        (0x7F7F, 1),
        (0x7F7F, 0x80),
        (0x7F80, 0),
        (0x7FC1, 0x3F80),
        (0x3F80, 0x7FC1),
        (0x3F80, 0x7F80),
        (0, 0x7F80),
        (0x2F80, 0x2F80),
        (0x4F7F, 0x4F7F),
        (0x2F7F, 0x3F80),
        (0x4F80, 0x3F80),
    )
    results: list[dict[str, object]] = []
    for query_bits, key_bits in patterns:
        query = np.zeros((1, 1, 8), np.uint16)
        query[0, 0, 0] = query_bits
        keys = np.zeros((256, 1, 8), np.uint16)
        keys[0, 0, 0] = key_bits
        values = np.zeros((256, 1, 8), np.uint16)
        values[0, 0, :] = 0x3F80
        inputs = (query.tobytes(), keys.tobytes(), values.tobytes())
        reference = _Invocation(
            baseline, "trillim_scalar_attention", parameters, inputs, output, 1
        )
        _, expected_bytes, _ = _invoke(reference, harness, 2**20)
        expected = np.frombuffer(expected_bytes[0], np.float32)
        role = "standard"
        name = "trillim_neon_attention"
        invocation = _Invocation(
            libraries["neon", role], name, parameters, inputs, output, 1
        )
        _, actual_bytes, _ = _invoke(invocation, harness, 2**20)
        actual = np.frombuffer(actual_bytes[0], np.float32)
        correct = bool(
            np.allclose(actual, expected, rtol=2e-4, atol=2e-5, equal_nan=True)
        )
        results.append(
            {
                "role": role,
                "query_bits": query_bits,
                "key_bits": key_bits,
                "correct": correct,
                "reference": [
                    float(value) if np.isfinite(value) else str(value)
                    for value in expected
                ],
                "actual": [
                    float(value) if np.isfinite(value) else str(value)
                    for value in actual
                ],
            }
        )
        if not correct:
            raise RuntimeError(f"BF16 edge mismatch: {results[-1]}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-library",
        type=Path,
        default=Path.home() / ".trillim/1.0.0/kernels/scalar/kernels.so",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8, 20])
    parser.add_argument("--neon-library", type=Path)
    parser.add_argument("--scalar-library", type=Path)
    parser.add_argument("--roles", choices=_ROLES, nargs="+", default=list(_ROLES))
    parser.add_argument("--output", type=Path, default=_ROOT / "results.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    arguments = parser.parse_args()
    output_dtype = cast(DType, arguments.output_dtype)
    roles = tuple(cast(list[str], arguments.roles))
    neon_library = cast(Path | None, arguments.neon_library)
    scalar_library = cast(Path | None, arguments.scalar_library)
    if (neon_library is None) != (scalar_library is None):
        parser.error("--neon-library and --scalar-library must be provided together")
    if neon_library is not None and scalar_library is not None:
        libraries = {
            (backend, role): neon_library if backend == "neon" else scalar_library
            for role in roles
            for backend in (
                ("neon",) if role in {"standard", "tiled"} else ("scalar", "neon")
            )
        }
    else:
        libraries = _compile_libraries(roles)
    graph_matches = _graph_matches(libraries, roles)
    harness = _build_cpu()
    bf16_edges = _bf16_edges(libraries, arguments.baseline_library, harness)
    rng = np.random.default_rng(829)
    workloads = [
        ("decode", 1, 2048, 32, 8, 128, 128, True, None, 1),
        ("decode_low_heads", 1, 8192, 8, 2, 128, 128, True, None, 1),
        ("decode_long", 1, 32768, 16, 8, 128, 128, True, None, 1),
        ("prefill", 128, 128, 32, 8, 128, 128, True, None, 1),
        ("small_prefill", 8, 128, 16, 8, 128, 128, True, None, 1),
        ("decode_batch", 4, 2048, 8, 2, 128, 128, True, None, 1),
        ("image", 256, 256, 16, 16, 128, 128, False, None, 1),
        ("window", 7, 91, 12, 3, 128, 128, True, 24, 1),
        ("irregular", 17, 91, 12, 3, 73, 37, False, None, 1),
        ("strided", 7, 91, 12, 3, 73, 37, True, None, 2),
        ("empty", 3, 0, 4, 2, 64, 64, False, None, 1),
        ("early_causal", 9, 4, 12, 3, 64, 64, True, None, 1),
        ("zero_window", 3, 9, 4, 2, 64, 64, True, 0, 1),
    ]
    if arguments.quick:
        workloads = [
            item
            for item in workloads
            if item[0] not in {"prefill", "image", "decode_long"}
        ]
    dtypes: tuple[DType, ...] = ("fp32", "fp16", "bf16")
    results: list[dict[str, object]] = []
    for (
        label,
        queries,
        length,
        heads,
        kv_heads,
        width,
        value_width,
        causal,
        window,
        stride,
    ) in workloads:
        for dtype in dtypes:
            stored_inputs = [
                _stored(rng.normal(0, 0.4, shape).astype(np.float32), dtype)
                for shape in (
                    (queries, heads, width * stride),
                    (length, kv_heads, width * stride),
                    (length, kv_heads, value_width * stride),
                )
            ]
            payloads = tuple(item[0] for item in stored_inputs)
            q, k, v = (item[1][:, :, ::stride] for item in stored_inputs)
            parameters: dict[str, _ConcreteParameter] = {
                "heads": heads,
                "kv_heads": kv_heads,
                "queries": queries,
                "length": length,
                "key_dim": width,
                "value_dim": value_width,
                "rank": 3,
                "q_strides": (width * stride, heads * width * stride, stride),
                "k_strides": (width * stride, kv_heads * width * stride, stride),
                "v_strides": (
                    value_width * stride,
                    kv_heads * value_width * stride,
                    stride,
                ),
                "q_rows": queries,
                "q_columns": heads * width * stride,
                "k_rows": length,
                "k_columns": kv_heads * width * stride,
                "v_rows": length,
                "v_columns": kv_heads * value_width * stride,
                "window": window if window is not None else 0,
                "output_dtype": _IDS[output_dtype],
                "q_dtype": _IDS[dtype],
                "k_dtype": _IDS[dtype],
                "v_dtype": _IDS[dtype],
                "causal": int(causal),
                "windowed": int(window is not None),
                "default_scale": 1,
                "scale": 1.0,
            }
            for role in roles:
                if output_dtype != "fp32" and role != "standard":
                    continue
                if role == "tiled" and stride != 1:
                    continue
                softcap = None if role in {"standard", "tiled"} else 3.0
                mask = (
                    rng.normal(-0.1, 0.3, (1, queries, length)).astype(np.float32)
                    if "mask" in role
                    else None
                )
                if mask is not None and length:
                    mask[:, :, ::7] = -np.inf
                    mask[:, 0, :] = -np.inf
                sinks = (
                    rng.normal(0, 1, (heads,)).astype(np.float32)
                    if "sinks" in role
                    else None
                )
                inputs = (
                    *payloads,
                    *((mask.tobytes(),) if mask is not None else ()),
                    *((sinks.tobytes(),) if sinks is not None else ()),
                )
                expected = _stored(
                    _reference(
                        q,
                        k,
                        v,
                        causal,
                        window,
                        softcap,
                        mask,
                        sinks,
                        dtype
                        if role not in {"standard", "tiled"} and dtype != "fp32"
                        else None,
                    ),
                    output_dtype,
                )[1].ravel()
                native_parameters = parameters
                if role not in {"standard", "tiled"}:
                    native_parameters = {
                        **parameters,
                        "softcapped": 1,
                        "softcap": 3.0,
                        "mask_strides": (0, length, 1)
                        if mask is not None
                        else (0, 0, 0),
                        "mask_rows": queries if mask is not None else 0,
                        "mask_columns": length if mask is not None else 0,
                        "mask_dtype": 11,
                        "sink_dtype": 11,
                    }
                suffix = "" if role in {"standard", "tiled"} else f"_{role}"
                baseline = _Invocation(
                    arguments.baseline_library
                    if role in {"standard", "tiled"}
                    else libraries["scalar", role],
                    f"trillim_scalar_attention{suffix}",
                    native_parameters,
                    inputs,
                    (_BenchmarkOutput(output_dtype, (queries, heads, value_width)),),
                    1,
                )
                for threads in arguments.threads:
                    baseline = replace(baseline, threads=threads)
                    old_times, old_outputs, _ = _invoke(baseline, harness, 2**31)
                    name = "trillim_neon_attention" + (
                        "" if role == "standard" else f"_{role}"
                    )
                    invocation = _Invocation(
                        libraries["neon", role],
                        name,
                        native_parameters,
                        inputs,
                        baseline.outputs,
                        threads,
                    )
                    times, outputs, _ = _invoke(invocation, harness, 2**31)
                    shape = (queries, heads, value_width)
                    actual = (
                        _decode(
                            (np.frombuffer(outputs[0], np.uint8), output_dtype), shape
                        )
                        .astype(np.float32)
                        .ravel()
                    )
                    old_actual = (
                        _decode(
                            (np.frombuffer(old_outputs[0], np.uint8), output_dtype),
                            shape,
                        )
                        .astype(np.float32)
                        .ravel()
                    )
                    relative_tolerance = (
                        2e-4
                        if output_dtype == "fp32"
                        else (0.001 if output_dtype == "fp16" else 0.008)
                    )
                    reference_correct = bool(
                        np.allclose(
                            actual,
                            expected,
                            rtol=relative_tolerance,
                            atol=2e-5,
                            equal_nan=True,
                        )
                    )
                    scalar_correct = bool(
                        np.allclose(
                            actual, old_actual, rtol=2e-4, atol=2e-5, equal_nan=True
                        )
                    )
                    correct = reference_correct and (
                        output_dtype == "fp32"
                        or scalar_correct
                        or label in {"empty", "early_causal", "zero_window"}
                    )
                    error = float(np.max(np.abs(actual - expected), initial=0))
                    item: dict[str, object] = {
                        "workload": label,
                        "shape": [queries, length, heads, kv_heads, width, value_width],
                        "role": role,
                        "dtype": dtype,
                        "output_dtype": output_dtype,
                        "threads": threads,
                        "correct": correct,
                        "reference_correct": reference_correct,
                        "matches_scalar": scalar_correct,
                        "equals_scalar": outputs[0] == old_outputs[0],
                        "maximum_error": error,
                        "scalar_correct": bool(
                            np.allclose(
                                old_actual,
                                expected,
                                rtol=2e-4,
                                atol=2e-5,
                                equal_nan=True,
                            )
                        ),
                        "scalar_samples_seconds": old_times,
                        "neon_samples_seconds": times,
                        "scalar_library": str(baseline.library),
                        "neon_library": str(invocation.library),
                        "speedup": median(old_times) / median(times),
                    }
                    results.append(item)
                    print(
                        f"{label:12s} {role:10s} {dtype:4s} t={threads:2d} scalar={median(old_times) * 1e3:9.3f}ms neon={median(times) * 1e3:8.3f}ms speedup={item['speedup']:6.2f} correct={correct}",
                        flush=True,
                    )
                    if not correct:
                        raise RuntimeError(f"numerical mismatch: {item}")
    report = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "seed": 829,
        "roles": roles,
        "build": "installed production kernels"
        if neon_library is not None
        else "cc -march=armv8-a -O3 via assembly",
        "graph_matches": graph_matches,
        "bf16_edges": bf16_edges,
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} workloads to {arguments.output}")


if __name__ == "__main__":
    main()
