"""Replay Parakeet relative attention against scalar kernels and NumPy."""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import time_ns
from typing import cast

import numpy as np
from numpy.typing import NDArray

from trillim.compiler.bench.bench_dataclasses import (
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
)
from trillim.compiler.bench.benchmark import _invoke
from trillim.compiler.kernels.build import _build_cpu
from trillim.compiler.kernels.kernel_dataclasses import (
    _Backend,
    _Kernel,
    _KernelBindingContext,
    _ParameterValue,
)
from trillim.compiler.kernels.native import _read_capabilities, _read_schema
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.match import _match
from trillim.compiler.passes.optimize import _optimize
from trillim.tdl import Graph, Input, Output, RelativeAttention, Tensor

_ROOT = Path.home() / ".trillim/kernel-perf-20261112/relative-attention"


def _compile() -> dict[str, Path]:
    sources = Path(__file__).parent.parent
    build = _ROOT / f"build-{time_ns()}"
    temporary = build / "tmp"
    temporary.mkdir(parents=True)
    environment = dict(os.environ)
    environment["TMPDIR"] = str(temporary)
    libraries: dict[str, Path] = {}
    for backend in ("scalar", "neon"):
        source = sources / backend / "relative_attention.c"
        assembly = build / f"{backend}.s"
        native_object = build / f"{backend}.o"
        library = build / f"{backend}.so"
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
            str(sources),
        ]
        subprocess.run(
            [*flags, "-S", str(source), "-o", str(assembly)],
            env=environment,
            check=True,
        )
        subprocess.run(
            ["cc", "-march=armv8-a", "-c", str(assembly), "-o", str(native_object)],
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                "cc",
                "-dynamiclib" if platform.system() == "Darwin" else "-shared",
                str(native_object),
                "-lm",
                "-o",
                str(library),
            ],
            env=environment,
            check=True,
        )
        libraries[backend] = library
    return libraries


def _graph_match(path: Path, backend: str) -> bool:
    name = f"trillim_{backend}_relative_attention"
    library = c.CDLL(str(path))
    function = getattr(library, name + "_tdl")
    function.restype = c.c_char_p
    namespace: dict[str, object] = {}
    exec(function().decode(), namespace)  # noqa: S102 - trusted built-in metadata
    kernel = _Kernel(
        name,
        cast(_Backend, backend),
        cast(Callable[..., tuple[Tensor, ...]], namespace["pattern"]),
        _read_schema(library, name),
        _read_capabilities(library, name),
        cast(
            Callable[[_KernelBindingContext], dict[str, _ParameterValue]],
            namespace["bind"],
        ),
        path,
        cast(tuple[str, ...], namespace["physical_inputs"]),
    )
    with Graph() as graph:
        x = Input(name="x", shape=(None, 2, 7), dtype="fp32")
        r = Input(name="relative_keys", shape=(255, 2, 7), dtype="fp32")
        u = Input(name="content_bias", shape=(2, 7), dtype="fp32")
        b = Input(name="position_bias", shape=(2, 7), dtype="fp32")
        Output(RelativeAttention(x, x, x, r, content_bias=u, position_bias=b), name="y")
    inference = _infer(graph, max_input_size=128)
    if _optimize(graph, inference):
        inference = _infer(graph, max_input_size=128)
    result = _match(graph, (kernel,), inference)
    if not any(result.candidates.values()):
        raise RuntimeError(f"{name} did not match: {result.rejections}")
    return True


def _reference(values: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    q, k, v, r, u, b = values
    length, heads, width = q.shape
    result = np.empty((length, heads, v.shape[2]), np.float32)
    indices = length - 1 - np.arange(length)[:, None] + np.arange(length)[None, :]
    for head in range(heads):
        content = (q[:, head] + u[head]) @ k[:, head].T
        relative = (q[:, head] + b[head]) @ r[:, head].T
        scores = (content + np.take_along_axis(relative, indices, axis=1)) / np.sqrt(
            np.float32(width)
        )
        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result[:, head] = probabilities @ v[:, head]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-library", type=Path)
    parser.add_argument("--neon-library", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 32, 100, 256])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8, 20])
    parser.add_argument("--output", type=Path, default=_ROOT / "results.json")
    arguments = parser.parse_args()
    scalar = cast(Path | None, arguments.scalar_library)
    neon = cast(Path | None, arguments.neon_library)
    if (scalar is None) != (neon is None):
        parser.error("provide both libraries or neither")
    libraries = (
        {"scalar": scalar, "neon": neon}
        if scalar is not None and neon is not None
        else _compile()
    )
    graph_matches = {
        backend: _graph_match(path, backend) for backend, path in libraries.items()
    }
    harness = _build_cpu()
    rng = np.random.default_rng(20261112)
    results: list[dict[str, object]] = []
    cases = [(length, 8, 128, 128, 1) for length in cast(list[int], arguments.lengths)]
    cases.extend(((1, 2, 7, 5, 1), (7, 3, 9, 5, 2), (33, 2, 73, 67, 1)))
    for length, heads, width, value_width, stride in cases:
        if length < 1:
            parser.error("lengths must be positive")
        shapes = (
            (length, heads, width * stride),
            (length, heads, width * stride),
            (length, heads, value_width * stride),
            (2 * length - 1, heads, width * stride),
            (heads, width * stride),
            (heads, width * stride),
        )
        stored = [rng.normal(0, 0.4, shape).astype(np.float32) for shape in shapes]
        values = [value[..., ::stride] for value in stored]
        expected = _reference(values).ravel()
        parameters: dict[str, _ConcreteParameter] = {
            "length": length,
            "relative_length": 2 * length - 1,
            "heads": heads,
            "key_dim": width,
            "value_dim": value_width,
            "rank": 3,
            "q_strides": (heads * width * stride, width * stride, stride),
            "k_strides": (heads * width * stride, width * stride, stride),
            "v_strides": (heads * value_width * stride, value_width * stride, stride),
            "r_strides": (heads * width * stride, width * stride, stride),
            "content_head_stride": width * stride,
            "content_channel_stride": stride,
            "position_head_stride": width * stride,
            "position_channel_stride": stride,
        }
        for threads in cast(list[int], arguments.threads):
            samples: dict[str, tuple[float, ...]] = {}
            maximum_error = 0.0
            for backend, path in libraries.items():
                invocation = _Invocation(
                    path,
                    f"trillim_{backend}_relative_attention",
                    parameters,
                    tuple(value.tobytes() for value in stored),
                    (_BenchmarkOutput("fp32", (length, heads, value_width)),),
                    threads,
                )
                times, outputs, _ = _invoke(invocation, harness, 2**30)
                actual = np.frombuffer(outputs[0], np.float32)
                maximum_error = max(
                    maximum_error, float(np.max(np.abs(actual - expected), initial=0))
                )
                if not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
                    raise RuntimeError(
                        f"{backend} mismatch at {(length, heads, width, value_width, stride)}: {maximum_error}"
                    )
                samples[backend] = times
            neon_seconds = median(samples["neon"])
            speedup = median(samples["scalar"]) / neon_seconds if neon_seconds else None
            results.append(
                {
                    "shape": [length, heads, width, value_width],
                    "stride": stride,
                    "threads": threads,
                    "correct": True,
                    "maximum_error": maximum_error,
                    "samples_seconds": samples,
                    "speedup": speedup,
                }
            )
            ratio = f"{speedup:.2f}" if speedup is not None else "below timer"
            print(
                f"T={length:4d} H={heads:2d} D={width:3d} stride={stride} threads={threads:2d} scalar={median(samples['scalar']) * 1e3:.3f}ms neon={neon_seconds * 1e3:.3f}ms speedup={ratio}",
                flush=True,
            )
    source_paths = [
        Path(__file__),
        *(
            Path(__file__).parent.parent / backend / "relative_attention.c"
            for backend in libraries
        ),
    ]
    report = {
        "command": sys.argv,
        "seed": 20261112,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "graph_matches": graph_matches,
        "libraries": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in libraries.values()
        },
        "current_sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(results)} cases to {arguments.output}")


if __name__ == "__main__":
    main()
