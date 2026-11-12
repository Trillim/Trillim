"""Replay Bonsai-2 signed normalized FWHT components with exact GGUF signs.

Activations and every butterfly stage use FP32. Inputs are rank-two token/channel
matrices with a full-width sign vector. This tool measures a necessary component;
Trillim does not yet import these GGUF models. Timings require --measure.
"""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import math
import platform
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from trillim.compiler.bench.bench_dataclasses import (
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
)
from trillim.compiler.bench.benchmark import _invoke
from trillim.compiler.kernels.kernel_dataclasses import (
    _Kernel,
    _KernelBindingContext,
    _ParameterValue,
)
from trillim.compiler.kernels.native import _read_capabilities, _read_schema
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.match import _match
from trillim.compiler.passes.optimize import _optimize
from trillim.tdl import (
    Add,
    Constant,
    Construct,
    Dim,
    Graph,
    Input,
    Mul,
    Output,
    Reshape,
    Stack,
    Sub,
    Take,
    Tensor,
)
from trillim.tdl.core.core_dataclasses import _TensorTuple


@Construct
def SignedHadamard(x: Tensor, signs: Tensor, *, block_size: int) -> Tensor:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    value = Reshape(Mul(x, signs), (-1, block_size))
    half = 1
    while half < block_size:
        pairs = Reshape(value, (-1, 2, half))
        left = Take(pairs, Constant(0), axis=1, rank=3, index_rank=0)
        right = Take(pairs, Constant(1), axis=1, rank=3, index_rank=0)
        value = Reshape(
            Stack((Add(left, right), Sub(left, right)), axis=1, rank=2),
            (-1, block_size),
        )
        half *= 2
    return Mul(Reshape(value, (Dim(x, 0), Dim(x, 1))), 1.0 / math.sqrt(block_size))


class _SignVector(BaseModel):
    path: Path
    elements: int = Field(gt=0)
    sha256: str


class _ModelSigns(BaseModel):
    model: str
    source_path: Path
    source_sha256: str
    revision: str
    block_size: int = Field(gt=0)
    sign_vectors_by_width: dict[str, _SignVector]


class _SignIndex(BaseModel):
    models: list[_ModelSigns]


class _Arguments(argparse.Namespace):
    signs: Path
    scalar_library: Path
    neon_library: Path
    harness: Path
    output: Path
    tokens: list[int]
    threads: int
    rounds: int
    measure: bool


def _reference(
    x: NDArray[np.float32], signs: NDArray[np.float32], block: int
) -> NDArray[np.float32]:
    value = np.multiply(x, signs, dtype=np.float32).reshape(-1, block)
    half = 1
    while half < block:
        pairs = value.reshape(-1, 2, half)
        left, right = pairs[:, 0].copy(), pairs[:, 1].copy()
        pairs[:, 0] = np.add(left, right, dtype=np.float32)
        pairs[:, 1] = np.subtract(left, right, dtype=np.float32)
        half *= 2
    return np.multiply(
        value.reshape(x.shape), np.float32(1.0 / math.sqrt(block)), dtype=np.float32
    )


def _kernel(library: Path, backend: str) -> _Kernel:
    symbol = f"trillim_{backend}_hadamard"
    native = c.CDLL(str(library))
    metadata = getattr(native, f"{symbol}_tdl")
    metadata.restype = c.c_char_p
    namespace: dict[str, object] = {}
    exec(compile(metadata().decode(), f"<{symbol}>", "exec"), namespace)  # noqa: S102
    return _Kernel(
        symbol,
        "scalar" if backend == "scalar" else "neon",
        cast(Callable[..., _TensorTuple], namespace["pattern"]),
        _read_schema(native, symbol),
        _read_capabilities(native, symbol),
        cast(
            Callable[[_KernelBindingContext], dict[str, _ParameterValue]],
            namespace["bind"],
        ),
        library,
    )


def _matching(kernel: _Kernel, width: int, block: int, maximum: int) -> int:
    with Graph() as graph:
        x = Input(name="x", shape=(None, width), dtype="fp32")
        signs = Input(name="signs", shape=(width,), dtype="fp32")
        Output(SignedHadamard(x, signs, block_size=block), name="y")
    facts = _infer(graph, max_input_size=maximum)
    if _optimize(graph, facts):
        facts = _infer(graph, max_input_size=maximum)
    matches = _match(graph, (kernel,), facts, algebraic=False)
    own = [
        match
        for candidates in matches.candidates.values()
        for match in candidates
        if match.kernel.name == kernel.name
    ]
    if not own:
        raise ValueError(f"faithful Hadamard graph did not match {kernel.name}")
    return max(len(match.covered) for match in own)


def _run(options: _Arguments) -> None:
    if (
        options.threads < 1
        or options.rounds < 1
        or not options.tokens
        or min(options.tokens) < 0
    ):
        raise ValueError(
            "positive threads/rounds and nonnegative token counts are required"
        )
    inventory = _SignIndex.model_validate_json(options.signs.read_bytes())
    kernels = (
        _kernel(options.scalar_library.resolve(), "scalar"),
        _kernel(options.neon_library.resolve(), "neon"),
    )
    records: list[dict[str, object]] = []
    report: dict[str, object] = {
        "scope": "Bonsai-2 signed normalized FWHT component; GGUF importer/model execution is not implemented",
        "machine": platform.platform(),
        "threads": options.threads,
        "measured": options.measure,
        "timing": "native CPU harness; allocation/parameter packing/pool creation excluded; first sample of each three-sample round discarded",
        "sign_inventory": str(options.signs.resolve()),
        "library_sha256": {
            kernel.name: hashlib.sha256(kernel.library.read_bytes()).hexdigest()
            for kernel in kernels
        },
        "measurements": records,
    }
    rng = np.random.default_rng(20261112)
    for model in inventory.models:
        for width_text, vector in model.sign_vectors_by_width.items():
            width = int(width_text)
            raw = vector.path.read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != vector.sha256
                or len(raw) != width * 4
                or vector.elements != width
            ):
                raise ValueError(f"invalid source sign vector: {vector.path}")
            signs = np.frombuffer(raw, dtype="<f4")
            if not np.all(np.isin(signs, (-1, 1))) or width % model.block_size:
                raise ValueError(
                    "sign metadata requires block-divisible width and explicit signs"
                )
            covered = {
                kernel.name: _matching(
                    kernel, width, model.block_size, max(1, max(options.tokens))
                )
                for kernel in kernels
            }
            for tokens in options.tokens:
                x = rng.normal(0, 0.5, size=(tokens, width)).astype(np.float32)
                expected = _reference(x, signs, model.block_size)
                parameters: dict[str, _ConcreteParameter] = {
                    "rows": tokens,
                    "width": width,
                    "row_stride": width,
                    "storage_rows": tokens,
                    "storage_columns": width,
                    "block_size": model.block_size,
                    "scale": 1.0 / math.sqrt(model.block_size),
                }
                for kernel in kernels:
                    invocation = _Invocation(
                        kernel.library,
                        kernel.name,
                        parameters,
                        (x.tobytes(), raw),
                        (_BenchmarkOutput("fp32", (tokens, width)),),
                        options.threads,
                    )
                    raw_samples: list[float] = []
                    for _ in range(options.rounds if options.measure else 1):
                        samples, outputs, _ = _invoke(
                            invocation, options.harness, 512 * 1024**2
                        )
                        actual = np.frombuffer(outputs[0], dtype="<f4").reshape(
                            tokens, width
                        )
                        if not np.array_equal(actual, expected, equal_nan=True):
                            raise ValueError(
                                f"Hadamard output differs from staged FP32 reference: {kernel.name}"
                            )
                        raw_samples.extend(samples)
                    record: dict[str, object] = {
                        "model": model.model,
                        "source": str(model.source_path),
                        "source_sha256": model.source_sha256,
                        "revision": model.revision,
                        "sign_path": str(vector.path),
                        "sign_sha256": vector.sha256,
                        "kernel": kernel.name,
                        "library": str(kernel.library),
                        "tokens": tokens,
                        "width": width,
                        "block_size": model.block_size,
                        "covered_operators": covered[kernel.name],
                        "exact": True,
                    }
                    if options.measure:
                        retained = [
                            value
                            for index, value in enumerate(raw_samples)
                            if index % 3
                        ]
                        record.update(
                            raw_samples_s=raw_samples,
                            discarded_sample_indices=list(
                                range(0, len(raw_samples), 3)
                            ),
                            median_s=median(retained),
                        )
                    records.append(record)
                    options.output.parent.mkdir(parents=True, exist_ok=True)
                    options.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(model.model, tokens, width, kernel.name, "exact", flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signs", type=Path, required=True)
    parser.add_argument("--scalar-library", type=Path, required=True)
    parser.add_argument("--neon-library", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--measure", action="store_true")
    _run(parser.parse_args(namespace=_Arguments()))


if __name__ == "__main__":
    _main()
