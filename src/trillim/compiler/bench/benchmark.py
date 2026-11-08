from __future__ import annotations

import ctypes as c
import hashlib
import mmap
import multiprocessing as mp
import os
import subprocess
import tempfile
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import mean, median, pvariance

import numpy as np

from ... import _VERSION_ROOT
from ..artifacts.parameters import _pack, _requirements
from ..dtype import _decode
from ..kernels.build import _KERNEL_ROOT, _run
from ..kernels.native import _read_schema
from .bench_dataclasses import (
    _BenchmarkFailure,
    _BenchmarkInput,
    _BenchmarkLevel,
    _BenchmarkResult,
    _Invocation,
    _NativeAllocation,
)

_MAX_BYTES = 512 * 1024**2


def _representative_tokens(level: _BenchmarkLevel) -> tuple[int, ...]:
    return {
        "none": (),
        "low": (1, 8, 64),
        "high": (1, 8, 64, 512, 4096, 32768),
    }[level]


def _harness() -> Path:
    source = Path(__file__).with_name("native.c")
    header = _KERNEL_ROOT / "common.h"
    compiler = os.environ.get("CC", "cc")
    flags = (
        "-O2",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-pthread",
        "-shared",
        "-fPIC",
    )
    identity = subprocess.check_output([compiler, "--version"])
    digest = hashlib.sha256(
        compiler.encode()
        + identity
        + repr(flags).encode()
        + Path(__file__).read_bytes()
        + source.read_bytes()
        + source.with_suffix(".h").read_bytes()
        + header.read_bytes()
    ).hexdigest()[:16]
    destination = _VERSION_ROOT / "bench" / digest / "native.so"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".bench-", dir=destination.parent
        ) as directory:
            temporary = Path(directory) / "native.so"
            _run(
                [
                    compiler,
                    *flags,
                    "-I",
                    str(header.parent),
                    str(source),
                    "-o",
                    str(temporary),
                ]
            )
            temporary.replace(destination)
    return destination


def _allocation(size: int, alignment: int = 64) -> _NativeAllocation:
    owner = c.create_string_buffer(size + alignment)
    address = (c.addressof(owner) + alignment - 1) // alignment * alignment
    return _NativeAllocation(owner, address, size)


def _invoke(
    invocation: _Invocation, harness: Path, max_bytes: int
) -> tuple[tuple[float, ...], tuple[bytes, ...], int]:
    library = c.CDLL(str(invocation.library))
    schema = _read_schema(library, invocation.kernel)
    packed = _pack(schema, invocation.parameters)
    footprint = _requirements(schema, invocation.parameters)
    input_sizes = footprint.inputs
    output_sizes = footprint.outputs
    scratch_size = footprint.scratch
    total = (
        sum(
            item.size if isinstance(item, _BenchmarkInput) else len(item)
            for item in invocation.inputs
        )
        + sum(output_sizes)
        + scratch_size
    )
    if total > max_bytes:
        raise ValueError(f"benchmark requires {total} bytes, limit is {max_bytes}")
    if (
        len(invocation.inputs) != schema.input_count
        or len(invocation.outputs) != schema.output_count
    ):
        raise ValueError("benchmark binding count mismatch")
    inputs: list[_NativeAllocation] = []
    mappings: dict[Path, mmap.mmap] = {}
    for payload, size in zip(invocation.inputs, input_sizes, strict=True):
        if isinstance(payload, _BenchmarkInput):
            if payload.offset < 0 or payload.size < size:
                raise ValueError(
                    "benchmark file binding is smaller than its required span"
                )
            path = payload.path.resolve()
            if path not in mappings:
                with path.open("rb") as stream:
                    mappings[path] = mmap.mmap(
                        stream.fileno(), 0, access=mmap.ACCESS_READ
                    )
            mapping = mappings[path]
            if payload.offset + payload.size > len(mapping):
                raise ValueError("benchmark file binding exceeds its payload")
            view = np.frombuffer(
                mapping, dtype=np.uint8, count=payload.size, offset=payload.offset
            )
            inputs.append(
                _NativeAllocation((mapping, view), view.ctypes.data, payload.size)
            )
        else:
            if len(payload) < size:
                raise ValueError("benchmark input is smaller than its required span")
            allocation = _allocation(size)
            c.memmove(allocation.address, payload, size)
            inputs.append(allocation)
    outputs = [_allocation(size) for size in output_sizes]
    scratch = _allocation(scratch_size)
    harness_library = c.CDLL(str(harness))
    create = harness_library.trillim_bench_create
    create.argtypes, create.restype = [c.c_size_t], c.c_void_p
    destroy = harness_library.trillim_bench_destroy
    destroy.argtypes, destroy.restype = [c.c_void_p], None
    if invocation.threads < 1:
        raise ValueError("benchmark threads must be positive")
    context = create(invocation.threads)
    if not context:
        raise MemoryError("cannot create benchmark thread pool")
    native = harness_library.trillim_bench_run
    native.argtypes = [
        c.c_void_p,
        c.c_void_p,
        c.POINTER(c.c_void_p),
        c.POINTER(c.c_void_p),
        c.c_void_p,
        c.c_void_p,
        c.c_size_t,
        c.POINTER(c.c_double),
    ]
    native.restype = c.c_int32
    samples = (c.c_double * 3)()
    output_pointers = (c.c_void_p * len(outputs))(*(item.address for item in outputs))
    input_pointers = (c.c_void_p * len(inputs))(*(item.address for item in inputs))
    try:
        status = native(
            context,
            c.cast(getattr(library, invocation.kernel), c.c_void_p),
            output_pointers,
            input_pointers,
            scratch.address,
            packed.address,
            3,
            samples,
        )
    finally:
        destroy(context)
    if status != 0:
        raise RuntimeError(f"native benchmark returned status {status}")
    return (
        tuple(samples),
        tuple(c.string_at(item.address, item.size) for item in outputs),
        total,
    )


def _worker(
    connection: Connection,
    invocation: _Invocation,
    golden: _Invocation | None,
    harness: Path,
    max_bytes: int,
) -> None:
    try:
        samples, outputs, allocated = _invoke(invocation, harness, max_bytes)
        correct = True
        maximum_error = 0.0
        normalized_error = 0.0
        reference_samples: tuple[float, ...] = ()
        if golden is not None:
            reference_samples, expected, _ = _invoke(golden, harness, max_bytes)
            for actual_bytes, expected_bytes, metadata in zip(
                outputs, expected, invocation.outputs, strict=True
            ):
                actual = _decode(
                    (np.frombuffer(actual_bytes, dtype=np.uint8), metadata.dtype),
                    metadata.shape,
                )
                reference = _decode(
                    (np.frombuffer(expected_bytes, dtype=np.uint8), metadata.dtype),
                    metadata.shape,
                )
                if actual.dtype.kind in "biu":
                    correct = correct and bool(np.array_equal(actual, reference))
                else:
                    actual_values = actual.astype(np.float64)
                    reference_values = reference.astype(np.float64)
                    finite_reference = reference_values[np.isfinite(reference_values)]
                    scale = max(
                        1.0,
                        float(np.sqrt(np.mean(finite_reference * finite_reference)))
                        if finite_reference.size
                        else 1.0,
                    )
                    correct = correct and bool(
                        np.allclose(
                            actual_values,
                            reference_values,
                            rtol=2e-4,
                            atol=2e-5 * scale,
                            equal_nan=True,
                        )
                    )
                    if actual.size:
                        same = (actual_values == reference_values) | (
                            np.isnan(actual_values) & np.isnan(reference_values)
                        )
                        with np.errstate(invalid="ignore"):
                            differences = np.where(
                                same, 0.0, np.abs(actual_values - reference_values)
                            )
                        error = float(
                            np.max(
                                np.nan_to_num(differences, nan=np.inf, posinf=np.inf)
                            )
                        )
                        maximum_error = max(maximum_error, error)
                        normalized_error = max(normalized_error, error / scale)
        result: _BenchmarkResult | _BenchmarkFailure = _BenchmarkResult(
            samples,
            mean(samples),
            median(samples),
            pvariance(samples),
            correct,
            maximum_error,
            allocated,
            normalized_error,
            reference_samples,
        )
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        MemoryError,
        OverflowError,
    ) as error:
        result = _BenchmarkFailure(str(error))
    try:
        connection.send(result)
    except (BrokenPipeError, EOFError):
        pass
    finally:
        connection.close()


def _benchmark(
    invocation: _Invocation,
    *,
    golden: _Invocation | None = None,
    timeout: float = 10.0,
    max_bytes: int = _MAX_BYTES,
) -> _BenchmarkResult:
    harness = _harness()
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(child, invocation, golden, harness, max_bytes)
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f"kernel benchmark exceeded {timeout:g} seconds")
        result = parent.recv()
        if not isinstance(result, (_BenchmarkResult, _BenchmarkFailure)):
            raise TypeError("invalid native benchmark result")
        if isinstance(result, _BenchmarkFailure):
            raise RuntimeError(result.message)  # noqa: TRY004 - reports a worker failure
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(0.2)
        if process.is_alive():
            process.kill()
            process.join()
