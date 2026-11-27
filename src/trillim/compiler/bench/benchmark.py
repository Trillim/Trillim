from __future__ import annotations

import ctypes as c
import mmap
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import mean, median, pvariance

import numpy as np

from ..artifacts.parameters import _pack, _requirements
from ..dtype import _decode
from ..kernels.bindings import _DTYPE_IDS
from ..kernels.native import _read_schema
from .bench_dataclasses import (
    _BenchmarkFailure,
    _BenchmarkInput,
    _BenchmarkLevel,
    _BenchmarkResult,
    _Invocation,
    _NativeAllocation,
)

_MAX_BYTES = 4 * 1024**3


def _representative_tokens(level: _BenchmarkLevel) -> tuple[int, ...]:
    return {
        "none": (),
        "low": (1, 64),
        "high": (1, 64, 512),
    }[level]


def _allocation(size: int, alignment: int = 64) -> _NativeAllocation:
    owner = c.create_string_buffer(size + alignment)
    address = (c.addressof(owner) + alignment - 1) // alignment * alignment
    return _NativeAllocation(owner, address, size)


def _invoke(
    invocation: _Invocation, harness: Path, max_bytes: int, repeats: int = 3
) -> tuple[tuple[float, ...], tuple[bytes, ...], int]:
    library = c.CDLL(str(invocation.library))
    schema = _read_schema(library, invocation.kernel)
    packed = _pack(schema, invocation.parameters)
    footprint = _requirements(schema, invocation.parameters, packed=packed)
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
    create = harness_library.trillim_cpu_create
    create.argtypes, create.restype = [c.c_size_t], c.c_void_p
    destroy = harness_library.trillim_cpu_destroy
    destroy.argtypes, destroy.restype = [c.c_void_p], None
    if invocation.threads < 1:
        raise ValueError("benchmark threads must be positive")
    context = create(invocation.threads)
    if not context:
        raise MemoryError("cannot create benchmark thread pool")
    native = harness_library.trillim_cpu_run
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
    samples = (c.c_double * repeats)()
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
            repeats,
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
        if golden is not None:
            _, expected, _ = _invoke(golden, harness, max_bytes, repeats=1)
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
                        float(np.finfo(np.float32).tiny),
                        float(np.sqrt(np.mean(finite_reference * finite_reference)))
                        if finite_reference.size
                        else 1.0,
                    )
                    epsilon = {"bf16": 2.0**-7, "fp16": 2.0**-10}.get(
                        metadata.dtype, 0.0
                    )
                    # Packed projections may round floating activations to INT8.
                    packed = {
                        _DTYPE_IDS[dtype]
                        for dtype in ("grp_int4", "grp_int5", "dubgrp_int6", "grp_int8")
                    }
                    if any(
                        name.endswith("dtype") and value in packed
                        for name, value in invocation.parameters.items()
                        if isinstance(value, int)
                    ):
                        epsilon = max(epsilon, 1 / 127)
                    correct = correct and bool(
                        np.allclose(
                            actual_values,
                            reference_values,
                            rtol=max(2e-4, 2 * epsilon),
                            atol=max(2e-5, 2 * epsilon) * scale,
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
    harness: Path,
    golden: _Invocation | None = None,
    timeout: float = 10.0,
    max_bytes: int = _MAX_BYTES,
) -> _BenchmarkResult:
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
