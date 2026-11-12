"""Build and measure convolution kernels, checking outputs before timing them.

Run from the repository with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
src/trillim/kernels/neon/benchmark_convolutions.py --threads 20.
All build products and result files stay beneath ~/.trillim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from trillim.compiler.bench.bench_dataclasses import (
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
)
from trillim.compiler.bench.benchmark import _invoke
from trillim.compiler.dtype import _decode, _encode
from trillim.compiler.kernels.build import _build_cpu
from trillim.compiler.kernels.kernel_pydantic_models import _KernelIndex
from trillim.runtime import Runtime
from trillim.tdl.core.types import DType

type _Shape = tuple[int, ...]
type _Specification = tuple[str, bool, int, int, _Shape, _Shape, _Shape, _Shape, int]


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    threads: int = Field(default=20, ge=1)
    directory: Path = Path.home() / ".trillim/convolution-validation"
    validation_only: bool = False
    blas: bool = False
    oracle_library: Path | None = None


def _build(directory: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "tmp"
    temporary.mkdir(exist_ok=True)
    library = directory / "convolution.so"
    names = (
        "conv_nd",
        "conv_nd_bias",
        "conv_transpose_nd",
        "conv_transpose_nd_bias",
    )
    sources = [root / "scalar" / f"{name}.c" for name in names]
    sources.extend(root / "neon" / f"{name}_neon.c" for name in names)
    sources.extend(
        root / "scalar" / f"{name}.c" for name in ("conv_blas", "conv_bias_blas")
    )
    subprocess.run(
        [
            "cc",
            "-march=armv8-a",
            "-O3",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
            "-fPIC",
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-I",
            str(root),
            *(str(source) for source in sources),
            "-lm",
            "-o",
            str(library),
        ],
        env=os.environ | {"TMPDIR": str(temporary)},
        check=True,
    )
    return library


def _reference(
    x: NDArray[np.float32],
    w: NDArray[np.float32],
    b: NDArray[np.float32] | None,
    shape: _Shape,
    kernel: _Shape,
    stride: _Shape,
    padding: _Shape,
    dilation: _Shape,
    groups: int,
    transpose: bool,
) -> NDArray[np.float32]:
    dimensions = len(kernel)
    channels = x.shape[0]
    outputs = w.shape[1] * groups if transpose else w.shape[0]
    result = np.zeros((outputs, *shape), dtype=np.float32)
    input_group = channels // groups
    output_group = outputs // groups
    if transpose:
        for offset in np.ndindex(*kernel):
            coordinates = tuple(
                np.arange(length) * step + k * d - pad
                for length, step, k, d, pad in zip(
                    x.shape[1:], stride, offset, dilation, padding, strict=True
                )
            )
            selected = tuple(
                np.flatnonzero((coordinate >= 0) & (coordinate < length))
                for coordinate, length in zip(coordinates, shape, strict=True)
            )
            destinations = tuple(
                coordinate[index]
                for coordinate, index in zip(coordinates, selected, strict=True)
            )
            for group in range(groups):
                first = group * input_group
                matrix = w[(slice(first, first + input_group), slice(None), *offset)]
                values = matrix.T @ x[first : first + input_group].reshape(
                    input_group, -1
                )
                values = values.reshape(output_group, *x.shape[1:])
                source = np.ix_(np.arange(output_group), *selected)
                target = np.ix_(
                    np.arange(group * output_group, (group + 1) * output_group),
                    *destinations,
                )
                result[target] += values[source]
    else:
        padded = np.pad(x, ((0, 0), *((pad, pad) for pad in padding)))
        effective = tuple(
            (k - 1) * d + 1 for k, d in zip(kernel, dilation, strict=True)
        )
        windows = padded
        for axis, extent in enumerate(effective, start=1):
            windows = np.lib.stride_tricks.sliding_window_view(
                windows, extent, axis=axis
            )
        windows = windows[
            (
                slice(None),
                *(slice(None, None, step) for step in stride),
                *(slice(None, None, d) for d in dilation),
            )
        ]
        axes = (
            *range(1, dimensions + 1),
            0,
            *range(dimensions + 1, 2 * dimensions + 1),
        )
        for group in range(groups):
            first = group * input_group
            matrix = windows[first : first + input_group].transpose(axes)
            matrix = matrix.reshape(-1, input_group * int(np.prod(kernel)))
            out = group * output_group
            values = w[out : out + output_group].reshape(output_group, -1) @ matrix.T
            result[out : out + output_group] = values.reshape(output_group, *shape)
    if b is not None:
        result += b.reshape(-1, *((1,) * dimensions))
    return result


def _measure(
    specification: _Specification,
    library: Path,
    harness: Path,
    threads: int,
    *,
    biased: bool = True,
    dtype: DType = "fp32",
    dilation: _Shape | None = None,
    extra: _Shape | None = None,
    reference: bool = False,
    blas: bool = False,
) -> dict[str, object]:
    name, transpose, channels, outputs, shape, kernel, stride, padding, groups = (
        specification
    )
    dimensions = len(shape)
    dilation = dilation or (1,) * dimensions
    extra = extra or (0,) * dimensions
    result_shape = tuple(
        (length - 1) * step - 2 * pad + d * (k - 1) + e + 1
        if transpose
        else (length + 2 * pad - d * (k - 1) - 1) // step + 1
        for length, k, step, pad, d, e in zip(
            shape, kernel, stride, padding, dilation, extra, strict=True
        )
    )
    weight_shape = (
        (channels, outputs // groups, *kernel)
        if transpose
        else (outputs, channels // groups, *kernel)
    )
    rng = np.random.default_rng(135)
    x = rng.normal(0, 0.2, (channels, *shape)).astype(np.float32)
    w = rng.normal(0, 0.2, weight_shape).astype(np.float32)
    b = rng.normal(0, 0.2, (outputs,)).astype(np.float32)
    dtype_id = {"fp16": 9, "bf16": 10, "fp32": 11}[dtype]
    parameters: dict[str, _ConcreteParameter] = {
        "dimensions": dimensions,
        "channels": channels,
        "outputs": outputs,
        "groups": groups,
        "input_shape": shape,
        "output_shape": result_shape,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "output_padding": extra,
        "input_dtype": dtype_id,
        "weight_dtype": dtype_id,
        "bias_dtype": dtype_id,
        "output_dtype": dtype_id,
    }
    payloads = tuple(
        _encode(value, dtype)[0].tobytes() for value in (x, w, b)[: 3 if biased else 2]
    )
    suffix = "conv_transpose_nd" if transpose else "conv_nd"
    if biased:
        suffix += "_bias"
    invocation = _Invocation(
        library,
        f"trillim_scalar_{suffix}",
        parameters,
        payloads,
        (_BenchmarkOutput(dtype, (outputs, *result_shape)),),
        threads=threads,
    )
    scalar_samples, scalar_bytes, allocated = _invoke(
        invocation, harness, 512 * 1024**2
    )
    neon_samples, neon_bytes, _ = _invoke(
        replace(invocation, kernel=f"trillim_neon_{suffix}"), harness, 512 * 1024**2
    )
    scalar = _decode(
        (np.frombuffer(scalar_bytes[0], dtype=np.uint8), dtype),
        (outputs, *result_shape),
    )
    neon = _decode(
        (np.frombuffer(neon_bytes[0], dtype=np.uint8), dtype), (outputs, *result_shape)
    )
    error = float(np.max(np.abs(scalar.astype(np.float64) - neon.astype(np.float64))))
    if not np.allclose(scalar, neon, rtol=2e-4, atol=2e-5):
        raise RuntimeError(f"{name}: scalar/NEON mismatch, maximum error {error}")
    reference_error = None
    if reference and dtype == "fp32":
        expected = _reference(
            x,
            w,
            b if biased else None,
            result_shape,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            transpose,
        )
        reference_error = float(np.max(np.abs(expected - scalar.astype(np.float32))))
        if not np.allclose(expected, scalar, rtol=2e-4, atol=2e-5):
            raise RuntimeError(
                f"{name}: NumPy mismatch, maximum error {reference_error}"
            )
    scalar_time = median(scalar_samples)
    neon_time = median(neon_samples)
    record = {
        "name": name,
        "transpose": transpose,
        "bias": biased,
        "dtype": dtype,
        "parameters": parameters,
        "threads": threads,
        "allocated_bytes": allocated,
        "scalar_samples_s": scalar_samples,
        "neon_samples_s": neon_samples,
        "scalar_median_s": scalar_time,
        "neon_median_s": neon_time,
        "speedup": scalar_time / neon_time,
        "maximum_error": error,
        "bitwise_equal": bool(np.array_equal(scalar, neon)),
        "numpy_maximum_error": reference_error,
    }
    if blas and not transpose and dtype == "fp32":
        candidate = replace(
            invocation,
            kernel="trillim_scalar_conv_bias_blas"
            if biased
            else "trillim_scalar_conv_blas",
        )
        try:
            blas_samples, blas_bytes, blas_allocated = _invoke(
                candidate, harness, 512 * 1024**2
            )
        except ValueError as error:
            record["blas_rejected"] = str(error)
        else:
            actual = np.frombuffer(blas_bytes[0], dtype=np.float32)
            expected = scalar.astype(np.float32).reshape(-1)
            maximum = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
                raise RuntimeError(f"{name}: BLAS mismatch, maximum error {maximum}")
            blas_time = median(blas_samples)
            record.update(
                blas_samples_s=blas_samples,
                blas_median_s=blas_time,
                blas_allocated_bytes=blas_allocated,
                blas_maximum_error=maximum,
                blas_speedup=scalar_time / blas_time,
            )
            print(
                f"{name} {dtype}: BLAS {blas_time * 1000:.3f} ms, "
                f"{scalar_time / blas_time:.2f}x scalar",
                flush=True,
            )
    print(
        f"{name} {dtype}: scalar {scalar_time * 1000:.3f} ms, "
        f"NEON {neon_time * 1000:.3f} ms, {scalar_time / neon_time:.2f}x",
        flush=True,
    )
    return record


def _primitive_oracle(options: _Options, library: Path, harness: Path) -> None:
    baseline = options.oracle_library
    if baseline is None:
        return
    baseline = baseline.resolve()
    index = _KernelIndex.model_validate_json(
        baseline.with_name("index.json").read_bytes()
    )
    if any("conv" in symbol for symbol in index.symbols):
        raise ValueError("the primitive oracle library must omit convolution fusions")
    oracle_home = options.directory / "oracle-home"
    installed = oracle_home / "1.0.0/kernels/scalar"
    installed.parent.mkdir(parents=True, exist_ok=True)
    if installed.exists():
        if installed.resolve() != baseline.parent:
            raise ValueError("the oracle directory already uses a different baseline")
    else:
        installed.symlink_to(baseline.parent, target_is_directory=True)
    sources = options.directory / "oracle-sources"
    sources.mkdir(exist_ok=True)
    environment = os.environ | {
        "TRILLIM_HOME": str(oracle_home),
        "TMPDIR": str(options.directory / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    compile_command = (
        "import sys\n"
        "from pathlib import Path\n"
        "from trillim.compiler.compile import _compile_model\n"
        "from trillim.compiler.compiler_pydantic_models import _CompileRequest\n"
        "_compile_model(_CompileRequest(tdl_path=Path(sys.argv[1]),checkpoints={},"
        "output=Path(sys.argv[2]),backend='scalar',depth='none',threads=4))\n"
    )
    rng = np.random.default_rng(1234)
    records: list[dict[str, object]] = []
    dtypes: tuple[tuple[DType, int], ...] = (("fp16", 9), ("bf16", 10), ("fp32", 11))
    for transpose in (False, True):
        for biased in (False, True):
            for dtype, dtype_id in dtypes:
                name = f"primitive-{transpose}-{biased}-{dtype}"
                source = sources / f"{name}.tdl"
                output = options.directory / name
                weight_shape = (4, 3, 3) if transpose else (6, 2, 3)
                result_shape = (6, 17) if transpose else (6, 5)
                operation = "ConvTranspose" if transpose else "Conv"
                text = (
                    "from trillim.tdl import *\n"
                    "with Graph() as graph:\n"
                    f"    x=Input(name='x',shape=(4,9),dtype='{dtype}')\n"
                    f"    w=Input(name='weight',shape={weight_shape!r},dtype='{dtype}')\n"
                )
                if biased:
                    text += f"    b=Input(name='bias',shape=(6,),dtype='{dtype}')\n"
                text += (
                    f"    y={operation}(x,w,kernel=(3,),stride=2,padding=1,groups=2"
                    + (",bias=b" if biased else "")
                    + ")\n    Output(y,name='output')\n"
                )
                if source.exists() and source.read_text() != text:
                    raise ValueError(f"oracle source changed: {source}")
                source.write_text(text)
                if not output.exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            compile_command,
                            str(source),
                            str(output),
                        ],
                        env=environment,
                        check=True,
                    )
                parameters: dict[str, _ConcreteParameter] = {
                    "dimensions": 1,
                    "channels": 4,
                    "outputs": 6,
                    "groups": 2,
                    "input_shape": (9,),
                    "output_shape": result_shape[1:],
                    "kernel": (3,),
                    "stride": (2,),
                    "padding": (1,),
                    "dilation": (1,),
                    "output_padding": (0,),
                    "input_dtype": dtype_id,
                    "weight_dtype": dtype_id,
                    "bias_dtype": dtype_id,
                    "output_dtype": dtype_id,
                }
                suffix = "conv_transpose_nd" if transpose else "conv_nd"
                if biased:
                    suffix += "_bias"
                maximum = 0.0
                equal = True
                with Runtime(output, threads=4) as runtime:
                    for _ in range(8):
                        x = rng.normal(0, 0.4, (4, 9)).astype(np.float32)
                        w = rng.normal(0, 0.4, weight_shape).astype(np.float32)
                        b = rng.normal(0, 0.4, (6,)).astype(np.float32)
                        inputs = {"x": x, "weight": w}
                        if biased:
                            inputs["bias"] = b
                        expected = runtime.run(inputs)["output"].astype(np.float32)
                        payloads = tuple(
                            _encode(value, dtype)[0].tobytes()
                            for value in (x, w, b)[: 3 if biased else 2]
                        )
                        for backend in ("scalar", "neon"):
                            invocation = _Invocation(
                                library,
                                f"trillim_{backend}_{suffix}",
                                parameters,
                                payloads,
                                (_BenchmarkOutput(dtype, result_shape),),
                                threads=4,
                            )
                            _, values, _ = _invoke(invocation, harness, 512 * 1024**2)
                            actual = _decode(
                                (np.frombuffer(values[0], dtype=np.uint8), dtype),
                                result_shape,
                            ).astype(np.float32)
                            error = float(np.max(np.abs(actual - expected)))
                            maximum = max(maximum, error)
                            equal = equal and bool(np.array_equal(actual, expected))
                            if not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
                                raise RuntimeError(
                                    f"{name}: {backend} primitive mismatch, error {error}"
                                )
                records.append(
                    {
                        "name": name, "maximum_error": maximum, "bitwise_equal": equal, "inputs": 8
                    }
                )
                print(f"{name}: primitive oracle maximum error {maximum}", flush=True)
    (options.directory / "primitive-oracle-results.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--directory", type=Path, default=_Options().directory)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--blas", action="store_true")
    parser.add_argument("--oracle-library", type=Path)
    options = _Options.model_validate(vars(parser.parse_args()))
    library = _build(options.directory)
    harness = _build_cpu()
    _primitive_oracle(options, library, harness)
    records: list[dict[str, object]] = []
    cases: tuple[_Specification, ...] = (
        ("one-dimensional", False, 4, 6, (31,), (3,), (1,), (1,), 1),
        ("strided", False, 4, 6, (17,), (5,), (2,), (2,), 1),
        ("grouped", False, 4, 6, (23,), (4,), (3,), (2,), 2),
        ("depthwise", False, 4, 8, (25,), (3,), (1,), (2,), 4),
        ("two-dimensional", False, 2, 3, (9, 13), (3, 3), (1, 1), (1, 1), 1),
        ("grouped-2d", False, 4, 6, (13, 17), (3, 2), (2, 3), (1, 1), 2),
        (
            "three-dimensional",
            False,
            2,
            3,
            (5, 7, 9),
            (2, 3, 2),
            (1, 2, 1),
            (0, 1, 0),
            1,
        ),
    )
    dtypes: tuple[DType, ...] = ("fp32", "fp16", "bf16")
    for transpose in (False, True):
        for biased in (False, True):
            for case in cases:
                case = (case[0], transpose, *case[2:])
                for dtype in dtypes:
                    records.append(
                        _measure(
                            case,
                            library,
                            harness,
                            options.threads,
                            biased=biased,
                            dtype=dtype,
                            reference=True,
                            blas=options.blas,
                        )
                    )
    if not options.validation_only:
        workloads: tuple[_Specification, ...] = (
            ("whisper-conv1", False, 80, 384, (3000,), (3,), (1,), (1,), 1),
            ("whisper-conv2", False, 384, 384, (3000,), (3,), (2,), (1,), 1),
            ("parakeet-depthwise", False, 1024, 1024, (256,), (9,), (1,), (4,), 1024),
            ("pocket-tts-upsample", True, 512, 256, (50,), (8,), (4,), (2,), 1),
            ("image-vae", False, 128, 128, (64, 64), (3, 3), (1, 1), (1, 1), 1),
        )
        for workload in workloads:
            records.append(
                _measure(workload, library, harness, options.threads, blas=options.blas)
            )
    destination = options.directory / "benchmark-results.json"
    destination.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} measurements to {destination}")


if __name__ == "__main__":
    main()
