from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import defaultdict
from collections.abc import Hashable
from math import isfinite, prod
from pathlib import Path
from statistics import mean, median, pvariance

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from ...tdl.core.core_pydantic_models import (
    Weight,
    _BinaryArgs,
    _BroadcastArgs,
    _CastArgs,
    _ConstantArgs,
    _ConstructArgs,
    _GatherArgs,
    _PermuteArgs,
    _ReshapeArgs,
    _ScatterArgs,
)
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from .. import dtype as _codec
from ..artifacts.artifact_dataclasses import _JSON
from ..artifacts.expressions import _number
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import _Inference, _ShapeExpression
from ..compiler_pydantic_models import _CompileRequest
from ..passes.passes_dataclasses import _KernelMatch, _MatchSet
from ..weights.weights_dataclasses import _WeightBinding, _WeightIndex
from ..weights.weights_pydantic_models import _WeightManifest
from .bench_dataclasses import (
    _BenchmarkInput,
    _BenchmarkOutput,
    _BenchmarkResult,
    _ConcreteParameter,
    _Invocation,
    _Tuning,
    _TuningState,
)
from .benchmark import _MAX_BYTES, _benchmark, _representative_tokens


def _bounded(value: _ShapeExpression, point: int) -> int:
    if value.minimum is None or value.maximum is None:
        raise ValueError(f"benchmark scalar {value.op!r} has no finite interval")
    return max(value.minimum, min(point, value.maximum))


def _evaluate(
    value: _ShapeExpression,
    point: int,
    state: _TuningState,
    active: set[_ShapeExpression] | None = None,
) -> int:
    key = value, point
    if key not in state.dimensions:
        state.dimensions[key] = _calculate(value, point, state, active)
    return state.dimensions[key]


def _calculate(
    value: _ShapeExpression,
    point: int,
    state: _TuningState,
    active: set[_ShapeExpression] | None = None,
) -> int:
    if value.op == "const":
        assert value.value is not None
        return value.value
    active = set() if active is None else active
    if value in active:
        return _bounded(value, point)
    active.add(value)
    try:
        if value.op == "dimension":
            assert value.tensor is not None and value.axis is not None
            key = value.tensor, value.axis, point
            if key not in state.dimensions:
                info = state.inference.tensors[value.tensor]
                actual = None if info.shape is None else info.shape[value.axis]
                state.dimensions[key] = (
                    _bounded(value, point)
                    if actual is None or actual == value
                    else _evaluate(actual, point, state, active)
                )
            return state.dimensions[key]
        if value.op == "scalar":
            assert value.tensor is not None
            values = state.inference.tensors[value.tensor].values
            index = value.axis or 0
            if values is not None and index < len(values) and values[index] != value:
                return _evaluate(values[index], point, state, active)
            return _bounded(value, point)
        if value.op == "cache_length":
            return _bounded(value, point)
        left, right = (
            _evaluate(argument, point, state, active) for argument in value.args
        )
        if value.op in {"floor_div", "ceil_div", "mod"} and right == 0:
            raise ValueError("benchmark shape expression divides by zero")
        if value.op == "add":
            return left + right
        if value.op == "sub":
            return left - right
        if value.op == "mul":
            return left * right
        if value.op == "floor_div":
            return left // right
        if value.op == "ceil_div":
            return -(-left // right)
        if value.op == "mod":
            return left % right
        if value.op == "min":
            return min(left, right)
        if value.op == "max":
            return max(left, right)
        if value.op == "broadcast":
            if left != right and left != 1 and right != 1:
                raise ValueError(
                    "benchmark point has incompatible broadcast dimensions"
                )
            return right if left == 1 else left
        if value.op == "broadcast_stride":
            return 0 if left == 1 else right
        raise ValueError(f"unsupported benchmark expression {value.op!r}")
    finally:
        active.remove(value)


def _parameters(
    candidate: _KernelMatch, point: int, state: _TuningState
) -> dict[str, _ConcreteParameter]:
    def scalar(value: float | _ShapeExpression) -> int | float:
        return (
            _evaluate(value, point, state)
            if isinstance(value, _ShapeExpression)
            else value
        )

    return {
        name: tuple(scalar(item) for item in value)
        if isinstance(value, tuple)
        else scalar(value)
        for name, value in candidate.parameters.items()
    }


def _shape(tensor: Tensor, point: int, state: _TuningState) -> tuple[int, ...]:
    shape = state.inference.tensors[tensor].shape
    if shape is None:
        raise ValueError("benchmark tensor has unresolved rank")
    result = tuple(_evaluate(dimension, point, state) for dimension in shape)
    if any(dimension < 0 for dimension in result):
        raise ValueError("benchmark tensor has negative dimensions")
    return result


def _base(tensor: Tensor) -> Tensor:
    while tensor._producer is not None and isinstance(
        tensor._producer.args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs)
    ):
        tensor = tensor._producer.args.x
    return tensor


def _weight(tensor: Tensor, state: _TuningState) -> _WeightBinding | None:
    tensor = _base(tensor)
    if tensor in state.weights.bindings:
        return state.weights.bindings[tensor]
    producer = tensor._producer
    if producer is not None and isinstance(producer.args, _CastArgs):
        return _weight(producer.args.x, state)
    if producer is not None and isinstance(producer.args, _ConstructArgs):
        return _weight(
            producer.regions[0].outputs[producer.outputs.index(tensor)], state
        )
    return None


def _mapped_view(tensor: Tensor, state: _TuningState) -> bool:
    producer = tensor._producer
    if producer is None:
        return True
    args = producer.args
    if isinstance(args, _ReshapeArgs) and producer not in state.views:
        return False
    if isinstance(args, (_ReshapeArgs, _PermuteArgs, _BroadcastArgs, _CastArgs)):
        return _mapped_view(args.x, state)
    return True


def _indices(
    tensor: Tensor, candidate: _KernelMatch, point: int, state: _TuningState
) -> NDArray[np.generic] | None:
    args = candidate.anchor.args
    if not isinstance(args, (_GatherArgs, _ScatterArgs)) or tensor is not args.indices:
        return None
    shape = _shape(tensor, point, state)
    source = _shape(args.x, point, state)
    if not shape or shape[-1] > len(source):
        raise ValueError("benchmark coordinates have invalid shape")
    width = shape[-1]
    count = prod(shape[:-1])
    if any(dimension == 0 for dimension in source[:width]) and count:
        raise ValueError(
            "benchmark cannot sample active coordinates from an empty axis"
        )
    values = np.empty((count, width), dtype=np.int64)
    if isinstance(args, _ScatterArgs) and args.mode == "replace":
        available = prod(source[:width])
        if count > available:
            raise ValueError(
                "benchmark scatter replacement lacks enough unique coordinates"
            )
        if width:
            flat = np.arange(count, dtype=np.int64)
            for axis in reversed(range(width)):
                values[:, axis] = flat % source[axis]
                flat //= source[axis]
    else:
        for axis in range(width):
            values[:, axis] = state.rng.integers(0, source[axis], size=count)
    return values.reshape(shape)


def _fixture(
    tensor: Tensor,
    candidate: _KernelMatch,
    point: int,
    state: _TuningState,
    required: int,
) -> tuple[bytes | _BenchmarkInput, dict[str, _JSON]]:
    dtype = state.inference.tensors[tensor].dtype
    if dtype is None:
        raise ValueError("benchmark tensor has unresolved dtype")
    binding = _weight(tensor, state)
    if binding is not None:
        if not _mapped_view(tensor, state):
            raise ValueError(
                "benchmark weight needs a non-contiguous materialized transformation"
            )
        record = next(
            (
                value
                for value in state.manifest.entries.values()
                if value.dtype == dtype
                and (
                    value.logical == binding.logical or binding.logical in value.aliases
                )
            ),
            None,
        )
        if record is None:
            raise ValueError(
                f"benchmark lacks real weight representation {binding.logical}:{dtype}"
            )
        if record.size < required:
            raise ValueError(
                "real benchmark weight does not cover its native footprint"
            )
        return _BenchmarkInput(
            state.bundle, state.manifest.data_start + record.offset, record.size
        ), {
            "source": "weight",
            "logical": binding.logical,
            "dtype": dtype,
            "shape": list(record.shape),
            "bytes": record.size,
        }
    base = _base(tensor)
    shape = _shape(base, point, state)
    estimated = sum(size for _, size in _codec._payload_sections(dtype, shape))
    if max(estimated, required) * 3 > _MAX_BYTES:
        raise ValueError("benchmark input preparation exceeds the memory budget")
    info = state.inference.tensors[base]
    producer = base._producer
    data: NDArray[np.generic]
    source = "normal"
    if (
        producer is not None
        and isinstance(producer.args, _ConstantArgs)
        and not isinstance(producer.args.value, Weight)
    ):
        data = np.asarray(producer.args.value)
        source = "constant"
    elif info.values is not None and len(info.values) == prod(shape):
        data = np.asarray(
            [_evaluate(value, point, state) for value in info.values], dtype=np.int64
        ).reshape(shape)
        source = "shape_or_scalar"
    elif dtype == "bool":
        data = state.rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.bool_)
        source = "boolean"
    elif dtype.startswith(("int", "uint")):
        coordinates = _indices(tensor, candidate, point, state)
        if coordinates is not None:
            data = coordinates.reshape(shape)
            source = "valid_coordinates"
        else:
            lower, upper = state.inference.value_ranges.get(base, (0, 127))
            low = max(
                0 if dtype.startswith("uint") else -3, lower if lower is not None else 0
            )
            high = min(127, upper if upper is not None else 127)
            if (
                isinstance(candidate.anchor.args, _BinaryArgs)
                and candidate.anchor.op in {"floor_div", "mod"}
                and tensor is candidate.anchor.args.b
            ):
                low = max(1, low)
            if high < low:
                raise ValueError("benchmark integer fixture has an empty range")
            data = state.rng.integers(
                low,
                high,
                size=shape,
                dtype=np.uint64 if dtype == "uint64" else np.int64,
                endpoint=True,
            )
            source = "bounded_integer"
    else:
        data = state.rng.standard_normal(shape, dtype=np.float32)
        if candidate.anchor.op in {"sqrt", "log"}:
            data = np.abs(data) + np.float32(1e-3)
            source = "positive_normal"
    encoded = _codec._encode(data, dtype)[0].tobytes()
    if len(encoded) < required:
        raise ValueError(
            "benchmark fixture does not cover the strided native footprint"
        )
    return encoded, {
        "source": source,
        "dtype": dtype,
        "shape": list(shape),
        "bytes": len(encoded),
    }


def _invocation(
    candidate: _KernelMatch,
    point: int,
    state: _TuningState,
    fixtures: dict[Tensor, bytes | _BenchmarkInput],
) -> tuple[_Invocation, list[dict[str, _JSON]]]:
    parameters = _parameters(candidate, point, state)
    footprint = _requirements(candidate.kernel.parameter_schema, parameters)
    comparison_bytes = sum(
        prod(_shape(tensor, point, state)) * 64 for tensor in candidate.outputs
    )
    if (
        sum(footprint.inputs) * 3 + footprint.scratch * 2 + comparison_bytes
        > _MAX_BYTES
    ):
        raise ValueError(
            "benchmark native footprint exceeds the preparation/reference memory budget"
        )
    inputs = []
    descriptions = []
    for tensor, size in zip(candidate.inputs, footprint.inputs, strict=True):
        fixture_key = _base(tensor)
        if fixture_key not in fixtures:
            payload, description = _fixture(tensor, candidate, point, state, size)
            fixtures[fixture_key] = payload
        else:
            payload = fixtures[fixture_key]
            description = {
                "source": "shared_reference_input",
                "bytes": payload.size
                if isinstance(payload, _BenchmarkInput)
                else len(payload),
            }
        inputs.append(payload)
        descriptions.append(description)
    outputs = []
    for tensor in candidate.outputs:
        dtype = state.inference.tensors[tensor].dtype
        if dtype is None:
            raise ValueError("benchmark output dtype is unresolved")
        outputs.append(_BenchmarkOutput(dtype, _shape(tensor, point, state)))
    return _Invocation(
        candidate.kernel.library,
        candidate.kernel.name,
        parameters,
        tuple(inputs),
        tuple(outputs),
        state.request.threads,
    ), descriptions


def _signature(candidate: _KernelMatch, point: int, state: _TuningState) -> Hashable:
    parameters = _parameters(candidate, point, state)
    return (
        candidate.kernel.library,
        candidate.kernel.name,
        tuple(sorted(parameters.items())),
        tuple(
            (state.inference.tensors[value].dtype, _shape(_base(value), point, state))
            for value in candidate.inputs
        ),
        tuple(
            (state.inference.tensors[value].dtype, _shape(value, point, state))
            for value in candidate.outputs
        ),
    )


def _reference(
    candidate: _KernelMatch, candidates: tuple[_KernelMatch, ...]
) -> _KernelMatch | None:
    def boundary(value: _KernelMatch) -> set[Tensor]:
        result = set()
        for tensor in value.inputs:
            base = _base(tensor)
            producer = base._producer
            if (
                producer is None
                or not isinstance(producer.args, _ConstantArgs)
                or isinstance(producer.args.value, Weight)
            ):
                result.add(base)
        return result

    expected = boundary(candidate)
    return next(
        (
            value
            for value in candidates
            if value.kernel.backend == "scalar"
            and boundary(value) == expected
            and value.outputs == candidate.outputs
        ),
        None,
    )


def _priority(
    candidates: tuple[_KernelMatch, ...], state: _TuningState
) -> tuple[int, int, int]:
    weighted = 0
    for candidate in candidates:
        for tensor in candidate.inputs:
            binding = _weight(tensor, state)
            if binding is not None:
                weighted = max(
                    weighted,
                    sum(
                        size
                        for _, size in _codec._payload_sections(
                            binding.dtype, binding.source.shape
                        )
                    ),
                )
    high_level = any(
        any(
            name in value.kernel.name.lower()
            for name in ("linear", "matmul", "attention", "norm")
        )
        for value in candidates
    )
    return (
        int(high_level),
        weighted,
        max((len(value.covered) for value in candidates), default=0),
    )


_RESULT = TypeAdapter(_BenchmarkResult)
_PROFILE_ROOT = Path(__file__).resolve().parents[4] / "build/kernelbench/profiles"


def _cpu_key() -> str:
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        lines = [platform.processor()]
    fields = sorted(
        {
            line.strip()
            for line in lines
            if line.partition(":")[0].strip()
            in {
                "vendor_id",
                "model name",
                "cpu family",
                "model",
                "stepping",
                "flags",
                "Features",
                "CPU implementer",
                "CPU part",
            }
        }
    )
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = list(range(os.cpu_count() or 1))
    return json.dumps(
        [platform.machine(), platform.release(), fields, affinity],
        separators=(",", ":"),
    )


def _code_key() -> str:
    directory = Path(__file__).resolve().parent
    sources = [
        Path(__file__),
        directory / "benchmark.py",
        directory / "bench_dataclasses.py",
        directory / "native.c",
        directory / "native.h",
        directory.parent / "artifacts/parameters.py",
        directory.parent / "dtype.py",
        directory.parent / "kernels/bindings.py",
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _json_key(value: object) -> _JSON:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float)):
        return _number(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return [_json_key(item) for item in value]
    raise TypeError("unsupported native profile signature value")


def _profile_key(
    candidate: _KernelMatch,
    reference: _KernelMatch | None,
    signature: Hashable,
    state: _TuningState,
    cpu: str,
    code: str,
    libraries: dict[Path, str],
) -> str:
    for kernel in (
        (candidate.kernel,)
        if reference is None
        else (candidate.kernel, reference.kernel)
    ):
        if kernel.library not in libraries:
            libraries[kernel.library] = hashlib.sha256(
                kernel.library.read_bytes()
            ).hexdigest()
    description = {
        "version": 1,
        "cpu": cpu,
        "threads": state.request.threads,
        "code": code,
        "library": libraries[candidate.kernel.library],
        "reference": None
        if reference is None
        else [reference.kernel.name, libraries[reference.kernel.library]],
        "signature": _json_key(signature),
    }
    return hashlib.sha256(
        json.dumps(
            description, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load_profile(key: str) -> _BenchmarkResult | None:
    try:
        profile = _RESULT.validate_json((_PROFILE_ROOT / f"{key}.json").read_bytes())
        if (
            len(profile.samples) != 3
            or any(
                not isfinite(value) or value < 0
                for value in (
                    *profile.samples,
                    profile.mean,
                    profile.median,
                    profile.variance,
                    profile.maximum_error,
                    profile.normalized_error,
                )
            )
            or profile.allocated_bytes < 0
        ):
            return None
        return profile
    except (OSError, ValueError, TypeError):
        return None


def _save_profile(key: str, profile: _BenchmarkResult) -> None:
    if not all(
        isfinite(value)
        for value in (
            *profile.samples,
            profile.mean,
            profile.median,
            profile.variance,
            profile.maximum_error,
            profile.normalized_error,
        )
    ):
        return
    temporary: Path | None = None
    try:
        _PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=_PROFILE_ROOT, prefix=".profile-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_RESULT.dump_json(profile))
        temporary.replace(_PROFILE_ROOT / f"{key}.json")
    except OSError:
        pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _increment(record: dict[str, _JSON], operator: int) -> None:
    uses = record["uses"]
    assert isinstance(uses, int)
    record["uses"] = uses + 1
    operators = record["operators"]
    assert isinstance(operators, list)
    if len(operators) < 8 and operator not in operators:
        operators.append(operator)


def _tune(
    matches: _MatchSet,
    inference: _Inference,
    weights: _WeightIndex,
    manifest: _WeightManifest,
    bundle: Path,
    request: _CompileRequest,
) -> _Tuning:
    """Measure unique native workloads; share verified costs across model layers."""
    result = _Tuning()
    points = _representative_tokens(request.depth)
    if not points:
        return result
    state = _TuningState(
        inference,
        weights,
        manifest,
        bundle.resolve(),
        request,
        np.random.default_rng(0),
        views=frozenset(matches.views),
    )
    cpu, code = _cpu_key(), _code_key()
    libraries: dict[Path, str] = {}
    keys: dict[tuple[Hashable, str | None], str] = {}
    cache: dict[str, _BenchmarkResult] = {}
    records: dict[tuple[Hashable, int], dict[str, _JSON]] = {}
    rejected: set[tuple[_Operator, str]] = set()
    aggregated: dict[_Operator, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    groups = sorted(
        enumerate(matches.candidates.items()),
        key=lambda item: _priority(item[1][1], state),
        reverse=True,
    )
    for operator_index, (operator, candidates) in groups:
        ordered = sorted(
            candidates, key=lambda value: -len(value.kernel.capabilities.hardware)
        )
        for point in points:
            costs = result.per_workload.setdefault(point, {})
            for candidate in ordered:
                signature: Hashable | None = None
                record_key: tuple[Hashable, int] | None = None
                record: dict[str, _JSON] | None = None
                try:
                    reference = _reference(candidate, candidates)
                    if candidate.kernel.backend != "scalar" and reference is None:
                        raise ValueError(
                            "no equivalent scalar candidate is available for numerical validation"
                        )
                    signature = _signature(candidate, point, state)
                    cache_identity = (
                        signature,
                        reference.kernel.name
                        if candidate.kernel.backend != "scalar"
                        and reference is not None
                        else None,
                    )
                    key = keys.get(cache_identity)
                    if key is None:
                        key = _profile_key(
                            candidate,
                            reference if candidate.kernel.backend != "scalar" else None,
                            signature,
                            state,
                            cpu,
                            code,
                            libraries,
                        )
                        keys[cache_identity] = key
                    profile = cache.get(key)
                    cache_source = "memory"
                    if profile is None:
                        profile = _load_profile(key)
                        cache_source = "persistent"
                        if profile is not None:
                            cache[key] = profile
                    record_key = key, point
                    record = records.get(record_key)
                    if record is None:
                        parameters = _parameters(candidate, point, state)
                        record = {
                            "tokens": point,
                            "operator": operator_index,
                            "operators": [operator_index],
                            "uses": 1,
                            "op": operator.op,
                            "kernel": candidate.kernel.name,
                            "backend": candidate.kernel.backend,
                            "threads": request.threads,
                            "profile_key": key,
                            "parameters": {
                                name: list(value) if isinstance(value, tuple) else value
                                for name, value in parameters.items()
                            },
                        }
                        records[record_key] = record
                        result.workloads.append(record)
                    else:
                        _increment(record, operator_index)
                    if profile is None:
                        fixtures: dict[Tensor, bytes | _BenchmarkInput] = {}
                        invocation, descriptions = _invocation(
                            candidate, point, state, fixtures
                        )
                        golden = None
                        if (
                            candidate.kernel.backend != "scalar"
                            and reference is not None
                        ):
                            golden, _ = _invocation(reference, point, state, fixtures)
                        profile = _benchmark(
                            invocation,
                            golden=golden,
                            max_bytes=_MAX_BYTES,
                        )
                        cache[key] = profile
                        _save_profile(key, profile)
                        cache_source = "measured"
                        inputs: list[_JSON] = [
                            description for description in descriptions
                        ]
                        record["inputs"] = inputs
                        if (
                            reference is not None
                            and candidate.kernel.backend != "scalar"
                            and len(profile.reference_samples) == 3
                        ):
                            reference_signature = _signature(reference, point, state)
                            reference_key = _profile_key(
                                reference,
                                None,
                                reference_signature,
                                state,
                                cpu,
                                code,
                                libraries,
                            )
                            keys[reference_signature, None] = reference_key
                            assert golden is not None
                            footprint = _requirements(
                                reference.kernel.parameter_schema, golden.parameters
                            )
                            size = (
                                sum(
                                    payload.size
                                    if isinstance(payload, _BenchmarkInput)
                                    else len(payload)
                                    for payload in golden.inputs
                                )
                                + sum(footprint.outputs)
                                + footprint.scratch
                            )
                            samples = profile.reference_samples
                            scalar_profile = _BenchmarkResult(
                                samples,
                                mean(samples),
                                median(samples),
                                pvariance(samples),
                                True,
                                0.0,
                                size,
                            )
                            cache[reference_key] = scalar_profile
                            _save_profile(reference_key, scalar_profile)
                    if not profile.correct:
                        rejected.add((operator, candidate.kernel.name))
                        costs.setdefault(operator, {})[candidate.kernel.name] = float(
                            "inf"
                        )
                        record.update(
                            status="rejected",
                            correct=False,
                            maximum_error=profile.maximum_error
                            if isfinite(profile.maximum_error)
                            else None,
                            normalized_error=profile.normalized_error
                            if isfinite(profile.normalized_error)
                            else None,
                            reason="candidate disagrees with equivalent scalar reference",
                        )
                        continue
                    if not isfinite(profile.median) or profile.median < 0:
                        raise ValueError("native profile contains an invalid timing")
                    costs.setdefault(operator, {})[candidate.kernel.name] = (
                        profile.median
                    )
                    aggregated[operator][candidate.kernel.name].append(profile.median)
                    parameters = _parameters(candidate, point, state)
                    count = parameters.get(
                        "tokens", parameters.get("queries", parameters.get("m", 1))
                    )
                    tokens = (
                        int(count)
                        if isinstance(count, (int, float)) and count > 0
                        else 1
                    )
                    if record.get("status") != "measured":
                        record.update(
                            status="measured",
                            cache=cache_source,
                            samples_seconds=list(profile.samples),
                            mean_seconds=profile.mean,
                            median_seconds=profile.median,
                            variance_seconds=profile.variance,
                            seconds_per_token=profile.median / tokens,
                            correct=True,
                            maximum_error=profile.maximum_error
                            if isfinite(profile.maximum_error)
                            else None,
                            normalized_error=profile.normalized_error
                            if isfinite(profile.normalized_error)
                            else None,
                            validation="scalar_reference"
                            if candidate.kernel.backend != "scalar"
                            else "assumed_scalar",
                            allocated_bytes=profile.allocated_bytes,
                        )
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    OSError,
                    OverflowError,
                    TimeoutError,
                ) as error:
                    fallback: Hashable = (
                        candidate.kernel.name,
                        tuple(
                            (
                                inference.tensors[value].dtype,
                                inference.tensors[value].maximum_shape,
                            )
                            for value in candidate.inputs
                        ),
                        str(error),
                    )
                    failed_key = record_key or (
                        signature if signature is not None else fallback,
                        point,
                    )
                    previous = records.get(failed_key)
                    if previous is not None and previous is not record:
                        _increment(previous, operator_index)
                    else:
                        if record is None:
                            record = {
                                "tokens": point,
                                "operator": operator_index,
                                "operators": [operator_index],
                                "uses": 1,
                                "op": operator.op,
                                "kernel": candidate.kernel.name,
                                "backend": candidate.kernel.backend,
                                "threads": request.threads,
                            }
                            records[failed_key] = record
                            result.workloads.append(record)
                        record.update(status="skipped", reason=str(error))
    result.costs = {
        operator: {name: mean(samples) for name, samples in values.items()}
        for operator, values in aggregated.items()
    }
    for operator, kernel in rejected:
        result.costs.setdefault(operator, {})[kernel] = float("inf")
        for costs in result.per_workload.values():
            costs.setdefault(operator, {})[kernel] = float("inf")
    return result
