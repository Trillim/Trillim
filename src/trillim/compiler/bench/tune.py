from __future__ import annotations

from collections.abc import Hashable
from math import prod
from pathlib import Path
from statistics import mean
from time import monotonic

import numpy as np
from numpy.typing import NDArray

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
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from .. import dtype as _codec
from ..artifacts.artifact_dataclasses import _JSON
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import _Inference, _ShapeExpression
from ..compiler_pydantic_models import _CompileRequest
from ..kernels.build import _build_cpu
from ..kernels.kernel_dataclasses import _ParameterValue
from ..passes.match import _shape_rewriter
from ..passes.passes_dataclasses import _KernelMatch, _MatchSet, _Selection
from ..passes.select import _INTRINSICS, _anchors, _greedy, _specialization
from ..weights.packing import _repack_row_int8
from ..weights.weights_dataclasses import _WeightBinding, _WeightIndex
from ..weights.weights_pydantic_models import _WeightManifest
from .bench_dataclasses import (
    _BenchmarkInput,
    _BenchmarkOutput,
    _ConcreteParameter,
    _Invocation,
    _Tuning,
    _TuningRegion,
    _TuningState,
)
from .benchmark import _MAX_BYTES, _benchmark, _representative_tokens

_LOW_SECONDS = 120.0
_HIGH_SECONDS = 300.0


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
    key = value, point, state.history
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
            key = value.tensor, value.axis, point, state.history
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
            return _bounded(value, state.history)
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
    anchor = candidate.anchor.args
    args: _GatherArgs | _ScatterArgs | None
    if isinstance(anchor, (_GatherArgs, _ScatterArgs)) and tensor is anchor.indices:
        args = anchor
    else:
        args = next(
            (
                operator.args
                for operator in candidate.covered
                if isinstance(operator.args, _GatherArgs)
                and tensor is operator.args.indices
            ),
            None,
        )
    if args is None:
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
        packing = candidate.packings.get(tensor, "canonical")
        records = [
            value
            for value in state.manifest.entries.values()
            if value.dtype == dtype
            and (value.logical == binding.logical or binding.logical in value.aliases)
        ]
        if not records:
            raise ValueError(
                f"benchmark lacks real weight representation {binding.logical}:{dtype}"
            )
        record = next(
            (value for value in records if value.layout == packing), records[0]
        )
        if record.size < required:
            raise ValueError(
                "real benchmark weight does not cover its native footprint"
            )
        payload: bytes | _BenchmarkInput = _BenchmarkInput(
            state.bundle, state.manifest.data_start + record.offset, record.size
        )
        if record.layout != packing:
            with state.bundle.open("rb") as stream:
                stream.seek(state.manifest.data_start + record.offset)
                packed_bytes = np.frombuffer(stream.read(record.size), dtype=np.uint8)
            if dtype != "row_int8":
                raise ValueError("benchmark cannot repack this logical dtype")
            payload = _repack_row_int8(
                packed_bytes, tuple(record.shape), record.layout, packing
            ).tobytes()
        return payload, {
            "source": "weight",
            "logical": binding.logical,
            "dtype": dtype,
            "packing": packing,
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
    fixtures: dict[tuple[Tensor, str], bytes | _BenchmarkInput],
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
        fixture_key = (_base(tensor), candidate.packings.get(tensor, "canonical"))
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
    family = candidate.kernel.name.removeprefix(
        f"trillim_{candidate.kernel.backend}_"
    ).removeprefix("scalar_")
    references = [
        value
        for value in candidates
        if value.kernel.backend == "scalar"
        and boundary(value) == expected
        and value.outputs == candidate.outputs
    ]
    return max(
        references,
        key=lambda value: (
            value.kernel.name == f"trillim_scalar_{family}",
            _specialization(value),
        ),
        default=None,
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
        operator.op in {"matmul", "conv", "conv_transpose", "softmax"}
        for candidate in candidates
        for operator in candidate.covered
    )
    return (
        int(high_level),
        weighted,
        max((len(value.covered) for value in candidates), default=0),
    )


def _regions(
    call: _KernelMatch,
    matches: _MatchSet,
    operators: tuple[_Operator, ...],
    *,
    high: bool,
) -> list[tuple[_KernelMatch, ...]]:
    choices: list[tuple[_KernelMatch, ...]] = [
        (candidate,)
        for candidate in matches.candidates.get(call.anchor, ())
        if candidate.covered == call.covered and candidate != call
    ]
    candidates = {
        operator: tuple(
            candidate
            for candidate in matches.candidates.get(operator, ())
            if candidate.covered < call.covered
        )
        for operator in call.covered
    }
    local = tuple(operator for operator in operators if operator in call.covered)
    cuts = [call.anchor] if isinstance(call.anchor.args, _CastArgs) else []
    if high:
        cuts.extend(
            operator
            for operator in local
            if operator is not call.anchor
            and not isinstance(operator.args, _INTRINSICS)
            and operator not in matches.views
        )
    partitions: set[frozenset[frozenset[_Operator]]] = set()
    for cut in cuts[:16]:
        neighborhood = _MatchSet(
            candidates={
                operator: tuple(
                    candidate
                    for candidate in group
                    if cut not in candidate.covered or candidate.anchor is cut
                )
                for operator, group in candidates.items()
            },
            views=matches.views,
        )
        try:
            selected = _greedy(local, neighborhood, None)
        except ValueError:
            continue
        missing = call.covered - selected.covered
        if any(
            operator not in matches.views and not isinstance(operator.args, _INTRINSICS)
            for operator in missing
        ):
            continue
        partition = frozenset(value.covered for value in selected.calls.values())
        if partition in partitions or not selected.calls:
            continue
        partitions.add(partition)
        choices.append(tuple(selected.calls.values()))
        if len(partitions) >= (4 if high else 1):
            break
    return choices


def _region_key(
    region: tuple[_KernelMatch, ...], original: _KernelMatch, state: _TuningState
) -> Hashable:
    tensors = dict.fromkeys((*original.inputs, *original.outputs))
    slots = {tensor: index for index, tensor in enumerate(tensors)}

    def parameter(value: _ParameterValue) -> Hashable:
        if isinstance(value, _ShapeExpression):
            return value if state.rewrite is None else state.rewrite(value)
        if isinstance(value, tuple):
            return tuple(parameter(item) for item in value)
        return value

    def binding(tensor: Tensor, candidate: _KernelMatch) -> Hashable:
        if tensor not in slots:
            slots[tensor] = len(slots)
        weight = _weight(tensor, state)
        info = state.inference.tensors[tensor]
        return (
            slots[tensor],
            info.dtype,
            None if info.shape is None else parameter(info.shape),
            info.maximum_shape,
            None if info.values is None else parameter(info.values),
            state.inference.value_ranges.get(tensor),
            candidate.packings.get(tensor, "canonical"),
            None
            if weight is None
            else (weight.dtype, weight.source.method, weight.source.linear_class),
        )

    # Substitute only shape equalities proved by the selected loop matches.
    # Unrelated symbols and their bounds remain distinct.
    return tuple(
        (
            candidate.kernel.library,
            candidate.kernel.name,
            candidate.kernel.backend,
            candidate.kernel.capabilities,
            tuple(
                (name, parameter(value))
                for name, value in sorted(candidate.parameters.items())
            ),
            tuple(sorted(operator.op for operator in candidate.covered)),
            tuple(binding(tensor, candidate) for tensor in candidate.inputs),
            tuple(binding(tensor, candidate) for tensor in candidate.outputs),
        )
        for candidate in region
    )


def _region_seconds(
    region: tuple[_KernelMatch, ...],
    references: _MatchSet,
    point: int,
    state: _TuningState,
    harness: Path,
    fixtures: dict[tuple[Tensor, str], bytes | _BenchmarkInput],
    deadline: float,
    result: _Tuning,
    *,
    incumbent: bool = False,
) -> float:
    elapsed = 0.0
    for candidate in region:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("tuning time budget exhausted")
        reference = _reference(
            candidate, references.candidates.get(candidate.anchor, ())
        )
        if candidate.kernel.backend != "scalar" and reference is None:
            raise ValueError("no equivalent scalar reference")
        invocation, _ = _invocation(candidate, point, state, fixtures)
        golden = None
        if candidate.kernel.backend != "scalar" and reference is not None:
            golden, _ = _invocation(reference, point, state, fixtures)
        profile = _benchmark(
            invocation,
            harness=harness,
            golden=golden,
            timeout=min(10.0, remaining),
            max_bytes=_MAX_BYTES,
        )
        result.measured += 1
        if not profile.correct:
            if incumbent:
                raise ArithmeticError(
                    f"incumbent {candidate.kernel.name} exceeds its numerical error "
                    f"budget at {point} rows: maximum={profile.maximum_error:g}, "
                    f"normalized={profile.normalized_error:g}"
                )
            raise ValueError("candidate exceeds its numerical error budget")
        elapsed += profile.median
    return elapsed


def _improve(
    graph: Graph,
    matches: _MatchSet,
    state: _TuningState,
    harness: Path,
    result: _Tuning,
    *,
    high: bool,
    seconds: float,
) -> None:
    request = state.request
    native = _MatchSet(
        candidates={
            operator: tuple(
                candidate
                for candidate in group
                if candidate.kernel.backend == request.backend
            )
            for operator, group in matches.candidates.items()
        },
        views=matches.views,
    )
    assert graph._region is not None
    operators = tuple(_anchors(graph._region))
    points = _representative_tokens("high" if high else "low")
    deadline = monotonic() + seconds
    state.rewrite = _shape_rewriter(
        {
            tensor: shape
            for call in result.selection.calls.values()
            for tensor, shape in call.loop_dimensions.items()
        }
    )
    groups: dict[Hashable, list[_TuningRegion]] = {}
    for original in result.selection.calls.values():
        alternatives = _regions(original, native, operators, high=high)
        if not alternatives:
            continue
        choices = {
            _region_key(region, original, state): region
            for region in ((original,), *alternatives)
        }
        decision_key: Hashable = (
            request.threads,
            points,
            _region_key((original,), original, state),
            frozenset(choices),
        )
        groups.setdefault(decision_key, []).append(_TuningRegion(original, choices))
    ordered = sorted(
        groups.items(),
        key=lambda item: _priority((item[1][0].original,), state),
        reverse=True,
    )
    for key, group in ordered:
        if monotonic() >= deadline:
            break
        representative = group[0]
        initial = _region_key(
            (representative.original,), representative.original, state
        )
        current = state.decisions.get(key, initial)
        representatives = group[:1] if len(group) == 1 else [group[0], group[-1]]
        for alternative in representative.choices if key not in state.decisions else ():
            if alternative == initial:
                continue
            ratios = []
            try:
                for member in representatives:
                    for point in points:
                        state.history = (
                            min(request.max_input_size, 512) if point == 1 else 0
                        )
                        state.dimensions.clear()
                        fixtures: dict[tuple[Tensor, str], bytes | _BenchmarkInput] = {}
                        baseline = _region_seconds(
                            member.choices[current],
                            matches,
                            point,
                            state,
                            harness,
                            fixtures,
                            deadline,
                            result,
                            incumbent=True,
                        )
                        proposed = _region_seconds(
                            member.choices[alternative],
                            matches,
                            point,
                            state,
                            harness,
                            fixtures,
                            deadline,
                            result,
                        )
                        if baseline <= 0:
                            raise ValueError("invalid incumbent timing")
                        ratios.append(proposed / baseline)
                        if ratios[-1] > 1.03:
                            raise ValueError("candidate regresses a representative")
            except (
                ValueError,
                TypeError,
                RuntimeError,
                OSError,
                OverflowError,
                TimeoutError,
            ):
                continue
            if (
                len(ratios) != len(points) * len(representatives)
                or mean(ratios) >= 0.97
            ):
                continue
            current = alternative
        state.decisions[key] = current
        if current == initial:
            continue
        calls = dict(result.selection.calls)
        covered = set(result.selection.covered)
        for member in group:
            replacement = member.choices[current]
            del calls[member.original.anchor]
            calls.update((candidate.anchor, candidate) for candidate in replacement)
            covered.difference_update(member.original.covered)
            covered.update(*(candidate.covered for candidate in replacement))
        result.accepted += len(group)
        result.propagated += max(0, len(group) - len(representatives))
        result.selection = _Selection(calls, covered)


def _tune(
    graph: Graph,
    matches: _MatchSet,
    inference: _Inference,
    weights: _WeightIndex,
    manifest: _WeightManifest,
    bundle: Path,
    request: _CompileRequest,
    selection: _Selection,
    structural: _MatchSet,
) -> _Tuning:
    """Refine the incumbent with freshly measured, non-regressing region choices."""
    result = _Tuning(selection)
    if request.depth == "none":
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
    with _build_cpu() as harness:
        _improve(
            graph, structural, state, harness, result, high=False, seconds=_LOW_SECONDS
        )
        if request.depth == "high":
            _improve(
                graph,
                matches,
                state,
                harness,
                result,
                high=True,
                seconds=_HIGH_SECONDS - _LOW_SECONDS,
            )
    return result
