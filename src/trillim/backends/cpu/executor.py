from __future__ import annotations

import ctypes as c
import math
import mmap
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import cast
from weakref import finalize

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ...compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from ...compiler.artifacts.expressions import _byte_size
from ...compiler.artifacts.parameters import _pack, _requirements
from ...compiler.dtype import _NATIVE_DTYPES, _QUANTIZED_DTYPES, _decode, _encode
from ...compiler.kernels.bindings import _DTYPE_IDS
from ...compiler.kernels.build import _build_cpu
from ...compiler.kernels.catalog import _available_capabilities
from ...compiler.kernels.kernel_dataclasses import _ParameterSchema, _ParameterValue
from ...compiler.kernels.native import _read_capabilities, _read_schema
from ...compiler.passes.verify import _verify
from ...compiler.weights.bundle import _read_manifest
from .cpu_dataclasses import _Allocation, _PreparedCall
from .errors import CancelledError, ExecutionError


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("expected manifest object")
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError("expected manifest list")
    return value


def _text(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise TypeError("expected manifest string")
    return value


def _integer(value: JsonValue) -> int:
    if not isinstance(value, int):
        raise TypeError("expected manifest integer")
    return value


def _dtype(name: str) -> np.dtype[np.generic]:
    names = {"bf16": "uint16", "fp16": "float16", "fp32": "float32"}
    return np.dtype(names.get(name, name))


class _CPU:
    def __init__(self, directory: Path, threads: int | None) -> None:
        self._lifecycle = Lock()
        self.context: int | None = None
        self._finalizer: Callable[[], object] | None = None
        self.mappings: list[mmap.mmap] = []
        self.mapping_views: list[NDArray[np.uint8]] = []
        self.inputs: dict[str, NDArray[np.generic]] = {}
        self._calls: dict[int, _PreparedCall] = {}
        self._expressions: dict[int, Callable[[], int]] = {}
        self._expression_values: dict[str, int] = {}
        try:
            self._load(directory, threads)
        except BaseException:
            self.close()
            raise

    def _load(self, directory: Path, threads: int | None) -> None:
        self.directory = directory
        _verify(directory)
        self.manifest = _ExecutableManifest.model_validate_json(
            (directory / "executable.json").read_bytes()
        )
        self.weights = _read_manifest(directory / self.manifest.weights)
        if self.manifest.backend not in {"scalar", "x86"}:
            raise ValueError(f"unsupported runtime backend: {self.manifest.backend}")
        if sys.byteorder != "little":
            raise ValueError("compiled models require a little-endian host")
        threads = self.manifest.threads if threads is None else threads
        if not 1 <= threads <= 65536:
            raise ValueError("threads must be between 1 and 65536")
        self._base_manifest = self.manifest
        self.arena = self._allocation(self.manifest.arena_size)
        self.cache = self._allocation(self.manifest.cache_size)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.lengths = dict.fromkeys(self.manifest.caches, 0)
        self.weight_address = self.map_file(directory / self.manifest.weights)
        self.constant_address = self.map_file(directory / self.manifest.constants)
        self.kernels: dict[str, tuple[c.CDLL, _ParameterSchema, int]] = {}
        libraries: dict[str, c.CDLL] = {}
        self.harness = c.CDLL(str(_build_cpu()))
        self.harness.trillim_cpu_features.argtypes = []
        self.harness.trillim_cpu_features.restype = c.c_uint64
        available = _available_capabilities(self.manifest.backend)
        if self.manifest.backend == "x86":
            available |= int(self.harness.trillim_cpu_features())
        for name, metadata in self.manifest.kernels.items():
            filename = _text(metadata["library"])
            library = libraries.setdefault(
                filename, c.CDLL(str((directory / filename).resolve()))
            )
            symbol = _text(metadata["symbol"])
            required = _read_capabilities(library, symbol)
            if (
                required != _integer(metadata["capabilities"])
                or required & available != required
            ):
                raise ValueError(
                    f"{name}: kernel capabilities unavailable on this device"
                )
            pointer = c.cast(getattr(library, symbol), c.c_void_p).value
            assert pointer is not None
            self.kernels[name] = library, _read_schema(library, symbol), pointer
        self.harness.trillim_cpu_create.argtypes = [c.c_size_t]
        self.harness.trillim_cpu_create.restype = c.c_void_p
        self.harness.trillim_cpu_destroy.argtypes = [c.c_void_p]
        self.harness.trillim_cpu_destroy.restype = None
        self.harness.trillim_cpu_cancel.argtypes = [c.c_void_p]
        self.harness.trillim_cpu_cancel.restype = None
        self.native = self.harness.trillim_cpu_invoke
        self.native.argtypes = [
            c.c_void_p,
            c.c_void_p,
            c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p),
            c.c_void_p,
            c.c_void_p,
        ]
        self.native.restype = c.c_int32
        self.context = self.harness.trillim_cpu_create(threads)
        if not self.context:
            raise MemoryError("cannot create native pool")
        self._finalizer = finalize(self, self.harness.trillim_cpu_destroy, self.context)
        self.harness.trillim_cpu_enter.argtypes = [c.c_void_p]
        self.harness.trillim_cpu_enter.restype = c.c_int32
        self.harness.trillim_cpu_leave.argtypes = [c.c_void_p]
        self.harness.trillim_cpu_leave.restype = None
        self.harness.trillim_cpu_cancelled.argtypes = [c.c_void_p]
        self.harness.trillim_cpu_cancelled.restype = c.c_bool
        for name, metadata in self.manifest.caches.items():
            initial = metadata["initial"]
            kind = _text(metadata["dtype"])
            if initial or kind in _QUANTIZED_DTYPES:
                shape = tuple(_integer(item) for item in _list(metadata["shape"]))
                payload = _encode(np.full(shape, self.scalar_parameter(initial)), kind)[
                    0
                ]
                c.memmove(
                    self.cache.address + _integer(metadata["offset"]),
                    payload.ctypes.data,
                    payload.nbytes,
                )

    @staticmethod
    def _allocation(size: int) -> _Allocation:
        owner = c.create_string_buffer(size + 64)
        return _Allocation(owner, (c.addressof(owner) + 63) // 64 * 64, size)

    def map_file(self, path: Path) -> int:
        if not path.stat().st_size:
            return 0
        with path.open("rb") as stream:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        view = np.frombuffer(mapping, dtype=np.uint8)
        self.mappings.append(mapping)
        self.mapping_views.append(view)
        return int(view.ctypes.data)

    def close(self) -> None:
        self.cancel()
        with self._lifecycle:
            if self.context:
                assert self._finalizer is not None
                self._finalizer()
                self.context = None
        self.inputs.clear()
        self._calls.clear()
        self._expressions.clear()
        self.mapping_views.clear()
        for mapping in self.mappings:
            mapping.close()
        self.mappings.clear()
        self.arena = self._allocation(0)
        self.cache = self._allocation(0)

    def cancel(self) -> None:
        with self._lifecycle:
            if self.context:
                self.harness.trillim_cpu_cancel(self.context)

    def check_cancelled(self) -> None:
        if self.harness.trillim_cpu_cancelled(self.context):
            raise CancelledError("model execution cancelled")

    def bind_inputs(
        self, inputs: Mapping[str, NDArray[np.generic] | JsonValue]
    ) -> None:
        if inputs.keys() != self._base_manifest.inputs.keys():
            raise ValueError(
                f"expected model inputs: {', '.join(self._base_manifest.inputs)}"
            )
        self.manifest = self._base_manifest
        self.shapes.clear()
        self.inputs.clear()
        self._expression_values.clear()
        for name, value in inputs.items():
            record = self.manifest.buffers[self.manifest.inputs[name]]
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or len(array.shape) != len(
                record.maximum_shape
            ):
                raise ValueError(f"{name}: invalid numeric input or rank")
            for actual, maximum, fixed in zip(
                array.shape, record.maximum_shape, record.shape, strict=True
            ):
                if actual > maximum or isinstance(fixed, int) and actual != fixed:
                    raise ValueError(
                        f"{name}: input shape exceeds or differs from its compiled dimensions"
                    )
            native = _NATIVE_DTYPES.get(record.dtype)
            if native is not None and native.kind in "biu" and array.size:
                low, high = (
                    (0, 1)
                    if native.kind == "b"
                    else (np.iinfo(native.name).min, np.iinfo(native.name).max)
                )
                if (
                    not low <= array.min().item()
                    or not array.max().item() <= high
                    or array.dtype.kind == "f"
                    and np.any(array != np.trunc(array))
                ):
                    raise ValueError(f"{name}: input values do not fit {record.dtype}")
            self.shapes[self.manifest.inputs[name]] = array.shape
            self.inputs[name] = _encode(array, record.dtype)[0]
        if self.manifest.workload:
            source = self._base_manifest.inputs[_text(self.manifest.workload["input"])]
            count = self.shapes[source][_integer(self.manifest.workload["axis"])]
            for plan in self.manifest.plans:
                if plan.minimum_tokens <= count <= plan.maximum_tokens:
                    self.manifest = self.manifest.model_copy(
                        update={
                            "buffers": plan.buffers,
                            "program": plan.program,
                            "expressions": plan.expressions,
                        }
                    )
                    break

    def execute(self) -> dict[str, NDArray[np.generic]]:
        if self.harness.trillim_cpu_enter(self.context):
            raise ExecutionError("another model is already executing")
        try:
            self.check_cancelled()
            self.program(list(self.manifest.program))
            return {
                name: self.logical_array(tensor).copy()
                for name, tensor in self.manifest.outputs.items()
            }
        finally:
            self.harness.trillim_cpu_leave(self.context)

    def logical_array(self, name: str) -> NDArray[np.generic]:
        record = self.manifest.buffers[name]
        if record.dtype in _QUANTIZED_DTYPES:
            rows, columns = self.storage_shape(name)
            size = _byte_size(record.dtype, (rows, columns))
            if size > self.span(name):
                raise ExecutionError("packed tensor exceeds storage")
            raw = (c.c_ubyte * size).from_address(self.address(name))
            decoded = _decode(
                (np.frombuffer(raw, dtype=np.uint8), record.dtype), (rows, columns)
            )
            shape, strides = self.shape(name), self.strides(name)
            if (
                math.prod(shape)
                and 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(shape, strides, strict=True)
                )
                > decoded.size
            ):
                raise ExecutionError("packed tensor view exceeds storage")
            return np.lib.stride_tricks.as_strided(
                decoded, shape, tuple(stride * 4 for stride in strides), writeable=False
            )
        value = self.array(name)
        if record.dtype == "bf16":
            return _decode(
                (np.ascontiguousarray(value).view(np.uint8).reshape(-1), "bf16"),
                value.shape,
            )
        return value

    @staticmethod
    def _broadcast(a: int, b: int) -> int:
        if a != b and a != 1 and b != 1:
            raise ValueError("incompatible broadcast")
        return b if a == 1 else a

    def _expression_source(self, value: JsonValue) -> str:
        if isinstance(value, int):
            return repr(value)
        expression = _object(value)
        if "expr" in expression:
            return f"backend._named_value({_text(expression['expr'])!r})"
        operation = _text(expression["op"])
        if operation in {"dimension", "scalar"}:
            tensor = repr(_text(expression["tensor"]))
            axis = _integer(expression.get("axis", 0))
            if operation == "dimension":
                return f"backend.shape({tensor})[{axis}]"
            return f"int(backend.array({tensor}).flat[{axis}])"
        if operation == "cache_length":
            return f"backend.lengths[{_text(expression['cache'])!r}]"
        a, b = (self._expression_source(item) for item in _list(expression["args"]))
        symbols = {"add": "+", "sub": "-", "mul": "*", "floor_div": "//", "mod": "%"}
        if operation in symbols:
            return f"({a} {symbols[operation]} {b})"
        if operation == "ceil_div":
            return f"(-(-{a} // {b}))"
        if operation in {"min", "max"}:
            return f"{operation}({a}, {b})"
        if operation == "broadcast_stride":
            return f"(0 if {a} == 1 else {b})"
        if operation == "broadcast":
            return f"backend._broadcast({a}, {b})"
        raise ValueError(f"unknown expression {operation}")

    def _named_value(self, name: str) -> int:
        if name not in self._expression_values:
            self._expression_values[name] = self.evaluate(self.manifest.expressions[name])
        return self._expression_values[name]

    def evaluate(self, value: JsonValue) -> int:
        if isinstance(value, int):
            return value
        identity = id(value)
        function = self._expressions.get(identity)
        if function is None:
            # Only fixed operators and quoted tensor names enter generated code.
            source = "lambda: " + self._expression_source(value)
            function = cast(
                Callable[[], int],
                eval(
                    source,
                    {
                        "__builtins__": {},
                        "backend": self,
                        "int": int,
                        "min": min,
                        "max": max,
                    },
                ),
            )
            self._expressions[identity] = function
        return function()

    def shape(self, name: str) -> tuple[int, ...]:
        if name in self.shapes:
            return self.shapes[name]
        return tuple(self.evaluate(item) for item in self.manifest.buffers[name].shape)

    def strides(self, name: str) -> tuple[int, ...]:
        return tuple(
            self.evaluate(item) for item in self.manifest.buffers[name].strides
        )

    def address(self, name: str) -> int:
        buffer = self.manifest.buffers[name]
        if buffer.storage == "alias":
            assert buffer.source is not None
            return self.address(buffer.source)
        if buffer.storage == "arena":
            return self.arena.address + buffer.offset
        if buffer.storage == "constant":
            return self.constant_address + buffer.offset
        if buffer.storage == "weight":
            return self.weight_address + self.weights.data_start + buffer.offset
        if buffer.storage == "input":
            assert buffer.source is not None
            return int(self.inputs[buffer.source].ctypes.data)
        if buffer.storage == "cache":
            assert buffer.source is not None
            return (
                self.cache.address
                + _integer(self.manifest.caches[buffer.source]["offset"])
                + self.evaluate(buffer.byte_offset)
            )
        raise ValueError(f"unknown storage {buffer.storage}")

    def span(self, name: str) -> int:
        buffer = self.manifest.buffers[name]
        if buffer.storage == "alias":
            assert buffer.source is not None
            return self.span(buffer.source)
        if buffer.storage == "cache":
            assert buffer.source is not None
            return _integer(
                self.manifest.caches[buffer.source]["size"]
            ) - self.evaluate(buffer.byte_offset)
        if buffer.storage == "input":
            assert buffer.source is not None
            return self.inputs[buffer.source].nbytes
        return buffer.size

    def array(self, name: str) -> NDArray[np.generic]:
        native = _dtype(self.manifest.buffers[name].dtype)
        raw = (c.c_ubyte * max(1, self.span(name))).from_address(self.address(name))
        return np.ndarray(
            self.shape(name),
            dtype=native,
            buffer=raw,
            strides=tuple(stride * native.itemsize for stride in self.strides(name)),
        )

    def cache_array(self, name: str, *, capacity: bool = False) -> NDArray[np.generic]:
        metadata = self.manifest.caches[name]
        shape = tuple(_integer(item) for item in _list(metadata["shape"]))
        kind = _text(metadata["dtype"])
        native = _dtype(kind) if kind not in _QUANTIZED_DTYPES else np.dtype(np.float32)
        raw = (c.c_ubyte * max(1, _integer(metadata["size"]))).from_address(
            self.cache.address + _integer(metadata["offset"])
        )
        result = (
            _decode((np.frombuffer(raw, dtype=np.uint8), kind), shape)
            if kind in _QUANTIZED_DTYPES
            else np.ndarray(shape, dtype=native, buffer=raw)
        )
        axis = metadata["axis"]
        if not capacity and isinstance(axis, int):
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(0, self.lengths[name])
            result = result[tuple(slices)]
        return result

    def scalar_parameter(self, value: JsonValue) -> int | float:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, dict) and "float" in value:
            return float(_text(value["float"]))
        return self.evaluate(value)

    def parameter(self, value: JsonValue) -> _ParameterValue:
        if isinstance(value, list):
            return tuple(self.scalar_parameter(item) for item in value)
        return self.scalar_parameter(value)

    def invoke(
        self,
        identity: int,
        kernel: str,
        parameters: dict[str, _ParameterValue],
        inputs: list[tuple[int, int]],
        outputs: list[tuple[int, int]],
        scratch: dict[str, JsonValue],
    ) -> None:
        _, schema, pointer = self.kernels[kernel]
        prepared = self._calls.get(identity)
        if prepared is None or prepared.parameters != parameters:
            prepared = _PreparedCall(
                parameters, _pack(schema, parameters), _requirements(schema, parameters)
            )
            self._calls[identity] = prepared
        footprint = prepared.requirements
        for (address, capacity), size, alignment in zip(
            inputs, footprint.inputs, schema.input_alignments, strict=True
        ):
            if size > capacity or address % alignment:
                raise ValueError(f"{kernel}: input footprint/alignment invalid")
        for (address, capacity), size, alignment in zip(
            outputs, footprint.outputs, schema.output_alignments, strict=True
        ):
            if size > capacity or address % alignment:
                raise ValueError(f"{kernel}: output footprint/alignment invalid")
        scratch_address = self.arena.address + _integer(scratch["offset"])
        if (
            footprint.scratch > _integer(scratch["size"])
            or scratch_address % schema.scratch_alignment
        ):
            raise ValueError(f"{kernel}: scratch footprint/alignment invalid")
        source = (c.c_void_p * len(inputs))(*(address for address, _ in inputs))
        target = (c.c_void_p * len(outputs))(*(address for address, _ in outputs))
        status = self.native(
            self.context,
            pointer,
            target,
            source,
            scratch_address,
            prepared.packed.address,
        )
        if status == 1:
            raise CancelledError(f"{kernel}: execution cancelled")
        if status:
            raise ExecutionError(
                f"{kernel} failed with status {status}, parameters={parameters}"
            )

    def call(self, instruction: dict[str, JsonValue]) -> None:
        for destination, source in _object(instruction.get("shape_from", {})).items():
            self.shapes[destination] = self.shape(_text(source))
        inputs = [_text(item) for item in _list(instruction["inputs"])]
        outputs = [_text(item) for item in _list(instruction["outputs"])]
        for name in outputs:
            if name not in _object(instruction.get("shape_from", {})):
                self.shapes.pop(name, None)
                self.shapes[name] = self.shape(name)
        parameters = {
            name: self.parameter(value)
            for name, value in _object(instruction["parameters"]).items()
        }
        self.invoke(
            id(instruction),
            _text(instruction["kernel"]),
            parameters,
            [(self.address(name), self.span(name)) for name in inputs],
            [(self.address(name), self.span(name)) for name in outputs],
            _object(instruction["scratch"]),
        )

    def storage_shape(self, name: str) -> tuple[int, int]:
        buffer = self.manifest.buffers[name]
        if buffer.storage == "alias":
            assert buffer.source is not None
            return self.storage_shape(buffer.source)
        if buffer.storage == "cache":
            assert buffer.source is not None
            shape = tuple(
                _integer(item)
                for item in _list(self.manifest.caches[buffer.source]["shape"])
            )
        else:
            shape = self.shape(name)
        return (
            (shape[0], math.prod(shape[1:]))
            if len(shape) > 1
            else (1, math.prod(shape))
        )

    def scalar_copy(
        self,
        source: str,
        destination: str,
        transfer: dict[str, JsonValue],
        *,
        source_offset: int = 0,
        destination_offset: int = 0,
    ) -> None:
        src, dst = self.manifest.buffers[source], self.manifest.buffers[destination]
        rows, columns = self.storage_shape(source)
        itemsize = _dtype(dst.dtype).itemsize
        parameters: dict[str, _ParameterValue] = {
            "count": 1,
            "rank": 0,
            "shape": (),
            "strides": (),
            "source_offset": source_offset,
            "source_rows": rows,
            "source_columns": columns,
            "output_rows": 1,
            "output_columns": 1,
            "input_dtype": _DTYPE_IDS[src.dtype],
            "output_dtype": _DTYPE_IDS[dst.dtype],
        }
        self.invoke(
            id(transfer),
            _text(transfer["kernel"]),
            parameters,
            [(self.address(source), self.span(source))],
            [
                (
                    self.address(destination) + destination_offset * itemsize,
                    self.span(destination) - destination_offset * itemsize,
                )
            ],
            _object(transfer["scratch"]),
        )

    def reduce(self, instruction: dict[str, JsonValue]) -> None:
        values = [_text(item) for item in _list(instruction["values"])]
        initial = [_text(item) for item in _list(instruction["initial"])]
        parameters = [_text(item) for item in _list(instruction["parameters"])]
        results = [_text(item) for item in _list(instruction["results"])]
        outputs = [_text(item) for item in _list(instruction["outputs"])]
        axes = tuple(_integer(item) for item in _list(instruction["axes"]))
        shape = self.shape(values[0])
        kept = tuple(axis for axis in range(len(shape)) if axis not in axes)
        transfer = _object(instruction["transfer"])
        for output_index in np.ndindex(tuple(shape[axis] for axis in kept)):
            self.check_cancelled()
            for src, dst in zip(initial, parameters[: len(values)], strict=True):
                self.scalar_copy(src, dst, transfer)
            for reduce_index in np.ndindex(tuple(shape[axis] for axis in axes)):
                coordinates = dict(zip(kept, output_index, strict=True)) | dict(
                    zip(axes, reduce_index, strict=True)
                )
                for src, dst in zip(values, parameters[len(values) :], strict=True):
                    offset = sum(
                        coordinates[axis] * stride
                        for axis, stride in enumerate(self.strides(src))
                    )
                    self.scalar_copy(src, dst, transfer, source_offset=offset)
                self.program(_list(instruction["program"]))
                # The reducer's result tuple is simultaneous, including identity carries.
                snapshots = [
                    c.string_at(
                        self.address(name),
                        _dtype(self.manifest.buffers[name].dtype).itemsize,
                    )
                    for name in results
                ]
                for payload, destination in zip(
                    snapshots, parameters[: len(values)], strict=True
                ):
                    c.memmove(self.address(destination), payload, len(payload))
            coordinates = dict(zip(kept, output_index, strict=True))
            for src, dst in zip(parameters[: len(values)], outputs, strict=True):
                if instruction["keepdims"]:
                    offset = sum(
                        coordinates.get(axis, 0) * stride
                        for axis, stride in enumerate(self.strides(dst))
                    )
                else:
                    offset = sum(
                        index * stride
                        for index, stride in zip(
                            output_index, self.strides(dst), strict=True
                        )
                    )
                self.scalar_copy(src, dst, transfer, destination_offset=offset)

    def copy_array(
        self, destination: NDArray[np.generic], source: NDArray[np.generic]
    ) -> None:
        if np.shares_memory(destination, source):
            source = source.copy()
        if not destination.ndim:
            np.copyto(destination, source, casting="no")
            return
        rows = max(
            1,
            1024**2 // max(1, math.prod(destination.shape[1:]) * destination.itemsize),
        )
        for start in range(0, destination.shape[0], rows):
            self.check_cancelled()
            np.copyto(
                destination[start : start + rows],
                source[start : start + rows],
                casting="no",
            )

    def program(self, program: list[JsonValue]) -> None:
        for value in program:
            self.check_cancelled()
            self._expression_values.clear()
            instruction = _object(value)
            operation = _text(instruction["op"])
            if operation == "call":
                self.call(instruction)
            elif operation == "guard":
                a, b = (
                    self.evaluate(instruction["left"]),
                    self.evaluate(instruction["right"]),
                )
                predicate = _text(instruction["predicate"])
                valid = {
                    "eq": a == b,
                    "le": a <= b,
                    "ge": a >= b,
                    "divisible": b != 0 and a % b == 0,
                    "broadcast": a == b or a == 1 or b == 1,
                }[predicate]
                if not valid:
                    raise ValueError(
                        f"shape guard failed: {instruction['reason']} ({a}, {b})"
                    )
            elif operation == "views":
                for name in _list(instruction["outputs"]):
                    tensor = _text(name)
                    self.shapes.pop(tensor, None)
                    self.shapes[tensor] = self.shape(tensor)
            elif operation == "if":
                condition_value = bool(
                    self.array(_text(instruction["condition"])).item()
                )
                branch = _object(
                    _list(instruction["branches"])[0 if condition_value else 1]
                )
                self.program(_list(branch["program"]))
                self.program(_list(branch["copies"]))
            elif operation == "while":
                condition = _object(instruction["condition"])
                body = _object(instruction["body"])
                self.program(_list(instruction["initial_copies"]))
                maximum = _integer(instruction["maximum_iterations"])
                for iteration in range(maximum + 1):
                    self.check_cancelled()
                    self.program(_list(condition["copies"]))
                    self.program(_list(condition["program"]))
                    if not bool(self.array(_text(condition["result"])).item()):
                        break
                    if iteration == maximum:
                        raise ExecutionError("compiled loop iteration bound exceeded")
                    self.program(_list(body["copies"]))
                    self.program(_list(body["program"]))
                    self.program(_list(body["updates"]))
            elif operation == "reduce":
                self.reduce(instruction)
            elif operation == "cache_read":
                cache = _text(instruction["cache"])
                output = _text(instruction["output"])
                snapshot = self.cache_array(cache)
                self.shapes[output] = snapshot.shape
                if instruction["snapshot"]:
                    if self.manifest.buffers[output].dtype in _QUANTIZED_DTYPES:
                        payload = _encode(
                            snapshot, self.manifest.buffers[output].dtype
                        )[0]
                        c.memmove(
                            self.address(output), payload.ctypes.data, payload.nbytes
                        )
                    else:
                        self.copy_array(self.array(output), snapshot)
            elif operation == "cache_write":
                cache = _text(instruction["cache"])
                source = _text(instruction["input"])
                metadata = self.manifest.caches[cache]
                shape = self.shape(source)
                axis = metadata["axis"]
                target = self.cache_array(cache, capacity=True)
                kind = _text(metadata["dtype"])
                values = (
                    self.logical_array(source)
                    if kind in _QUANTIZED_DTYPES
                    else self.array(source)
                )
                if isinstance(axis, int):
                    begin = (
                        self.lengths[cache] if instruction["action"] == "append" else 0
                    )
                    end = begin + shape[axis]
                    capacity = target.shape[axis]
                    if (
                        end > capacity
                        and metadata["mode"] == "sliding"
                        and instruction["action"] == "append"
                    ):
                        previous = self.cache_array(cache)
                        values = np.concatenate((previous, values), axis=axis)
                        slices = [slice(None)] * len(shape)
                        slices[axis] = slice(max(0, end - capacity), end)
                        values = values[tuple(slices)]
                        begin, end = 0, min(end, capacity)
                    elif end > capacity:
                        raise ExecutionError("cache capacity exceeded")
                    slices = [slice(None)] * len(shape)
                    slices[axis] = slice(begin, end)
                    if not instruction["metadata_only"]:
                        self.copy_array(target[tuple(slices)], values)
                    self.lengths[cache] = end
                else:
                    if target.shape != shape:
                        raise ExecutionError(
                            "overwrite cache shape differs from its capacity"
                        )
                    if not instruction["metadata_only"]:
                        self.copy_array(target, values)
                if kind in _QUANTIZED_DTYPES and not instruction["metadata_only"]:
                    payload = _encode(target, kind)[0]
                    c.memmove(
                        self.cache.address + _integer(metadata["offset"]),
                        payload.ctypes.data,
                        payload.nbytes,
                    )
            else:
                raise ValueError(f"unknown instruction {operation}")
            self._expression_values.clear()
        self.check_cancelled()
