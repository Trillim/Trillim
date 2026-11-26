from __future__ import annotations

import json
import os
import platform
import signal
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping
from io import FileIO
from math import prod
from pathlib import Path
from threading import Event, RLock, Thread, Timer, current_thread, main_thread
from types import FrameType
from typing import cast
from weakref import ReferenceType, WeakSet, finalize, ref

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue, ValidationError

from .. import _VERSION_ROOT
from ..compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from ..compiler.dtype import _NATIVE_DTYPES, _QUANTIZED_DTYPES, _encode
from ..compiler.kernels.catalog import _available_capabilities
from ..compiler.kernels.native import _CAPABILITY_FLAGS
from .errors import CancelledError, ExecutionError
from .runtime_dataclasses import TokenEvent
from .runtime_pydantic_models import _Response

_HEADER_LENGTH = struct.Struct("<Q")
_MAX_HEADER_LENGTH = 64 * 1024**2
_STARTUP_TIMEOUT = 60.0
_CANCEL_GRACE = 0.1
_CLOSE_GRACE = 0.2

_execution = RLock()
_active: ReferenceType[_CPU] | None = None
_sessions: WeakSet[_CPU] = WeakSet()
_handlers: dict[int, int | Callable[[int, FrameType | None], object] | None] = {}


def _worker_path() -> Path:
    override = os.environ.get("TRILLIM_CPU_WORKER")
    name = "worker.exe" if sys.platform == "win32" else "worker"
    path = (
        Path(override).expanduser().resolve()
        if override
        else _VERSION_ROOT / "backends/cpu" / name
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"CPU worker executable not found: {path}. Run 'uv run task build-cpu' "
            "or set TRILLIM_CPU_WORKER to a prebuilt executable."
        )
    return path


def _relay_signal(signum: int, frame: FrameType | None) -> None:
    active = _active() if _active is not None else None
    if active is not None:
        active._cancelled = True
        Thread(target=active.cancel, daemon=True).start()
        return
    previous = _handlers.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
    elif previous == signal.SIG_DFL:
        signal.signal(signum, previous)
        signal.raise_signal(signum)


def _register_signals(session: _CPU) -> None:
    _sessions.add(session)
    if current_thread() is not main_thread():
        return
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(signum)
        if previous is not None and previous is not _relay_signal:
            _handlers[signum] = previous
            signal.signal(signum, _relay_signal)


def _unregister_signals(session: _CPU) -> None:
    _sessions.discard(session)
    if _sessions or current_thread() is not main_thread():
        return
    for signum, previous in _handlers.items():
        if signal.getsignal(signum) is _relay_signal:
            signal.signal(signum, previous)
    _handlers.clear()


def _wait_for_worker(process: subprocess.Popen[bytes], timeout: float) -> None:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _close_worker(process: subprocess.Popen[bytes]) -> None:
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.close()
        _wait_for_worker(process, _CLOSE_GRACE)
    finally:
        process.stdout.close()


class _CPU:
    def __init__(
        self,
        directory: Path,
        manifest: _ExecutableManifest,
        threads: int | None,
    ) -> None:
        if sys.byteorder != "little":
            raise ValueError("compiled models require a little-endian host")
        threads = manifest.threads if threads is None else threads
        if not 1 <= threads <= 65536:
            raise ValueError("threads must be between 1 and 65536")
        machine = platform.machine().lower()
        if manifest.backend == "x86" and machine not in {
            "x86_64",
            "amd64",
            "i386",
            "i686",
            "x86",
        }:
            raise ValueError("x86 backend requires an x86 device")
        if manifest.backend == "neon" and machine not in {"aarch64", "arm64"}:
            raise ValueError("ARM backend requires an ARM64 device")
        capabilities = _available_capabilities(manifest.backend)
        if manifest.backend == "neon":
            capabilities |= _CAPABILITY_FLAGS["neon"]["neon"]

        self.manifest = manifest
        self.last_stats: dict[str, JsonValue] = {}
        self._inputs: dict[str, tuple[tuple[int, ...], NDArray[np.uint8]]] = {}
        self._profile = "TRILLIM_EXPERIMENTAL_CPP_PROFILE" in os.environ
        self._probe_outputs = [
            name.strip()
            for name in os.environ.get(
                "TRILLIM_EXPERIMENTAL_CPP_PROBE_OUTPUTS", ""
            ).split(",")
            if name.strip()
        ]
        self._lifecycle = RLock()
        self._executing = False
        self._sending = False
        self._cancelled = False
        self._stopping = False
        executable = _worker_path()
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        self._process = subprocess.Popen(
            [
                str(executable),
                "--bundle",
                str(directory),
                "--threads",
                str(threads),
                "--capabilities",
                str(capabilities),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name == "posix",
            creationflags=creation_flags,
        )
        self._stdin = cast(FileIO, self._process.stdin)
        self._stdout = cast(FileIO, self._process.stdout)
        self._finish = finalize(self, _close_worker, self._process)
        self._await_ready()
        _register_signals(self)

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("model execution cancelled")

    def _release_execution(self) -> bool:
        global _active
        with _execution:
            active = _active() if _active is not None else None
            if active is not self:
                return False
            _active = None
            return self._cancelled

    def _pipe_error(self, message: str) -> ExecutionError:
        self._check_cancelled()
        status = self._process.poll()
        if status is None:
            try:
                status = self._process.wait(timeout=_CANCEL_GRACE)
            except subprocess.TimeoutExpired:
                pass
        self._check_cancelled()
        if status in (130, -signal.SIGINT, -signal.SIGTERM):
            return CancelledError("model execution cancelled")
        return ExecutionError(f"{message} (status {status})")

    def _await_ready(self) -> None:
        expired = Event()

        def kill_unready_worker() -> None:
            expired.set()
            self._process.kill()

        watchdog = Timer(_STARTUP_TIMEOUT, kill_unready_worker)
        watchdog.daemon = True
        try:
            watchdog.start()
            try:
                response = self._read_response()
            finally:
                watchdog.cancel()
                watchdog.join()
            if expired.is_set():
                raise ExecutionError("CPU worker startup timed out")
            if (
                not response.ready
                or response.event is not None
                or response.outputs is not None
            ):
                raise ExecutionError("expected readiness frame from C++ runtime")
        except BaseException as error:
            self._process.kill()
            self._finish()
            if expired.is_set() and isinstance(error, ExecutionError):
                raise ExecutionError("CPU worker startup timed out") from error
            raise

    def _write(self, data: bytes | memoryview[int]) -> None:
        remaining = memoryview(data)
        while remaining:
            self._check_cancelled()
            try:
                count = self._stdin.write(remaining)
            except OSError as error:
                raise self._pipe_error("cannot write to CPU worker") from error
            if not count:
                raise self._pipe_error("C++ runtime did not accept request bytes")
            remaining = remaining[count:]

    def _read_into(self, buffer: bytearray | memoryview[int]) -> None:
        remaining = memoryview(buffer)
        while remaining:
            try:
                count = self._stdout.readinto(remaining)
            except OSError as error:
                raise self._pipe_error("cannot read from CPU worker") from error
            if not count:
                raise self._pipe_error("C++ runtime closed its output pipe")
            remaining = remaining[count:]

    def _write_header(self, message: dict[str, JsonValue]) -> None:
        header = json.dumps(message, separators=(",", ":")).encode()
        if len(header) > _MAX_HEADER_LENGTH:
            raise ExecutionError("C++ runtime request header is too large")
        self._write(_HEADER_LENGTH.pack(len(header)) + header)

    def _read_response(self) -> _Response:
        self._check_cancelled()
        prefix = bytearray(_HEADER_LENGTH.size)
        self._read_into(prefix)
        (size,) = _HEADER_LENGTH.unpack(prefix)
        if not 0 < size <= _MAX_HEADER_LENGTH:
            raise ExecutionError("invalid C++ runtime response header length")
        header = bytearray(size)
        self._read_into(header)
        try:
            response = _Response.model_validate_json(header)
        except ValidationError as error:
            raise ExecutionError("invalid C++ runtime response header") from error
        self._check_cancelled()
        if not response.ok:
            error_type = CancelledError if response.cancelled else ExecutionError
            raise error_type(response.error or "C++ runtime request failed")
        return response

    def bind_inputs(
        self, inputs: Mapping[str, NDArray[np.generic] | JsonValue]
    ) -> None:
        if inputs.keys() != self.manifest.inputs.keys():
            raise ValueError(
                f"expected model inputs: {', '.join(self.manifest.inputs)}"
            )
        encoded: dict[str, tuple[tuple[int, ...], NDArray[np.uint8]]] = {}
        for name, value in inputs.items():
            record = self.manifest.buffers[self.manifest.inputs[name]]
            native = _NATIVE_DTYPES.get(record.dtype)
            json_integer = (
                native is not None
                and native.kind in "biu"
                and not isinstance(value, np.ndarray)
            )
            array = np.asarray(value, dtype=object if json_integer else None)
            if (
                not json_integer
                and array.dtype.kind not in "biuf"
                or len(array.shape) != len(record.maximum_shape)
            ):
                raise ValueError(f"{name}: invalid numeric input or rank")
            for actual, maximum, fixed in zip(
                array.shape, record.maximum_shape, record.shape, strict=True
            ):
                if actual > maximum or isinstance(fixed, int) and actual != fixed:
                    raise ValueError(
                        f"{name}: input shape exceeds or differs from its compiled dimensions"
                    )
            if native is not None and native.kind in "biu":
                low, high = (
                    (0, 1)
                    if native.kind == "b"
                    else (np.iinfo(native.name).min, np.iinfo(native.name).max)
                )
                if json_integer:
                    for item in array.flat:
                        if (
                            not (
                                isinstance(item, int)
                                or isinstance(item, float)
                                and item.is_integer()
                            )
                            or not low <= item <= high
                        ):
                            raise ValueError(
                                f"{name}: input values do not fit {record.dtype}"
                            )
                    array = array.astype(native)
                elif array.size and (
                    not low <= array.min().item()
                    or not array.max().item() <= high
                    or array.dtype.kind == "f"
                    and np.any(array != np.trunc(array))
                ):
                    raise ValueError(f"{name}: input values do not fit {record.dtype}")
            encoded[name] = array.shape, _encode(array, record.dtype)[0]
        self._inputs = encoded

    def execute(
        self, on_token: Callable[[TokenEvent], None] | None
    ) -> dict[str, NDArray[np.generic]]:
        try:
            with self._lifecycle:
                self._check_cancelled()
                global _active
                with _execution:
                    if _active is not None and _active() is not None:
                        raise ExecutionError("another model is already executing")
                    _active = ref(self)
                self._executing = True
                self._sending = True
            self.last_stats = {}
            descriptors: list[JsonValue] = [
                {"name": name, "shape": list(shape), "nbytes": payload.nbytes}
                for name, (shape, payload) in self._inputs.items()
            ]
            message: dict[str, JsonValue] = {"op": "run", "inputs": descriptors}
            if on_token is not None:
                message["stream"] = True
            if self._profile:
                message["profile"] = True
            if self._probe_outputs:
                message["probe_outputs"] = list(self._probe_outputs)
            self._write_header(message)
            for _, payload in self._inputs.values():
                self._write(memoryview(payload))
            with self._lifecycle:
                self._sending = False
            response = self._read_response()
            if on_token is None:
                outputs = self._read_outputs(response)
            else:
                response, outputs = self._read_token_stream(response, on_token)
            with self._lifecycle:
                self._check_cancelled()
                if self._release_execution():
                    raise CancelledError("model execution cancelled")
                self._executing = False
            self.last_stats = response.stats
            return outputs
        except BaseException:
            self.close()
            raise
        finally:
            self._inputs.clear()
            with self._lifecycle:
                self._release_execution()
                self._executing = False
                self._sending = False

    def _read_token_stream(
        self, response: _Response, on_token: Callable[[TokenEvent], None]
    ) -> tuple[_Response, dict[str, NDArray[np.generic]]]:
        tensor = self.manifest.outputs.get("tokens")
        record = self.manifest.buffers[tensor] if tensor is not None else None
        dtype = _NATIVE_DTYPES.get(record.dtype) if record is not None else None
        if (
            record is None
            or len(record.maximum_shape) != 1
            or dtype is None
            or dtype.kind not in "iu"
        ):
            raise ExecutionError(
                "token streaming requires a rank-one integer tokens output"
            )
        capacity = record.maximum_shape[0]
        bounds = np.iinfo(dtype.name)
        tokens: list[int] = []
        while response.event == "token":
            index, token = response.index, response.token
            if (
                response.ready
                or response.outputs is not None
                or response.output != "tokens"
                or index is None
                or index != len(tokens)
                or len(tokens) >= capacity
                or token is None
                or not bounds.min <= token <= bounds.max
            ):
                raise ExecutionError("invalid token frame from CPU worker")
            self._check_cancelled()
            tokens.append(token)
            on_token(TokenEvent("tokens", index, token))
            response = self._read_response()
        outputs = self._read_outputs(response)
        if not np.array_equal(outputs["tokens"], np.asarray(tokens, dtype=dtype)):
            raise ExecutionError(
                "CPU worker token stream disagrees with the final output"
            )
        return response, outputs

    def _read_outputs(self, response: _Response) -> dict[str, NDArray[np.generic]]:
        if (
            response.outputs is None
            or response.ready
            or response.event == "token"
            or response.output is not None
            or response.index is not None
            or response.token is not None
        ):
            raise ExecutionError("expected outputs from C++ runtime")
        outputs: dict[str, NDArray[np.generic]] = {}
        for item in response.outputs:
            tensor = self.manifest.outputs.get(item.name)
            if tensor is None or item.name in outputs:
                raise ExecutionError(f"unexpected C++ runtime output {item.name!r}")
            record = self.manifest.buffers[tensor]
            dtype = (
                "fp32"
                if record.dtype == "bf16" or record.dtype in _QUANTIZED_DTYPES
                else record.dtype
            )
            shape = tuple(item.shape)
            if (
                item.dtype != dtype
                or len(shape) != len(record.maximum_shape)
                or any(
                    not 0 <= actual <= maximum
                    or isinstance(fixed, int)
                    and actual != fixed
                    for actual, maximum, fixed in zip(
                        shape, record.maximum_shape, record.shape, strict=True
                    )
                )
            ):
                raise ExecutionError(
                    f"invalid C++ runtime output metadata for {item.name!r}"
                )
            native = _NATIVE_DTYPES[dtype]
            if item.nbytes != prod(shape) * native.itemsize:
                raise ExecutionError(
                    f"invalid C++ runtime output size for {item.name!r}"
                )
            output = np.empty(shape, dtype=native)
            raw = output.reshape(-1).view(np.uint8)
            self._read_into(memoryview(raw))
            outputs[item.name] = output
        if outputs.keys() != self.manifest.outputs.keys():
            raise ExecutionError("C++ runtime response is missing model outputs")
        return {name: outputs[name] for name in self.manifest.outputs}

    def cancel(self) -> None:
        self._cancelled = True
        with self._lifecycle:
            if self._stopping:
                return
            self._stopping = True
            self._release_execution()
            if self._executing:
                Thread(
                    target=_wait_for_worker,
                    args=(self._process, _CANCEL_GRACE),
                    daemon=True,
                ).start()
                if os.name == "posix":
                    self._process.send_signal(signal.SIGINT)
            # A concurrent close may block behind a pipe write on Windows.
            if not self._sending:
                self._stdin.close()

    def close(self) -> None:
        try:
            self.cancel()
        finally:
            self._finish()
            self._inputs.clear()
            _unregister_signals(self)
