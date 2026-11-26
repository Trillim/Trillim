from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ..compiler.passes.verify import _verify
from .runtime_dataclasses import TensorSpec, TokenEvent
from .runtime_pydantic_models import _RunRequest


class Runtime:
    """Run a compiled model with persistent caches and independently owned outputs."""

    def __init__(self, directory: str | Path, *, threads: int | None = None) -> None:
        from .executor import _CPU

        self._lock = RLock()
        self._closed = False
        self._failed = False
        self._running = False
        request = _RunRequest(
            directory=Path(directory).expanduser().resolve(), threads=threads
        )
        manifest = _verify(request.directory)
        self._backend = _CPU(request.directory, manifest, request.threads)

    @property
    def inputs(self) -> Mapping[str, TensorSpec]:
        return self._specs(self._backend.manifest.inputs)

    @property
    def outputs(self) -> Mapping[str, TensorSpec]:
        return self._specs(self._backend.manifest.outputs)

    def _specs(self, ports: Mapping[str, str]) -> dict[str, TensorSpec]:
        return {
            name: TensorSpec(
                record.dtype,
                tuple(
                    value if isinstance(value, int) else None for value in record.shape
                ),
                tuple(record.maximum_shape),
            )
            for name, tensor in ports.items()
            for record in (self._backend.manifest.buffers[tensor],)
        }

    def run(
        self,
        inputs: Mapping[str, NDArray[np.generic] | JsonValue],
        *,
        on_token: Callable[[TokenEvent], None] | None = None,
    ) -> dict[str, NDArray[np.generic]]:
        """Run a request, optionally reporting its rank-one integer tokens output.

        The callback runs on this thread as tokens arrive and should return promptly.
        A callback exception aborts execution and closes the worker.
        """
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("model session is closed or failed")
            if self._running:
                raise RuntimeError("model session is already running")
            if on_token is not None and not callable(on_token):
                raise TypeError("on_token must be callable")
            self._running = True
            try:
                self._backend.bind_inputs(inputs)
                try:
                    return self._backend.execute(on_token)
                except BaseException:
                    self._failed = True
                    raise
            finally:
                self._running = False

    def cancel(self) -> None:
        """Request cancellation without waiting for an active native kernel."""
        self._backend.cancel()

    def close(self) -> None:
        self.cancel()
        with self._lock:
            if not self._closed:
                self._closed = True
                self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
