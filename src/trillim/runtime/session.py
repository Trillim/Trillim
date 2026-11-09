from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ..backends.cpu.executor import _CPU
from .runtime_dataclasses import TensorSpec
from .runtime_pydantic_models import _RunRequest


class Runtime:
    """Run a compiled model with persistent caches and independently owned outputs."""

    def __init__(self, directory: str | Path, *, threads: int | None = None) -> None:
        self._lock = RLock()
        self._closed = False
        self._failed = False
        request = _RunRequest(
            directory=Path(directory).expanduser().resolve(), threads=threads
        )
        self._backend = _CPU(request.directory, request.threads)

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
        self, inputs: Mapping[str, NDArray[np.generic] | JsonValue]
    ) -> dict[str, NDArray[np.generic]]:
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("model session is closed or failed")
            self._backend.bind_inputs(inputs)
            try:
                return self._backend.execute()
            except BaseException:
                self._failed = True
                self._backend.cancel()
                raise

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
