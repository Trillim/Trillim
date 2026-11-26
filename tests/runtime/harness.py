from __future__ import annotations

import json
import os
import struct
import traceback
from collections.abc import Mapping
from multiprocessing.connection import Connection
from select import select
from threading import Event, Thread
from time import monotonic
from typing import IO

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import JsonValue, TypeAdapter

from trillim.runtime import CancelledError, Runtime

from .runtime_dataclasses import ModelArtifact

_FRAME_LENGTH = struct.Struct("<Q")
_FRAME = TypeAdapter(dict[str, JsonValue])


def read_bytes(stream: IO[bytes], length: int) -> bytes:
    received = bytearray()
    deadline = monotonic() + 5
    while len(received) < length:
        wait = max(deadline - monotonic(), 0)
        assert select([stream], [], [], wait)[0], "timed out reading worker pipe"
        chunk = os.read(stream.fileno(), length - len(received))
        assert chunk, "worker closed its pipe before the response was complete"
        received.extend(chunk)
    return bytes(received)


def read_frame(stream: IO[bytes]) -> dict[str, JsonValue]:
    (size,) = _FRAME_LENGTH.unpack(read_bytes(stream, _FRAME_LENGTH.size))
    assert 0 < size <= 64 * 1024**2
    return _FRAME.validate_json(read_bytes(stream, size))


def write_request(
    stream: IO[bytes], inputs: Mapping[str, NDArray[np.generic]], *, stream_tokens: bool
) -> None:
    header = json.dumps(
        {
            "op": "run",
            "stream": stream_tokens,
            "inputs": [
                {"name": name, "shape": value.shape, "nbytes": value.nbytes}
                for name, value in inputs.items()
            ],
        }
    ).encode()
    request = _FRAME_LENGTH.pack(len(header)) + header
    request += b"".join(value.tobytes() for value in inputs.values())
    assert stream.write(request) == len(request)


def inputs(
    artifact: ModelArtifact, tokens: tuple[int, ...], *, reset: bool
) -> dict[str, NDArray[np.generic]]:
    indices = np.asarray(tokens, dtype=np.int32)
    return {
        "tokens": indices,
        "hidden": artifact.embeddings[indices],
        "reset": np.asarray(reset),
    }


def cancel_worker(artifact: ModelArtifact, mode: str, channel: Connection) -> None:
    try:
        with Runtime(artifact.directory, threads=2) as model:
            dispatched = Event()
            resume = Event()
            failures: list[BaseException] = []
            process = model._backend._process
            read_response = model._backend._read_response

            request = inputs(artifact, tuple(i % 8 for i in range(512)), reset=True)

            def infer() -> None:
                try:
                    model.run(request)
                except BaseException as error:  # noqa: BLE001 - transfer worker failures to the parent
                    failures.append(error)

            with pytest.MonkeyPatch.context() as patch:

                def observe() -> object:
                    dispatched.set()
                    assert resume.wait(5), "test did not release the response pipe"
                    return read_response()

                patch.setattr(model._backend, "_read_response", observe)
                worker = Thread(target=infer, daemon=True)
                worker.start()
                try:
                    assert dispatched.wait(10), (
                        f"request never reached worker: {failures!r}"
                    )
                    channel.send(("running", process.pid))
                    assert channel.poll(5), "test did not interrupt the worker"
                    assert channel.recv() == "cancel"
                    if mode == "api":
                        model.cancel()
                finally:
                    resume.set()
                worker.join(5)
                assert not worker.is_alive(), "execution did not stop"
                assert len(failures) == 1 and isinstance(failures[0], CancelledError), (
                    failures
                )
                with pytest.raises(RuntimeError, match="closed or failed"):
                    model.run(request)
        assert process.poll() is not None, "runtime worker survived shutdown"
        channel.send("cancelled")
    except BaseException:  # noqa: BLE001 - report child failures before the watchdog kills it
        channel.send(traceback.format_exc())
    finally:
        channel.close()
