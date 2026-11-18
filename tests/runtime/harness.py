from __future__ import annotations

import traceback
from multiprocessing.connection import Connection
from threading import Event, Thread

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import JsonValue

from trillim.runtime import CancelledError, Runtime

from .runtime_dataclasses import ModelArtifact


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
            entered = Event()
            failures: list[BaseException] = []
            original = model._backend.call

            def observe(instruction: dict[str, JsonValue]) -> None:
                bindings = instruction["inputs"]
                assert isinstance(bindings, list)
                for name in bindings:
                    assert isinstance(name, str)
                    record = model._backend.manifest.buffers[name]
                    while record.storage == "alias":
                        assert record.source is not None
                        record = model._backend.manifest.buffers[record.source]
                    if record.storage == "weight" and len(record.maximum_shape) == 2:
                        entered.set()
                        break
                original(instruction)

            request = inputs(artifact, tuple(i % 8 for i in range(512)), reset=True)

            def infer() -> None:
                try:
                    while True:
                        model.run(request)
                except BaseException as error:  # noqa: BLE001 - transfer worker failures to the parent
                    failures.append(error)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(model._backend, "call", observe)
                worker = Thread(target=infer)
                worker.start()
                assert entered.wait(10), f"projection never started: {failures!r}"
                channel.send("running")
                if mode == "api":
                    assert channel.recv() == "cancel"
                    model.cancel()
                worker.join(5)
                assert not worker.is_alive(), "execution did not stop"
                assert len(failures) == 1 and isinstance(failures[0], CancelledError), (
                    failures
                )
        channel.send("cancelled")
    except BaseException:  # noqa: BLE001 - report child failures before the watchdog kills it
        channel.send(traceback.format_exc())
    finally:
        channel.close()
