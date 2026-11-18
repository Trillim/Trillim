from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time

import numpy as np
import pytest

from trillim.runtime import Runtime

from .harness import cancel_worker, inputs
from .runtime_dataclasses import ModelArtifact


def test_qwen_cache_persists_between_requests(qwen_layer: ModelArtifact) -> None:
    prompts = (
        ((0, 1, 2), 0, False),
        ((0, 1, 2, 3, 4), 3, False),
        ((0, 1, 6, 7), 2, False),
        ((0, 1), 1, False),
        ((0, 1, 2), 0, True),
    )
    with Runtime(qwen_layer.directory, threads=2) as session:
        for tokens, reused, reset in prompts:
            actual = session.run(inputs(qwen_layer, tokens, reset=reset))
            assert actual["reused"].item() == reused
            assert actual["cache_length"].item() == len(tokens)
            with Runtime(qwen_layer.directory, threads=2) as fresh:
                expected = fresh.run(inputs(qwen_layer, tokens, reset=True))
            assert np.isfinite(expected["hidden"]).all()
            np.testing.assert_allclose(
                actual["hidden"].astype(np.float32),
                expected["hidden"][reused:].astype(np.float32),
                rtol=2e-4,
                atol=2e-5,
            )


@pytest.mark.parametrize("mode", ("api", "sigint", "sigterm"))
def test_qwen_cancellation_stops_promptly(qwen_layer: ModelArtifact, mode: str) -> None:
    if mode != "api" and os.name == "nt":
        pytest.skip("POSIX signal delivery requires a POSIX host")
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=cancel_worker, args=(qwen_layer, mode, child))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "model did not start"
        assert parent.recv() == "running"
        # Allow the observed projection to enter native code before cancelling.
        time.sleep(0.02)
        started = time.perf_counter()
        if mode == "api":
            parent.send("cancel")
        else:
            assert process.pid is not None
            os.kill(process.pid, signal.SIGINT if mode == "sigint" else signal.SIGTERM)
        assert parent.poll(0.25), "cancellation and worker shutdown exceeded 250 ms"
        assert parent.recv() == "cancelled"
        assert time.perf_counter() - started < 0.25
        process.join(2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        parent.close()
