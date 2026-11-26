from __future__ import annotations

import os
import platform
import signal
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import Never

import numpy as np
import pytest

from trillim.compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from trillim.compiler.kernels.native import _CAPABILITY_FLAGS
from trillim.runtime import CancelledError, ExecutionError, Runtime, executor

from .harness import read_frame, write_request


def test_worker_launch_uses_prebuilt_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags = _CAPABILITY_FLAGS["x86"]
    software = flags["m"]
    manifest = _ExecutableManifest(
        backend="x86",
        threads=1,
        arena_size=0,
        cache_size=0,
        inputs={},
        outputs={},
        buffers={},
        caches={},
        kernels={},
        program=[],
    )

    class LaunchObserved(Exception):
        pass

    def observe_launch(command: list[str], **_options: object) -> Never:
        assert command[command.index("--capabilities") + 1] == str(software)
        raise LaunchObserved

    monkeypatch.delenv("TRILLIM_CPU_WORKER", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(executor, "_available_capabilities", lambda backend: software)
    monkeypatch.setattr(executor, "_worker_path", lambda: tmp_path / "worker")
    monkeypatch.setattr(subprocess, "Popen", observe_launch)
    with pytest.raises(LaunchObserved):
        executor._CPU(tmp_path, manifest, threads=None)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX pipes")
def test_worker_exits_if_parent_closes_request_pipe_under_backpressure(
    compile_tdl: Callable[[str], Path],
) -> None:
    directory = compile_tdl("""
        from trillim.tdl import Broadcast, Graph, Input, Output

        with Graph() as graph:
            x = Input(name="x", shape=(), dtype="fp32")
            Output(Broadcast(x, (1024 * 1024,)), name="result")
    """)
    with Runtime(directory) as model:
        process = model._backend._process
        assert process.stdin is not None and process.stdout is not None
        write_request(
            process.stdin, {"x": np.asarray(9, dtype=np.float32)}, stream_tokens=False
        )
        response = read_frame(process.stdout)
        assert response["ok"] is True
        assert response["outputs"] == [
            {"name": "result", "dtype": "fp32", "shape": [1048576], "nbytes": 4194304}
        ]
        process.stdin.close()
        process.wait(timeout=2)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signaled_idle_worker_reports_cancellation_on_next_request(
    token_model: Path, signum: int
) -> None:
    with Runtime(token_model) as model:
        process = model._backend._process
        process.send_signal(signum)
        assert process.wait(timeout=2) == -signum
        with pytest.raises(CancelledError):
            model.run({"script": np.arange(8, dtype=np.int32), "cap": 4})


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signals")
def test_runtime_restores_application_signal_handlers(token_model: Path) -> None:
    script = dedent("""
        import signal
        import sys

        from trillim.runtime import Runtime

        received = []

        def handle(signum, frame):
            received.append(signum)

        signals = (signal.SIGINT, signal.SIGTERM)
        for signum in signals:
            signal.signal(signum, handle)

        with Runtime(sys.argv[1]) as model:
            model.run({"script": list(range(8)), "cap": 2})
            for signum in signals:
                signal.raise_signal(signum)
            assert received == list(signals)
    """)
    subprocess.run(
        [sys.executable, "-c", script, str(token_model)], check=True, timeout=10
    )


@pytest.mark.skipif(os.name != "posix", reason="requires an executable test script")
@pytest.mark.parametrize(
    "startup, payload, length, message",
    [
        (
            True,
            b'{"ok":false,"error":"test startup failed"}',
            None,
            "test startup failed",
        ),
        (
            False,
            b'{"ok":true,"event":"unexpected"}',
            None,
            "invalid .* response header",
        ),
        (False, b'{"ok":true', 100, "closed its output pipe"),
    ],
)
def test_runtime_rejects_worker_protocol_errors_and_reaps_child(
    token_model: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    startup: bool,
    payload: bytes,
    length: int | None,
    message: str,
) -> None:
    with Runtime(token_model) as runtime:
        backend = sys.modules[type(runtime._backend).__module__]
    path = tmp_path / "fake-worker"
    pid_file = tmp_path / "pid"
    wire = struct.pack("<Q", len(payload) if length is None else length) + payload
    path.write_text(
        f"#!{sys.executable}\n"
        + dedent(f"""
        import json
        import os
        import struct
        import sys
        from pathlib import Path

        Path({str(pid_file)!r}).write_text(str(os.getpid()))
        if not {startup!r}:
            ready = b'{{"ok":true,"ready":true}}'
            sys.stdout.buffer.write(struct.pack("<Q", len(ready)) + ready)
            sys.stdout.buffer.flush()
            (size,) = struct.unpack("<Q", sys.stdin.buffer.read(8))
            request = json.loads(sys.stdin.buffer.read(size))
            sys.stdin.buffer.read(sum(item["nbytes"] for item in request["inputs"]))
        sys.stdout.buffer.write({wire!r})
        sys.stdout.buffer.flush()
    """)
    )
    path.chmod(0o755)
    monkeypatch.delenv("TRILLIM_CPU_WORKER", raising=False)
    monkeypatch.setattr(backend, "_worker_path", lambda: path)

    if startup:
        with pytest.raises(ExecutionError, match=message):
            Runtime(token_model)
    else:
        with Runtime(token_model) as model:
            inputs = {"script": np.arange(8, dtype=np.int32), "cap": np.asarray(4)}
            with pytest.raises(ExecutionError, match=message):
                model.run(inputs)
            with pytest.raises(RuntimeError, match="closed or failed"):
                model.run(inputs)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
