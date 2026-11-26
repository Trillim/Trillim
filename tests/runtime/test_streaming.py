from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from trillim.compiler.artifacts.artifact_pydantic_models import _ExecutableManifest
from trillim.runtime import ExecutionError, Runtime, TokenEvent

from .harness import read_bytes, read_frame, write_request


@pytest.mark.parametrize(
    "script, cap, expected",
    [
        ([6, 3, 1, 2, 3, 4, 5, 0], 8, [6]),
        ([1, 5, 7, 2, 3, 4, 5, 0], 8, [1, 5, 7]),
        ([1, 2, 3, 4, 5, 6, 7, 0], 3, [1, 2, 3]),
        ([1, 2, 3, 4, 5, 6, 7, 0], 0, []),
    ],
)
def test_stream_emits_each_committed_token(
    token_model: Path, script: list[int], cap: int, expected: list[int]
) -> None:
    events: list[TokenEvent] = []
    with Runtime(token_model) as model:
        inputs = {"script": np.asarray(script, dtype=np.int32), "cap": np.asarray(cap)}
        streamed = model.run(inputs, on_token=events.append)["tokens"]
        batch = model.run(inputs)["tokens"]
    assert events == [
        TokenEvent("tokens", index, token) for index, token in enumerate(expected)
    ]
    np.testing.assert_array_equal(streamed, expected)
    np.testing.assert_array_equal(streamed, batch)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX pipes")
def test_worker_streams_without_token_acknowledgements(token_model: Path) -> None:
    script = np.arange(8, dtype=np.int32)
    with Runtime(token_model) as model:
        process = model._backend._process
        assert process.stdin is not None and process.stdout is not None
        write_request(
            process.stdin,
            {"script": script, "cap": np.asarray(4, dtype=np.int64)},
            stream_tokens=True,
        )
        for index in range(4):
            frame = read_frame(process.stdout)
            assert frame == {
                "ok": True,
                "event": "token",
                "output": "tokens",
                "index": index,
                "token": index,
            }
        response = read_frame(process.stdout)
        assert response["ok"] is True
        assert response["outputs"] == [
            {"name": "tokens", "dtype": "int32", "shape": [4], "nbytes": 16}
        ]
        assert read_bytes(process.stdout, 16) == script[:4].tobytes()


def test_callback_error_ends_session_and_reaps_worker(token_model: Path) -> None:
    error = LookupError("callback aborted")

    def fail(event: TokenEvent) -> None:
        assert event.index == 0
        raise error

    with Runtime(token_model) as model:
        process = model._backend._process
        inputs = {"script": np.arange(8, dtype=np.int32), "cap": np.asarray(4)}
        with pytest.raises(LookupError) as failure:
            model.run(inputs, on_token=fail)
        assert failure.value is error
        with pytest.raises(RuntimeError, match="closed or failed"):
            model.run(inputs)
    assert process.returncode is not None


def test_prefill_callback_precedes_decode_error(
    token_model: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "bundle"
    shutil.copytree(token_model, directory)
    path = directory / "executable.json"
    manifest = _ExecutableManifest.model_validate_json(path.read_bytes())
    loop = next(step for step in manifest.program if step["op"] == "while")
    body = loop["body"]
    assert isinstance(body, dict)
    program = body["program"]
    assert isinstance(program, list)
    program.insert(
        0,
        {
            "op": "guard",
            "left": 0,
            "right": 1,
            "predicate": "eq",
            "reason": "decode failed",
        },
    )
    path.write_text(manifest.model_dump_json())

    events: list[TokenEvent] = []
    with (
        Runtime(directory) as runtime,
        pytest.raises(ExecutionError, match="decode failed"),
    ):
        runtime.run(
            {"script": np.arange(8, dtype=np.int32), "cap": 4}, on_token=events.append
        )
    assert events == [TokenEvent("tokens", 0, 0)]


def test_revising_loop_streams_final_tokens(compile_tdl: Callable[[str], Path]) -> None:
    directory = compile_tdl("""
        import numpy as np
        from trillim.tdl import Add, Cast, Compare, Constant, Graph, Iota, Minimum, Output, Reshape, Scatter, Select, Take, Tensor, While

        with Graph() as graph:
            def body(state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
                step, count, tokens = state
                token = Cast(Add(step, 1), "int32")
                updated = Scatter(tokens, indices=Reshape(count, (1,)), values=token)
                length = Select(Compare(step, 1, predicate="eq"), 0, Minimum(Add(count, 1), 4))
                return Add(step, 1), length, updated

            _, count, tokens = While(
                (Constant(0), Constant(0), Constant(np.zeros(4, dtype=np.int32))),
                condition=lambda state: Compare(state[0], 4, predicate="lt"),
                body=body,
            )
            Output(Take(tokens, Iota(count)), name="tokens")
    """)
    events: list[TokenEvent] = []
    with Runtime(directory) as runtime:
        actual = runtime.run({}, on_token=events.append)["tokens"]
    np.testing.assert_array_equal(actual, [3, 4])
    assert events == [TokenEvent("tokens", 0, 3), TokenEvent("tokens", 1, 4)]
