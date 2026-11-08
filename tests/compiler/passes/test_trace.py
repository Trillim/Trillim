from __future__ import annotations

from pathlib import Path
from traceback import extract_tb

import pytest

from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.compiler.passes.trace import _trace

FIXTURES = Path(__file__).parents[1] / "fixtures" / "trace"


@pytest.mark.parametrize(
    ("filename", "input_name", "output_name"),
    [("valid.tdl", "x", "y"), ("valid_loop.tdl", "limit", "history")],
)
def test_valid_graph(filename: str, input_name: str, output_name: str) -> None:
    graph = _trace(_CompileRequest(tdl_path=FIXTURES / filename))

    assert set(graph._inputs) == {input_name}
    assert set(graph._outputs) == {output_name}
    assert graph._region is not None
    assert graph._region.outputs == (graph._outputs[output_name],)


def test_independent_traces() -> None:
    request = _CompileRequest(tdl_path=FIXTURES / "valid.tdl")

    first = _trace(request)
    second = _trace(request)

    assert first is not second
    assert first._inputs["x"] is not second._inputs["x"]


def test_runtime_error_preserves_source() -> None:
    path = FIXTURES / "buggy.tdl"

    with pytest.raises(NameError) as raised:
        _trace(_CompileRequest(tdl_path=path))

    assert raised.value.name == "missing_value"
    assert any(str(path) in note for note in raised.value.__notes__)
    frame = extract_tb(raised.value.__traceback__)[-1]
    assert frame.filename == str(path)
    assert frame.line == 'Output(Add(x, missing_value), name="y")'


def test_syntax_error_preserves_source() -> None:
    path = FIXTURES / "syntax_error.tdl"

    with pytest.raises(SyntaxError) as raised:
        _trace(_CompileRequest(tdl_path=path))

    assert raised.value.filename == str(path)
    assert raised.value.text is not None
    assert raised.value.text.strip() == "Output("
    assert any(str(path) in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    ("filename", "error_type"),
    [
        ("missing_graph.tdl", TypeError),
        ("wrong_graph.tdl", TypeError),
        ("unfinished_graph.tdl", ValueError),
    ],
)
def test_invalid_graph(filename: str, error_type: type[Exception]) -> None:
    _trace(_CompileRequest(tdl_path=FIXTURES / "valid.tdl"))
    path = FIXTURES / filename

    with pytest.raises(error_type) as raised:
        _trace(_CompileRequest(tdl_path=path))

    assert str(path) in str(raised.value)


def test_missing_file_preserves_source(tmp_path: Path) -> None:
    path = tmp_path / "missing.tdl"

    with pytest.raises(FileNotFoundError) as raised:
        _trace(_CompileRequest(tdl_path=path))

    assert raised.value.filename == str(path)
    assert any(str(path) in note for note in raised.value.__notes__)


def test_failure_restores_graph_context() -> None:
    with pytest.raises(NameError):
        _trace(_CompileRequest(tdl_path=FIXTURES / "buggy.tdl"))

    graph = _trace(_CompileRequest(tdl_path=FIXTURES / "valid.tdl"))

    assert graph._region is not None
    assert "y" in graph._outputs
