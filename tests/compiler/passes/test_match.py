from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from trillim.compiler.kernels import catalog
from trillim.compiler.kernels.kernel_dataclasses import _Kernel
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.match import _match
from trillim.compiler.passes.passes_dataclasses import _KernelMatch
from trillim.compiler.passes.select import _select
from trillim.tdl import (
    Add,
    Cache,
    Cast,
    Construct,
    Div,
    Exp,
    Graph,
    Input,
    Matmul,
    Mul,
    Output,
    Permute,
    Silu,
    Sub,
    Tensor,
    TensorShape,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "kernels"
_KERNEL_HEADERS = Path(__file__).parents[3] / "src" / "trillim" / "kernels"
_BINDER = """
import inspect
from trillim.compiler.artifacts.expressions import _bound
from trillim.compiler.kernels.bindings import _contiguous_strides

def bind(context):
    tensors = (*context.inputs.values(), *context.outputs)
    for tensor in tensors:
        if tensor.dtype != "fp32":
            raise ValueError("fixture requires fp32")
        if tensor.strides != _contiguous_strides(tensor.shape):
            raise ValueError("fixture requires contiguous tensors")
    count = 1
    for dimension in context.outputs[0].shape:
        count *= _bound(dimension)
    counts = []
    for name in inspect.signature(pattern).parameters:
        if name not in context.inputs: continue
        size = 1
        for dimension in context.inputs[name].shape: size *= _bound(dimension)
        counts.append(size)
    return {"count": count, "input_count": len(counts), "input_counts": tuple(counts)}
"""


def _kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    *,
    inputs: int = 2,
    outputs: int = 1,
) -> tuple[_Kernel, ...]:
    directory = tmp_path / "kernels" / "scalar"
    directory.mkdir(parents=True)
    source = (_FIXTURES / "matching.c").read_text()
    source = source.replace("INPUT_COUNT", str(inputs)).replace(
        "OUTPUT_COUNT", str(outputs)
    )
    source = source.replace("INPUT_ALIGNMENTS", ", ".join("1" for _ in range(inputs)))
    source = source.replace("OUTPUT_ALIGNMENTS", ", ".join("1" for _ in range(outputs)))
    source = source.replace(
        "TDL_SOURCE", json.dumps("from trillim.tdl import *\n" + pattern + _BINDER)
    )
    path = tmp_path / "matching.c"
    path.write_text(source)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-I",
            str(_KERNEL_HEADERS),
            str(path),
            "-o",
            str(directory / "kernels.so"),
        ],
        check=True,
        capture_output=True,
    )
    (directory / "index.json").write_text(
        json.dumps({"version": 1, "symbols": ["matching"]})
    )
    monkeypatch.setattr(catalog, "_VERSION_ROOT", tmp_path)
    kernels = catalog._discover()
    assert len(kernels) == 1
    assert kernels[0].parameter_schema.input_count == inputs
    assert kernels[0].parameter_schema.output_count == outputs
    return kernels


def _candidates(
    graph: Graph, kernels: tuple[_Kernel, ...], output: Tensor
) -> tuple[_KernelMatch, ...]:
    producer = output._producer
    assert producer is not None
    return _match(graph, kernels, _infer(graph)).candidates[producer]


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(lambda a, b: Add(a, b), id="direct"),
        pytest.param(lambda a, b: Add(b, a), id="commuted"),
        pytest.param(lambda a, b: Sub(a, Mul(b, -1)), id="subtraction-negation"),
    ],
)
def test_binary_math_from_native_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expression: Callable[[Tensor, Tensor], Tensor],
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Add(a, b),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = expression(a, b)
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert set(matches[0].inputs) == {a, b}
    assert matches[0].outputs == (result,)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(lambda a, b: Mul(Add(a, b), Add(a, b)), id="factored-square"),
        pytest.param(
            lambda a, b: Add(Add(Mul(a, a), Mul(Mul(a, b), 2)), Mul(b, b)),
            id="expanded-square",
        ),
        pytest.param(
            lambda a, b: Cast(Mul(Add(Cast(a, "fp16"), b), Add(b, a)), "fp32"),
            id="pure-math-casts",
        ),
    ],
)
def test_polynomial_fusion_preserves_physical_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expression: Callable[[Tensor, Tensor], Tensor],
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Mul(Add(a, b), Add(a, b)),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = expression(a, b)
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert set(matches[0].inputs) == {a, b}
    assert all(value._producer not in matches[0].covered for value in matches[0].inputs)


@Construct
def _square_sum(a: Tensor, b: Tensor) -> Tensor:
    return Mul(Add(a, b), Add(a, b))


def test_core_kernel_can_cover_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Mul(Add(a, b), Add(a, b)),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = _square_sum(a, b)
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert result._producer in matches[0].covered
    assert set(matches[0].inputs) == {a, b}


@pytest.mark.parametrize("restriction", ["dtype", "stride"])
def test_binder_rejects_unsupported_physical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restriction: str
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Add(a, b),)\n",
    )
    with Graph() as graph:
        a = Input(
            name="a",
            shape=TensorShape((3, 2)),
            dtype="fp16" if restriction == "dtype" else "fp32",
        )
        b = Input(
            name="b",
            shape=TensorShape((2, 3)),
            dtype="fp16" if restriction == "dtype" else "fp32",
        )
        result = Add(Permute(a, (1, 0)), b)
        Output(result, name="result")
    producer = result._producer
    assert producer is not None
    matched = _match(graph, kernels, _infer(graph))
    assert not matched.candidates[producer]
    assert ("fp32" if restriction == "dtype" else "contiguous") in matched.rejections[
        producer
    ]["matching"]


def test_shared_intermediate_prevents_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Mul(Add(a, b), Add(a, b)),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        shared = Add(a, b)
        result = Mul(shared, shared)
        Output(result, name="result")
        Output(Div(shared, 2), name="live")
    assert not _candidates(graph, kernels, result)


@Construct
def _sum_difference(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    return Add(a, b), Sub(a, b)


def test_multioutput_core_pattern_covers_one_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Add(a, b), Sub(a, b))\n",
        outputs=2,
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        total, difference = _sum_difference(a, b)
        Output(total, name="total")
        Output(difference, name="difference")
    matches = _candidates(graph, kernels, total)
    assert len(matches) == 1
    assert matches[0].outputs == (total, difference)
    assert set(matches[0].inputs) == {a, b}


def test_cache_reads_remain_physical_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Add(a, b),)\n",
    )
    with Graph() as graph:
        cache = Cache(name="state", shape=(2, 3), dtype="fp32")
        before = cache.read()
        x = Input(name="x", shape=TensorShape((2, 3)), dtype="fp32")
        cache.assign(x)
        result = Add(before, x)
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert before in matches[0].inputs
    assert all(
        node.op not in {"cache", "cache_assign", "cache_read"}
        for node in matches[0].covered
    )


@Construct
def _triple_sum(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    return Add(c, Add(b, a))


@pytest.mark.parametrize("algebraic", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_associative_reordering_binds_each_input_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, algebraic: bool, composed: bool
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor, c: Tensor):\n    return (Add(Add(a, b), c),)\n",
        inputs=3,
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        c = Input(name="c", shape=TensorShape((2, 3)), dtype="fp32")
        result = _triple_sum(a, b, c) if composed else Add(c, Add(b, a))
        Output(result, name="result")
    producer = result._producer
    assert producer is not None
    matched = _match(graph, kernels, _infer(graph), algebraic=algebraic)
    matches = matched.candidates[producer]
    assert len(matches) == 1
    assert set(matches[0].inputs) == {a, b, c}


def test_composition_kernel_matches_core_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (SiluMul(a, b),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = Mul(Silu(a), b)
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert set(matches[0].inputs) == {a, b}


@Construct
def _effectful_sum(a: Tensor, b: Tensor, cache: Cache) -> Tensor:
    cache.assign(a)
    return Add(a, b)


def test_composition_with_cache_write_cannot_be_replaced_by_pure_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Add(a, b),)\n",
    )
    with Graph() as graph:
        cache = Cache(name="state", shape=(2, 3), dtype="fp32")
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = _effectful_sum(a, b, cache)
        Output(result, name="result")
        Output(cache.read(), name="state")
    assert not _candidates(graph, kernels, result)


def test_metadata_can_define_its_own_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "@Construct\ndef square(a: Tensor, b: Tensor) -> Tensor:\n"
        "    return Mul(Add(a, b), Add(a, b))\n"
        "def pattern(a: Tensor, b: Tensor):\n    return (square(a, b),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = Mul(Add(a, b), Add(a, b))
        Output(result, name="result")
    matches = _candidates(graph, kernels, result)
    assert len(matches) == 1
    assert set(matches[0].inputs) == {a, b}


@Construct
def _square_with_unused_pure_work(a: Tensor, b: Tensor) -> Tensor:
    Exp(a)
    return Mul(Add(a, b), Add(a, b))


def test_fused_composition_does_not_select_eliminated_region_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor):\n    return (Mul(Add(a, b), Add(a, b)),)\n",
    )
    with Graph() as graph:
        a = Input(name="a", shape=TensorShape((2, 3)), dtype="fp32")
        b = Input(name="b", shape=TensorShape((2, 3)), dtype="fp32")
        result = _square_with_unused_pure_work(a, b)
        Output(result, name="result")
    matches = _match(graph, kernels, _infer(graph))
    selection = _select(graph, matches)
    producer = result._producer
    assert producer is not None
    assert tuple(selection.calls) == (producer,)
    assert selection.calls[producer].outputs == (result,)


@pytest.mark.parametrize("bias_first", (False, True))
def test_core_projection_bias_with_different_operand_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bias_first: bool
) -> None:
    kernels = _kernel(
        tmp_path,
        monkeypatch,
        "def pattern(a: Tensor, b: Tensor, bias: Tensor):\n    return (Add(Matmul(a,b),bias),)\n",
        inputs=3,
    )
    with Graph() as graph:
        x = Input(name="x", shape=TensorShape((2, 3)), dtype="fp32")
        weight = Input(name="weight", shape=TensorShape((3, 4)), dtype="fp32")
        bias = Input(name="bias", shape=TensorShape((4,)), dtype="fp32")
        projected = Matmul(x, weight)
        result = Add(bias, projected) if bias_first else Add(projected, bias)
        Output(result, name="result")
    candidates = _candidates(graph, kernels, result)
    assert candidates
    assert set(candidates[0].inputs) == {x, weight, bias}
