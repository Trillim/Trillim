from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.optimize import _optimize
from trillim.compiler.passes.validate import _validate
from trillim.compiler.passes.walk import _walk
from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _read_values
from trillim.tdl import (
    Add,
    Cache,
    Cast,
    Constant,
    Cos,
    Exp,
    FloorDiv,
    Gelu,
    Graph,
    If,
    Input,
    Mul,
    Output,
    Permute,
    Reshape,
    Silu,
    Sin,
    Sub,
    Tensor,
    TensorShape,
)
from trillim.tdl.core.core_pydantic_models import _BinaryArgs, _ConstantArgs

_CHECKPOINT = (
    Path(__file__).parents[1] / "fixtures" / "weights" / "Qwen3-0.6B.safetensors"
)


def _values(name: str) -> NDArray[np.float32]:
    checkpoint = _load_checkpoint("Qwen3-0.6B", _CHECKPOINT)
    source = checkpoint.tensors[f"model.layers.0.{name}.weight"]
    return np.asarray(
        _read_values(source, 0, math.prod(source.shape)).reshape(source.shape),
        dtype=np.float32,
    )


def _fold_case(name: str) -> tuple[Tensor, NDArray[np.float32]]:
    gate = _values("mlp.gate_proj")
    up = _values("mlp.up_proj")
    x, y = Constant(gate), Constant(up)
    if name == "cast":
        return Cast(x, "fp32"), gate
    if name == "head-views":
        return Reshape(Reshape(x, (8, 4, 32)), (32, 32)), gate
    if name == "transpose":
        return Permute(x, (1, 0)), gate.T
    if name == "silu":
        return Silu(x), gate / (1 + np.exp(-gate))
    if name == "swiglu":
        return Mul(Silu(x), y), gate / (1 + np.exp(-gate)) * up
    if name == "gelu-tanh":
        result = (
            0.5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3)))
        )
        return Gelu(x, approximate="tanh"), result
    if name == "rotary-pair":
        return Sub(Mul(x, Cos(y)), Mul(y, Sin(y))), gate * np.cos(up) - up * np.sin(up)
    assert name == "attention-exp"
    return Exp(Sub(x, y)), np.exp(gate - up)


@pytest.mark.parametrize(
    "algorithm",
    [
        "cast",
        "head-views",
        "transpose",
        "silu",
        "swiglu",
        "gelu-tanh",
        "rotary-pair",
        "attention-exp",
    ],
)
def test_folded_checkpoint_computations_match_numpy(algorithm: str) -> None:
    with Graph() as graph:
        result, expected = _fold_case(algorithm)
        Output(result, name="result")
    inference = _infer(graph)
    _optimize(graph, inference)
    _validate(graph)
    output = graph._outputs["result"]
    producer = output._producer
    assert producer is not None
    assert isinstance(producer.args, _ConstantArgs)
    actual = producer.args.value
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), expected, rtol=3e-6, atol=2e-8
    )
    assert actual.shape == expected.shape
    assert not _optimize(graph, _infer(graph))


def test_effectful_checkpoint_cache_update_survives_dead_code_elimination() -> None:
    with Graph() as graph:
        cache = Cache(name="hidden", shape=(32, 32), dtype="fp32")
        value = Constant(_values("self_attn.q_proj"))
        cache.assign(value)
        Add(value, value)
        Output(cache.read(), name="hidden")
    _optimize(graph, _infer(graph))
    _validate(graph)
    assert graph._region is not None
    operators = tuple(node for _, node in _walk(graph._region))
    assert any(node.op == "cache_assign" for node in operators)
    assert any(node.op == "cache_read" for node in operators)
    assert not any(node.op == "add" for node in operators)


def test_partial_integer_division_remains_live_after_common_expression_elimination() -> (
    None
):
    with Graph() as graph:
        tokens = Input(name="token_ids", shape=TensorShape((8,)), dtype="int64")
        divisor = Input(name="group_size", shape=TensorShape(()), dtype="int64")
        quotient = FloorDiv(tokens, divisor)
        duplicate = FloorDiv(tokens, divisor)
        Output(Add(quotient, duplicate), name="groups")
    _optimize(graph, _infer(graph))
    _validate(graph)
    assert graph._region is not None
    divisions = tuple(
        node for _, node in _walk(graph._region) if node.op == "floor_div"
    )
    assert len(divisions) == 1
    args = divisions[0].args
    assert isinstance(args, _BinaryArgs)
    assert args.a is tokens
    assert args.b is divisor


@pytest.mark.parametrize("chosen", [True, False])
def test_constant_model_branch_keeps_only_selected_checkpoint(chosen: bool) -> None:
    gate = _values("mlp.gate_proj")
    up = _values("mlp.up_proj")
    with Graph() as graph:
        selected = If(
            Constant(chosen),
            then_branch=lambda: (Constant(gate),),
            else_branch=lambda: (Constant(up),),
        )[0]
        Output(selected, name="selected")
    _optimize(graph, _infer(graph))
    _validate(graph)
    producer = graph._outputs["selected"]._producer
    assert producer is not None
    assert isinstance(producer.args, _ConstantArgs)
    np.testing.assert_array_equal(producer.args.value, gate if chosen else up)
