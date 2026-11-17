from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from trillim.compiler.passes.canonicalize import _normal
from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.passes_dataclasses import _MatchingState
from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _read_values
from trillim.tdl import (
    Add,
    Cast,
    Constant,
    Cos,
    Div,
    Erf,
    Gelu,
    Graph,
    Matmul,
    Mean,
    Mul,
    Output,
    Permute,
    Reshape,
    RMSNorm,
    Sigmoid,
    Silu,
    Sin,
    Sqrt,
    Tensor,
)

_CHECKPOINT = (
    Path(__file__).parents[1] / "fixtures" / "weights" / "Qwen3-0.6B.safetensors"
)


def _weight(name: str) -> Tensor:
    checkpoint = _load_checkpoint("Qwen3-0.6B", _CHECKPOINT)
    source = checkpoint.tensors[name]
    return Constant(
        _read_values(source, 0, math.prod(source.shape)).reshape(source.shape)
    )


def _projection(name: str) -> Tensor:
    return _weight(f"model.layers.0.self_attn.{name}_proj.weight")


def _assert_equivalent(graph: Graph, left: Tensor, right: Tensor) -> None:
    inference = _infer(graph)
    state = _MatchingState(inference)
    left_form = _normal(left, inference, state)
    right_form = _normal(right, inference, state)
    assert left_form.key == right_form.key
    assert left_form.origins
    assert right_form.origins
    assert inference.tensors[left].shape == inference.tensors[right].shape


@pytest.mark.parametrize(
    "identity",
    [
        pytest.param(lambda x: Cast(x, "fp32"), id="identity-cast"),
        pytest.param(
            lambda x: Reshape(Reshape(x, (8, 4, 32)), (32, 32)), id="split-merge-heads"
        ),
        pytest.param(
            lambda x: Permute(Permute(x, (1, 0)), (1, 0)),
            id="inverse-projection-transposes",
        ),
    ],
)
def test_projection_views_preserve_mathematics(
    identity: Callable[[Tensor], Tensor],
) -> None:
    with Graph() as graph:
        q = _projection("q")
        left = Mul(q, 1)
        right = identity(q)
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_qwen_rmsnorm_scale_reassociation() -> None:
    with Graph() as graph:
        q = _projection("q")
        weight = _weight("model.layers.0.input_layernorm.weight")
        left = RMSNorm(q, weight=weight, eps=1e-6)
        variance = Mean(Mul(q, q), axes=(-1,), keepdims=True)
        right = Mul(q, Div(weight, Sqrt(Add(variance, 1e-6))))
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_qwen_swiglu_gate_reassociation() -> None:
    with Graph() as graph:
        gate = _weight("model.layers.0.mlp.gate_proj.weight")
        up = _weight("model.layers.0.mlp.up_proj.weight")
        left = Mul(Silu(gate), up)
        right = Mul(gate, Mul(up, Sigmoid(gate)))
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_gemma_gelu_projection_factoring() -> None:
    with Graph() as graph:
        gate = _weight("model.layers.0.mlp.gate_proj.weight")
        up = _weight("model.layers.0.mlp.up_proj.weight")
        left = Mul(Gelu(gate), up)
        cdf = Erf(Mul(gate, 0.7071067811865476))
        right = Mul(Mul(up, gate), Add(Mul(cdf, 0.5), 0.5))
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_attention_projection_transpose_reverses_matmul_factors() -> None:
    with Graph() as graph:
        q, k = _projection("q"), _projection("k")
        left = Permute(Matmul(q, k), (1, 0))
        right = Matmul(Permute(k, (1, 0)), Permute(q, (1, 0)))
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_rotary_phase_sine_cosine_preserve_norm() -> None:
    with Graph() as graph:
        phase = _projection("q")
        sine, cosine = Sin(phase), Cos(phase)
        left = Add(Mul(sine, sine), Mul(cosine, cosine))
        right = Add(Mul(phase, 0), 1)
        Output(left, name="left")
        Output(right, name="right")
    _assert_equivalent(graph, left, right)


def test_norm_epsilon_and_projection_order_remain_distinct() -> None:
    with Graph() as graph:
        q, k = _projection("q"), _projection("k")
        norm_a = RMSNorm(q, eps=1e-6)
        norm_b = RMSNorm(q, eps=1e-5)
        qk, kq = Matmul(q, k), Matmul(k, q)
        for name, tensor in (("a", norm_a), ("b", norm_b), ("qk", qk), ("kq", kq)):
            Output(tensor, name=name)
    inference = _infer(graph)
    state = _MatchingState(inference)
    assert (
        _normal(norm_a, inference, state).key != _normal(norm_b, inference, state).key
    )
    assert _normal(qk, inference, state).key != _normal(kq, inference, state).key
