from __future__ import annotations

import math
from pathlib import Path

import pytest

from trillim.compiler.passes.infer import _infer
from trillim.compiler.passes.validate import _validate
from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _read_values
from trillim.tdl import (
    Add,
    Cache,
    Cast,
    Constant,
    Graph,
    If,
    Input,
    Matmul,
    Mean,
    Mul,
    Output,
    Permute,
    Reshape,
    RMSNorm,
    Silu,
    Softmax,
    Tensor,
    TensorShape,
)
from trillim.tdl.core.core_pydantic_models import _CacheAssignArgs

_CHECKPOINT = (
    Path(__file__).parents[1] / "fixtures" / "weights" / "Qwen3-0.6B.safetensors"
)


def _weight(name: str) -> Tensor:
    checkpoint = _load_checkpoint("Qwen3-0.6B", _CHECKPOINT)
    source = checkpoint.tensors[f"model.layers.0.{name}.weight"]
    return Constant(
        _read_values(source, 0, math.prod(source.shape)).reshape(source.shape)
    )


def _architecture(name: str) -> tuple[Tensor, tuple[int, ...]]:
    q = _weight("self_attn.q_proj")
    if name == "cast":
        return Cast(q, "fp32"), (32, 32)
    if name == "transpose":
        return Permute(q, (1, 0)), (32, 32)
    if name == "head-split":
        return Reshape(q, (32, 4, 8)), (32, 4, 8)
    k = _weight("self_attn.k_proj")
    if name == "projection":
        return Matmul(q, Permute(k, (1, 0))), (32, 32)
    if name == "rmsnorm":
        return RMSNorm(q, weight=_weight("input_layernorm")), (32, 32)
    if name == "swiglu":
        gate = Matmul(q, _weight("mlp.gate_proj"))
        up = Matmul(q, _weight("mlp.up_proj"))
        return Matmul(Mul(Silu(gate), up), _weight("mlp.down_proj")), (32, 32)
    if name == "attention":
        logits = Mul(Matmul(q, Permute(k, (1, 0))), 1 / math.sqrt(32))
        return Softmax(logits, axis=-1), (32, 32)
    assert name == "mean-square"
    return Mean(Mul(q, q), axes=(-1,), keepdims=True), (32, 1)


@pytest.mark.parametrize(
    "architecture",
    [
        "cast",
        "transpose",
        "head-split",
        "projection",
        "rmsnorm",
        "swiglu",
        "attention",
        "mean-square",
    ],
)
def test_model_subgraphs_have_complete_stable_facts(architecture: str) -> None:
    with Graph() as graph:
        output, expected_shape = _architecture(architecture)
        Output(output, name="output")
    first = _infer(graph)
    _validate(graph)
    info = first.tensors[output]
    assert info.shape is not None
    assert tuple(dimension.value for dimension in info.shape) == expected_shape
    assert info.maximum_shape == expected_shape
    assert info.dtype == "fp32"
    second = _infer(graph)
    assert second.tensors[output] == info
    assert second.constraints == first.constraints
    _validate(graph)


@pytest.mark.parametrize("bound", [None, 64])
def test_dynamic_token_bound_propagates_through_projection(bound: int | None) -> None:
    with Graph() as graph:
        tokens = Input(name="hidden", shape=TensorShape((None, 32)), dtype="fp32")
        result = Matmul(tokens, _weight("self_attn.q_proj"))
        Output(result, name="query")
    inference = _infer(graph, max_input_size=bound)
    info = inference.tensors[result]
    assert info.shape is not None
    assert info.shape[0].minimum == 0
    assert info.shape[0].maximum == bound
    assert info.maximum_shape == (bound, 32)
    assert info.dtype == "fp32"


def test_projection_inner_dimension_contradiction_is_rejected() -> None:
    with Graph() as graph:
        hidden = Input(name="hidden", shape=TensorShape((8, 31)), dtype="fp32")
        Output(Matmul(hidden, _weight("self_attn.q_proj")), name="query")
    with pytest.raises(ValueError, match="matmul|dimension"):
        _infer(graph)


def test_mixed_activation_dtype_promotes_and_cache_storage_stays_declared() -> None:
    with Graph() as graph:
        cache = Cache(name="state", shape=(32, 32), dtype="bf16")
        query = _weight("self_attn.q_proj")
        mixed = Add(Cast(query, "fp16"), query)
        cache.assign(mixed)
        snapshot = cache.read()
        Output(mixed, name="mixed")
        Output(snapshot, name="snapshot")
    inference = _infer(graph)
    _validate(graph)
    assert inference.tensors[mixed].dtype == "fp32"
    assert inference.tensors[snapshot].dtype == "bf16"
    assert inference.tensors[snapshot].maximum_shape == (32, 32)
    assert graph._region is not None
    assignments = tuple(
        node
        for node in graph._region.operators
        if isinstance(node.args, _CacheAssignArgs)
    )
    assert len(assignments) == 1
    args = assignments[0].args
    assert isinstance(args, _CacheAssignArgs)
    assert inference.tensors[args.value].dtype == "bf16"


def test_branch_result_dtype_joins_real_projection_dtypes() -> None:
    with Graph() as graph:
        condition = Input(name="use_fp16", shape=TensorShape(()), dtype="bool")
        q = _weight("self_attn.q_proj")
        result = If(
            condition, then_branch=lambda: (Cast(q, "fp16"),), else_branch=lambda: (q,)
        )[0]
        Output(result, name="query")
    inference = _infer(graph)
    _validate(graph)
    assert inference.tensors[result].dtype == "fp32"
    assert inference.tensors[result].maximum_shape == (32, 32)
