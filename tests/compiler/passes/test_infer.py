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
    Dim,
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
    Shape,
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


@pytest.mark.parametrize("bound", [None, 17])
@pytest.mark.parametrize("middle", [0, 5])
def test_shape_round_trip_preserves_dimension_identity(
    bound: int | None, middle: int
) -> None:
    with Graph() as graph:
        hidden = Input(name="hidden", shape=(None, middle, 6), dtype="fp32")
        dimensions = Shape(hidden)
        restored = Reshape(hidden, dimensions)
        restored_dimensions = Shape(restored)
        result = Reshape(hidden, restored_dimensions)
        Output(result, name="result")

    inference = _infer(graph, max_input_size=bound)
    _validate(graph)
    expected_maximum = bound if bound is not None else (1 << 63) - 1
    values = inference.tensors[dimensions].values
    assert values is not None
    dynamic, fixed, width = values
    assert dynamic.op == "dimension"
    assert dynamic.tensor is hidden
    assert dynamic.axis == 0
    assert dynamic.minimum == 0
    assert dynamic.maximum == expected_maximum
    assert fixed.value == middle
    assert width.value == 6
    assert inference.tensors[dimensions].dtype == "int64"
    assert inference.tensors[restored_dimensions].values == values
    assert inference.tensors[restored].shape == values
    assert inference.tensors[result].shape == values
    assert inference.tensors[result].maximum_shape == (expected_maximum, middle, 6)
    assert (
        _infer(graph, max_input_size=bound).tensors[result] == inference.tensors[result]
    )


def test_dynamic_reshape_bounds_keep_zero_and_fractional_divisor_guards() -> None:
    with Graph() as graph:
        hidden = Input(name="hidden", shape=(None,), maximum_shape=(91,), dtype="fp32")
        groups = Input(name="groups", shape=(None,), maximum_shape=(7,), dtype="fp32")
        grouped = Reshape(hidden, (Dim(groups, 0), -1))
        dimensions = Shape(grouped)
        result = Reshape(hidden, dimensions)
        Output(result, name="result")

    inference = _infer(graph)
    _validate(graph)
    source_shape = inference.tensors[hidden].shape
    group_shape = inference.tensors[groups].shape
    shape = inference.tensors[grouped].shape
    assert source_shape is not None and group_shape is not None and shape is not None
    divisor, quotient = shape
    assert divisor == group_shape[0]
    assert divisor.minimum == 0
    assert quotient.op == "floor_div"
    assert quotient.args == (source_shape[0], divisor)
    # One group can contain all 91 elements; dividing the upper bounds gives 13.
    assert (quotient.minimum, quotient.maximum) == (0, 91)
    assert inference.tensors[dimensions].values == shape
    assert inference.tensors[result].shape == shape
    assert inference.tensors[result].maximum_shape == (7, 91)

    guards = [c for c in inference.constraints if c.operator is grouped._producer]
    positive, exact = guards
    assert positive.predicate == "ge"
    assert positive.left == divisor
    assert positive.right.value == 1
    assert exact.predicate == "eq"
    assert exact.left == source_shape[0]
    assert exact.right.op == "mul"
    assert exact.right.args == (divisor, quotient)
    # This equality must survive: 91 // 2 fits the bounds but loses an element.
    repeated = _infer(graph)
    assert repeated.tensors[result] == inference.tensors[result]
    assert repeated.constraints == inference.constraints


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
