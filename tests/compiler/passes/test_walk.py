from __future__ import annotations

from collections.abc import Sequence
from graphlib import CycleError
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trillim.compiler.passes.walk import _collect_checkpoints, _walk
from trillim.tdl import (
    Add,
    Cache,
    Compare,
    Constant,
    Construct,
    Graph,
    If,
    Input,
    Linear,
    Model,
    Mul,
    Output,
    Sub,
    Tensor,
    Weight,
    While,
)


@given(permutation=st.permutations(range(7)))
def test_shuffled_dependency_order(permutation: Sequence[int]) -> None:
    orders: list[tuple[int, ...]] = []
    for _ in range(2):
        with Graph() as graph:
            x = Input(name="x", shape=(2,), dtype="fp32")
            shared = Add(x, 1)
            left = Mul(shared, 2)
            right = Sub(shared, 3)
            Output(Add(left, right), name="y")
            Constant(99)

        region = graph._region
        assert region is not None
        original = region.operators
        assert len(original) == len(permutation)
        region.operators = tuple(original[index] for index in permutation)

        entries = list(_walk(region))
        positions = {operator: index for index, (_, operator) in enumerate(entries)}
        assert set(positions) == set(original)
        assert len(entries) == len(original)
        assert all(owner is region for owner, _ in entries)
        assert entries == list(_walk(region))
        for operator in original:
            for value in operator.inputs:
                if value._producer is not None:
                    assert positions[value._producer] < positions[operator]
        labels = {operator: index for index, operator in enumerate(original)}
        orders.append(tuple(labels[operator] for _, operator in entries))

    assert orders[0] == orders[1]


def test_empty_region() -> None:
    with Graph() as graph:
        pass

    assert graph._region is not None
    assert list(_walk(graph._region)) == []


def test_cache_effect_order() -> None:
    with Graph() as graph:
        state = Cache(name="state", shape=(), dtype="fp32")
        state.assign(Constant(1.0))
        snapshot = state.read()
        state.assign(snapshot)
        state.assign(Constant(2.0))
        Output(snapshot, name="snapshot")

    region = graph._region
    assert region is not None
    effects = tuple(
        operator
        for operator in region.operators
        if operator.op in ("cache", "cache_assign", "cache_read")
    )
    region.operators = tuple(reversed(region.operators))

    entries = list(_walk(region))

    assert tuple(operator for _, operator in entries if operator in effects) == effects
    assert len(entries) == len(region.operators)


def test_nested_regions_and_captures() -> None:
    @Construct
    def update(value: Tensor, offset: Tensor) -> Tensor:
        return Add(value, offset)

    with Graph() as graph:
        x = Input(name="x", shape=(), dtype="int64")

        def body(state: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            return If(
                Compare(state[0], 2, predicate="lt"),
                then_branch=lambda: (update(state[0], x),),
                else_branch=lambda: (Sub(state[0], x),),
            )

        result = While(
            (x,),
            condition=lambda state: Compare(state[0], 4, predicate="lt"),
            body=body,
        )[0]
        Output(result, name="y")

    region = graph._region
    assert region is not None
    input_op, loop_op, output_op = region.operators
    condition, body_region = loop_op.regions
    compare_op, branch_op = body_region.operators
    then_region, else_region = branch_op.regions
    (composition,) = then_region.operators
    (composition_body,) = composition.regions

    expected = [
        (region, input_op),
        (condition, condition.operators[0]),
        (body_region, compare_op),
        (composition_body, composition_body.operators[0]),
        (then_region, composition),
        (else_region, else_region.operators[0]),
        (body_region, branch_op),
        (region, loop_op),
        (region, output_op),
    ]
    entries = list(_walk(region))

    assert entries == expected
    for (owner, _), (expected_owner, _) in zip(entries, expected, strict=True):
        assert owner is expected_owner


def test_nested_cache_dependency_stays_in_parent_region() -> None:
    with Graph() as graph:
        state = Cache(name="state", shape=(), dtype="fp32")
        result = If(
            Constant(True),
            then_branch=lambda: (state.read(),),
            else_branch=lambda: (Constant(0.0),),
        )[0]

    assert graph._region is not None
    assert result._producer is not None
    body = result._producer.regions[0]

    assert list(_walk(body)) == [(body, body.operators[0])]


@pytest.mark.parametrize("self_cycle", [True, False], ids=["self", "two-node"])
def test_dependency_cycles(self_cycle: bool) -> None:
    with Graph() as graph:
        x = Constant(1)
        Add(x, 1)

    assert graph._region is not None
    first, second = graph._region.operators
    first.dependencies = (first,) if self_cycle else (second,)

    with pytest.raises(CycleError):
        list(_walk(graph._region))


def test_no_weight_references() -> None:
    with Graph() as graph:
        Output(Constant(1), name="value")

    assert graph._region is not None
    assert _collect_checkpoints(graph._region, {}) == ()


@pytest.mark.parametrize("missing", [None, "base", "adapter"])
def test_base_and_adapter_checkpoints(tmp_path: Path, missing: str | None) -> None:
    base = Model("base")
    adapter = Model("adapter")
    with Graph() as graph:
        x = Input(name="x", shape=(1, 2), dtype="fp32")
        main = Linear(x, base["linear.weight"], bias=base["linear.bias"])
        down = Linear(x, adapter["down.weight"], bias=adapter["down.bias"])
        delta = Linear(down, adapter["up.weight"], bias=adapter["up.bias"])
        Output(Add(main, delta), name="y")

    assert graph._region is not None
    bindings = {name: tmp_path for name in ("base", "adapter") if name != missing}
    if missing is None:
        assert _collect_checkpoints(graph._region, bindings) == ("base", "adapter")
    else:
        with pytest.raises(ValueError, match=missing):
            _collect_checkpoints(graph._region, bindings)


def test_checkpoints_in_nested_branches(tmp_path: Path) -> None:
    base, left, right = Model("base"), Model("left"), Model("right")
    with Graph() as graph:
        result = While(
            (Constant(base["initial"]),),
            condition=lambda _state: Constant(False),
            body=lambda state: If(
                Constant(True),
                then_branch=lambda: (Add(state[0], Constant(left["bias"])),),
                else_branch=lambda: (Add(state[0], Constant(right["bias"])),),
            ),
        )[0]
        Output(result, name="result")

    assert graph._region is not None
    bindings = {name: tmp_path for name in ("base", "left", "right")}
    assert _collect_checkpoints(graph._region, bindings) == ("base", "left", "right")


def test_unused_argument_metadata_is_not_a_weight_reference() -> None:
    @Construct
    def identity(x: Tensor, _weight: Weight) -> Tensor:
        return x

    unused = Model("unused")
    with Graph() as graph:
        Output(identity(Constant(1), unused["weight"]), name="value")

    assert graph._region is not None
    assert _collect_checkpoints(graph._region, {}) == ()
