from __future__ import annotations

from dataclasses import replace

import pytest

from trillim.compiler.passes.validate import validate
from trillim.tdl.core import (
    Add,
    Cache,
    Compare,
    Constant,
    Construct,
    Graph,
    If,
    Input,
    Output,
    Reduce,
    Tensor,
    While,
)
from trillim.tdl.core.core_dataclasses import CacheMode
from trillim.tdl.core.core_pydantic_models import (
    BinaryArgs,
    CacheAssignArgs,
    CacheReadArgs,
)
from trillim.tdl.core.operator import Operator, _inputs


def test_nested_graph() -> None:
    @Construct
    def update(value: Tensor, offset: Tensor) -> Tensor:
        return Add(value, offset)

    with Graph() as graph:
        x = Input(name="x", shape=(), dtype="int64")
        cache = Cache(name="state", shape=(), dtype="int64")

        def body(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            result = If(
                Compare(values[0], 2, predicate="lt"),
                then_branch=lambda: (update(values[0], x),),
                else_branch=lambda: (x,),
            )
            cache.assign(result[0])
            return result

        result = While(
            (x,),
            condition=lambda values: Compare(values[0], cache.read(), predicate="lt"),
            body=body,
        )[0]
        Output(result, name="result")
        Output(result, name="alias")

    assert graph.region is not None
    graph.region.operators = tuple(reversed(graph.region.operators))
    validate(graph)


def test_tuple_reduction() -> None:
    with Graph() as graph:
        x = Input(name="x", shape=(4,), dtype="fp32")
        results = Reduce(
            (x, x),
            initial=(Constant(0.0), Constant(1.0)),
            axes=(0,),
            reducer=lambda accumulators, elements: (
                Add(accumulators[0], elements[0]),
                Add(accumulators[1], elements[1]),
            ),
        )
        Output(results[0], name="sum")
        Output(results[1], name="offset")

    validate(graph)


def test_empty_graph_and_loop() -> None:
    with Graph() as empty:
        pass
    validate(empty)

    with Graph() as graph:
        While((), condition=lambda _: Constant(False), body=lambda _: ())
    validate(graph)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("producer", "producer link"),
        ("missing-consumer", "consumer links"),
        ("duplicate-consumer", "consumer links"),
        ("foreign-consumer", "consumer links"),
        ("stale-input", "stale input links"),
        ("duplicate-operator", "multiple positions"),
        ("duplicate-output", "multiple definitions"),
        ("input-binding", "bindings do not match"),
        ("output-binding", "bindings do not match"),
        ("root-result", "results do not match"),
    ],
)
def test_tensor_links(fault: str, message: str) -> None:
    with Graph() as graph:
        x = Input(name="x", shape=(), dtype="fp32")
        y = Add(x, 1)
        Output(y, name="y")

    assert graph.region is not None
    input_op, add_op, _ = graph.region.operators
    if fault == "producer":
        y.producer = input_op
    elif fault == "missing-consumer":
        x.consumers.clear()
    elif fault == "duplicate-consumer":
        x.consumers.append(add_op)
    elif fault == "foreign-consumer":
        Operator("foreign", BinaryArgs(a=x, b=1))
    elif fault == "stale-input":
        add_op.inputs = ()
    elif fault == "duplicate-operator":
        graph.region.operators += (add_op,)
    elif fault == "duplicate-output":
        add_op.outputs = (y, y)
    elif fault == "input-binding":
        graph.inputs["x"] = y
    elif fault == "output-binding":
        graph.outputs["y"] = x
    elif fault == "root-result":
        graph.region.outputs = ()
    else:
        raise AssertionError(fault)

    with pytest.raises(ValueError, match=message):
        validate(graph)


@pytest.mark.parametrize("kind", ["if", "while", "reduce", "construct"])
@pytest.mark.parametrize("fault", ["regions", "parameters", "results"])
def test_region_bindings(kind: str, fault: str) -> None:
    @Construct
    def identity(x: Tensor) -> Tensor:
        return x

    with Graph() as graph:
        x = Input(name="x", shape=(), dtype="fp32")
        if kind == "if":
            y = If(Constant(True), then_branch=lambda: (x,), else_branch=lambda: (x,))[
                0
            ]
        elif kind == "while":
            y = While((x,), condition=lambda _: Constant(False), body=lambda v: v)[0]
        elif kind == "reduce":
            y = Reduce(
                (x,), initial=(x,), axes=(), reducer=lambda a, b: (Add(a[0], b[0]),)
            )[0]
        else:
            y = identity(x)
        Output(y, name="y")

    assert y.producer is not None
    operator = y.producer
    region = operator.regions[0]
    if fault == "regions":
        operator.regions = ()
        operator.inputs = _inputs(operator.args, operator.regions)
    elif fault == "parameters":
        region.parameters += (Tensor(_parameter=True),)
    else:
        region.outputs = ()

    with pytest.raises(ValueError, match="regions|region bindings"):
        validate(graph)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "sibling"])
def test_region_captures(fault: str) -> None:
    with Graph() as graph:
        x = Input(name="x", shape=(), dtype="fp32")
        y = If(
            Constant(True),
            then_branch=lambda: (Add(x, 1),),
            else_branch=lambda: (Add(x, 2),),
        )[0]
        Output(y, name="y")

    assert y.producer is not None
    assert graph.region is not None
    then, otherwise = y.producer.regions
    if fault == "missing":
        then.captures = ()
        message = "captures do not match"
    elif fault == "duplicate":
        then.captures = (x, x)
        message = "captures do not match"
    else:
        replacement = otherwise.outputs[0] if fault == "sibling" else Constant(0)
        add = then.operators[0]
        add.args = BinaryArgs(a=replacement, b=1)
        add.inputs = (replacement,)
        then.captures = (replacement,)
        y.producer.inputs += (replacement,)
        message = "captures do not match"

    with pytest.raises(ValueError, match=message):
        validate(graph)


@pytest.mark.parametrize("fault", ["reuse", "recursive", "dependency", "parameter"])
def test_region_ownership(fault: str) -> None:
    with Graph() as graph:
        x = Constant(0)
        y = While(
            (x,),
            condition=lambda values: Compare(values[0], 3, predicate="lt"),
            body=lambda values: (Add(values[0], 1),),
        )[0]

    assert graph.region is not None
    assert y.producer is not None
    loop = y.producer
    condition, body = loop.regions
    if fault == "reuse":
        loop.regions = (condition, condition)
        message = "region is reused"
    elif fault == "recursive":
        loop.regions = (condition, graph.region)
        message = "region is reused"
    elif fault == "dependency":
        body.operators[0].dependencies = (condition.operators[0],)
        message = "dependency outside its scope"
    else:
        body.parameters[0].producer = loop
        message = "parameter cannot have a producer"

    with pytest.raises(ValueError, match=message):
        validate(graph)


@pytest.mark.parametrize("mode", ["overwrite", "append", "sliding"])
def test_cache_modes(mode: CacheMode) -> None:
    with Graph() as graph:
        cache = Cache(
            name="state",
            shape=(4,),
            dtype="fp32",
            mode=mode,
            axis=None if mode == "overwrite" else -1,
        )
        value = Input(
            name="value", shape=(4,) if mode == "overwrite" else (1,), dtype="fp32"
        )
        cache.assign(value)
        cache.read()
        if mode != "overwrite":
            cache.append(value)
    validate(graph)


def test_cache_order_through_tensor_flow_and_parallel_reads() -> None:
    with Graph() as graph:
        cache = Cache(name="state", shape=(), dtype="fp32")
        first = cache.read()
        second = cache.read()
        cache.assign(Add(first, second))

    assert graph.region is not None
    declaration, read1, read2, _, write = graph.region.operators
    read1.dependencies = (declaration,)
    read2.dependencies = (declaration,)
    write.dependencies = ()
    validate(graph)


@pytest.mark.parametrize("fault", ["ownership", "declaration", "summary", "order"])
def test_cache_validation(fault: str) -> None:
    with Graph() as graph:
        cache = Cache(name="state", shape=(), dtype="fp32")
        snapshot = cache.read()
        cache.assign(Constant(1.0))
        Output(snapshot, name="snapshot")

    assert graph.region is not None
    declaration, read, _, write, _ = graph.region.operators
    if fault == "ownership":
        read.args = CacheReadArgs(cache=replace(cache.spec))
        message = "belongs to another graph"
    elif fault == "declaration":
        graph.caches["state"] = replace(cache.spec, dtype="fp16")
        message = "invalid cache declaration"
    elif fault == "summary":
        graph.region.reads = ()
        message = "effects do not match"
    else:
        write.dependencies = (declaration,)
        message = "unordered cache effects"

    with pytest.raises(ValueError, match=message):
        validate(graph)


def test_nested_cache_effect_order() -> None:
    @Construct
    def reset(cache: Cache) -> tuple[Tensor, ...]:
        cache.assign(Constant(0.0))
        return ()

    with Graph() as graph:
        cache = Cache(name="state", shape=(), dtype="fp32")
        cache.read()
        reset(cache)

    validate(graph)
    assert graph.region is not None
    declaration, _, reset_op = graph.region.operators
    reset_op.dependencies = (declaration,)
    with pytest.raises(ValueError, match="unordered cache effects"):
        validate(graph)


@pytest.mark.parametrize("kind", ["while", "reduce"])
def test_region_effect_restrictions(kind: str) -> None:
    with Graph() as graph:
        cache = Cache(name="state", shape=(), dtype="fp32")
        x = Constant(0.0)
        if kind == "while":
            y = While((x,), condition=lambda _: Constant(False), body=lambda v: v)[0]
        else:
            y = Reduce((x,), initial=(x,), axes=(), reducer=lambda a, _: a)[0]

    assert graph.region is not None
    assert y.producer is not None
    operator = y.producer
    region = operator.regions[0]
    declaration = graph.region.operators[0]
    if kind == "while":
        effect = Operator(
            "cache_assign",
            CacheAssignArgs(cache=cache.spec, value=x, action="assign"),
            outputs=(),
        )
        region.captures = (x,)
        region.writes = graph.region.writes = (cache.spec,)
        message = "while condition cannot update caches"
    else:
        effect = Operator("cache_read", CacheReadArgs(cache=cache.spec))
        region.reads = graph.region.reads = (cache.spec,)
        message = "reduce body cannot access caches"
    effect.dependencies = operator.dependencies = (declaration,)
    region.operators += (effect,)

    with pytest.raises(ValueError, match=message):
        validate(graph)


def test_unfinished_graph() -> None:
    with pytest.raises(ValueError, match="completed"):
        validate(Graph())
