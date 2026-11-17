from __future__ import annotations

from pathlib import Path

import pytest

from trillim.compiler.weights.resolve import _resolve_payloads, _resolve_weights
from trillim.tdl import Constant, Graph, Output, Weight

from .helpers import FIXTURES, PROJECTION


def test_binding_preserves_source_dtype_and_explicit_override() -> None:
    with Graph() as graph:
        native = Constant(Weight(f"Model.{PROJECTION}"))
        overridden = Constant(Weight(f"Model.{PROJECTION}", dtype="fp16"), dtype="fp32")
        Output(native, name="native")
        Output(overridden, name="overridden")
    index = _resolve_weights(graph, {"Model": FIXTURES / "Qwen3-0.6B.safetensors"})
    assert native.dtype == "bf16"
    assert overridden.dtype == "fp16"
    assert native.shape is not None and native.shape.dimensions == (32, 32)
    assert index.bindings[native].source is index.bindings[overridden].source
    resolved = list(_resolve_payloads(index))
    assert {weight.record.dtype for weight in resolved} == {"bf16", "fp16"}
    assert all("@" in weight.record.name for weight in resolved)


def test_binding_errors_and_checkpoint_identity() -> None:
    with Graph() as graph:
        first = Constant(Weight(f"First.{PROJECTION}"))
        second = Constant(Weight(f"Second.{PROJECTION}"))
        Output(first, name="first")
        Output(second, name="second")
    path = FIXTURES / "Qwen3-0.6B.safetensors"
    with pytest.raises(ValueError, match="missing checkpoint binding"):
        _resolve_weights(graph, {})
    index = _resolve_weights(graph, {"First": path, "Second": path})
    assert (
        index.bindings[first].source.identity != index.bindings[second].source.identity
    )
    with pytest.raises(ValueError, match="completed graph"):
        _resolve_weights(Graph(), {})


def test_tied_metadata_rejects_changed_real_copy(tmp_path: Path) -> None:
    from .helpers import checkpoint, raw, source

    entry = source()
    data = raw(entry)
    changed = data[2:4] + data[2:]
    path = checkpoint(
        tmp_path / "tied.safetensors",
        {
            "embedding": ("BF16", entry.shape, data),
            "head": ("BF16", entry.shape, changed),
        },
        {"head": "embedding"},
    )
    with Graph() as graph:
        Output(Constant(Weight("Model.head")), name="head")
    index = _resolve_weights(graph, {"Model": path})
    with pytest.raises(ValueError, match="different values"):
        list(_resolve_payloads(index))


def test_config_embedding_tie_and_shape_mismatch(tmp_path: Path) -> None:
    import json

    from trillim.tdl.core.core_dataclasses import Shape

    from .helpers import checkpoint, raw, source

    entry = source()
    path = checkpoint(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", entry.shape, raw(entry))},
    )
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    weight = Weight("Model.lm_head.weight")
    weight.shape = Shape((31, 32))
    with Graph() as graph:
        Output(Constant(weight), name="head")
    with pytest.raises(ValueError, match="declared weight shape"):
        _resolve_weights(graph, {"Model": path})
    weight.shape = Shape((None, 32))
    index = _resolve_weights(graph, {"Model": path})
    assert (
        index.sources["Model.lm_head.weight"].identity
        == "Model.model.embed_tokens.weight"
    )
