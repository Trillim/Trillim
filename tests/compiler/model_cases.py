from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trillim.compiler.weights.reader import _load_checkpoint
from trillim.compiler.weights.resolve import _read_values
from trillim.tdl import (
    FFN,
    Add,
    Attention,
    Cache,
    Compare,
    Constant,
    Conv,
    Dim,
    FullLike,
    Graph,
    If,
    Input,
    Iota,
    Linear,
    MergeHeads,
    Model,
    Mul,
    Output,
    Permute,
    Reduce,
    Reshape,
    RMSNorm,
    Rotary,
    Scatter,
    Slice,
    Softmax,
    SplitHeads,
    Tensor,
    While,
)

from .compiler_dataclasses import ModelCase

FIXTURES = Path(__file__).parent / "fixtures" / "weights"
CASE_NAMES = (
    "projection",
    "views",
    "softmax",
    "swiglu",
    "gqa",
    "kv_cache",
    "recurrence",
    "conv_branch_reduce",
)


def values(name: str, model: str = "Qwen3-0.6B") -> NDArray[np.float32]:
    checkpoint = _load_checkpoint(model, FIXTURES / f"{model}.safetensors")
    source = checkpoint.tensors[name]
    return (
        _read_values(source, 0, math.prod(source.shape))
        .reshape(source.shape)
        .astype(np.float32)
    )


def model_case(name: str) -> ModelCase:
    hidden = values("model.embed_tokens.weight")[:4].copy()
    norm = values("model.layers.0.input_layernorm.weight")
    q, k, v, out = (
        values(f"model.layers.0.self_attn.{projection}_proj.weight")
        for projection in ("q", "k", "v", "o")
    )
    up, gate, down = (
        values(f"model.layers.0.mlp.{projection}_proj.weight")
        for projection in ("up", "gate", "down")
    )
    inputs: dict[str, NDArray[np.generic]] = {"hidden": hidden}
    oracle: NDArray[np.float32] | None = None
    shape = hidden.shape
    with Graph() as graph:
        x = Input(name="hidden", shape=(None, 32), dtype="fp32")
        if name == "projection":
            result = Linear(
                x, Model("checkpoint")["model.layers.0.self_attn.q_proj.weight"]
            )
            oracle = hidden @ q.T
        elif name == "views":
            result = MergeHeads(SplitHeads(x, heads=4))
            oracle = hidden
        elif name == "softmax":
            result = Softmax(x)
            exp = np.exp(hidden - np.max(hidden, axis=-1, keepdims=True))
            oracle = (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
        elif name == "swiglu":
            normalized = RMSNorm(x, weight=Constant(norm), eps=1e-6)
            result = Add(
                x,
                FFN(
                    normalized,
                    up=Constant(up),
                    gate=Constant(gate),
                    down=Constant(down),
                ),
            )
            normalized_np = (
                hidden
                / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-6)
                * norm
            )
            g = normalized_np @ gate.T
            oracle = hidden + (g / (1 + np.exp(-g)) * (normalized_np @ up.T)) @ down.T
        elif name in {"gqa", "kv_cache"}:
            normalized = RMSNorm(x, weight=Constant(norm), eps=1e-6)
            queries = SplitHeads(Linear(normalized, Constant(q)), heads=4)
            keys = SplitHeads(
                Slice(Linear(normalized, Constant(k)), axis=1, stop=16, rank=2), heads=2
            )
            vals = SplitHeads(
                Slice(Linear(normalized, Constant(v)), axis=1, stop=16, rank=2), heads=2
            )
            positions = Iota(Dim(x, 0))
            queries = Rotary(queries, positions, dim=8, theta=1_000_000.0)
            keys = Rotary(keys, positions, dim=8, theta=1_000_000.0)
            if name == "kv_cache":
                key = Cache(
                    name="key", shape=(8, 2, 8), dtype="fp32", mode="append", axis=0
                )
                value = Cache(
                    name="value", shape=(8, 2, 8), dtype="fp32", mode="append", axis=0
                )
                key.append(Slice(keys, axis=0, stop=2, rank=3))
                snapshot = key.read()
                key.append(Slice(keys, axis=0, start=2, rank=3))
                Output(snapshot, name="keys_prefix")
                value.append(vals)
                keys, vals = key.read(), value.read()
            result = Add(
                x, Linear(MergeHeads(Attention(queries, keys, vals)), Constant(out))
            )
            normalized_np = (
                hidden
                / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-6)
                * norm
            )
            q_np = (normalized_np @ q.T).reshape(4, 4, 8)
            k_np = (normalized_np @ k.T)[:, :16].reshape(4, 2, 8)
            v_np = (normalized_np @ v.T)[:, :16].reshape(4, 2, 8)
            angles = np.arange(4)[:, None, None] * np.exp(
                np.arange(4) * (-2 * math.log(1_000_000.0) / 8)
            )
            cosine, sine = np.cos(angles), np.sin(angles)
            q_np = np.concatenate(
                (
                    q_np[:, :, :4] * cosine - q_np[:, :, 4:] * sine,
                    q_np[:, :, 4:] * cosine + q_np[:, :, :4] * sine,
                ),
                axis=-1,
            )
            k_np = np.concatenate(
                (
                    k_np[:, :, :4] * cosine - k_np[:, :, 4:] * sine,
                    k_np[:, :, 4:] * cosine + k_np[:, :, :4] * sine,
                ),
                axis=-1,
            )
            scores = np.einsum(
                "thd,shd->hts", q_np, np.repeat(k_np, 2, axis=1)
            ) / math.sqrt(8)
            scores = np.where(
                np.arange(4)[None, :] <= np.arange(4)[:, None], scores, -np.inf
            )
            probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            context = np.einsum(
                "hts,shd->thd", probabilities, np.repeat(v_np, 2, axis=1)
            ).reshape(4, 32)
            oracle = (hidden + context @ out.T).astype(np.float32)

        elif name == "recurrence":
            state = Cache(name="state", shape=(32,), dtype="fp32")
            initial = Reshape(Slice(x, axis=0, start=-1, rank=2), (32,))

            def body(s: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
                index, previous, output = s
                updated = Add(
                    Linear(
                        RMSNorm(previous, weight=Constant(norm), eps=1e-6), Constant(q)
                    ),
                    state.read(),
                )
                state.assign(updated)
                return (
                    Add(index, 1),
                    updated,
                    Scatter(output, indices=Reshape(index, (1,)), values=updated),
                )

            generated = While(
                (Constant(0), initial, FullLike(x, shape=(3, 32))),
                condition=lambda s: Compare(s[0], 3, predicate="lt"),
                body=body,
            )
            result = generated[2]
            shape = (3, 32)
            previous = hidden[-1].copy()
            rows = []
            persistent = np.zeros(32, np.float32)
            for _ in range(3):
                normalized_np = (
                    previous / np.sqrt(np.mean(previous * previous) + 1e-6) * norm
                )
                previous = normalized_np @ q.T + persistent
                persistent = previous.copy()
                rows.append(previous)
            oracle = np.stack(rows)
        elif name == "conv_branch_reduce":
            inputs["condition"] = np.asarray(True)
            condition = Input(name="condition", shape=(), dtype="bool")
            weight = values("mimi.decoder.model.9.block.1.conv.weight", "PocketTTS")
            channels = Permute(x, (1, 0))
            channels = Slice(channels, axis=0, stop=4, rank=2)
            y = Conv(channels, Constant(weight), kernel=(3,), padding=1)
            y = If(
                condition, then_branch=lambda: (Mul(y, 0.5),), else_branch=lambda: (y,)
            )[0]
            moments = Reduce(
                (y, y),
                initial=(Constant(0.0), Constant(0.0)),
                axes=(1,),
                reducer=lambda a, b: (Add(a[0], b[0]), Add(a[1], Mul(b[1], b[1]))),
            )
            result = moments[0]
            Output(moments[1], name="energy")
            shape = (4,)
            padded = np.pad(hidden[:, :4].T, ((0, 0), (1, 1)))
            convolved = np.stack(
                [
                    np.sum(padded[:, i : i + 3][None, :, :] * weight, axis=(1, 2))
                    for i in range(4)
                ],
                axis=1,
            )
            oracle = np.sum(convolved * 0.5, axis=1)
        else:
            raise ValueError(name)
        Output(result, name="result")
    return ModelCase(graph, inputs, shape, oracle)
