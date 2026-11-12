"""Validate and benchmark the Gemma 4 text decoder with native checkpoint weights.

Includes per-layer embeddings, shared KV, proportional RoPE and logit softcap.
Vision/audio encoders, tokenization and media inputs are outside the timing scope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ModelMetadata, Options, benchmark
from trillim.kernels.neon.benchmark_qwen35 import (
    ModelFactory,
    ReferenceMetadata,
    TorchAPI,
    prompt_ids,
)
from trillim.runtime import Runtime

_REFERENCE_RTOL = 2e-4
_REFERENCE_ATOL = 2e-4
_STRICT_REFERENCE_ATOL = 2e-5


class RopeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rope_type: Literal["default", "proportional"]
    rope_theta: float = Field(gt=0)
    partial_rotary_factor: float = Field(default=1.0, gt=0, le=1)


class TextConfig(ModelMetadata):
    vocab_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    global_head_dim: int = Field(gt=0)
    rms_norm_eps: float = Field(gt=0)
    hidden_activation: Literal["gelu_pytorch_tanh"]
    attention_bias: Literal[False]
    attention_k_eq_v: Literal[False]
    enable_moe_block: Literal[False]
    use_bidirectional_attention: None
    tie_word_embeddings: Literal[True]
    use_double_wide_mlp: bool
    layer_types: list[Literal["sliding_attention", "full_attention"]]
    sliding_window: int = Field(gt=0)
    num_kv_shared_layers: int = Field(ge=0)
    hidden_size_per_layer_input: int = Field(gt=0)
    vocab_size_per_layer_input: int = Field(gt=0)
    final_logit_softcapping: float = Field(gt=0)
    rope_parameters: dict[str, RopeConfig]

    @model_validator(mode="after")
    def supported_architecture(self) -> Self:
        if self.model_type != "gemma4_text":
            raise ValueError("expected a Gemma 4 text configuration")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer types must describe every decoder layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must divide into KV groups")
        first_shared = self.num_hidden_layers - self.num_kv_shared_layers
        if not 0 < first_shared <= self.num_hidden_layers:
            raise ValueError("KV sharing needs at least one unshared decoder layer")
        if not set(self.layer_types) <= set(self.layer_types[:first_shared]):
            raise ValueError("shared layers need an earlier KV source of each type")
        for layer_type in set(self.layer_types):
            rope = self.rope_parameters[layer_type]
            width = (
                self.head_dim
                if layer_type == "sliding_attention"
                else self.global_head_dim
            )
            if width % 2 or not int(width * rope.partial_rotary_factor / 2):
                raise ValueError("rotary dimensions must be positive and even")
            if rope.rope_type == "default" and rope.partial_rotary_factor != 1.0:
                raise ValueError("default rotary requires the complete head")
        return self


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: Literal["gemma4"]
    text_config: TextConfig


class GemmaOptions(Options):
    backend: Literal["scalar", "neon"] = "scalar"
    run_reference: bool = False
    validate_only: bool = False
    validate_reference: Path | None = None


class GemmaReferenceMetadata(ReferenceMetadata):
    checkpoint_sha256: str
    reference_scope: str = "Full text decoder; no vision/audio inputs"


def read_config(checkpoint: Path) -> TextConfig:
    return CheckpointConfig.model_validate_json(
        (checkpoint / "config.json").read_bytes()
    ).text_config


def graph_source(config: TextConfig, capacity: int) -> str:
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    frequency_source: list[str] = []
    for layer_type in sorted(set(config.layer_types)):
        rope = config.rope_parameters[layer_type]
        width = (
            config.head_dim
            if layer_type == "sliding_attention"
            else config.global_head_dim
        )
        rotated = int(width * rope.partial_rotary_factor / 2)
        frequency_source.append(
            f"frequencies[{layer_type!r}] = np.concatenate((np.reciprocal(np.power(np.float32({rope.rope_theta!r}), np.arange(0, {2 * rotated}, 2, dtype=np.float32) / np.float32({width}))), np.zeros({width // 2 - rotated}, dtype=np.float32)))"
        )
    frequencies = "\n".join(frequency_source)
    return f"""from trillim.tdl import *
import numpy as np

weights = Model("model")
layer_types = {tuple(config.layer_types)!r}
frequencies = {{}}
{frequencies}
with Graph() as graph:
    tokens = Input(name="tokens", shape=(None,), dtype="int64")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    hidden = Mul(Cast(Embedding(tokens, weights["model.language_model.embed_tokens.weight"]), "fp32"), {config.hidden_size**0.5!r})
    identity = Reshape(Mul(Cast(Embedding(tokens, weights["model.language_model.embed_tokens_per_layer.weight"]), "fp32"), {config.hidden_size_per_layer_input**0.5!r}), (Dim(tokens, 0), {config.num_hidden_layers}, {config.hidden_size_per_layer_input}))
    projected = Linear(hidden, weights["model.language_model.per_layer_model_projection.weight"])
    projected = Reshape(Mul(projected, {config.hidden_size**-0.5!r}), (Dim(tokens, 0), {config.num_hidden_layers}, {config.hidden_size_per_layer_input}))
    projected = RMSNorm(projected, weight=Cast(Constant(weights["model.language_model.per_layer_projection_norm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
    per_layer_inputs = Mul(Add(projected, identity), {2**-0.5!r})
    shared = {{}}
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{{layer}}."
        head_dim = {config.head_dim} if layer_type == "sliding_attention" else {config.global_head_dim}
        window = {config.sliding_window} if layer_type == "sliding_attention" else None
        residual = hidden
        normed = RMSNorm(hidden, weight=Cast(Constant(weights[prefix + "input_layernorm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        query = SplitHeads(Linear(normed, weights[prefix + "self_attn.q_proj.weight"]), heads={config.num_attention_heads}, head_dim=head_dim)
        query = RMSNorm(query, weight=Cast(Constant(weights[prefix + "self_attn.q_norm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        query = Rotary(query, positions, dim=head_dim, frequencies=Constant(frequencies[layer_type]))
        if layer < {first_shared}:
            key = SplitHeads(Linear(normed, weights[prefix + "self_attn.k_proj.weight"]), heads={config.num_key_value_heads}, head_dim=head_dim)
            key = RMSNorm(key, weight=Cast(Constant(weights[prefix + "self_attn.k_norm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
            key = Rotary(key, positions, dim=head_dim, frequencies=Constant(frequencies[layer_type]))
            value = SplitHeads(Linear(normed, weights[prefix + "self_attn.v_proj.weight"]), heads={config.num_key_value_heads}, head_dim=head_dim)
            value = RMSNorm(value, eps={config.rms_norm_eps!r})
            keys = Cache(name=f"key_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, head_dim), dtype="fp32", mode="append", axis=0)
            values = Cache(name=f"value_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, head_dim), dtype="fp32", mode="append", axis=0)
            keys.append(key)
            values.append(value)
            shared[layer_type] = keys.read(), values.read()
        key, value = shared[layer_type]
        attended = Attention(query, key, value, scale=1.0, window=window)
        attended = Linear(MergeHeads(attended), weights[prefix + "self_attn.o_proj.weight"])
        attended = RMSNorm(attended, weight=Cast(Constant(weights[prefix + "post_attention_layernorm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        hidden = Add(residual, attended)
        residual = hidden
        normed = RMSNorm(hidden, weight=Cast(Constant(weights[prefix + "pre_feedforward_layernorm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        gate = Gelu(Linear(normed, weights[prefix + "mlp.gate_proj.weight"]), approximate="tanh")
        feedforward = Linear(Mul(gate, Linear(normed, weights[prefix + "mlp.up_proj.weight"])), weights[prefix + "mlp.down_proj.weight"])
        feedforward = RMSNorm(feedforward, weight=Cast(Constant(weights[prefix + "post_feedforward_layernorm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        hidden = Add(residual, feedforward)
        residual = hidden
        per_layer_input = Reshape(Slice(per_layer_inputs, axis=1, start=layer, stop=layer + 1, rank=3), (Dim(tokens, 0), {config.hidden_size_per_layer_input}))
        gate = Gelu(Linear(hidden, weights[prefix + "per_layer_input_gate.weight"]), approximate="tanh")
        projected = Linear(Mul(gate, per_layer_input), weights[prefix + "per_layer_projection.weight"])
        projected = RMSNorm(projected, weight=Cast(Constant(weights[prefix + "post_per_layer_input_norm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
        hidden = Mul(Add(residual, projected), Cast(Constant(weights[prefix + "layer_scalar"]), "fp32"))
    hidden = RMSNorm(hidden, weight=Cast(Constant(weights["model.language_model.norm.weight"]), "fp32"), eps={config.rms_norm_eps!r})
    logits = Linear(Slice(hidden, axis=0, start=-1, rank=2), weights["model.language_model.embed_tokens.weight"])
    Output(Mul(Tanh(Div(logits, {config.final_logit_softcapping!r})), {config.final_logit_softcapping!r}), name="logits")
"""


def compile_bundle(options: GemmaOptions, config: TextConfig) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = options.output.with_name(options.output.name + ".tdl")
    source = graph_source(config, options.cache)
    if source_path.exists() and source_path.read_text() != source:
        raise ValueError(f"existing graph source differs: {source_path}")
    source_path.write_text(source)
    if options.output.exists():
        return
    _compile_model(
        _CompileRequest(
            tdl_path=source_path,
            checkpoints={"model": options.checkpoint},
            output=options.output,
            backend=options.backend,
            threads=options.threads,
            max_input_size=options.prompt,
            depth="none",
        ),
        progress=lambda message: print(message, flush=True),
    )


def checkpoint_hash(checkpoint: Path) -> str:
    with (checkpoint / "model.safetensors").open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_reference(options: GemmaOptions) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path.home() / "Trillim/models/.hf-cache"))
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".trillim/gemma/torch-cache"))
    torch = cast(TorchAPI, importlib.import_module("torch"))
    transformers = importlib.import_module("transformers")
    factory = cast(ModelFactory, transformers.Gemma4ForConditionalGeneration)
    torch.set_num_threads(options.threads)
    torch.set_default_device("cpu")
    model = factory.from_pretrained(
        options.checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    source_module = importlib.import_module(
        "transformers.models.gemma4.modeling_gemma4"
    )
    if source_module.__file__ is None:
        raise ValueError("cannot identify the Transformers architecture source")
    source_path = Path(source_module.__file__)
    prompt = prompt_ids(options.prompt)
    token_ids = prompt
    positions = list(range(options.prompt))
    past: object | None = None
    generated: list[int] = []
    logits: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for step in range(options.decode + 1):
            output = model(
                input_ids=torch.tensor([token_ids], dtype=torch.long),
                position_ids=torch.tensor([positions], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            last = output.logits[0, -1, :].detach().float().cpu().numpy()
            if not np.isfinite(last).all():
                raise ValueError("nonfinite Transformers reference logits")
            logits.append(last.copy())
            generated.append(int(np.argmax(last)))
            token_ids = [generated[-1]]
            positions = [options.prompt + step]
            past = output.past_key_values
    metadata = GemmaReferenceMetadata(
        checkpoint=str(options.checkpoint),
        checkpoint_sha256=checkpoint_hash(options.checkpoint),
        config_sha256=hashlib.sha256(
            (options.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        transformers_version=str(transformers.__version__),
        torch_version=torch.__version__,
        architecture_source=str(source_path),
        architecture_url="https://github.com/huggingface/transformers/blob/v"
        + str(transformers.__version__)
        + "/src/transformers/models/gemma4/modeling_gemma4.py",
        architecture_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    path = options.output.with_name(options.output.name + "-transformers.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, logits=np.stack(logits))
    path.with_suffix(".json").write_text(metadata.model_dump_json(indent=2) + "\n")
    print(metadata.model_dump_json(indent=2), flush=True)


def validate_reference(options: GemmaOptions, path: Path) -> dict[str, object]:
    metadata = GemmaReferenceMetadata.model_validate_json(
        path.with_suffix(".json").read_bytes()
    )
    if Path(metadata.checkpoint).resolve() != options.checkpoint.resolve():
        raise ValueError("reference checkpoint path differs")
    if metadata.checkpoint_sha256 != checkpoint_hash(options.checkpoint):
        raise ValueError("reference checkpoint weights differ")
    if (
        metadata.config_sha256
        != hashlib.sha256((options.checkpoint / "config.json").read_bytes()).hexdigest()
    ):
        raise ValueError("reference checkpoint configuration differs")
    if metadata.prompt_token_ids != prompt_ids(options.prompt):
        raise ValueError("reference prompt differs")
    with np.load(path) as archive:
        expected = np.asarray(archive["logits"], dtype=np.float32)
    if len(expected) != options.decode + 1:
        raise ValueError("reference decode length differs")
    comparisons: list[dict[str, object]] = []
    actual_logits: list[NDArray[np.float32]] = []
    matches = True
    with Runtime(options.output, threads=options.threads) as runtime:
        for step, reference in enumerate(expected):
            tokens = (
                metadata.prompt_token_ids
                if not step
                else [metadata.generated_token_ids[step - 1]]
            )
            positions = (
                list(range(options.prompt)) if not step else [options.prompt + step - 1]
            )
            actual = np.asarray(
                runtime.run(
                    {
                        "tokens": np.asarray(tokens, dtype=np.int64),
                        "positions": np.asarray(positions, dtype=np.int64),
                    }
                )["logits"],
                dtype=np.float32,
            ).ravel()
            difference = actual.astype(np.float64) - reference.astype(np.float64)
            actual_logits.append(actual.copy())
            rms_error = float(np.sqrt(np.mean(difference * difference)))
            reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
            greedy_matches = int(np.argmax(actual)) == int(np.argmax(reference))
            close = bool(
                np.allclose(
                    actual, reference, rtol=_REFERENCE_RTOL, atol=_REFERENCE_ATOL
                )
            )
            matches = matches and greedy_matches and close
            comparisons.append(
                {
                    "step": step,
                    "max_absolute_error": float(np.max(np.abs(difference))),
                    "rms_error": rms_error,
                    "normalized_rms_error": rms_error / max(reference_rms, 1.0),
                    "outside_tolerance": int(
                        np.count_nonzero(
                            np.abs(difference)
                            > _REFERENCE_ATOL + _REFERENCE_RTOL * np.abs(reference)
                        )
                    ),
                    "outside_strict_tolerance": int(
                        np.count_nonzero(
                            np.abs(difference)
                            > _STRICT_REFERENCE_ATOL
                            + _REFERENCE_RTOL * np.abs(reference)
                        )
                    ),
                    "greedy_matches": greedy_matches,
                    "allclose": close,
                }
            )
    np.save(
        options.output.with_name(options.output.name + "-validation-logits.npy"),
        np.stack(actual_logits),
    )
    return {
        "transformers_reference": metadata.model_dump(),
        "comparisons": comparisons,
        "reference_matches": matches,
        "reference_relative_tolerance": _REFERENCE_RTOL,
        "reference_absolute_tolerance": _REFERENCE_ATOL,
        "strict_reference_absolute_tolerance": _STRICT_REFERENCE_ATOL,
        "reference_tolerance_scope": "Whole decoder FP32 accumulation; strict counts retained, per-kernel tolerances unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="gemma4-e2b")
    parser.add_argument("--backend", choices=("scalar", "neon"), default="scalar")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--prompt", type=int, default=4)
    parser.add_argument("--decode", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache", type=int, default=64)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    parser.add_argument("--run-reference", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-reference", type=Path)
    options = GemmaOptions.model_validate(vars(parser.parse_args()))
    if (
        not options.output.expanduser()
        .resolve()
        .is_relative_to(Path.home() / ".trillim")
    ):
        raise ValueError("benchmark artifacts must stay beneath ~/.trillim")
    if options.prompt + options.decode > options.cache:
        raise ValueError("cache must fit prompt and measured decode")
    config = read_config(options.checkpoint)
    if options.run_reference:
        run_reference(options)
        return
    compile_bundle(options, config)
    if options.compile_only:
        return
    if options.backend != "scalar":
        raise ValueError("runtime currently requires scalar registration")
    if options.validate_reference is None:
        raise ValueError(
            "validate against independent Transformers logits before timing"
        )
    results = validate_reference(options, options.validate_reference)
    path = options.output.with_name(options.output.name + "-results.json")
    if results["reference_matches"] is not True:
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        raise ValueError("Gemma 4 logits do not match the Transformers reference")
    if not options.validate_only:
        results.update(benchmark(options, config=config))
        results["text_scope"] = (
            f"All {config.num_hidden_layers} decoder layers, per-layer embeddings and full vocabulary logits"
        )
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
