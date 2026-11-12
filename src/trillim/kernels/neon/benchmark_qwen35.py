"""Compile and validate the Qwen3.5 text decoder with native checkpoint weights.

Architecture reference: Hugging Face Transformers' modeling_qwen3_5.py.
Text positions use identical temporal/height/width coordinates, so its MRoPE
reduces to ordinary partial rotary embedding. Vision and MTP are not evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ModelMetadata, Options, benchmark
from trillim.runtime import Runtime


class RopeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    rope_type: Literal["default"]
    rope_theta: float = Field(gt=0)
    partial_rotary_factor: float = Field(gt=0, le=1)
    mrope_section: list[int]
    mrope_interleaved: bool = True


class TextConfig(ModelMetadata):
    vocab_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    rms_norm_eps: float = Field(gt=0)
    hidden_act: Literal["silu"]
    attention_bias: Literal[False] = False
    attn_output_gate: Literal[True] = True
    tie_word_embeddings: bool = False
    layer_types: list[Literal["linear_attention", "full_attention"]]
    linear_conv_kernel_dim: int = Field(gt=0)
    linear_key_head_dim: int = Field(gt=0)
    linear_value_head_dim: int = Field(gt=0)
    linear_num_key_heads: int = Field(gt=0)
    linear_num_value_heads: int = Field(gt=0)
    mamba_ssm_dtype: Literal["float32"] = "float32"
    rope_parameters: RopeConfig

    @model_validator(mode="after")
    def supported_architecture(self) -> Self:
        if self.model_type != "qwen3_5_text":
            raise ValueError("expected a Qwen3.5 text configuration")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer types must describe every decoder layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must divide into KV groups")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must divide into key groups")
        rotary_dim = int(self.head_dim * self.rope_parameters.partial_rotary_factor)
        if not rotary_dim or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be positive and even")
        if sum(self.rope_parameters.mrope_section) != rotary_dim // 2:
            raise ValueError("MRoPE sections must partition the rotary frequencies")
        return self


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: Literal["qwen3_5"]
    text_config: TextConfig


class Qwen35Options(Options):
    backend: Literal["scalar", "neon"] = "scalar"
    run_reference: bool = False
    validate_only: bool = False
    validate_reference: Path | None = None


class ReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: str
    config_sha256: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    transformers_version: str
    torch_version: str
    architecture_source: str
    architecture_url: str
    architecture_sha256: str
    reference_dtype: Literal["fp32"] = "fp32"
    weight_conversion: str = "Exact FP32 expansion of native checkpoint values"


class ReferenceTensor(Protocol):
    def __getitem__(self, key: object) -> ReferenceTensor: ...

    def detach(self) -> ReferenceTensor: ...

    def float(self) -> ReferenceTensor: ...

    def cpu(self) -> ReferenceTensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


class ReferenceOutput(Protocol):
    logits: ReferenceTensor
    past_key_values: object


class ReferenceModel(Protocol):
    def eval(self) -> Self: ...

    def __call__(
        self,
        *,
        input_ids: ReferenceTensor,
        position_ids: ReferenceTensor,
        past_key_values: object | None,
        use_cache: bool,
    ) -> ReferenceOutput: ...


class ModelFactory(Protocol):
    def from_pretrained(
        self,
        checkpoint: Path,
        *,
        dtype: object,
        attn_implementation: str,
        local_files_only: bool,
    ) -> ReferenceModel: ...


class TorchAPI(Protocol):
    __version__: str
    float32: object
    long: object

    def set_num_threads(self, threads: int) -> None: ...

    def set_default_device(self, device: str) -> None: ...

    def tensor(self, data: list[list[int]], *, dtype: object) -> ReferenceTensor: ...

    def inference_mode(self) -> AbstractContextManager[object]: ...


def read_config(checkpoint: Path) -> TextConfig:
    return CheckpointConfig.model_validate_json(
        (checkpoint / "config.json").read_bytes()
    ).text_config


def graph_source(config: TextConfig, capacity: int) -> str:
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    conv_width = 2 * key_width + value_width
    rotary_dim = int(config.head_dim * config.rope_parameters.partial_rotary_factor)
    output_weight = (
        "model.language_model.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight"
    )
    return f"""from trillim.tdl import *
import numpy as np

weights = Model("model")
layer_types = {tuple(config.layer_types)!r}
frequencies = np.reciprocal(np.power(np.float32({config.rope_parameters.rope_theta!r}), np.arange(0, {rotary_dim}, 2, dtype=np.float32) / np.float32({rotary_dim})))
with Graph() as graph:
    tokens = Input(name="tokens", shape=(None,), dtype="int64")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    hidden = Cast(Embedding(tokens, weights["model.language_model.embed_tokens.weight"]), "fp32")
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{{layer}}."
        residual = hidden
        normed = RMSNorm(hidden, weight=Cast(Constant(weights[prefix + "input_layernorm.weight"]), "fp32"), weight_offset=1.0, eps={config.rms_norm_eps!r})
        if layer_type == "linear_attention":
            prefix_linear = prefix + "linear_attn."
            projected = Linear(normed, weights[prefix_linear + "in_proj_qkv.weight"])
            history = Cache(name=f"conv_{{layer}}", shape=({conv_width}, {config.linear_conv_kernel_dim}), dtype="fp32")
            sequence = Concat((history.read(), Permute(projected, (1, 0))), axis=1, rank=2)
            convolved = Conv(sequence, weights[prefix_linear + "conv1d.weight"], kernel=({config.linear_conv_kernel_dim},), groups={conv_width})
            history.assign(Slice(sequence, axis=1, start=-{config.linear_conv_kernel_dim}, rank=2))
            mixed = Silu(Permute(Slice(convolved, axis=1, start=1, rank=2), (1, 0)))
            query = Reshape(Slice(mixed, axis=1, stop={key_width}, rank=2), (Dim(tokens, 0), {config.linear_num_key_heads}, {config.linear_key_head_dim}))
            key = Reshape(Slice(mixed, axis=1, start={key_width}, stop={2 * key_width}, rank=2), (Dim(tokens, 0), {config.linear_num_key_heads}, {config.linear_key_head_dim}))
            value = Reshape(Slice(mixed, axis=1, start={2 * key_width}, rank=2), (Dim(tokens, 0), {config.linear_num_value_heads}, {config.linear_value_head_dim}))
            if {config.linear_num_key_heads} != {config.linear_num_value_heads}:
                groups = FloorDiv(Iota({config.linear_num_value_heads}), {config.linear_num_value_heads // config.linear_num_key_heads})
                query = Take(query, groups, axis=1, rank=3)
                key = Take(key, groups, axis=1, rank=3)
            beta = Sigmoid(Linear(normed, weights[prefix_linear + "in_proj_b.weight"]))
            time_step = Softplus(Add(Linear(normed, weights[prefix_linear + "in_proj_a.weight"]), Cast(Constant(weights[prefix_linear + "dt_bias"]), "fp32")))
            log_decay = Mul(Mul(Exp(Constant(weights[prefix_linear + "A_log"])), -1), time_step)
            memory = Cache(name=f"recurrent_{{layer}}", shape=({config.linear_num_value_heads}, {config.linear_key_head_dim}, {config.linear_value_head_dim}), dtype="fp32")
            attended = GatedDelta(query, key, value, beta=beta, log_decay=log_decay, state=memory)
            gate = SplitHeads(Linear(normed, weights[prefix_linear + "in_proj_z.weight"]), heads={config.linear_num_value_heads}, head_dim={config.linear_value_head_dim})
            attended = Mul(RMSNorm(attended, weight=weights[prefix_linear + "norm.weight"], eps={config.rms_norm_eps!r}), Silu(gate))
            mixed_output = Linear(MergeHeads(attended), weights[prefix_linear + "out_proj.weight"])
        else:
            prefix_attn = prefix + "self_attn."
            projected = Reshape(Linear(normed, weights[prefix_attn + "q_proj.weight"]), (Dim(tokens, 0), {config.num_attention_heads}, {2 * config.head_dim}))
            query = Slice(projected, axis=2, stop={config.head_dim}, rank=3)
            gate = Slice(projected, axis=2, start={config.head_dim}, rank=3)
            query = RMSNorm(query, weight=Cast(Constant(weights[prefix_attn + "q_norm.weight"]), "fp32"), weight_offset=1.0, eps={config.rms_norm_eps!r})
            key = SplitHeads(Linear(normed, weights[prefix_attn + "k_proj.weight"]), heads={config.num_key_value_heads}, head_dim={config.head_dim})
            key = RMSNorm(key, weight=Cast(Constant(weights[prefix_attn + "k_norm.weight"]), "fp32"), weight_offset=1.0, eps={config.rms_norm_eps!r})
            value = SplitHeads(Linear(normed, weights[prefix_attn + "v_proj.weight"]), heads={config.num_key_value_heads}, head_dim={config.head_dim})
            query = Rotary(query, positions, dim={rotary_dim}, theta={config.rope_parameters.rope_theta!r}, frequencies=Constant(frequencies))
            key = Rotary(key, positions, dim={rotary_dim}, theta={config.rope_parameters.rope_theta!r}, frequencies=Constant(frequencies))
            keys = Cache(name=f"key_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {config.head_dim}), dtype="fp32", mode="append", axis=0)
            values = Cache(name=f"value_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {config.head_dim}), dtype="fp32", mode="append", axis=0)
            keys.append(key)
            values.append(value)
            attended = Attention(query, keys.read(), values.read())
            mixed_output = Linear(Mul(MergeHeads(attended), Sigmoid(MergeHeads(gate))), weights[prefix_attn + "o_proj.weight"])
        hidden = Add(residual, mixed_output)
        residual = hidden
        normed = RMSNorm(hidden, weight=Cast(Constant(weights[prefix + "post_attention_layernorm.weight"]), "fp32"), weight_offset=1.0, eps={config.rms_norm_eps!r})
        hidden = Add(residual, FFN(normed, up=weights[prefix + "mlp.up_proj.weight"], gate=weights[prefix + "mlp.gate_proj.weight"], down=weights[prefix + "mlp.down_proj.weight"]))
    hidden = RMSNorm(hidden, weight=Cast(Constant(weights["model.language_model.norm.weight"]), "fp32"), weight_offset=1.0, eps={config.rms_norm_eps!r})
    Output(Linear(Slice(hidden, axis=0, start=-1, rank=2), weights[{output_weight!r}]), name="logits")
"""


def compile_bundle(options: Qwen35Options, config: TextConfig) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = options.output.with_name(options.output.name + ".tdl")
    source = graph_source(config, options.cache)
    if source_path.exists() and source_path.read_text() != source:
        raise ValueError(f"existing graph source differs: {source_path}")
    source_path.write_text(source)
    if not options.output.exists():
        _compile_model(
            _CompileRequest(
                tdl_path=source_path,
                checkpoints={"model": options.checkpoint},
                output=options.output,
                backend=options.backend,
                threads=options.threads,
                max_input_size=options.prompt,
            ),
            progress=lambda message: print(message, flush=True),
        )


def prompt_ids(length: int) -> list[int]:
    original = [
        9707,
        11,
        358,
        1097,
        264,
        1646,
        13,
        576,
        4871,
        374,
        220,
        17,
        10,
        17,
        28,
        220,
    ]
    return [original[index % len(original)] for index in range(length)]


def run_reference(options: Qwen35Options) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path.home() / "Trillim/models/.hf-cache"))
    os.environ.setdefault(
        "TORCH_HOME", str(Path.home() / ".trillim/qwen35/torch-cache")
    )
    torch = cast(TorchAPI, importlib.import_module("torch"))
    transformers = importlib.import_module("transformers")
    factory = cast(ModelFactory, transformers.Qwen3_5ForConditionalGeneration)
    torch.set_num_threads(options.threads)
    torch.set_default_device("cpu")
    model = factory.from_pretrained(
        options.checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    source_module = importlib.import_module(
        "transformers.models.qwen3_5.modeling_qwen3_5"
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
    metadata = ReferenceMetadata(
        checkpoint=str(options.checkpoint),
        config_sha256=hashlib.sha256(
            (options.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        transformers_version=str(transformers.__version__),
        torch_version=torch.__version__,
        architecture_source=str(source_path),
        architecture_url=(
            "https://github.com/huggingface/transformers/blob/v"
            + str(transformers.__version__)
            + "/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
        ),
        architecture_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    path = options.output.with_name(options.output.name + "-transformers.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, logits=np.stack(logits))
    path.with_suffix(".json").write_text(metadata.model_dump_json(indent=2) + "\n")
    print(metadata.model_dump_json(indent=2), flush=True)


def validate_reference(options: Qwen35Options, path: Path) -> dict[str, object]:
    metadata = ReferenceMetadata.model_validate_json(
        path.with_suffix(".json").read_bytes()
    )
    if Path(metadata.checkpoint).resolve() != options.checkpoint.resolve():
        raise ValueError("reference checkpoint path differs")
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
    fixed_matches = True
    with Runtime(options.output, threads=options.threads) as runtime:
        for step, reference in enumerate(expected):
            tokens = (
                metadata.prompt_token_ids
                if step == 0
                else [metadata.generated_token_ids[step - 1]]
            )
            positions = (
                list(range(options.prompt))
                if step == 0
                else [options.prompt + step - 1]
            )
            actual = runtime.run(
                {
                    "tokens": np.asarray(tokens, dtype=np.int64),
                    "positions": np.asarray(positions, dtype=np.int64),
                }
            )["logits"].ravel()
            actual_logits.append(np.asarray(actual, dtype=np.float32).copy())
            reference64 = reference.astype(np.float64)
            reference_rms = float(np.sqrt(np.mean(reference64 * reference64)))
            absolute_tolerance = 2e-5 * max(1.0, reference_rms)
            difference = actual.astype(np.float64) - reference64
            greedy_matches = int(np.argmax(actual)) == int(np.argmax(reference))
            fixed_close = np.isclose(actual, reference, rtol=2e-4, atol=2e-5)
            scaled_close = np.isclose(
                actual, reference, rtol=2e-4, atol=absolute_tolerance
            )
            close = bool(np.all(scaled_close))
            matches = matches and greedy_matches and close
            fixed_matches = (
                fixed_matches and greedy_matches and bool(np.all(fixed_close))
            )
            comparisons.append(
                {
                    "step": step,
                    "max_absolute_error": float(np.max(np.abs(difference))),
                    "rms_error": float(np.sqrt(np.mean(difference * difference))),
                    "greedy_matches": greedy_matches,
                    "reference_rms": reference_rms,
                    "absolute_tolerance": absolute_tolerance,
                    "fixed_absolute_tolerance_allclose": bool(np.all(fixed_close)),
                    "fixed_absolute_tolerance_failure_count": int(
                        np.count_nonzero(~fixed_close)
                    ),
                    "scaled_absolute_tolerance_failure_count": int(
                        np.count_nonzero(~scaled_close)
                    ),
                    "allclose": close,
                }
            )
    logits_path = options.output.with_name(
        options.output.name + "-validation-logits.npz"
    )
    np.savez(logits_path, logits=np.stack(actual_logits))
    return {
        "transformers_reference": metadata.model_dump(),
        "comparisons": comparisons,
        "reference_matches": matches,
        "fixed_reference_matches": fixed_matches,
        "reference_relative_tolerance": 2e-4,
        "reference_absolute_tolerance_base": 2e-5,
        "reference_absolute_tolerance_policy": "2e-5 * max(1, RMS reference logits) per step",
        "validation_logits": str(logits_path),
        "reference_logits": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="qwen3.5-0.8b")
    parser.add_argument(
        "--backend",
        choices=("scalar", "neon"),
        default="scalar",
        help="runtime uses scalar registration, which may include ARM symbols; neon is for compile-only inspection",
    )
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
    options = Qwen35Options.model_validate(vars(parser.parse_args()))
    if (
        not options.output.expanduser()
        .resolve()
        .is_relative_to(Path.home() / ".trillim")
    ):
        raise ValueError("benchmark artifacts must stay beneath ~/.trillim")
    config = read_config(options.checkpoint)
    if (
        options.backend == "neon"
        and not options.compile_only
        and not options.run_reference
    ):
        raise ValueError("neon registration supports compile-only inspection")
    if options.run_reference:
        run_reference(options)
        return
    compile_bundle(options, config)
    if options.compile_only:
        return
    if options.validate_only:
        if options.validate_reference is None:
            raise ValueError("--validate-only requires --validate-reference")
        results = validate_reference(options, options.validate_reference)
    else:
        results = benchmark(options, config=config)
        results["backend"] = options.backend
        results["recurrent_state_dtype"] = "fp32"
        results["causal_conv_state_dtype"] = "fp32"
    if options.validate_reference is not None and not options.validate_only:
        results.update(validate_reference(options, options.validate_reference))
    path = options.output.with_name(options.output.name + "-results.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if results.get("reference_matches") is False:
        raise ValueError("Qwen3.5 logits do not match the Transformers reference")


if __name__ == "__main__":
    main()
