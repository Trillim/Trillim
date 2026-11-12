"""Reproduce decoder model timings without modifying the compiler or runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pydantic
from pydantic import BaseModel, ConfigDict, Field

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.compiler.weights.reader import _load_checkpoint
from trillim.runtime import Runtime


class RopeScaling(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    rope_type: str
    factor: float
    original_max_position_embeddings: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None
    truncate: bool = True


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: str
    hidden_size: int
    num_hidden_layers: int


class DecoderConfig(ModelMetadata):
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int | None = None
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool = False
    rope_scaling: RopeScaling | None = None
    is_bitnet_config: bool = False
    num_local_experts: int = 0
    num_experts_per_tok: int = 0
    layer_types: list[str] = Field(default_factory=list)
    sliding_window: int | None = None
    swiglu_limit: float = 7.0
    hidden_act: str = "silu"


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    threads: int = Field(default=20, ge=1)
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=8, ge=1)
    repeats: int = Field(default=3, ge=1)
    cache: int = Field(default=512, ge=1)
    compile_only: bool = False
    reference_bundle: Path | None = None


class ReferenceResults(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    generated_token_ids: list[list[int]]


class KernelSymbol(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    symbol: str
    library: str


class InstructionSymbol(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    op: str
    kernel: str | None = None
    program: list[InstructionSymbol] = Field(default_factory=list)
    body: ProgramBlock | None = None
    condition: str | ProgramBlock | None = None
    branches: list[ProgramBlock] = Field(default_factory=list)


class ProgramBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    program: list[InstructionSymbol] = Field(default_factory=list)


class ManifestSymbols(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    kernels: dict[str, KernelSymbol]
    program: list[InstructionSymbol]


def instruction_calls(program: list[InstructionSymbol]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for instruction in program:
        if instruction.op == "call" and instruction.kernel is not None:
            counts[instruction.kernel] += 1
        counts.update(instruction_calls(instruction.program))
        if instruction.body is not None:
            counts.update(instruction_calls(instruction.body.program))
        if isinstance(instruction.condition, ProgramBlock):
            counts.update(instruction_calls(instruction.condition.program))
        for branch in instruction.branches:
            counts.update(instruction_calls(branch.program))
    return counts


def graph_source(config: DecoderConfig, capacity: int) -> str:
    if config.model_type == "gpt_oss":
        return gptoss_graph_source(config, capacity)
    if (
        config.model_type == "bitnet"
        or config.model_type == "llama"
        and config.is_bitnet_config
    ):
        return bitnet_graph_source(config, capacity)
    if config.model_type not in {"qwen3", "qwen2", "llama"} or config.is_bitnet_config:
        raise ValueError(
            "this benchmark graph supports floating Qwen3, Qwen2 and Llama checkpoints"
        )
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    biased = config.attention_bias or config.model_type == "qwen2"
    q_bias = ', bias=weights[prefix + "self_attn.q_proj.bias"]' if biased else ""
    k_bias = ', bias=weights[prefix + "self_attn.k_proj.bias"]' if biased else ""
    v_bias = ', bias=weights[prefix + "self_attn.v_proj.bias"]' if biased else ""
    qk_norm = ""
    if config.model_type == "qwen3":
        qk_norm = f"""        query = RMSNorm(query, weight=weights[prefix + "self_attn.q_norm.weight"], eps={config.rms_norm_eps!r})
        key = RMSNorm(key, weight=weights[prefix + "self_attn.k_norm.weight"], eps={config.rms_norm_eps!r})
"""
    rotary = "Rotary"
    rotary_args = f"positions, dim={head_dim}, theta={config.rope_theta!r}"
    if config.rope_scaling is not None:
        scaling = config.rope_scaling
        if scaling.rope_type != "yarn":
            raise ValueError("this benchmark supports unscaled rotary or YaRN")
        rotary = "YaRN"
        rotary_args += f", factor={scaling.factor!r}, original_context={scaling.original_max_position_embeddings}, beta_fast={scaling.beta_fast!r}, beta_slow={scaling.beta_slow!r}, attention_factor={scaling.attention_factor!r}, truncate={scaling.truncate!r}"
    output_weight = (
        "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    )
    return f"""from trillim.tdl import *

weights = Model("model")
with Graph() as graph:
    tokens = Input(name="tokens", shape=(None,), dtype="int64")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    hidden = Cast(Embedding(tokens, weights["model.embed_tokens.weight"]), "fp32")
    for layer in range({config.num_hidden_layers}):
        prefix = f"model.layers.{{layer}}."
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "input_layernorm.weight"], eps={config.rms_norm_eps!r})
        query = SplitHeads(Linear(normed, weights[prefix + "self_attn.q_proj.weight"]{q_bias}), heads={config.num_attention_heads}, head_dim={head_dim})
        key = SplitHeads(Linear(normed, weights[prefix + "self_attn.k_proj.weight"]{k_bias}), heads={config.num_key_value_heads}, head_dim={head_dim})
        value = SplitHeads(Linear(normed, weights[prefix + "self_attn.v_proj.weight"]{v_bias}), heads={config.num_key_value_heads}, head_dim={head_dim})
{qk_norm}        query = {rotary}(query, {rotary_args})
        key = {rotary}(key, {rotary_args})
        keys = Cache(name=f"key_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {head_dim}), dtype="fp32", mode="append", axis=0)
        values = Cache(name=f"value_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {head_dim}), dtype="fp32", mode="append", axis=0)
        keys.append(key)
        values.append(value)
        attended = Attention(query, keys.read(), values.read())
        hidden = Add(residual, Linear(MergeHeads(attended), weights[prefix + "self_attn.o_proj.weight"]))
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "post_attention_layernorm.weight"], eps={config.rms_norm_eps!r})
        hidden = Add(residual, FFN(normed, up=weights[prefix + "mlp.up_proj.weight"], down=weights[prefix + "mlp.down_proj.weight"], gate=weights[prefix + "mlp.gate_proj.weight"]))
    hidden = RMSNorm(hidden, weight=weights["model.norm.weight"], eps={config.rms_norm_eps!r})
    last = Slice(hidden, axis=0, start=-1, rank=2)
    Output(Linear(last, weights[{output_weight!r}]), name="logits")
"""


def gptoss_graph_source(config: DecoderConfig, capacity: int) -> str:
    scaling = config.rope_scaling
    if (
        config.head_dim is None
        or scaling is None
        or scaling.rope_type != "yarn"
        or not config.attention_bias
        or not 0 < config.num_experts_per_tok <= config.num_local_experts
        or len(config.layer_types) != config.num_hidden_layers
        or any(
            layer not in {"full_attention", "sliding_attention"}
            for layer in config.layer_types
        )
    ):
        raise ValueError(
            "GPT-oss benchmark requires biased attention, YaRN and valid expert/layer configuration"
        )
    if "sliding_attention" in config.layer_types and config.sliding_window is None:
        raise ValueError("GPT-oss sliding layers need a window")
    windows = tuple(
        config.sliding_window if layer == "sliding_attention" else None
        for layer in config.layer_types
    )
    rotary_args = f"positions, dim={config.head_dim}, theta={config.rope_theta!r}, factor={scaling.factor!r}, original_context={scaling.original_max_position_embeddings}, beta_fast={scaling.beta_fast!r}, beta_slow={scaling.beta_slow!r}, attention_factor={scaling.attention_factor!r}, truncate={scaling.truncate!r}"
    return f"""from trillim.tdl import *

weights = Model("model")
windows = {windows!r}
with Graph() as graph:
    tokens = Input(name="tokens", shape=(None,), dtype="int64")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    hidden = Cast(Embedding(tokens, weights["model.embed_tokens.weight"]), "fp32")
    for layer in range({config.num_hidden_layers}):
        prefix = f"model.layers.{{layer}}."
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "input_layernorm.weight"], eps={config.rms_norm_eps!r})
        query = SplitHeads(Linear(normed, weights[prefix + "self_attn.q_proj.weight"], bias=weights[prefix + "self_attn.q_proj.bias"]), heads={config.num_attention_heads}, head_dim={config.head_dim})
        key = SplitHeads(Linear(normed, weights[prefix + "self_attn.k_proj.weight"], bias=weights[prefix + "self_attn.k_proj.bias"]), heads={config.num_key_value_heads}, head_dim={config.head_dim})
        value = SplitHeads(Linear(normed, weights[prefix + "self_attn.v_proj.weight"], bias=weights[prefix + "self_attn.v_proj.bias"]), heads={config.num_key_value_heads}, head_dim={config.head_dim})
        query = YaRN(query, {rotary_args})
        key = YaRN(key, {rotary_args})
        keys = Cache(name=f"key_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {config.head_dim}), dtype="fp32", mode="append", axis=0)
        values = Cache(name=f"value_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {config.head_dim}), dtype="fp32", mode="append", axis=0)
        keys.append(key)
        values.append(value)
        attended = Attention(query, keys.read(), values.read(), window=windows[layer], sinks=weights[prefix + "self_attn.sinks"])
        hidden = Add(residual, Linear(MergeHeads(attended), weights[prefix + "self_attn.o_proj.weight"], bias=weights[prefix + "self_attn.o_proj.bias"]))
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "post_attention_layernorm.weight"], eps={config.rms_norm_eps!r})
        mixed = GptOssMoE(normed, router=weights[prefix + "mlp.router.weight"], router_bias=weights[prefix + "mlp.router.bias"], gate_up=weights[prefix + "mlp.experts.gate_up_proj"], down=weights[prefix + "mlp.experts.down_proj"], gate_up_bias=weights[prefix + "mlp.experts.gate_up_proj_bias"], down_bias=weights[prefix + "mlp.experts.down_proj_bias"], top_k={config.num_experts_per_tok}, limit={config.swiglu_limit!r})
        hidden = Add(residual, mixed)
    hidden = RMSNorm(hidden, weight=weights["model.norm.weight"], eps={config.rms_norm_eps!r})
    last = Slice(hidden, axis=0, start=-1, rank=2)
    Output(Linear(last, weights["lm_head.weight"]), name="logits")
"""


def bitnet_graph_source(config: DecoderConfig, capacity: int) -> str:
    microsoft = config.model_type == "bitnet"
    activation = "ReluSquared" if microsoft else "Silu"
    expected_activation = "relu2" if microsoft else "silu"
    if (
        config.attention_bias
        or config.rope_scaling is not None
        or config.hidden_act != expected_activation
    ):
        raise ValueError(
            "BitNet benchmark requires bias-free attention, unscaled rotary and the model's expected activation"
        )
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    output_weight = (
        "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    )
    attended = "MergeHeads(Attention(query, keys.read(), values.read()))"
    expanded = f"Mul({activation}(gate), up)"
    if microsoft:
        attended = f'RMSNorm({attended}, weight=weights[prefix + "self_attn.attn_sub_norm.weight"], eps={config.rms_norm_eps!r})'
        expanded = f'RMSNorm({expanded}, weight=weights[prefix + "mlp.ffn_sub_norm.weight"], eps={config.rms_norm_eps!r})'
    return f"""from trillim.tdl import *

weights = Model("model")
with Graph() as graph:
    tokens = Input(name="tokens", shape=(None,), dtype="int64")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    hidden = Cast(Embedding(tokens, weights["model.embed_tokens.weight"]), "fp32")
    for layer in range({config.num_hidden_layers}):
        prefix = f"model.layers.{{layer}}."
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "input_layernorm.weight"], eps={config.rms_norm_eps!r})
        codes, scale = QuantizeInt8(normed)
        query = SplitHeads(QuantizedLinear(codes, weights[prefix + "self_attn.q_proj.weight"], scale), heads={config.num_attention_heads}, head_dim={head_dim})
        key = SplitHeads(QuantizedLinear(codes, weights[prefix + "self_attn.k_proj.weight"], scale), heads={config.num_key_value_heads}, head_dim={head_dim})
        value = SplitHeads(QuantizedLinear(codes, weights[prefix + "self_attn.v_proj.weight"], scale), heads={config.num_key_value_heads}, head_dim={head_dim})
        query = Rotary(query, positions, dim={head_dim}, theta={config.rope_theta!r})
        key = Rotary(key, positions, dim={head_dim}, theta={config.rope_theta!r})
        keys = Cache(name=f"key_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {head_dim}), dtype="fp32", mode="append", axis=0)
        values = Cache(name=f"value_{{layer}}", shape=({capacity}, {config.num_key_value_heads}, {head_dim}), dtype="fp32", mode="append", axis=0)
        keys.append(key)
        values.append(value)
        attended = {attended}
        hidden = Add(residual, BitLinear(attended, weights[prefix + "self_attn.o_proj.weight"]))
        residual = hidden
        normed = RMSNorm(hidden, weight=weights[prefix + "post_attention_layernorm.weight"], eps={config.rms_norm_eps!r})
        codes, scale = QuantizeInt8(normed)
        gate = QuantizedLinear(codes, weights[prefix + "mlp.gate_proj.weight"], scale)
        up = QuantizedLinear(codes, weights[prefix + "mlp.up_proj.weight"], scale)
        expanded = {expanded}
        hidden = Add(residual, BitLinear(expanded, weights[prefix + "mlp.down_proj.weight"]))
    hidden = RMSNorm(hidden, weight=weights["model.norm.weight"], eps={config.rms_norm_eps!r})
    last = Slice(hidden, axis=0, start=-1, rank=2)
    Output(Linear(last, weights[{output_weight!r}]), name="logits")
"""


def compile_bundle(options: Options) -> None:
    config = DecoderConfig.model_validate_json(
        (options.checkpoint / "config.json").read_bytes()
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = options.output.with_name(options.output.name + ".tdl")
    source = graph_source(config, options.cache)
    if source_path.exists() and source_path.read_text() != source:
        raise ValueError(f"existing graph source differs: {source_path}")
    source_path.write_text(source)
    if options.output.exists():
        return
    checkpoint = _load_checkpoint("model", options.checkpoint)
    files = {value.path: value.file_size for value in checkpoint.tensors.values()}
    if sum(files.values()) > 32_000_000_000:
        raise ValueError("checkpoint weights exceed the 32 GB size limit")
    request = _CompileRequest(
        tdl_path=source_path,
        checkpoints={"model": options.checkpoint},
        output=options.output,
        backend="scalar",
        max_input_size=options.prompt,
        depth="none",
        threads=options.threads,
    )
    _compile_model(request, progress=lambda message: print(message, flush=True))


def benchmark(
    options: Options, *, config: ModelMetadata | None = None
) -> dict[str, object]:
    if options.prompt + options.decode > options.cache:
        raise ValueError("cache must fit prompt and measured decode")
    if config is None:
        config = ModelMetadata.model_validate_json(
            (options.checkpoint / "config.json").read_bytes()
        )
    prompt = np.array(
        [9707, 11, 358, 1097, 264, 1646, 13, 576, 4871, 374, 220, 17, 10, 17, 28, 220],
        dtype=np.int64,
    )
    prompt = np.resize(prompt, options.prompt)
    prefill: list[float] = []
    decode: list[float] = []
    generations: list[list[int]] = []
    logits_first: list[float] = []
    output_hashes: list[str] = []
    for repeat in range(options.repeats):
        with Runtime(options.output, threads=options.threads) as runtime:
            started = time.perf_counter()
            logits = runtime.run(
                {
                    "tokens": prompt,
                    "positions": np.arange(options.prompt, dtype=np.int64),
                }
            )["logits"]
            prefill.append(time.perf_counter() - started)
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite model logits")
            if repeat == 0:
                logits_first = logits.ravel().astype(float).tolist()
                np.save(
                    options.output.with_name(
                        options.output.name + "-prefill-logits.npy"
                    ),
                    logits,
                )
            output_hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
            generated = [int(np.argmax(logits))]
            for step in range(options.decode):
                started = time.perf_counter()
                logits = runtime.run(
                    {
                        "tokens": np.array([generated[-1]], dtype=np.int64),
                        "positions": np.array([options.prompt + step], dtype=np.int64),
                    }
                )["logits"]
                decode.append(time.perf_counter() - started)
                if not np.isfinite(logits).all():
                    raise ValueError("nonfinite model decode logits")
                generated.append(int(np.argmax(logits)))
            generations.append(generated)
        print(
            f"repeat {repeat + 1}: prefill {prefill[-1]:.6f}s, decode {1 / statistics.median(decode[-options.decode :]):.3f} token/s",
            flush=True,
        )
    executable_bytes = (options.output / "executable.json").read_bytes()
    manifest = ManifestSymbols.model_validate_json(executable_bytes)
    symbols = sorted({metadata.symbol for metadata in manifest.kernels.values()})
    counts = instruction_calls(manifest.program)
    result: dict[str, object] = {
        "label": options.label,
        "checkpoint": str(options.checkpoint),
        "bundle": str(options.output),
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "host": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pydantic_version": pydantic.__version__,
        "threads": options.threads,
        "prompt_tokens": options.prompt,
        "prompt_token_ids": [int(token) for token in prompt],
        "decode_tokens": options.decode,
        "cache_capacity": options.cache,
        "repeats": options.repeats,
        "activation_dtype": "fp32",
        "kv_dtype": "fp32",
        "backend": "scalar",
        "timing_scope": "Runtime.run including binding, Python executor, native kernels, and output copy; excludes tokenization, greedy token selection, model loading, and compilation",
        "prefill_seconds": prefill,
        "prefill_median_seconds": statistics.median(prefill),
        "prefill_tokens_per_second": options.prompt / statistics.median(prefill),
        "decode_seconds": decode,
        "decode_median_seconds": statistics.median(decode),
        "decode_tokens_per_second": 1 / statistics.median(decode),
        "generated_token_ids": generations,
        "prefill_output_hashes": output_hashes,
        "prefill_logit_min": min(logits_first),
        "prefill_logit_max": max(logits_first),
        "kernels": symbols,
        "executable_sha256": hashlib.sha256(executable_bytes).hexdigest(),
        "kernel_library_sha256": {
            name: hashlib.sha256((options.output / name).read_bytes()).hexdigest()
            for name in sorted({value.library for value in manifest.kernels.values()})
        },
        "kernel_call_counts": dict(counts),
        "kernel_call_count_scope": "static explicit native call instructions, including nested loop/branch/reduction programs; each body counted once",
        "neon_kernel_call_count": sum(
            count for name, count in counts.items() if name.startswith("trillim_neon_")
        ),
        "graph_sha256": hashlib.sha256(
            options.output.with_name(options.output.name + ".tdl").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (options.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
    }
    if options.reference_bundle is not None:
        reference_name = options.reference_bundle.name
        reference_logits_path = options.reference_bundle.with_name(
            reference_name + "-prefill-logits.npy"
        )
        reference_results_path = options.reference_bundle.with_name(
            reference_name + "-results.json"
        )
        expected = np.load(reference_logits_path).ravel()
        actual = np.asarray(logits_first, dtype=np.float32)
        if expected.shape != actual.shape:
            raise ValueError("reference logit shape differs")
        differences = actual.astype(np.float64) - expected.astype(np.float64)
        result["reference_bundle"] = str(options.reference_bundle)
        result["prefill_max_absolute_error"] = float(np.max(np.abs(differences)))
        result["prefill_rms_error"] = float(np.sqrt(np.mean(differences * differences)))
        reference = ReferenceResults.model_validate_json(
            reference_results_path.read_bytes()
        )
        expected_generation = reference.generated_token_ids[0]
        result["greedy_matches_reference"] = all(
            tokens == expected_generation for tokens in generations
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache", type=int, default=512)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    options = Options.model_validate(vars(parser.parse_args()))
    compile_bundle(options)
    if not options.compile_only:
        result = benchmark(options)
        result_path = options.output.with_name(options.output.name + "-results.json")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
