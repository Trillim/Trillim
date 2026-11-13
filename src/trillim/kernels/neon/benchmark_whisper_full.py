"""Replay a compiled Whisper encoder/decoder with explicit cache partitions.

Run each scalar/ARM invocation in a separate process. --forced-token-file replaces
the initial prefix; --teacher supplies subsequent tokens, matching whisper-debug.
Raw endpoint logits are retained at --checkpoints. --trace-full-logits retains
the full vocabulary vector after every decoder call, including prefill calls.
Elapsed times are diagnostic unless the caller coordinates an isolated run.

Download the pinned native FP32 checkpoint from a source checkout:
  HF_HOME=~/Trillim/models/hf-cache UV_CACHE_DIR=~/.trillim/uv-cache uv run --with huggingface_hub python -c 'from pathlib import Path; from huggingface_hub import snapshot_download; snapshot_download("openai/whisper-base", revision="e37978b90ca9030d5170a5c07aadb050351a65bb", local_dir=Path.home()/"Trillim/models/arm-matrix/openai/whisper-base", allow_patterns=["*.json", "merges.txt", "model.safetensors"])'

On macOS/AArch64, prepare separate catalogs using the current CPU registration.
Use distinct kernel homes:
  TRILLIM_HOME=~/.trillim/whisper-scalar-home uv run trillim kernels --backend scalar
  TRILLIM_HOME=~/.trillim/whisper-arm-home uv run trillim kernels --backend scalar src/trillim/kernels/scalar/*.c src/trillim/kernels/neon/*.c

Invoke this module with --compile-only --checkpoint CHECKPOINT --label scalar
under the scalar home, and --label arm under the ARM home. Both compile backend scalar;
the ARM catalog contains the NEON implementations. --max-input-size 128 supports
wide prefill while the checkpoint retains its 448-token decoder cache.
Compilation writes complete encoder/decoder TDL and bundles below --directory.
--prepare-only writes TDL without compiling. --parts decoder avoids encoder copies.

Finally run two separate processes with --label scalar/arm, the same --checkpoint,
--directory, supplied --features mel.npy (FP32[80,3000]), and distinct --output
paths. The resulting bundles include their kernels and need no installed catalog
at execution time. Compare the first differing raw vector and its top-two margin
before comparing generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ManifestSymbols
from trillim.kernels.neon.benchmark_whisper import WhisperConfig, graph_source
from trillim.runtime import Runtime

_ROOT = Path.home() / ".trillim/whisper-full"
_CHECKPOINT = Path.home() / "Trillim/models/arm-matrix/openai/whisper-base"


class FullWhisperConfig(WhisperConfig):
    vocab_size: int = Field(gt=0)
    decoder_attention_heads: int = Field(gt=0)
    decoder_layers: int = Field(gt=0)
    decoder_ffn_dim: int = Field(gt=0)
    decoder_start_token_id: int = Field(ge=0)
    eos_token_id: int = Field(ge=0)
    max_target_positions: int = Field(gt=0)
    forced_decoder_ids: list[list[int]] = Field(default_factory=list)
    suppress_tokens: list[int] = Field(default_factory=list)
    begin_suppress_tokens: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decoder_heads(self) -> Self:
        if self.d_model % self.decoder_attention_heads:
            raise ValueError("hidden size must be divisible by decoder heads")
        return self


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str = Field(default="arm", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    directory: Path = _ROOT
    prepare_only: bool = False
    compile_only: bool = False
    max_input_size: int = Field(default=128, gt=0)
    parts: list[Literal["encoder", "decoder"]] = ["encoder", "decoder"]
    checkpoint: Path = _CHECKPOINT
    encoder: Path | None = None
    decoder: Path | None = None
    features: Path | None = None
    encoded: Path | None = None
    teacher: Path | None = None
    forced_token_file: Path | None = None
    prefix: list[int] | None = None
    bos_only: bool = False
    prefill_chunk: int = Field(default=16, gt=0)
    checkpoints: list[int] = Field(default_factory=list)
    steps: int = Field(default=64, gt=0)
    threads: int = Field(default=20, gt=0)
    trace_full_logits: bool = False
    output: Path | None = None

    @model_validator(mode="after")
    def validate_paths_and_prefix(self) -> Self:
        if self.prepare_only and self.compile_only:
            raise ValueError("choose either prepare-only or compile-only")
        if not self.parts or len(set(self.parts)) != len(self.parts):
            raise ValueError("compile parts must be nonempty and unique")
        if (
            sum(
                (
                    self.prefix is not None,
                    self.forced_token_file is not None,
                    self.bos_only,
                )
            )
            > 1
        ):
            raise ValueError("choose only one prefix override")
        if (
            self.output is not None
            and not self.output.expanduser()
            .resolve()
            .is_relative_to(Path.home() / ".trillim")
        ):
            raise ValueError("benchmark artifacts must remain under ~/.trillim")
        if (
            not self.directory.expanduser()
            .resolve()
            .is_relative_to(Path.home() / ".trillim")
        ):
            raise ValueError("benchmark bundles must remain under ~/.trillim")
        if self.compile_only:
            for path in (self.encoder, self.decoder):
                if path is not None and not path.expanduser().resolve().is_relative_to(
                    Path.home() / ".trillim"
                ):
                    raise ValueError("compiled bundles must remain under ~/.trillim")
        if not self.prepare_only and not self.compile_only:
            if self.output is None:
                raise ValueError("running the model requires --output")
            if self.encoded is None and self.features is None:
                raise ValueError("supply --features or --encoded")
        if any(value < 1 for value in self.checkpoints):
            raise ValueError("checkpoints must be positive token counts")
        return self


def decoder_graph_source(config: FullWhisperConfig) -> str:
    heads = config.decoder_attention_heads
    width = config.d_model // heads
    scale = width**-0.5
    return f"""from trillim.tdl import *
weights=Model("model")
with Graph() as graph:
    tokens=Input(name="tokens",shape=(None,),dtype="int64")
    positions=Input(name="positions",shape=(None,),dtype="int64")
    encoded=Input(name="encoded",shape=({config.max_source_positions},{config.d_model}),dtype="fp32")
    initialize=Input(name="initialize",shape=(),dtype="bool")
    x=Add(Cast(Embedding(tokens,weights["model.decoder.embed_tokens.weight"]),"fp32"),Cast(Embedding(positions,weights["model.decoder.embed_positions.weight"]),"fp32"))
    for layer in range({config.decoder_layers}):
        prefix=f"model.decoder.layers.{{layer}}."
        sk=Cache(name=f"sk_{{layer}}",shape=({config.max_target_positions},{heads},{width}),dtype="fp32",mode="append",axis=0)
        sv=Cache(name=f"sv_{{layer}}",shape=({config.max_target_positions},{heads},{width}),dtype="fp32",mode="append",axis=0)
        ck=Cache(name=f"ck_{{layer}}",shape=({config.max_source_positions},{heads},{width}),dtype="fp32")
        cv=Cache(name=f"cv_{{layer}}",shape=({config.max_source_positions},{heads},{width}),dtype="fp32")
        def initialize_cross():
            ck.assign(SplitHeads(Linear(encoded,weights[prefix+"encoder_attn.k_proj.weight"]),heads={heads}))
            cv.assign(SplitHeads(Linear(encoded,weights[prefix+"encoder_attn.v_proj.weight"],bias=weights[prefix+"encoder_attn.v_proj.bias"]),heads={heads}))
            return ()
        If(initialize,then_branch=initialize_cross,else_branch=lambda:())
        n=LayerNorm(x,weight=weights[prefix+"self_attn_layer_norm.weight"],bias=weights[prefix+"self_attn_layer_norm.bias"],eps=1e-5)
        q=SplitHeads(Mul(Linear(n,weights[prefix+"self_attn.q_proj.weight"],bias=weights[prefix+"self_attn.q_proj.bias"]),{scale!r}),heads={heads})
        k=SplitHeads(Linear(n,weights[prefix+"self_attn.k_proj.weight"]),heads={heads})
        v=SplitHeads(Linear(n,weights[prefix+"self_attn.v_proj.weight"],bias=weights[prefix+"self_attn.v_proj.bias"]),heads={heads})
        sk.append(k)
        sv.append(v)
        a=MergeHeads(Attention(q,sk.read(),sv.read(),scale=1.0))
        x=Add(x,Linear(a,weights[prefix+"self_attn.out_proj.weight"],bias=weights[prefix+"self_attn.out_proj.bias"]))
        n=LayerNorm(x,weight=weights[prefix+"encoder_attn_layer_norm.weight"],bias=weights[prefix+"encoder_attn_layer_norm.bias"],eps=1e-5)
        q=SplitHeads(Mul(Linear(n,weights[prefix+"encoder_attn.q_proj.weight"],bias=weights[prefix+"encoder_attn.q_proj.bias"]),{scale!r}),heads={heads})
        a=MergeHeads(Attention(q,ck.read(),cv.read(),causal=False,scale=1.0))
        x=Add(x,Linear(a,weights[prefix+"encoder_attn.out_proj.weight"],bias=weights[prefix+"encoder_attn.out_proj.bias"]))
        n=LayerNorm(x,weight=weights[prefix+"final_layer_norm.weight"],bias=weights[prefix+"final_layer_norm.bias"],eps=1e-5)
        n=Gelu(Linear(n,weights[prefix+"fc1.weight"],bias=weights[prefix+"fc1.bias"]))
        x=Add(x,Linear(n,weights[prefix+"fc2.weight"],bias=weights[prefix+"fc2.bias"]))
    x=LayerNorm(x,weight=weights["model.decoder.layer_norm.weight"],bias=weights["model.decoder.layer_norm.bias"],eps=1e-5)
    Output(Linear(Slice(x,axis=0,start=-1,rank=2),weights["model.decoder.embed_tokens.weight"]),name="logits")
"""


def prepare(options: Options, config: FullWhisperConfig) -> None:
    options.directory.mkdir(parents=True, exist_ok=True)
    sources = {"encoder": graph_source(config), "decoder": decoder_graph_source(config)}
    overrides = {"encoder": options.encoder, "decoder": options.decoder}
    for part in options.parts:
        tdl = options.directory / f"{options.label}-{part}.tdl"
        tdl.write_text(sources[part])
        if options.compile_only:
            output = (
                overrides[part]
                or options.directory / f"whisper-base-{options.label}-{part}"
            )
            _compile_model(
                _CompileRequest(
                    tdl_path=tdl,
                    checkpoints={"model": options.checkpoint},
                    output=output,
                    backend="scalar",
                    max_input_size=options.max_input_size,
                    depth="none",
                    threads=options.threads,
                ),
                progress=lambda message: print(message, flush=True),
            )
    print(f"Prepared {', '.join(options.parts)} under {options.directory}", flush=True)


def _load_tokens(path: Path, key: str) -> NDArray[np.int64]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = archive[key]
    elif path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        payload: object = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload[key]
        validated = TypeAdapter(list[int]).validate_python(payload, strict=True)
        values = np.asarray(validated, dtype=np.int64)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("token files must contain a one-dimensional integer array")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError("token identifier exceeds int64")
    result = np.asarray(values, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("token identifiers must be nonnegative")
    return result


def _prefix(options: Options, config: FullWhisperConfig) -> NDArray[np.int64]:
    if options.forced_token_file is not None:
        result = _load_tokens(options.forced_token_file, "tokens")
    elif options.prefix is not None:
        result = np.asarray(options.prefix, dtype=np.int64)
    else:
        values = [config.decoder_start_token_id]
        if not options.bos_only:
            values.extend(pair[1] for pair in config.forced_decoder_ids)
        result = np.asarray(values, dtype=np.int64)
    if (
        not result.size
        or result.size > config.max_target_positions
        or np.any(result < 0)
        or np.any(result >= config.vocab_size)
    ):
        raise ValueError("prefix must be nonempty and fit the decoder position limit")
    return result


def _encoded(options: Options, encoder: Path) -> tuple[NDArray[np.float32], float]:
    if options.encoded is not None:
        if options.encoded.suffix == ".npz":
            with np.load(options.encoded, allow_pickle=False) as archive:
                values = archive["encoded"]
        else:
            values = np.load(options.encoded, allow_pickle=False)
        return np.asarray(values, dtype=np.float32), 0.0
    if options.features is None:
        raise ValueError("encoder inference requires --features")
    mel = np.load(options.features, allow_pickle=False).astype(np.float32)
    with Runtime(encoder, threads=options.threads) as runtime:
        start = time.perf_counter()
        values = runtime.run({"mel": mel})["encoder_hidden_states"]
        seconds = time.perf_counter() - start
    return np.asarray(values, dtype=np.float32), seconds


def _select(logits: NDArray[np.float32], config: FullWhisperConfig, step: int) -> int:
    values = logits.copy()
    values[config.suppress_tokens] = -np.inf
    values[50364:] = -np.inf
    if step == 0:
        values[config.begin_suppress_tokens] = -np.inf
    return int(np.argmax(values))


def _summary(logits: NDArray[np.float32]) -> dict[str, JsonValue]:
    top = np.argsort(logits)[-5:][::-1]
    return {
        "raw_top_ids": [int(value) for value in top],
        "raw_top_logits": [
            float(logits[value]) if np.isfinite(logits[value]) else str(logits[value])
            for value in top
        ],
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(logits))),
        "raw_top_two_margin": float(logits[top[0]] - logits[top[1]])
        if top.size >= 2 and np.all(np.isfinite(logits[top[:2]]))
        else None,
    }


def run(options: Options) -> None:
    config = FullWhisperConfig.model_validate_json(
        (options.checkpoint / "config.json").read_bytes()
    )
    if options.prepare_only or options.compile_only:
        prepare(options, config)
        return
    if options.output is None:
        raise ValueError("model inference requires --output")
    prefix = _prefix(options, config)
    teacher = (
        _load_tokens(options.teacher, "generated")
        if options.teacher is not None
        else None
    )
    if teacher is not None and teacher.size < options.steps:
        raise ValueError("teacher file has fewer tokens than --steps")
    if teacher is not None and np.any(teacher >= config.vocab_size):
        raise ValueError("teacher token exceeds the checkpoint vocabulary")
    if any(value > prefix.size for value in options.checkpoints):
        raise ValueError("prefill checkpoints must fit the initial prefix")
    encoder = (
        options.encoder or options.directory / f"whisper-base-{options.label}-encoder"
    )
    decoder = (
        options.decoder or options.directory / f"whisper-base-{options.label}-decoder"
    )
    encoded, encoder_seconds = _encoded(options, encoder)
    calls: list[dict[str, JsonValue]] = []
    traces: dict[str, NDArray[np.float32]] = {}
    generated: list[int] = []
    checkpoints = sorted(set(options.checkpoints))
    with Runtime(decoder, threads=options.threads) as runtime:
        maximum = runtime.inputs["tokens"].maximum_shape[0]
        if options.prefill_chunk > maximum:
            raise ValueError(
                f"prefill chunk {options.prefill_chunk} exceeds compiled token bound {maximum}"
            )

        def invoke(
            tokens: NDArray[np.int64], position: int, phase: str
        ) -> NDArray[np.float32]:
            start = time.perf_counter()
            output = (
                runtime.run(
                    {
                        "tokens": tokens,
                        "positions": np.arange(
                            position, position + tokens.size, dtype=np.int64
                        ),
                        "encoded": encoded,
                        "initialize": np.asarray(position == 0, dtype=np.bool_),
                    }
                )["logits"]
                .ravel()
                .astype(np.float32)
            )
            record: dict[str, JsonValue] = {
                "phase": phase,
                "position": position,
                "tokens": [int(value) for value in tokens],
                "end_position": position + tokens.size,
                "diagnostic_seconds": time.perf_counter() - start,
                **_summary(output),
            }
            calls.append(record)
            if options.trace_full_logits or position + tokens.size in checkpoints:
                traces[f"call_{len(calls) - 1}_position_{position + tokens.size}"] = (
                    output.copy()
                )
            return output

        position = 0
        logits = np.empty(0, np.float32)
        while position < prefix.size:
            end = min(prefix.size, position + options.prefill_chunk)
            for checkpoint in checkpoints:
                if position < checkpoint < end:
                    end = checkpoint
                    break
            logits = invoke(prefix[position:end], position, "prefill")
            position = end
        for step in range(options.steps):
            selected = _select(logits, config, step)
            generated.append(selected)
            if (
                teacher is None and selected == config.eos_token_id
            ) or step + 1 == options.steps:
                break
            if position >= config.max_target_positions:
                raise ValueError("decoder cache reached its checkpoint position limit")
            token = selected if teacher is None else int(teacher[step])
            logits = invoke(np.asarray([token], np.int64), position, "decode")
            position += 1
    options.output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {
        "encoded": encoded,
        "generated": np.asarray(generated, np.int64),
        "prefix": prefix,
        "last_logits": logits,
        **traces,
    }
    np.savez(options.output.with_suffix(".npz"), allow_pickle=False, **arrays)
    manifest = decoder / "executable.json"
    symbols = ManifestSymbols.model_validate_json(manifest.read_bytes())
    report: dict[str, JsonValue] = {
        "scope": "Correctness replay; elapsed times are diagnostic, not isolated performance",
        "command": [value for value in sys.argv],
        "options": options.model_dump(mode="json"),
        "encoder_seconds": encoder_seconds,
        "compiled_token_bound": maximum,
        "selection_policy": {
            "suppress_tokens": [value for value in config.suppress_tokens],
            "begin_suppress_tokens": [value for value in config.begin_suppress_tokens],
            "begin_suppression_step": 0,
            "timestamp_token_start": 50364,
            "eos_token_id": config.eos_token_id,
        },
        "decoder_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selected_kernel_symbols": [
            value
            for value in sorted({kernel.symbol for kernel in symbols.kernels.values()})
        ],
        "kernel_libraries": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((decoder / "kernels").glob("*.so"))
        },
        "config_sha256": hashlib.sha256(
            (options.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "encoded_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "prefix": [int(value) for value in prefix],
        "generated": [value for value in generated],
        "calls": [record for record in calls],
    }
    options.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"generated": generated, "calls": len(calls), "output": str(options.output)}
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default="arm")
    parser.add_argument("--directory", type=Path, default=_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--parts",
        choices=("encoder", "decoder"),
        nargs="+",
        default=["encoder", "decoder"],
    )
    parser.add_argument("--max-input-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--encoded", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--forced-token-file", type=Path)
    parser.add_argument("--prefix", type=int, nargs="+")
    parser.add_argument("--bos-only", action="store_true")
    parser.add_argument("--prefill-chunk", type=int, default=16)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--trace-full-logits", action="store_true")
    parser.add_argument("--output", type=Path)
    run(Options.model_validate(vars(parser.parse_args())))


if __name__ == "__main__":
    main()
