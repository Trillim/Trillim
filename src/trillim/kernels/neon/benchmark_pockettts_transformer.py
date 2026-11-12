"""Measure PocketTTS's complete embedded-input autoregressive backbone.

Includes all six flow_lm.transformer layers, persistent KV caches and out_norm.
Excludes token embedding, latent input projection, EOS head, flow head and Mimi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ManifestSymbols, instruction_calls
from trillim.kernels.neon.benchmark_pockettts import file_hash
from trillim.runtime import Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=4, ge=1)
    cache: int = Field(default=512, ge=1)
    threads: int = Field(default=20, ge=1)
    warmups: int = Field(default=2, ge=0)
    repeats: int = Field(default=9, ge=1)
    compile_only: bool = False
    validate_only: bool = False
    torch_reference: bool = False
    reference_bundle: Path | None = None

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.prompt + self.decode > self.cache:
            raise ValueError("cache must fit prefill and all decode steps")
        return self


class ReferenceGraph(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt: int
    decode: int
    cache: int
    threads: int
    graph_sha256: str
    input_sha256: str
    checkpoint_sha256: str


class ReferenceTiming(ReferenceGraph):
    prefill_median_seconds: float
    decode_median_seconds: float


def graph_source(capacity: int) -> str:
    return f"""from trillim.tdl import *

weights = Model("model")
with Graph() as graph:
    x = Input(name="embedded", shape=(None,1024), dtype="fp32")
    positions = Input(name="positions", shape=(None,), dtype="int64")
    for layer in range(6):
        name = f"flow_lm.transformer.layers.{{layer}}."
        n = LayerNorm(x, weight=weights[name+"norm1.weight"], bias=weights[name+"norm1.bias"], eps=1e-5)
        qkv = Linear(n, weights[name+"self_attn.in_proj.weight"])
        q = SplitHeads(Slice(qkv, axis=1, start=0, stop=1024, rank=2), heads=16)
        k = SplitHeads(Slice(qkv, axis=1, start=1024, stop=2048, rank=2), heads=16)
        v = SplitHeads(Slice(qkv, axis=1, start=2048, stop=3072, rank=2), heads=16)
        q = Rotary(q, positions, dim=64, theta=10000.0, interleaved=True)
        k = Rotary(k, positions, dim=64, theta=10000.0, interleaved=True)
        keys = Cache(name=f"key_{{layer}}", shape=({capacity},16,64), dtype="fp32", mode="append", axis=0)
        values = Cache(name=f"value_{{layer}}", shape=({capacity},16,64), dtype="fp32", mode="append", axis=0)
        keys.append(k)
        values.append(v)
        attended = MergeHeads(Attention(q, keys.read(), values.read(), causal=True))
        x = Add(x, Linear(attended, weights[name+"self_attn.out_proj.weight"]))
        n = LayerNorm(x, weight=weights[name+"norm2.weight"], bias=weights[name+"norm2.bias"], eps=1e-5)
        x = Add(x, Linear(Gelu(Linear(n, weights[name+"linear1.weight"])), weights[name+"linear2.weight"]))
    x = LayerNorm(x, weight=weights["flow_lm.out_norm.weight"], bias=weights["flow_lm.out_norm.bias"], eps=1e-5)
    Output(x, name="conditioning")
"""


def compile_bundle(options: Options) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source = options.output.with_name(options.output.name + ".tdl")
    text = graph_source(options.cache)
    if source.exists() and source.read_text() != text:
        raise ValueError(f"existing graph differs: {source}")
    source.write_text(text)
    if not options.output.exists():
        _compile_model(
            _CompileRequest(
                tdl_path=source,
                checkpoints={"model": options.checkpoint},
                output=options.output,
                backend="scalar",
                depth="none",
                threads=options.threads,
                max_input_size=options.prompt,
            ),
            progress=lambda message: print(message, flush=True),
        )


def compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError("backbone reference shape differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    correct = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    if not correct:
        raise ValueError(f"backbone differs: maximum {maximum}, RMS {rms}")
    return {
        "maximum_absolute_error": maximum,
        "rms_error": rms,
        "allclose": correct,
        "comparison_rtol": 2e-4,
        "comparison_atol": 2e-4,
    }


def torch_reference(options: Options, features: Path) -> Path:
    destination = options.output.with_name(options.output.name + "-torch-output.npz")
    if destination.exists() and destination.with_suffix(".json").exists():
        return destination
    code = """import hashlib
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.stateful_module import init_states,increment_steps
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.modules.rope import apply_rope
torch.set_num_threads(int(sys.argv[4]))
prompt,steps,capacity = (int(v) for v in sys.argv[5:8])
model = StreamingTransformer(1024,16,6,dim_feedforward=4096,max_period=10000.0,kind="flow_lm").float().cpu().eval()
norm = torch.nn.LayerNorm(1024,eps=1e-5).float().cpu().eval()
prefix = "flow_lm.transformer."
with safe_open(sys.argv[1],framework="pt",device="cpu") as checkpoint:
    state = {name.removeprefix(prefix):checkpoint.get_tensor(name) for name in checkpoint.keys() if name.startswith(prefix)}
    norm_state = {name.removeprefix("flow_lm.out_norm."):checkpoint.get_tensor(name) for name in checkpoint.keys() if name.startswith("flow_lm.out_norm.")}
assert len(state)==48 and len(norm_state)==2
assert all(t.dtype==torch.bfloat16 for t in (*state.values(),*norm_state.values()))
model.load_state_dict(state,strict=True)
norm.load_state_dict(norm_state,strict=True)
features = np.load(sys.argv[2])
cache = init_states(model,batch_size=1,sequence_length=capacity)
results = []
with torch.inference_mode():
    prefill = norm(model(torch.from_numpy(features[:prompt]).unsqueeze(0),cache)).squeeze(0).numpy()
    increment_steps(model,cache,increment=prompt)
    for step in range(steps):
        chunk = torch.from_numpy(features[prompt+step:prompt+step+1]).unsqueeze(0)
        results.append(norm(model(chunk,cache)).squeeze(0).numpy())
        increment_steps(model,cache,increment=1)
np.savez(sys.argv[3],prefill=prefill,decode=np.concatenate(results,axis=0))
sources = {str(Path(inspect.getfile(inspect.unwrap(symbol)))):hashlib.sha256(Path(inspect.getfile(inspect.unwrap(symbol))).read_bytes()).hexdigest() for symbol in (StreamingTransformer,StreamingMultiheadAttention,apply_rope,init_states)}
metadata = dict(torch_version=torch.__version__,device="cpu",activation_dtype="fp32",sources_sha256=sources,native_weight_count=50,native_weight_dtype="bf16",native_weight_bytes=sum(t.numel()*t.element_size() for t in (*state.values(),*norm_state.values())),prompt=prompt,decode=steps,cache_capacity=capacity)
Path(sys.argv[3]).with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\\n")
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(options.checkpoint),
            str(features),
            str(destination),
            str(options.threads),
            str(options.prompt),
            str(options.decode),
            str(options.cache),
        ],
        env=os.environ
        | {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TMPDIR": os.environ.get("TMPDIR", str(Path.home() / ".trillim/ci/tmp")),
            "XDG_CACHE_HOME": str(Path.home() / ".trillim/ci/cache"),
            "TORCH_HOME": str(Path.home() / ".trillim/ci/torch"),
            "HF_HOME": str(Path.home() / ".trillim/ci/hf"),
        },
        check=True,
    )
    return destination


def benchmark(options: Options) -> dict[str, object]:
    rng = np.random.default_rng(1337)
    features = rng.normal(0, 0.5, (options.prompt + options.decode, 1024)).astype(
        np.float32
    )
    feature_path = options.output.with_name(options.output.name + "-inputs.npy")
    np.save(feature_path, features)
    prefill_times: list[float] = []
    decode_times: list[float] = []
    prefill: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    decoded: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    repetitions = 1 if options.validate_only else options.warmups + options.repeats
    for repeat in range(repetitions):
        steps: list[NDArray[np.float32]] = []
        with Runtime(options.output, threads=options.threads) as runtime:
            started = time.perf_counter() if not options.validate_only else 0.0
            prefill = runtime.run(
                {
                    "embedded": features[: options.prompt],
                    "positions": np.arange(options.prompt, dtype=np.int64),
                }
            )["conditioning"].astype(np.float32)
            if not options.validate_only and repeat >= options.warmups:
                prefill_times.append(time.perf_counter() - started)
            for step in range(options.decode):
                position = options.prompt + step
                started = time.perf_counter() if not options.validate_only else 0.0
                value = runtime.run(
                    {
                        "embedded": features[position : position + 1],
                        "positions": np.array([position], dtype=np.int64),
                    }
                )["conditioning"].astype(np.float32)
                if not options.validate_only and repeat >= options.warmups:
                    decode_times.append(time.perf_counter() - started)
                steps.append(value)
        decoded = np.concatenate(steps, axis=0)
        if not np.isfinite(prefill).all() or not np.isfinite(decoded).all():
            raise ValueError("backbone produced nonfinite outputs")
        print(f"backbone trial {repeat + 1} complete", flush=True)
    np.savez(
        options.output.with_name(options.output.name + "-output.npz"),
        prefill=prefill,
        decode=decoded,
    )
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    graph_digest = hashlib.sha256(graph_source(options.cache).encode()).hexdigest()
    input_digest = hashlib.sha256(features.tobytes()).hexdigest()
    checkpoint_digest = file_hash(options.checkpoint)
    record: dict[str, object] = {
        "label": options.label,
        "component": "PocketTTS embedded-input flow_lm.transformer plus out_norm",
        "scope": "All six backbone layers and persistent KV caches; excludes embedding, latent input projection, EOS head, flow head, solver and Mimi waveform decoder",
        "bundle": str(options.output),
        "checkpoint": str(options.checkpoint),
        "activation_dtype": "fp32",
        "native_weight_dtype": "bf16",
        "native_weight_count": 50,
        "native_weight_bytes": 151048192,
        "layers": 6,
        "hidden_size": 1024,
        "attention_heads": 16,
        "head_dimension": 64,
        "ffn_size": 4096,
        "prompt": options.prompt,
        "decode": options.decode,
        "cache": options.cache,
        "threads": options.threads,
        "host": platform.platform(),
        "backend_registration": "scalar",
        "compilation_depth": "none",
        "input_description": "Seed 1337 synthetic Gaussian embedded FP32 vectors, standard deviation0.5; consecutive absolute RoPE positions",
        "input_sha256": input_digest,
        "graph_sha256": graph_digest,
        "checkpoint_sha256": checkpoint_digest,
        "benchmark_source_sha256": file_hash(Path(__file__)),
        "compiled_weights_sha256": file_hash(options.output / "weights.bin"),
        "executable_sha256": file_hash(options.output / "executable.json"),
        "kernel_library_sha256": {
            name: file_hash(options.output / name)
            for name in sorted({kernel.library for kernel in manifest.kernels.values()})
        },
        "kernel_call_counts": dict(instruction_calls(manifest.program)),
        "validation_only": options.validate_only,
    }
    if prefill_times:
        record.update(
            timing_scope="Uninstrumented Runtime.run, fresh Runtime per trial with persistent cache across prefill and decode; excludes loading and compilation; no audio throughput claim",
            warmup_trials=options.warmups,
            measured_trials=options.repeats,
            prefill_seconds=prefill_times,
            decode_seconds=decode_times,
            prefill_median_seconds=statistics.median(prefill_times),
            decode_median_seconds=statistics.median(decode_times),
            decode_vectors_per_second=1 / statistics.median(decode_times),
        )
    if options.torch_reference:
        reference = torch_reference(options, feature_path)
        with np.load(reference) as expected:
            record["torch_prefill_comparison"] = compare(
                prefill, np.asarray(expected["prefill"], dtype=np.float32)
            )
            record["torch_decode_comparison"] = compare(
                decoded, np.asarray(expected["decode"], dtype=np.float32)
            )
        record["torch_reference_path"] = str(reference)
    if options.reference_bundle is not None:
        suffix = "-validation.json" if options.validate_only else "-results.json"
        metadata_bytes = options.reference_bundle.with_name(
            options.reference_bundle.name + suffix
        ).read_bytes()
        metadata = ReferenceGraph.model_validate_json(metadata_bytes)
        if (
            metadata.prompt != options.prompt
            or metadata.decode != options.decode
            or metadata.cache != options.cache
            or metadata.threads != options.threads
            or metadata.graph_sha256 != graph_digest
            or metadata.input_sha256 != input_digest
            or metadata.checkpoint_sha256 != checkpoint_digest
        ):
            raise ValueError("reference graph, input, weights or workload differs")
        with np.load(
            options.reference_bundle.with_name(
                options.reference_bundle.name + "-output.npz"
            )
        ) as expected:
            record["scalar_prefill_comparison"] = compare(
                prefill, np.asarray(expected["prefill"], dtype=np.float32)
            )
            record["scalar_decode_comparison"] = compare(
                decoded, np.asarray(expected["decode"], dtype=np.float32)
            )
        if prefill_times:
            timing = ReferenceTiming.model_validate_json(metadata_bytes)
            record["prefill_speedup_over_reference"] = (
                timing.prefill_median_seconds / statistics.median(prefill_times)
            )
            record["decode_speedup_over_reference"] = (
                timing.decode_median_seconds / statistics.median(decode_times)
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--decode", type=int, default=4)
    parser.add_argument("--cache", type=int, default=512)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--torch-reference", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    options = Options.model_validate(vars(parser.parse_args()))
    compile_bundle(options)
    if not options.compile_only:
        record = benchmark(options)
        suffix = "-validation.json" if options.validate_only else "-results.json"
        options.output.with_name(options.output.name + suffix).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
