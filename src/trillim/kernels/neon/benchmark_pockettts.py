"""Measure PocketTTS's complete flow_lm.flow_net component with native weights.

This excludes the autoregressive backbone, flow solver, audio decoder and text
processing. Choose the installed kernel library through TRILLIM_HOME.
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

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ManifestSymbols, instruction_calls
from trillim.runtime import Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    batch: int = Field(default=1, ge=1)
    threads: int = Field(default=20, ge=1)
    repeats: int = Field(default=9, ge=1)
    warmups: int = Field(default=2, ge=0)
    compile_only: bool = False
    validate_only: bool = False
    torch_reference: bool = False
    reference_bundle: Path | None = None


class ReferenceGraph(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    batch: int
    threads: int
    graph_sha256: str
    input_sha256: str
    checkpoint_sha256: str


class ReferenceTiming(ReferenceGraph):
    flow_head_median_seconds: float


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def graph_source(batch: int) -> str:
    return f"""from trillim.tdl import *

weights = Model("model")
prefix = "flow_lm.flow_net."
with Graph() as graph:
    c = Input(name="c", shape=({batch},1024), dtype="fp32")
    s = Input(name="s", shape=({batch},1), dtype="fp32")
    t = Input(name="t", shape=({batch},1), dtype="fp32")
    x = Input(name="x", shape=({batch},32), dtype="fp32")
    x = Linear(x, weights[prefix+"input_proj.weight"], bias=weights[prefix+"input_proj.bias"])
    times = []
    for index, value in enumerate((s,t)):
        name = prefix+f"time_embed.{{index}}."
        args = Mul(value, Cast(Constant(weights[name+"freqs"]), "fp32"))
        embedding = Concat((Cos(args), Sin(args)), axis=1, rank=2)
        embedding = Silu(Linear(embedding, weights[name+"mlp.0.weight"], bias=weights[name+"mlp.0.bias"]))
        embedding = Linear(embedding, weights[name+"mlp.2.weight"], bias=weights[name+"mlp.2.bias"])
        times.append(VarianceNorm(embedding, weight=weights[name+"mlp.3.alpha"], correction=1, eps=1e-5))
    combined = Div(Add(times[0], times[1]), 2)
    y = Add(combined, Linear(c, weights[prefix+"cond_embed.weight"], bias=weights[prefix+"cond_embed.bias"]))
    for index in range(6):
        name = prefix+f"res_blocks.{{index}}."
        modulation = Linear(Silu(y), weights[name+"adaLN_modulation.1.weight"], bias=weights[name+"adaLN_modulation.1.bias"])
        shift = Slice(modulation, axis=1, start=0, stop=512, rank=2)
        scale = Slice(modulation, axis=1, start=512, stop=1024, rank=2)
        gate = Slice(modulation, axis=1, start=1024, stop=1536, rank=2)
        h = LayerNorm(x, weight=weights[name+"in_ln.weight"], bias=weights[name+"in_ln.bias"], eps=1e-6)
        h = Add(Mul(h, Add(scale, 1)), shift)
        h = Silu(Linear(h, weights[name+"mlp.0.weight"], bias=weights[name+"mlp.0.bias"]))
        h = Linear(h, weights[name+"mlp.2.weight"], bias=weights[name+"mlp.2.bias"])
        x = Add(x, Mul(gate, h))
    name = prefix+"final_layer."
    modulation = Linear(Silu(y), weights[name+"adaLN_modulation.1.weight"], bias=weights[name+"adaLN_modulation.1.bias"])
    shift = Slice(modulation, axis=1, start=0, stop=512, rank=2)
    scale = Slice(modulation, axis=1, start=512, stop=1024, rank=2)
    x = Add(Mul(LayerNorm(x, eps=1e-6), Add(scale, 1)), shift)
    x = Linear(x, weights[name+"linear.weight"], bias=weights[name+"linear.bias"])
    Output(x, name="velocity")
"""


def compile_bundle(options: Options) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source = options.output.with_name(options.output.name + ".tdl")
    text = graph_source(options.batch)
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
            ),
            progress=lambda message: print(message, flush=True),
        )


def inputs(batch: int) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(1337)
    return {
        "c": rng.normal(0, 0.5, (batch, 1024)).astype(np.float32),
        "s": np.linspace(0.0, 0.4, batch, dtype=np.float32).reshape(batch, 1),
        "t": np.linspace(0.6, 1.0, batch, dtype=np.float32).reshape(batch, 1),
        "x": rng.normal(0, 0.5, (batch, 32)).astype(np.float32),
    }


def compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError("flow head reference shape differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    correct = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    if not correct:
        raise ValueError(f"flow head differs: maximum {maximum}, RMS {rms}")
    return dict(
        maximum_absolute_error=maximum,
        rms_error=rms,
        allclose=correct,
        comparison_rtol=2e-4,
        comparison_atol=2e-4,
    )


def torch_reference(options: Options, features: Path) -> Path:
    destination = options.output.with_name(options.output.name + "-torch-velocity.npy")
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
from pocket_tts.modules.mlp import SimpleMLPAdaLN
torch.set_num_threads(int(sys.argv[4]))
model = SimpleMLPAdaLN(32,512,32,1024,6,num_time_conds=2).float().cpu().eval()
prefix = "flow_lm.flow_net."
with safe_open(sys.argv[1],framework="pt",device="cpu") as checkpoint:
    state = {name.removeprefix(prefix):checkpoint.get_tensor(name) for name in checkpoint.keys() if name.startswith(prefix)}
assert len(state)==68
assert all(tensor.dtype==torch.bfloat16 for tensor in state.values())
model.load_state_dict(state,strict=True)
features = np.load(sys.argv[2])
with torch.inference_mode():
    result = model(*(torch.from_numpy(features[name]) for name in ("c","s","t","x"))).numpy()
np.save(sys.argv[3],result)
source = Path(inspect.getfile(SimpleMLPAdaLN))
metadata = dict(torch_version=torch.__version__, device="cpu", activation_dtype="fp32", source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), weight_count=len(state), native_weight_dtype="bf16", native_weight_bytes=sum(t.numel()*t.element_size() for t in state.values()))
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
    features = inputs(options.batch)
    feature_path = options.output.with_name(options.output.name + "-inputs.npz")
    np.savez(
        feature_path, c=features["c"], s=features["s"], t=features["t"], x=features["x"]
    )
    samples: list[float] = []
    warmups: list[float] = []
    encoded: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    iterations = 1 if options.validate_only else options.warmups + options.repeats
    with Runtime(options.output, threads=options.threads) as runtime:
        for iteration in range(iterations):
            if options.validate_only:
                encoded = runtime.run(features)["velocity"].astype(np.float32)
            else:
                started = time.perf_counter()
                encoded = runtime.run(features)["velocity"].astype(np.float32)
                elapsed = time.perf_counter() - started
                (warmups if iteration < options.warmups else samples).append(elapsed)
                print(
                    f"flow head iteration {iteration + 1}: {elapsed:.6f}s", flush=True
                )
            if encoded.shape != (options.batch, 32) or not np.isfinite(encoded).all():
                raise ValueError("flow head output shape or values are invalid")
    np.save(options.output.with_name(options.output.name + "-velocity.npy"), encoded)
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    counts = instruction_calls(manifest.program)
    input_digest = hashlib.sha256(
        b"".join(features[name].tobytes() for name in ("c", "s", "t", "x"))
    ).hexdigest()
    graph_digest = hashlib.sha256(graph_source(options.batch).encode()).hexdigest()
    checkpoint_digest = file_hash(options.checkpoint)
    result: dict[str, object] = dict(
        label=options.label,
        checkpoint=str(options.checkpoint),
        bundle=str(options.output),
        component="PocketTTS flow_lm.flow_net",
        scope="Complete flow head only; excludes autoregressive backbone, flow solver, audio decoder and text processing",
        activation_dtype="fp32",
        native_weight_dtype="bf16",
        native_weight_count=68,
        native_weight_bytes=19518016,
        residual_blocks=6,
        hidden_size=512,
        condition_size=1024,
        latent_size=32,
        time_conditions=2,
        batch=options.batch,
        threads=options.threads,
        backend_registration="scalar",
        compilation_depth="none",
        input_description="Seed 1337 Gaussian FP32 c/x, fixed linspace s/t, native checkpoint BF16 frequency values",
        input_sha256=input_digest,
        graph_sha256=graph_digest,
        checkpoint_sha256=checkpoint_digest,
        benchmark_source_sha256=file_hash(Path(__file__)),
        executable_sha256=file_hash(options.output / "executable.json"),
        compiled_weights_sha256=file_hash(options.output / "weights.bin"),
        compiled_constants_sha256=file_hash(options.output / "constants.bin"),
        kernel_library_sha256={
            name: file_hash(options.output / name)
            for name in sorted({kernel.library for kernel in manifest.kernels.values()})
        },
        output_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
        kernel_call_counts=dict(counts),
        host=platform.platform(),
        validation_only=options.validate_only,
    )
    if samples:
        median = statistics.median(samples)
        result.update(
            timing_scope="Uninstrumented Runtime.run including input binding, executor, native kernels and output copy; excludes loading and compilation",
            warmup_seconds=warmups,
            flow_head_seconds=samples,
            flow_head_median_seconds=median,
            flow_head_calls_per_second=1 / median,
        )
    if options.torch_reference:
        reference = torch_reference(options, feature_path)
        result["torch_comparison"] = compare(
            encoded, np.asarray(np.load(reference), dtype=np.float32)
        )
        result["torch_reference_path"] = str(reference)
    if options.reference_bundle is not None:
        suffix = "-validation.json" if options.validate_only else "-results.json"
        metadata_bytes = options.reference_bundle.with_name(
            options.reference_bundle.name + suffix
        ).read_bytes()
        metadata = ReferenceGraph.model_validate_json(metadata_bytes)
        if (
            metadata.batch != options.batch
            or metadata.threads != options.threads
            or metadata.graph_sha256 != graph_digest
            or metadata.input_sha256 != input_digest
            or metadata.checkpoint_sha256 != checkpoint_digest
        ):
            raise ValueError("reference graph, input, batch or threads differ")
        reference = options.reference_bundle.with_name(
            options.reference_bundle.name + "-velocity.npy"
        )
        result["scalar_comparison"] = compare(
            encoded, np.asarray(np.load(reference), dtype=np.float32)
        )
        if samples:
            timing = ReferenceTiming.model_validate_json(metadata_bytes)
            result["speedup_over_reference"] = (
                timing.flow_head_median_seconds / statistics.median(samples)
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--torch-reference", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    options = Options.model_validate(vars(parser.parse_args()))
    compile_bundle(options)
    if not options.compile_only:
        result = benchmark(options)
        suffix = "-validation.json" if options.validate_only else "-results.json"
        options.output.with_name(options.output.name + suffix).write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
