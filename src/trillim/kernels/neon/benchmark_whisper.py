"""Measure the genuine Whisper encoder on a full 30-second mel input.

Choose the installed kernel library through TRILLIM_HOME. Both scalar and ARM
custom-kernel libraries use the supported scalar runtime registration. This
measures encoder inference only, excluding audio preprocessing and decoding.
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
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ManifestSymbols, instruction_calls
from trillim.runtime import Runtime


class WhisperConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: Literal["whisper"]
    d_model: int = Field(gt=0)
    encoder_attention_heads: int = Field(gt=0)
    encoder_ffn_dim: int = Field(gt=0)
    encoder_layers: int = Field(gt=0)
    num_mel_bins: int = Field(gt=0)
    max_source_positions: int = Field(gt=0)
    activation_function: Literal["gelu"]

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if self.d_model % self.encoder_attention_heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        return self


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    sampling_rate: int = Field(gt=0)
    hop_length: int = Field(gt=0)


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    threads: int = Field(default=20, ge=1)
    repeats: int = Field(default=3, ge=1)
    warmups: int = Field(default=1, ge=0)
    compile_only: bool = False
    torch_reference: bool = False
    reference_bundle: Path | None = None


class ReferenceTiming(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    threads: int
    frames: int
    graph_sha256: str
    input_sha256: str
    config_sha256: str
    encoder_median_seconds: float


def graph_source(config: WhisperConfig) -> str:
    hidden = config.d_model
    heads = config.encoder_attention_heads
    frames = config.max_source_positions * 2
    scale = (hidden // heads) ** -0.5
    return f"""from trillim.tdl import *

weights = Model("model")
with Graph() as graph:
    mel = Input(name="mel", shape=({config.num_mel_bins}, {frames}), dtype="fp32")
    x = Gelu(Conv(mel, weights["model.encoder.conv1.weight"], bias=weights["model.encoder.conv1.bias"], kernel=(3,), stride=1, padding=1))
    x = Gelu(Conv(x, weights["model.encoder.conv2.weight"], bias=weights["model.encoder.conv2.bias"], kernel=(3,), stride=2, padding=1))
    x = Add(Permute(x, (1, 0)), Constant(weights["model.encoder.embed_positions.weight"]))
    for layer in range({config.encoder_layers}):
        prefix = f"model.encoder.layers.{{layer}}."
        n = LayerNorm(x, weight=weights[prefix + "self_attn_layer_norm.weight"], bias=weights[prefix + "self_attn_layer_norm.bias"], eps=1e-5)
        q = Mul(Linear(n, weights[prefix + "self_attn.q_proj.weight"], bias=weights[prefix + "self_attn.q_proj.bias"]), {scale!r})
        k = Linear(n, weights[prefix + "self_attn.k_proj.weight"])
        v = Linear(n, weights[prefix + "self_attn.v_proj.weight"], bias=weights[prefix + "self_attn.v_proj.bias"])
        q, k, v = (SplitHeads(value, heads={heads}) for value in (q, k, v))
        a = MergeHeads(Attention(q, k, v, causal=False, scale=1.0))
        x = Add(x, Linear(a, weights[prefix + "self_attn.out_proj.weight"], bias=weights[prefix + "self_attn.out_proj.bias"]))
        n = LayerNorm(x, weight=weights[prefix + "final_layer_norm.weight"], bias=weights[prefix + "final_layer_norm.bias"], eps=1e-5)
        n = Gelu(Linear(n, weights[prefix + "fc1.weight"], bias=weights[prefix + "fc1.bias"]))
        x = Add(x, Linear(n, weights[prefix + "fc2.weight"], bias=weights[prefix + "fc2.bias"]))
    x = LayerNorm(x, weight=weights["model.encoder.layer_norm.weight"], bias=weights["model.encoder.layer_norm.bias"], eps=1e-5)
    Output(x, name="encoder_hidden_states")
"""


def compile_bundle(options: Options, config: WhisperConfig) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = options.output.with_name(options.output.name + ".tdl")
    source = graph_source(config)
    if source_path.exists() and source_path.read_text() != source:
        raise ValueError(f"existing graph source differs: {source_path}")
    source_path.write_text(source)
    if options.output.exists():
        return
    request = _CompileRequest(
        tdl_path=source_path,
        checkpoints={"model": options.checkpoint},
        output=options.output,
        backend="scalar",
        max_input_size=config.max_source_positions * 2,
        depth="none",
        threads=options.threads,
    )
    _compile_model(request, progress=lambda message: print(message, flush=True))


def compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError("reference encoder shape differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    scale = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12)
    correct = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    if not correct:
        raise ValueError(f"encoder differs from reference: max {maximum}, RMS {rms}")
    return dict(
        maximum_absolute_error=maximum,
        rms_error=rms,
        normalized_rms_error=rms / scale,
        allclose=correct,
        comparison_rtol=2e-4,
        comparison_atol=2e-4,
    )


def torch_reference(options: Options, features: Path) -> Path:
    destination = options.output.with_name(options.output.name + "-torch-encoder.npy")
    metadata = destination.with_suffix(".json")
    if destination.exists() and metadata.exists():
        return destination
    code = """import hashlib
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
torch.set_num_threads(int(sys.argv[4]))
model = WhisperModel.from_pretrained(sys.argv[1], local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").cpu().eval()
features = np.load(sys.argv[2])
with torch.inference_mode():
    result = model.encoder(torch.from_numpy(features).unsqueeze(0)).last_hidden_state.squeeze(0).numpy()
np.save(sys.argv[3], result)
source = Path(inspect.getfile(WhisperEncoder))
metadata = dict(torch_version=torch.__version__, transformers_version=transformers.__version__, attention_implementation="eager", device="cpu", dtype="fp32", input_sha256=hashlib.sha256(features.tobytes()).hexdigest(), modeling_whisper_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
Path(sys.argv[3]).with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\\n")
"""
    temporary = options.output.parent / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = os.environ | {
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(options.checkpoint.parent / ".hf-cache"),
    }
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
        env=environment,
        check=True,
    )
    return destination


def benchmark(options: Options, config: WhisperConfig) -> dict[str, object]:
    frames = config.max_source_positions * 2
    rng = np.random.default_rng(1337)
    mel = rng.normal(0, 0.5, (config.num_mel_bins, frames)).astype(np.float32)
    feature_path = options.output.with_name(options.output.name + "-mel.npy")
    np.save(feature_path, mel)
    feature_config = FeatureConfig.model_validate_json(
        (options.checkpoint / "preprocessor_config.json").read_bytes()
    )
    audio_seconds = frames * feature_config.hop_length / feature_config.sampling_rate
    samples: list[float] = []
    warmups: list[float] = []
    hashes: list[str] = []
    encoded: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    with Runtime(options.output, threads=options.threads) as runtime:
        for iteration in range(options.warmups + options.repeats):
            started = time.perf_counter()
            encoded = runtime.run({"mel": mel})["encoder_hidden_states"].astype(
                np.float32, copy=False
            )
            elapsed = time.perf_counter() - started
            if not np.isfinite(encoded).all():
                raise ValueError("encoder produced nonfinite values")
            if encoded.shape != (config.max_source_positions, config.d_model):
                raise ValueError("encoder output shape differs from the checkpoint")
            if iteration < options.warmups:
                warmups.append(elapsed)
            else:
                samples.append(elapsed)
                hashes.append(hashlib.sha256(encoded.tobytes()).hexdigest())
            print(f"encoder iteration {iteration + 1}: {elapsed:.6f}s", flush=True)
    encoder_path = options.output.with_name(options.output.name + "-encoder.npy")
    np.save(encoder_path, encoded)
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    counts = instruction_calls(manifest.program)
    median = statistics.median(samples)
    input_digest = hashlib.sha256(mel.tobytes()).hexdigest()
    graph_digest = hashlib.sha256(
        options.output.with_name(options.output.name + ".tdl").read_bytes()
    ).hexdigest()
    config_digest = hashlib.sha256(
        (options.checkpoint / "config.json").read_bytes()
    ).hexdigest()
    result: dict[str, object] = dict(
        label=options.label,
        checkpoint=str(options.checkpoint),
        bundle=str(options.output),
        model_type="whisper",
        timing_scope="Runtime.run for the encoder only, including input binding, executor, native kernels and output copy; excludes model loading, compilation, audio preprocessing and decoder",
        input_type="deterministic synthetic FP32 mel features, seed 1337, normal standard deviation 0.5",
        input_shape=mel.shape,
        output_shape=encoded.shape,
        represented_audio_seconds=audio_seconds,
        frames=frames,
        encoder_positions=config.max_source_positions,
        layers=config.encoder_layers,
        hidden_size=config.d_model,
        attention_heads=config.encoder_attention_heads,
        ffn_size=config.encoder_ffn_dim,
        threads=options.threads,
        backend_registration="scalar",
        activation_dtype="fp32",
        compilation_depth="none",
        host=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        warmup_seconds=warmups,
        encoder_seconds=samples,
        encoder_median_seconds=median,
        encoder_real_time_factor=median / audio_seconds,
        audio_seconds_per_encoder_second=audio_seconds / median,
        output_sha256=hashes,
        input_sha256=input_digest,
        graph_sha256=graph_digest,
        config_sha256=config_digest,
        kernels=sorted({metadata.symbol for metadata in manifest.kernels.values()}),
        kernel_call_counts=dict(counts),
        neon_kernel_call_count=sum(
            count for name, count in counts.items() if name.startswith("trillim_neon_")
        ),
    )
    if options.reference_bundle is not None:
        reference_metadata = ReferenceTiming.model_validate_json(
            options.reference_bundle.with_name(
                options.reference_bundle.name + "-results.json"
            ).read_bytes()
        )
        if (
            reference_metadata.threads != options.threads
            or reference_metadata.frames != frames
            or reference_metadata.graph_sha256 != graph_digest
            or reference_metadata.input_sha256 != input_digest
            or reference_metadata.config_sha256 != config_digest
        ):
            raise ValueError("reference graph, input, config or thread count differs")
        reference_path = options.reference_bundle.with_name(
            options.reference_bundle.name + "-encoder.npy"
        )
        expected = np.asarray(np.load(reference_path), dtype=np.float32)
        result["scalar_reference"] = compare(encoded, expected)
        result["reference_bundle"] = str(options.reference_bundle)
        result["encoder_speedup_over_reference"] = (
            reference_metadata.encoder_median_seconds / median
        )
    if options.torch_reference:
        reference_path = torch_reference(options, feature_path)
        expected = np.asarray(np.load(reference_path), dtype=np.float32)
        result["torch_reference"] = compare(encoded, expected)
        result["torch_reference_path"] = str(reference_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--torch-reference", action="store_true")
    parser.add_argument("--reference-bundle", type=Path)
    options = Options.model_validate(vars(parser.parse_args()))
    config = WhisperConfig.model_validate_json(
        (options.checkpoint / "config.json").read_bytes()
    )
    compile_bundle(options, config)
    if not options.compile_only:
        result = benchmark(options, config)
        destination = options.output.with_name(options.output.name + "-results.json")
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
