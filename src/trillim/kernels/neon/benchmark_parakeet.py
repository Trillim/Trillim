"""Compile and measure the Parakeet TDT Conformer encoder.

The graph includes all subsampling convolutions and Conformer blocks, and returns
the encoder hidden states before the TDT encoder projector. It excludes feature
extraction, padding, the predictor, joint network and token/duration decoding.
Use TRILLIM_HOME to select the installed scalar or custom ARM kernel library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
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
from trillim.kernels.neon.benchmark_models import ManifestSymbols, instruction_calls
from trillim.runtime import Runtime

_METADATA = TypeAdapter(dict[str, JsonValue])


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: Literal["parakeet_encoder"]
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    num_mel_bins: int = Field(gt=0)
    conv_kernel_size: int = Field(gt=0)
    subsampling_conv_channels: int = Field(gt=0)
    subsampling_conv_kernel_size: int = Field(gt=0)
    subsampling_conv_stride: int = Field(gt=0)
    subsampling_factor: int = Field(ge=2)
    hidden_act: Literal["silu"]
    attention_bias: Literal[False]
    convolution_bias: Literal[False]
    scale_input: bool

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads != self.num_key_value_heads
        ):
            raise ValueError(
                "encoder requires equal query/key/value heads dividing hidden size"
            )
        if (
            self.subsampling_factor & (self.subsampling_factor - 1)
            or self.subsampling_conv_stride != 2
        ):
            raise ValueError("subsampling requires a power-of-two factor and stride 2")
        if self.num_mel_bins % self.subsampling_factor:
            raise ValueError("mel dimensions must divide evenly through subsampling")
        if self.subsampling_conv_kernel_size % 2 != 1 or self.conv_kernel_size % 2 != 1:
            raise ValueError("encoder convolution kernels must be odd")
        return self


class ParakeetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: Literal["parakeet_tdt"]
    encoder_config: EncoderConfig


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    sampling_rate: int = Field(gt=0)
    hop_length: int = Field(gt=0)


class ProcessorConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    feature_extractor: FeatureConfig


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    frames: int = Field(default=256, ge=8)
    threads: int = Field(default=20, ge=1)
    repeats: int = Field(default=3, ge=1)
    warmups: int = Field(default=1, ge=0)
    seed: int = Field(default=20261112, ge=0)
    compile_only: bool = False
    correctness_only: bool = False
    torch_reference: bool = False
    reference_python: Path = Path.home() / ".trillim/qwen35/reference-env/bin/python"
    reference_bundle: Path | None = None


class ReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    input_sha256: str
    config_sha256: str


class PreviousResult(ReferenceMetadata):
    frames: int
    threads: int
    graph_sha256: str
    encoder_median_seconds: float | None = None


def graph_source(config: EncoderConfig, frames: int) -> str:
    channels = config.subsampling_conv_channels
    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads
    stages = config.subsampling_factor.bit_length() - 1
    frequency = config.num_mel_bins // config.subsampling_factor
    scale = math.sqrt(hidden) if config.scale_input else 1.0
    return f"""from trillim.tdl import *

weights = Model("model")
with Graph() as graph:
    features = Input(name="features", shape=({frames}, {config.num_mel_bins}), dtype="fp32")
    x = Reshape(features, (1, {frames}, {config.num_mel_bins}))
    x = Relu(Conv(x, weights["encoder.subsampling.layers.0.weight"], bias=weights["encoder.subsampling.layers.0.bias"], kernel=({config.subsampling_conv_kernel_size}, {config.subsampling_conv_kernel_size}), stride={config.subsampling_conv_stride}, padding={config.subsampling_conv_kernel_size // 2}))
    for stage in range({stages - 1}):
        index = 2 + stage * 3
        prefix = f"encoder.subsampling.layers.{{index}}."
        x = Conv(x, weights[prefix + "weight"], bias=weights[prefix + "bias"], kernel=({config.subsampling_conv_kernel_size}, {config.subsampling_conv_kernel_size}), stride={config.subsampling_conv_stride}, padding={config.subsampling_conv_kernel_size // 2}, groups={channels})
        prefix = f"encoder.subsampling.layers.{{index + 1}}."
        x = Relu(Conv(x, weights[prefix + "weight"], bias=weights[prefix + "bias"], kernel=(1, 1)))
    x = Reshape(Permute(x, (1, 0, 2)), (Dim(x, 1), {channels * frequency}))
    x = Mul(Linear(x, weights["encoder.subsampling.linear.weight"], bias=weights["encoder.subsampling.linear.bias"]), {scale!r})
    length = Dim(x, 0)
    positions = Iota(Mul(length, -1), start=Sub(length, 1), step=-1)
    positional = Sinusoidal(positions, dim={hidden}, interleaved=True, cos_first=False)
    for layer in range({config.num_hidden_layers}):
        prefix = f"encoder.layers.{{layer}}."
        n = LayerNorm(x, weight=weights[prefix + "norm_feed_forward1.weight"], bias=weights[prefix + "norm_feed_forward1.bias"], eps=1e-5)
        f = FFN(n, up=weights[prefix + "feed_forward1.linear1.weight"], down=weights[prefix + "feed_forward1.linear2.weight"])
        x = Add(x, Mul(f, 0.5))
        n = LayerNorm(x, weight=weights[prefix + "norm_self_att.weight"], bias=weights[prefix + "norm_self_att.bias"], eps=1e-5)
        q = SplitHeads(Linear(n, weights[prefix + "self_attn.q_proj.weight"]), heads={config.num_attention_heads}, head_dim={head_dim})
        k = SplitHeads(Linear(n, weights[prefix + "self_attn.k_proj.weight"]), heads={config.num_attention_heads}, head_dim={head_dim})
        v = SplitHeads(Linear(n, weights[prefix + "self_attn.v_proj.weight"]), heads={config.num_attention_heads}, head_dim={head_dim})
        r = SplitHeads(Linear(positional, weights[prefix + "self_attn.relative_k_proj.weight"]), heads={config.num_attention_heads}, head_dim={head_dim})
        a = RelativeAttention(q, k, v, r, content_bias=Constant(weights[prefix + "self_attn.bias_u"]), position_bias=Constant(weights[prefix + "self_attn.bias_v"]))
        x = Add(x, Linear(MergeHeads(a), weights[prefix + "self_attn.o_proj.weight"]))
        n = LayerNorm(x, weight=weights[prefix + "norm_conv.weight"], bias=weights[prefix + "norm_conv.bias"], eps=1e-5)
        c = Conv(Permute(n, (1, 0)), weights[prefix + "conv.pointwise_conv1.weight"], kernel=(1,))
        c = Mul(Slice(c, axis=0, stop={hidden}, rank=2), Sigmoid(Slice(c, axis=0, start={hidden}, rank=2)))
        c = Conv(c, weights[prefix + "conv.depthwise_conv.weight"], kernel=({config.conv_kernel_size},), padding={config.conv_kernel_size // 2}, groups={hidden})
        c = BatchNorm(c, running_mean=weights[prefix + "conv.norm.running_mean"], running_var=weights[prefix + "conv.norm.running_var"], weight=weights[prefix + "conv.norm.weight"], bias=weights[prefix + "conv.norm.bias"], eps=1e-5, rank=2)
        c = Conv(Silu(c), weights[prefix + "conv.pointwise_conv2.weight"], kernel=(1,))
        x = Add(x, Permute(c, (1, 0)))
        n = LayerNorm(x, weight=weights[prefix + "norm_feed_forward2.weight"], bias=weights[prefix + "norm_feed_forward2.bias"], eps=1e-5)
        f = FFN(n, up=weights[prefix + "feed_forward2.linear1.weight"], down=weights[prefix + "feed_forward2.linear2.weight"])
        x = LayerNorm(Add(x, Mul(f, 0.5)), weight=weights[prefix + "norm_out.weight"], bias=weights[prefix + "norm_out.bias"], eps=1e-5)
    Output(x, name="encoder_hidden_states")
"""


def compile_bundle(options: Options, config: EncoderConfig) -> Path:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source = graph_source(config, options.frames)
    source_path = options.output.with_name(options.output.name + ".tdl")
    if (
        options.output.exists()
        and source_path.exists()
        and source_path.read_text() != source
    ):
        raise ValueError(f"existing graph source differs: {source_path}")
    source_path.write_text(source)
    if not options.output.exists():
        _compile_model(
            _CompileRequest(
                tdl_path=source_path,
                checkpoints={"model": options.checkpoint},
                output=options.output,
                backend="scalar",
                depth="none",
                max_input_size=options.frames,
                threads=options.threads,
            ),
            progress=lambda message: print(message, flush=True),
        )
    return source_path


def _compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError("reference encoder shape differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    scale = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12)
    correct = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    result: dict[str, object] = {
        "maximum_absolute_error": maximum,
        "rms_error": rms,
        "normalized_rms_error": rms / scale,
        "allclose": correct,
        "comparison_rtol": 2e-4,
        "comparison_atol": 2e-4,
    }
    if not correct:
        raise ValueError(f"Parakeet encoder differs from reference: {result}")
    return result


def torch_reference(options: Options, features_path: Path) -> Path:
    destination = options.output.with_name(options.output.name + "-torch-encoder.npy")
    metadata_path = destination.with_suffix(".json")
    features = np.asarray(np.load(features_path), dtype=np.float32)
    input_digest = hashlib.sha256(features.tobytes()).hexdigest()
    config_digest = hashlib.sha256(
        (options.checkpoint / "config.json").read_bytes()
    ).hexdigest()
    if destination.exists() and metadata_path.exists():
        previous = ReferenceMetadata.model_validate_json(metadata_path.read_bytes())
        if (
            previous.input_sha256 != input_digest
            or previous.config_sha256 != config_digest
        ):
            raise ValueError(
                "cached Torch reference has different inputs or checkpoint config"
            )
        return destination
    code = """import hashlib
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import ParakeetForTDT
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoder
torch.set_num_threads(int(sys.argv[4]))
model = ParakeetForTDT.from_pretrained(sys.argv[1], local_files_only=True, dtype=torch.float32, attn_implementation="eager").cpu().eval()
features = np.load(sys.argv[2])
with torch.inference_mode():
    result = model.encoder(torch.from_numpy(features).unsqueeze(0), attention_mask=None).last_hidden_state.squeeze(0).numpy()
np.save(sys.argv[3], result)
source = Path(inspect.getfile(ParakeetEncoder))
metadata = dict(torch_version=torch.__version__, transformers_version=transformers.__version__, device="cpu", dtype="fp32", threads=torch.get_num_threads(), attention_implementation="eager", attention_mask=None, input_sha256=hashlib.sha256(features.tobytes()).hexdigest(), config_sha256=hashlib.sha256((Path(sys.argv[1])/"config.json").read_bytes()).hexdigest(), modeling_parakeet_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
Path(sys.argv[3]).with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\\n")
"""
    temporary = options.output.parent / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = os.environ | {
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(options.checkpoint.parent / ".hf-cache"),
        "XDG_CACHE_HOME": str(temporary / "xdg"),
        "TORCH_HOME": str(temporary / "torch"),
    }
    subprocess.run(
        [
            str(options.reference_python),
            "-c",
            code,
            str(options.checkpoint),
            str(features_path),
            str(destination),
            str(options.threads),
        ],
        env=environment,
        check=True,
    )
    return destination


def benchmark(
    options: Options, config: EncoderConfig, source_path: Path
) -> dict[str, object]:
    features = (
        np.random.default_rng(options.seed)
        .normal(0.0, 0.5, (options.frames, config.num_mel_bins))
        .astype(np.float32)
    )
    feature_path = options.output.with_name(options.output.name + "-features.npy")
    np.save(feature_path, features)
    input_digest = hashlib.sha256(features.tobytes()).hexdigest()
    config_digest = hashlib.sha256(
        (options.checkpoint / "config.json").read_bytes()
    ).hexdigest()
    graph_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    feature_config = ProcessorConfig.model_validate_json(
        (options.checkpoint / "processor_config.json").read_bytes()
    ).feature_extractor
    audio_seconds = (
        options.frames * feature_config.hop_length / feature_config.sampling_rate
    )
    reference_path = (
        torch_reference(options, feature_path) if options.torch_reference else None
    )
    samples: list[float] = []
    warmups: list[float] = []
    hashes: list[str] = []
    output: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    output_frames = options.frames
    for _ in range(config.subsampling_factor.bit_length() - 1):
        output_frames = (output_frames - 1) // config.subsampling_conv_stride + 1
    iterations = 1 if options.correctness_only else options.warmups + options.repeats
    with Runtime(options.output, threads=options.threads) as runtime:
        for index in range(iterations):
            started = time.perf_counter()
            output = runtime.run({"features": features})[
                "encoder_hidden_states"
            ].astype(np.float32, copy=False)
            elapsed = time.perf_counter() - started
            if (
                output.shape != (output_frames, config.hidden_size)
                or not np.isfinite(output).all()
            ):
                raise ValueError(
                    "encoder produced unexpected dimensions or nonfinite values"
                )
            if options.correctness_only:
                print("encoder correctness invocation complete", flush=True)
            elif index < options.warmups:
                warmups.append(elapsed)
            else:
                samples.append(elapsed)
                hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
                print(f"encoder iteration {index + 1}: {elapsed:.6f}s", flush=True)
    if hashlib.sha256(features.tobytes()).hexdigest() != input_digest:
        raise ValueError("encoder modified caller input features")
    if len(set(hashes)) > 1:
        raise ValueError("encoder output changed between identical invocations")
    output_path = options.output.with_name(options.output.name + "-encoder.npy")
    np.save(output_path, output)
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    counts = instruction_calls(manifest.program)
    symbols: Counter[str] = Counter()
    for name, count in counts.items():
        symbols[manifest.kernels[name].symbol] += count
    median_seconds = statistics.median(samples) if samples else None
    result: dict[str, object] = {
        "label": options.label,
        "checkpoint": str(options.checkpoint),
        "bundle": str(options.output),
        "model_type": "parakeet_tdt",
        "scope": "full Conformer encoder before encoder_projector; one unpadded sequence, no attention mask",
        "timing_scope": "Runtime.run encoder including input binding, executor, kernels and output copy; excludes compilation, model loading, feature extraction, TDT predictor/joint/token-duration decoding",
        "correctness_only": options.correctness_only,
        "input_type": "deterministic synthetic FP32 features, normal mean0 std0.5",
        "seed": options.seed,
        "input_shape": features.shape,
        "output_shape": output.shape,
        "frames": options.frames,
        "encoder_positions": output_frames,
        "nominal_audio_seconds": audio_seconds,
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "threads": options.threads,
        "activation_dtype": "fp32",
        "backend_registration": "scalar",
        "compilation_depth": "none",
        "host": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "warmup_seconds": warmups,
        "encoder_seconds": samples,
        "encoder_median_seconds": median_seconds,
        "encoder_real_time_factor": median_seconds / audio_seconds
        if median_seconds is not None
        else None,
        "input_sha256": input_digest,
        "config_sha256": config_digest,
        "graph_sha256": graph_digest,
        "output_sha256": hashes,
        "final_output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "selected_kernel_symbols": dict(sorted(symbols.items())),
    }
    if reference_path is not None:
        reference = np.asarray(np.load(reference_path), dtype=np.float32)
        result["torch_reference_comparison"] = _compare(output, reference)
        result["torch_reference_metadata"] = _METADATA.validate_json(
            reference_path.with_suffix(".json").read_bytes()
        )
    if options.reference_bundle is not None:
        previous_path = options.reference_bundle.with_name(
            options.reference_bundle.name + "-results.json"
        )
        previous = PreviousResult.model_validate_json(previous_path.read_bytes())
        if (
            previous.frames,
            previous.threads,
            previous.input_sha256,
            previous.config_sha256,
            previous.graph_sha256,
        ) != (
            options.frames,
            options.threads,
            input_digest,
            config_digest,
            graph_digest,
        ):
            raise ValueError(
                "reference bundle uses different graph, checkpoint, workload or threads"
            )
        expected = np.asarray(
            np.load(
                options.reference_bundle.with_name(
                    options.reference_bundle.name + "-encoder.npy"
                )
            ),
            dtype=np.float32,
        )
        result["reference_bundle_comparison"] = _compare(output, expected)
        if median_seconds is not None and previous.encoder_median_seconds is not None:
            result["speedup_vs_reference"] = (
                previous.encoder_median_seconds / median_seconds
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "Trillim/models/arm-matrix/nvidia/parakeet-tdt-0.6b-v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="parakeet-encoder")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--torch-reference", action="store_true")
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path.home() / ".trillim/qwen35/reference-env/bin/python",
    )
    parser.add_argument("--reference-bundle", type=Path)
    options = Options.model_validate(vars(parser.parse_args()))
    config = ParakeetConfig.model_validate_json(
        (options.checkpoint / "config.json").read_bytes()
    ).encoder_config
    source_path = compile_bundle(options, config)
    if options.compile_only:
        return
    result = benchmark(options, config, source_path)
    destination = options.output.with_name(options.output.name + "-results.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
