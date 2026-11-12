"""Measure PocketTTS's complete initial-state latent-to-waveform decoder.

Includes latent affine, quantizer projection, upsampling, decoder transformer
and SEANet waveform synthesis. Each invocation starts with zero history and
zero transposed-convolution partials. This does not measure streaming or TTS.
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
from trillim.kernels.neon.benchmark_pockettts import file_hash
from trillim.runtime import Runtime


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    label: str
    frames: int = Field(default=1, ge=1, le=32)
    threads: int = Field(default=20, ge=1)
    warmups: int = Field(default=2, ge=0)
    repeats: int = Field(default=9, ge=1)
    compile_only: bool = False
    validate_only: bool = False
    torch_reference: bool = False
    reference_bundle: Path | None = None


class ReferenceGraph(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    frames: int
    threads: int
    graph_sha256: str
    input_sha256: str
    checkpoint_sha256: str


class ReferenceTiming(ReferenceGraph):
    decoder_median_seconds: float


def graph_source() -> str:
    return """from trillim.tdl import *

weights = Model("model")
def causal_conv(x, name, kernel):
    x = Pad(x, padding=((0,0),(kernel-1,0)))
    return Conv(x, weights[name+"weight"], bias=weights[name+"bias"], kernel=(kernel,))
with Graph() as graph:
    latent = Input(name="latent", shape=(None,32), dtype="fp32")
    x = Add(Mul(latent, Constant(weights["flow_lm.emb_std"])), Constant(weights["flow_lm.emb_mean"]))
    x = Permute(x, (1,0))
    x = Conv(x, weights["mimi.quantizer.output_proj.weight"], kernel=(1,))
    x = ConvTranspose(x, weights["mimi.upsample.convtr.convtr.weight"], kernel=(32,), stride=16, groups=512)
    x = Slice(x, axis=1, stop=-16, rank=2)
    x = Permute(x, (1,0))
    positions = Iota(Dim(x,0))
    for layer in range(2):
        name = f"mimi.decoder_transformer.transformer.layers.{layer}."
        n = LayerNorm(x, weight=weights[name+"norm1.weight"], bias=weights[name+"norm1.bias"], eps=1e-5)
        qkv = Linear(n, weights[name+"self_attn.in_proj.weight"])
        q = SplitHeads(Slice(qkv, axis=1, start=0, stop=512, rank=2), heads=8)
        k = SplitHeads(Slice(qkv, axis=1, start=512, stop=1024, rank=2), heads=8)
        v = SplitHeads(Slice(qkv, axis=1, start=1024, stop=1536, rank=2), heads=8)
        q = Rotary(q, positions, dim=64, theta=10000.0, interleaved=True)
        k = Rotary(k, positions, dim=64, theta=10000.0, interleaved=True)
        attended = MergeHeads(Attention(q,k,v,causal=True,window=250))
        update = Linear(attended, weights[name+"self_attn.out_proj.weight"])
        x = Add(x, Mul(update, Constant(weights[name+"layer_scale_1.scale"])))
        n = LayerNorm(x, weight=weights[name+"norm2.weight"], bias=weights[name+"norm2.bias"], eps=1e-5)
        update = Linear(Gelu(Linear(n, weights[name+"linear1.weight"])), weights[name+"linear2.weight"])
        x = Add(x, Mul(update, Constant(weights[name+"layer_scale_2.scale"])))
    x = Permute(x,(1,0))
    x = causal_conv(x, "mimi.decoder.model.0.conv.", 7)
    for ratio, upsample, residual in ((6,2,3),(5,5,6),(4,8,9)):
        name = f"mimi.decoder.model.{upsample}.convtr."
        x = ConvTranspose(Elu(x), weights[name+"weight"], bias=weights[name+"bias"], kernel=(2*ratio,), stride=ratio)
        x = Slice(x, axis=1, stop=-ratio, rank=2)
        name = f"mimi.decoder.model.{residual}.block."
        h = causal_conv(Elu(x), name+"1.conv.", 3)
        h = causal_conv(Elu(h), name+"3.conv.", 1)
        x = Add(x,h)
    x = causal_conv(Elu(x), "mimi.decoder.model.11.conv.", 3)
    Output(x,name="waveform")
"""


def compile_bundle(options: Options) -> None:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source = options.output.with_name(options.output.name + ".tdl")
    text = graph_source()
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
                max_input_size=options.frames,
            ),
            progress=lambda message: print(message, flush=True),
        )


def compare(
    actual: NDArray[np.float32], expected: NDArray[np.float32]
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise ValueError("waveform reference shape differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    correct = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-4))
    if not correct:
        raise ValueError(f"waveform differs: maximum {maximum}, RMS {rms}")
    return {
        "maximum_absolute_error": maximum,
        "rms_error": rms,
        "allclose": correct,
        "comparison_rtol": 2e-4,
        "comparison_atol": 2e-4,
    }


def torch_reference(options: Options, features: Path) -> Path:
    destination = options.output.with_name(options.output.name + "-torch-waveform.npy")
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
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.seanet import SEANetEncoder,SEANetDecoder
from pocket_tts.modules.mimi_transformer import ProjectedTransformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.resample import ConvTrUpsample1d
from pocket_tts.modules.conv import StreamingConv1d,StreamingConvTranspose1d
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.modules.rope import apply_rope
from pocket_tts.modules.layer_scale import LayerScale
torch.set_num_threads(int(sys.argv[4]))
seanet = dict(channels=1,dimension=512,n_filters=64,n_residual_layers=1,ratios=[6,5,4],kernel_size=7,last_kernel_size=3,residual_kernel_size=3,dilation_base=2,pad_mode="constant",compress=2)
transformer = dict(input_dimension=512,output_dimensions=(512,),d_model=512,num_heads=8,num_layers=2,layer_scale=0.01,context=250,max_period=10000.0,dim_feedforward=2048)
model = MimiModel(encoder=SEANetEncoder(**seanet),decoder=SEANetDecoder(**seanet),quantizer=DummyQuantizer(32,512),frame_rate=12.5,encoder_frame_rate=200.0,sample_rate=24000,channels=1,encoder_transformer=ProjectedTransformer(**transformer),decoder_transformer=ProjectedTransformer(**transformer)).float().cpu().eval()
all_state = {}
with safe_open(sys.argv[1],framework="pt",device="cpu") as checkpoint:
    for name in ("quantizer","upsample","decoder_transformer","decoder"):
        prefix = "mimi."+name+"."
        state = {key.removeprefix(prefix):checkpoint.get_tensor(key) for key in checkpoint.keys() if key.startswith(prefix)}
        getattr(model,name).load_state_dict(state,strict=True)
        all_state.update({prefix+k:v for k,v in state.items()})
    mean = checkpoint.get_tensor("flow_lm.emb_mean")
    std = checkpoint.get_tensor("flow_lm.emb_std")
assert len(all_state)==44
assert all(t.dtype==torch.bfloat16 for t in all_state.values())
features = torch.from_numpy(np.load(sys.argv[2])).unsqueeze(0)
with torch.inference_mode():
    latent = (features*std.float()+mean.float()).transpose(1,2)
    projected = model.quantizer(latent)
    state = init_states(model,batch_size=1,sequence_length=1000)
    result = model.decode_from_latent(projected,state).squeeze(0).numpy()
np.save(sys.argv[3],result)
symbols = (MimiModel,SEANetDecoder,ProjectedTransformer,DummyQuantizer,ConvTrUpsample1d,StreamingConv1d,StreamingConvTranspose1d,init_states,apply_rope,LayerScale)
sources = {str(Path(inspect.getfile(inspect.unwrap(s)))):hashlib.sha256(Path(inspect.getfile(inspect.unwrap(s))).read_bytes()).hexdigest() for s in symbols}
metadata = dict(torch_version=torch.__version__,device="cpu",activation_dtype="fp32",sources_sha256=sources,native_decoder_weight_count=len(all_state),native_decoder_weight_bytes=sum(t.numel()*t.element_size() for t in all_state.values()),affine_statistics_dtypes=[str(mean.dtype),str(std.dtype)],initial_state="zero convolution history and transposed-convolution partials; empty causal attention cache",sample_rate=24000)
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
    features = (
        np.random.default_rng(1337)
        .normal(0, 0.5, (options.frames, 32))
        .astype(np.float32)
    )
    feature_path = options.output.with_name(options.output.name + "-inputs.npy")
    np.save(feature_path, features)
    encoded: NDArray[np.float32] = np.empty((0,), dtype=np.float32)
    samples: list[float] = []
    repetitions = 1 if options.validate_only else options.warmups + options.repeats
    with Runtime(options.output, threads=options.threads) as runtime:
        for repeat in range(repetitions):
            started = time.perf_counter() if not options.validate_only else 0.0
            encoded = runtime.run({"latent": features})["waveform"].astype(np.float32)
            if not options.validate_only and repeat >= options.warmups:
                samples.append(time.perf_counter() - started)
            if (
                encoded.shape != (1, 1920 * options.frames)
                or not np.isfinite(encoded).all()
            ):
                raise ValueError("waveform output shape or values are invalid")
            print(f"waveform invocation {repeat + 1} complete", flush=True)
    np.save(options.output.with_name(options.output.name + "-waveform.npy"), encoded)
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    graph_digest = hashlib.sha256(graph_source().encode()).hexdigest()
    input_digest = hashlib.sha256(features.tobytes()).hexdigest()
    checkpoint_digest = file_hash(options.checkpoint)
    record: dict[str, object] = {
        "label": options.label,
        "component": "PocketTTS initial-state flow-latent to Mimi waveform decoder",
        "scope": "Latent affine, quantizer projection, upsample, two-layer decoder transformer and complete SEANet waveform synthesis; excludes text, autoregressive backbone and flow solver",
        "initial_state": "Zero convolution history and transposed-convolution partials; empty causal attention history. Every invocation independently decodes one complete chunk.",
        "streaming": False,
        "checkpoint": str(options.checkpoint),
        "bundle": str(options.output),
        "activation_dtype": "fp32",
        "native_decoder_weight_dtype": "bf16",
        "native_decoder_weight_count": 44,
        "native_decoder_weight_bytes": 20610626,
        "additional_native_statistics": ["flow_lm.emb_mean", "flow_lm.emb_std"],
        "frames": options.frames,
        "threads": options.threads,
        "sample_rate": 24000,
        "waveform_samples": 1920 * options.frames,
        "represented_audio_seconds": options.frames / 12.5,
        "host": platform.platform(),
        "backend_registration": "scalar",
        "compilation_depth": "none",
        "input_description": "Seed 1337 Gaussian FP32 raw flow latents, standard deviation0.5",
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
    if samples:
        median = statistics.median(samples)
        record.update(
            timing_scope="Uninstrumented Runtime.run for independent initial-state waveform chunks, including input binding, executor, kernels and output copy; excludes loading and compilation",
            warmups=options.warmups,
            decoder_seconds=samples,
            decoder_median_seconds=median,
            decoder_real_time_factor=median / (options.frames / 12.5),
        )
    if options.torch_reference:
        reference = torch_reference(options, feature_path)
        record["torch_comparison"] = compare(
            encoded, np.asarray(np.load(reference), dtype=np.float32)
        )
        record["torch_reference_path"] = str(reference)
    if options.reference_bundle is not None:
        suffix = "-validation.json" if options.validate_only else "-results.json"
        metadata_bytes = options.reference_bundle.with_name(
            options.reference_bundle.name + suffix
        ).read_bytes()
        metadata = ReferenceGraph.model_validate_json(metadata_bytes)
        if (
            metadata.frames != options.frames
            or metadata.threads != options.threads
            or metadata.graph_sha256 != graph_digest
            or metadata.input_sha256 != input_digest
            or metadata.checkpoint_sha256 != checkpoint_digest
        ):
            raise ValueError("reference graph, input, weights or workload differs")
        reference = options.reference_bundle.with_name(
            options.reference_bundle.name + "-waveform.npy"
        )
        record["scalar_comparison"] = compare(
            encoded, np.asarray(np.load(reference), dtype=np.float32)
        )
        if samples:
            timing = ReferenceTiming.model_validate_json(metadata_bytes)
            record["speedup_over_reference"] = (
                timing.decoder_median_seconds / statistics.median(samples)
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frames", type=int, default=1)
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
