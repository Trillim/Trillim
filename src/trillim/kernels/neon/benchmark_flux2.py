"""Validate and time one complete FLUX.2 Klein denoiser forward.

Uses native checkpoint weights and deterministic packed latents, text embeddings,
timestep and four-axis positions. Excludes text encoding, VAE and sampling loops.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trillim.compiler.compile import _compile_model
from trillim.compiler.compiler_pydantic_models import _CompileRequest
from trillim.kernels.neon.benchmark_models import ManifestSymbols, instruction_calls
from trillim.kernels.neon.benchmark_qwen35 import ReferenceTensor
from trillim.runtime import Runtime

_RTOL = 2e-4
_ATOL = 2e-4
_STRICT_ATOL = 2e-5


class FluxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    attention_head_dim: int = Field(gt=0)
    axes_dims_rope: list[int]
    eps: float = Field(gt=0)
    guidance_embeds: Literal[False]
    in_channels: int = Field(gt=0)
    joint_attention_dim: int = Field(gt=0)
    mlp_ratio: float = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_single_layers: int = Field(gt=0)
    out_channels: int | None = None
    patch_size: Literal[1]
    rope_theta: int = Field(gt=0)
    timestep_guidance_channels: int = Field(gt=0)

    @model_validator(mode="after")
    def supported_architecture(self) -> Self:
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError("rotary axes must partition each attention head")
        if not self.axes_dims_rope or any(v <= 0 or v % 2 for v in self.axes_dims_rope):
            raise ValueError("rotary axis dimensions must be positive and even")
        if self.timestep_guidance_channels % 2:
            raise ValueError("timestep embedding width must be even")
        return self


class FluxOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    output: Path
    image_tokens: int = Field(default=4, ge=1)
    text_tokens: int = Field(default=2, ge=1)
    timestep: float = Field(default=0.5, ge=0, le=1)
    seed: int = Field(default=20261112, ge=0)
    threads: int = Field(default=20, ge=1)
    repeats: int = Field(default=3, ge=1)
    backend: Literal["scalar", "neon"] = "scalar"
    compile_only: bool = False
    run_reference: bool = False
    validate_only: bool = False
    validate_reference: Path | None = None

    @model_validator(mode="after")
    def square_image_grid(self) -> Self:
        if math.isqrt(self.image_tokens) ** 2 != self.image_tokens:
            raise ValueError("image tokens must form a square packed-latent grid")
        if (
            not self.output.expanduser()
            .resolve()
            .is_relative_to(Path.home() / ".trillim")
        ):
            raise ValueError("benchmark artifacts must stay beneath ~/.trillim")
        return self


class FluxReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    input_sha256: dict[str, str]
    image_tokens: int
    text_tokens: int
    timestep: float
    seed: int
    diffusers_version: str
    torch_version: str
    source_sha256: dict[str, str]
    architecture_url: str
    reference_dtype: Literal["fp32"] = "fp32"
    reference_attention: str = (
        "Diffusers default PyTorch CPU scaled_dot_product_attention"
    )
    scope: str = (
        "One complete denoiser forward; excludes text encoder, VAE and sampling"
    )
    rotary_precision: str = "Reference CPU phase FP64 then FP32 cos/sin; TDL phase FP32"


class FluxModel(Protocol):
    def eval(self) -> Self: ...

    def __call__(
        self,
        *,
        hidden_states: ReferenceTensor,
        encoder_hidden_states: ReferenceTensor,
        timestep: ReferenceTensor,
        img_ids: ReferenceTensor,
        txt_ids: ReferenceTensor,
        return_dict: Literal[False],
    ) -> tuple[ReferenceTensor, ...]: ...


class FluxFactory(Protocol):
    def from_pretrained(
        self, checkpoint: Path, *, torch_dtype: object, local_files_only: bool
    ) -> FluxModel: ...


class TorchAPI(Protocol):
    __version__: str
    float32: object

    def set_num_threads(self, count: int) -> None: ...

    def set_default_device(self, device: str) -> None: ...

    def from_numpy(self, value: NDArray[np.float32]) -> ReferenceTensor: ...

    def inference_mode(self) -> AbstractContextManager[object]: ...


def graph_source(config: FluxConfig, image_tokens: int, text_tokens: int) -> str:
    hidden = config.num_attention_heads * config.attention_head_dim
    mlp = int(hidden * config.mlp_ratio)
    return f"""from trillim.tdl import *

weights = Model("model")

def chunk(x, index, width):
    return Slice(x, axis=1, start=index * width, stop=(index + 1) * width, rank=2)

def modulate(x, shift, scale):
    return Add(Mul(LayerNorm(x, eps={config.eps!r}), Add(scale, 1.0)), shift)

def project_heads(x, name):
    return SplitHeads(Linear(x, weights[name]), heads={config.num_attention_heads}, head_dim={config.attention_head_dim})

def rotary(x, positions):
    return AxialRotary(x, positions, sections={tuple(config.axes_dims_rope)!r}, theta={float(config.rope_theta)!r}, interleaved=True)

def swiglu(x, width):
    return Mul(Silu(chunk(x, 0, width)), chunk(x, 1, width))

with Graph() as graph:
    image = Input(name="image", shape=({image_tokens}, {config.in_channels}), dtype="fp32")
    text = Input(name="text", shape=({text_tokens}, {config.joint_attention_dim}), dtype="fp32")
    timestep = Input(name="timestep", shape=(1,), dtype="fp32")
    image_positions = Input(name="image_positions", shape=({image_tokens}, {len(config.axes_dims_rope)}), dtype="fp32")
    text_positions = Input(name="text_positions", shape=({text_tokens}, {len(config.axes_dims_rope)}), dtype="fp32")
    time_features = Sinusoidal(Mul(timestep, 1000.0), dim={config.timestep_guidance_channels}, frequency_shift=0.0)
    time_features = Silu(Linear(time_features, weights["time_guidance_embed.timestep_embedder.linear_1.weight"]))
    temb = Linear(time_features, weights["time_guidance_embed.timestep_embedder.linear_2.weight"])
    activated_time = Silu(temb)
    img_mod = Linear(activated_time, weights["double_stream_modulation_img.linear.weight"])
    txt_mod = Linear(activated_time, weights["double_stream_modulation_txt.linear.weight"])
    single_mod = Linear(activated_time, weights["single_stream_modulation.linear.weight"])
    image = Linear(image, weights["x_embedder.weight"])
    text = Linear(text, weights["context_embedder.weight"])
    positions = Concat((text_positions, image_positions), axis=0, rank=2)
    for layer in range({config.num_layers}):
        prefix = f"transformer_blocks.{{layer}}."
        img_norm = modulate(image, chunk(img_mod, 0, {hidden}), chunk(img_mod, 1, {hidden}))
        txt_norm = modulate(text, chunk(txt_mod, 0, {hidden}), chunk(txt_mod, 1, {hidden}))
        iq = RMSNorm(project_heads(img_norm, prefix + "attn.to_q.weight"), weight=weights[prefix + "attn.norm_q.weight"], eps={config.eps!r})
        ik = RMSNorm(project_heads(img_norm, prefix + "attn.to_k.weight"), weight=weights[prefix + "attn.norm_k.weight"], eps={config.eps!r})
        iv = project_heads(img_norm, prefix + "attn.to_v.weight")
        tq = RMSNorm(project_heads(txt_norm, prefix + "attn.add_q_proj.weight"), weight=weights[prefix + "attn.norm_added_q.weight"], eps={config.eps!r})
        tk = RMSNorm(project_heads(txt_norm, prefix + "attn.add_k_proj.weight"), weight=weights[prefix + "attn.norm_added_k.weight"], eps={config.eps!r})
        tv = project_heads(txt_norm, prefix + "attn.add_v_proj.weight")
        query = rotary(Concat((tq, iq), axis=0, rank=3), positions)
        key = rotary(Concat((tk, ik), axis=0, rank=3), positions)
        value = Concat((tv, iv), axis=0, rank=3)
        attended = MergeHeads(Attention(query, key, value, causal=False))
        txt_attn = Linear(Slice(attended, axis=0, stop=Dim(text, 0), rank=2), weights[prefix + "attn.to_add_out.weight"])
        img_attn = Linear(Slice(attended, axis=0, start=Dim(text, 0), rank=2), weights[prefix + "attn.to_out.0.weight"])
        image = Add(image, Mul(chunk(img_mod, 2, {hidden}), img_attn))
        img_norm = modulate(image, chunk(img_mod, 3, {hidden}), chunk(img_mod, 4, {hidden}))
        img_ff = Linear(swiglu(Linear(img_norm, weights[prefix + "ff.linear_in.weight"]), {mlp}), weights[prefix + "ff.linear_out.weight"])
        image = Add(image, Mul(chunk(img_mod, 5, {hidden}), img_ff))
        text = Add(text, Mul(chunk(txt_mod, 2, {hidden}), txt_attn))
        txt_norm = modulate(text, chunk(txt_mod, 3, {hidden}), chunk(txt_mod, 4, {hidden}))
        txt_ff = Linear(swiglu(Linear(txt_norm, weights[prefix + "ff_context.linear_in.weight"]), {mlp}), weights[prefix + "ff_context.linear_out.weight"])
        text = Add(text, Mul(chunk(txt_mod, 5, {hidden}), txt_ff))
    text_length = Dim(text, 0)
    combined = Concat((text, image), axis=0, rank=2)
    for layer in range({config.num_single_layers}):
        prefix = f"single_transformer_blocks.{{layer}}.attn."
        normed = modulate(combined, chunk(single_mod, 0, {hidden}), chunk(single_mod, 1, {hidden}))
        projected = Linear(normed, weights[prefix + "to_qkv_mlp_proj.weight"])
        query = RMSNorm(SplitHeads(chunk(projected, 0, {hidden}), heads={config.num_attention_heads}, head_dim={config.attention_head_dim}), weight=weights[prefix + "norm_q.weight"], eps={config.eps!r})
        key = RMSNorm(SplitHeads(chunk(projected, 1, {hidden}), heads={config.num_attention_heads}, head_dim={config.attention_head_dim}), weight=weights[prefix + "norm_k.weight"], eps={config.eps!r})
        value = SplitHeads(chunk(projected, 2, {hidden}), heads={config.num_attention_heads}, head_dim={config.attention_head_dim})
        attended = MergeHeads(Attention(rotary(query, positions), rotary(key, positions), value, causal=False))
        feedforward = swiglu(Slice(projected, axis=1, start={3 * hidden}, rank=2), {mlp})
        mixed = Linear(Concat((attended, feedforward), axis=1, rank=2), weights[prefix + "to_out.weight"])
        combined = Add(combined, Mul(chunk(single_mod, 2, {hidden}), mixed))
    image = Slice(combined, axis=0, start=text_length, rank=2)
    output_mod = Linear(activated_time, weights["norm_out.linear.weight"])
    image = modulate(image, chunk(output_mod, 1, {hidden}), chunk(output_mod, 0, {hidden}))
    Output(Linear(image, weights["proj_out.weight"]), name="denoised")
"""


def make_inputs(
    options: FluxOptions, config: FluxConfig
) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(options.seed)
    axes = len(config.axes_dims_rope)
    if axes != 4:
        raise ValueError("deterministic input positions require four rotary axes")
    image_positions = np.zeros((options.image_tokens, axes), dtype=np.float32)
    side = math.isqrt(options.image_tokens)
    image_positions[:, 1] = np.arange(options.image_tokens) // side
    image_positions[:, 2] = np.arange(options.image_tokens) % side
    text_positions = np.zeros((options.text_tokens, axes), dtype=np.float32)
    text_positions[:, 3] = np.arange(options.text_tokens)
    return {
        "image": rng.normal(size=(options.image_tokens, config.in_channels)).astype(
            np.float32
        ),
        "text": rng.normal(
            size=(options.text_tokens, config.joint_attention_dim)
        ).astype(np.float32),
        "timestep": np.asarray([options.timestep], dtype=np.float32),
        "image_positions": image_positions,
        "text_positions": text_positions,
    }


def checkpoint_hash(checkpoint: Path) -> str:
    with (checkpoint / "diffusion_pytorch_model.safetensors").open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def input_hashes(inputs: dict[str, NDArray[np.float32]]) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in inputs.items()
    }


def compile_bundle(options: FluxOptions, config: FluxConfig) -> None:
    source_path = options.output.with_name(options.output.name + ".tdl")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source = graph_source(config, options.image_tokens, options.text_tokens)
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
            max_input_size=max(options.image_tokens, options.text_tokens),
            depth="none",
        ),
        progress=lambda message: print(message, flush=True),
    )


def run_reference(options: FluxOptions, config: FluxConfig) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path.home() / "Trillim/models/.hf-cache"))
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".trillim/flux2/torch-cache"))
    torch = cast(TorchAPI, importlib.import_module("torch"))
    diffusers = importlib.import_module("diffusers")
    factory = cast(FluxFactory, diffusers.Flux2Transformer2DModel)
    torch.set_num_threads(options.threads)
    torch.set_default_device("cpu")
    model = factory.from_pretrained(
        options.checkpoint, torch_dtype=torch.float32, local_files_only=True
    ).eval()
    inputs = make_inputs(options, config)
    with torch.inference_mode():
        result = model(
            hidden_states=torch.from_numpy(inputs["image"][None]),
            encoder_hidden_states=torch.from_numpy(inputs["text"][None]),
            timestep=torch.from_numpy(inputs["timestep"]),
            img_ids=torch.from_numpy(inputs["image_positions"]),
            txt_ids=torch.from_numpy(inputs["text_positions"]),
            return_dict=False,
        )
        expected = result[0][0].detach().float().cpu().numpy()
    if not np.isfinite(expected).all():
        raise ValueError("nonfinite Diffusers reference output")
    sources: dict[str, str] = {}
    for name in (
        "diffusers.models.transformers.transformer_flux2",
        "diffusers.models.normalization",
        "diffusers.models.embeddings",
    ):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"cannot identify reference source: {name}")
        sources[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    metadata = FluxReferenceMetadata(
        checkpoint=str(options.checkpoint),
        checkpoint_sha256=checkpoint_hash(options.checkpoint),
        config_sha256=hashlib.sha256(
            (options.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        input_sha256=input_hashes(inputs),
        image_tokens=options.image_tokens,
        text_tokens=options.text_tokens,
        timestep=options.timestep,
        seed=options.seed,
        diffusers_version=str(diffusers.__version__),
        torch_version=torch.__version__,
        source_sha256=sources,
        architecture_url=f"https://github.com/huggingface/diffusers/blob/v{diffusers.__version__}/src/diffusers/models/transformers/transformer_flux2.py",
    )
    path = options.output.with_name(options.output.name + "-diffusers.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        denoised=expected,
        image=inputs["image"],
        text=inputs["text"],
        timestep=inputs["timestep"],
        image_positions=inputs["image_positions"],
        text_positions=inputs["text_positions"],
    )
    path.with_suffix(".json").write_text(metadata.model_dump_json(indent=2) + "\n")
    print(metadata.model_dump_json(indent=2), flush=True)


def validate_reference(
    options: FluxOptions, inputs: dict[str, NDArray[np.float32]]
) -> tuple[dict[str, object], NDArray[np.float32]]:
    if options.validate_reference is None:
        raise ValueError("independent Diffusers reference required before timing")
    path = options.validate_reference
    metadata = FluxReferenceMetadata.model_validate_json(
        path.with_suffix(".json").read_bytes()
    )
    if (
        Path(metadata.checkpoint).resolve() != options.checkpoint.resolve()
        or metadata.checkpoint_sha256 != checkpoint_hash(options.checkpoint)
    ):
        raise ValueError("reference checkpoint weights differ")
    if (
        metadata.config_sha256
        != hashlib.sha256((options.checkpoint / "config.json").read_bytes()).hexdigest()
    ):
        raise ValueError("reference checkpoint configuration differs")
    if metadata.input_sha256 != input_hashes(inputs):
        raise ValueError("reference input values differ")
    with np.load(path) as archive:
        expected = np.asarray(archive["denoised"], dtype=np.float32)
    with Runtime(options.output, threads=options.threads) as runtime:
        actual = np.asarray(runtime.run(inputs)["denoised"], dtype=np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("denoiser output shape or finiteness differs")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rms_error = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    matches = bool(np.allclose(actual, expected, rtol=_RTOL, atol=_ATOL))
    np.save(
        options.output.with_name(options.output.name + "-validation-output.npy"), actual
    )
    return {
        "diffusers_reference": metadata.model_dump(),
        "reference_matches": matches,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rms_error": rms_error,
        "normalized_rms_error": rms_error / max(reference_rms, 1.0),
        "outside_tolerance": int(
            np.count_nonzero(np.abs(difference) > _ATOL + _RTOL * np.abs(expected))
        ),
        "outside_strict_tolerance": int(
            np.count_nonzero(
                np.abs(difference) > _STRICT_ATOL + _RTOL * np.abs(expected)
            )
        ),
        "reference_relative_tolerance": _RTOL,
        "reference_absolute_tolerance": _ATOL,
        "strict_reference_absolute_tolerance": _STRICT_ATOL,
    }, expected


def benchmark(
    options: FluxOptions,
    config: FluxConfig,
    inputs: dict[str, NDArray[np.float32]],
    expected: NDArray[np.float32],
) -> dict[str, object]:
    samples: list[float] = []
    hashes: list[str] = []
    with Runtime(options.output, threads=options.threads) as runtime:
        for _ in range(options.repeats):
            start = time.perf_counter()
            output = runtime.run(inputs)["denoised"]
            samples.append(time.perf_counter() - start)
            if not np.allclose(output, expected, rtol=_RTOL, atol=_ATOL):
                raise ValueError("timed denoiser output differs from the reference")
            hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
    manifest = ManifestSymbols.model_validate_json(
        (options.output / "executable.json").read_bytes()
    )
    counts = instruction_calls(manifest.program)
    return {
        "label": options.checkpoint.parent.name + "-denoiser",
        "checkpoint": str(options.checkpoint),
        "bundle": str(options.output),
        "model_type": "flux2",
        "scope": "One full denoiser forward; no text encoder, VAE, sampling loop or image generation",
        "host": platform.platform(),
        "numpy_version": np.__version__,
        "threads": options.threads,
        "image_tokens": options.image_tokens,
        "text_tokens": options.text_tokens,
        "timestep": options.timestep,
        "seed": options.seed,
        "dual_stream_blocks": config.num_layers,
        "single_stream_blocks": config.num_single_layers,
        "hidden_size": config.num_attention_heads * config.attention_head_dim,
        "activation_dtype": "fp32",
        "checkpoint_weight_dtype": "bf16",
        "repeats": options.repeats,
        "latency_seconds": samples,
        "median_seconds": statistics.median(samples),
        "forwards_per_second": 1.0 / statistics.median(samples),
        "output_sha256": hashes,
        "timing_scope": "Runtime.run input binding, Python dispatch, native kernels and output copy; excludes model/reference loading and compilation",
        "kernel_call_counts": dict(counts),
        "neon_kernel_call_count": sum(
            count for name, count in counts.items() if name.startswith("trillim_neon_")
        ),
        "kernels": sorted({k.symbol for k in manifest.kernels.values()}),
        "graph_sha256": hashlib.sha256(
            options.output.with_name(options.output.name + ".tdl").read_bytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-tokens", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=2)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=("scalar", "neon"), default="scalar")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-reference", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-reference", type=Path)
    options = FluxOptions.model_validate(vars(parser.parse_args()))
    config = FluxConfig.model_validate_json(
        (options.checkpoint / "config.json").read_bytes()
    )
    if options.run_reference:
        run_reference(options, config)
        return
    compile_bundle(options, config)
    if options.compile_only:
        return
    if options.backend != "scalar":
        raise ValueError("runtime currently requires scalar registration")
    inputs = make_inputs(options, config)
    results, expected = validate_reference(options, inputs)
    path = options.output.with_name(options.output.name + "-results.json")
    if results["reference_matches"] is not True:
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        raise ValueError("FLUX.2 denoiser output differs from Diffusers reference")
    if not options.validate_only:
        results.update(benchmark(options, config, inputs, expected))
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
