"""Matched eager PyTorch CPU controls for Trillim's component benchmarks.

Use saved Trillim input arrays and independent reference outputs. Model loading,
FP32 expansion of native weights and cache allocation are outside timed calls.
Every forward includes an independent FP32 NumPy output copy. No model bundles
or converted weight files are written. These are named official-module controls,
not a claim about the fastest possible PyTorch implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from trillim.kernels.neon.benchmark_torch_components_dataclasses import (
    Adapter,
    Archive,
    Arrays,
    EncoderModel,
    EncoderOutput,
    Factory,
    IncrementSteps,
    InitializeStates,
    MimiModel,
    Module,
    PretrainedFactory,
    SafeOpen,
    State,
    Tensor,
    TorchAPI,
)

_ROOT = Path.home() / ".trillim/cpu-compare-20261113"
type Component = Literal[
    "parakeet_encoder",
    "pocket_flow",
    "pocket_backbone",
    "pocket_mimi",
    "flux2_denoiser",
    "bonsai_image_denoiser",
]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    component: Component
    checkpoint: Path
    inputs: Path
    reference: Path
    output: Path
    threads: int = Field(default=20, ge=1)
    interop_threads: int = Field(default=1, ge=1)
    warmups: int = Field(default=2, ge=0)
    repeats: int | None = Field(default=None, ge=1)
    attention: Literal["sdpa", "eager"] = "sdpa"
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=4, ge=1)
    cache: int = Field(default=512, ge=1)
    rtol: float = Field(default=2e-4, ge=0)
    atol: float = Field(default=2e-4, ge=0)
    plan_only: bool = False
    validate_only: bool = False

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        self.output = self.output.expanduser().resolve()
        self.checkpoint = self.checkpoint.expanduser().resolve()
        self.inputs = self.inputs.expanduser().resolve()
        self.reference = self.reference.expanduser().resolve()
        if not self.output.is_relative_to(_ROOT):
            raise ValueError(f"all outputs must stay beneath {_ROOT}")
        if not self.checkpoint.exists():
            raise ValueError("native checkpoint is missing")
        if not self.inputs.is_file() or not self.reference.is_file():
            raise ValueError("saved matched inputs and reference outputs are required")
        if (
            self.component == "pocket_backbone"
            and self.prompt + self.decode > self.cache
        ):
            raise ValueError("cache must fit the matched prefill and decode inputs")
        return self


class TensorHeader(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]


class FluxShape(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    num_layers: Literal[5]
    num_single_layers: Literal[20]
    in_channels: int
    joint_attention_dim: int


class ReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    input_sha256: str | dict[str, str] | None = None
    config_sha256: str | None = None
    checkpoint_sha256: str | None = None


_HEADER = TypeAdapter(dict[str, TensorHeader | dict[str, str]])


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_environment() -> None:
    sys.dont_write_bytecode = True
    for name in (
        "tmp",
        "hf-cache",
        "torch-cache",
        "xdg-cache",
        "inductor-cache",
        "torch-extensions",
        "triton-cache",
    ):
        (_ROOT / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(_ROOT / "tmp"),
            "HF_HOME": str(_ROOT / "hf-cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_HOME": str(_ROOT / "torch-cache"),
            "XDG_CACHE_HOME": str(_ROOT / "xdg-cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(_ROOT / "inductor-cache"),
            "TORCH_EXTENSIONS_DIR": str(_ROOT / "torch-extensions"),
            "TRITON_CACHE_DIR": str(_ROOT / "triton-cache"),
        }
    )


def load_inputs(options: Options) -> Arrays:
    names: tuple[str, ...]
    if options.component == "pocket_flow":
        names = ("c", "s", "t", "x")
    elif options.component in ("flux2_denoiser", "bonsai_image_denoiser"):
        names = ("image", "text", "timestep", "image_positions", "text_positions")
    else:
        name = {
            "parakeet_encoder": "features",
            "pocket_backbone": "embedded",
            "pocket_mimi": "latent",
        }[options.component]
        array = np.load(options.inputs, allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise ValueError("matched inputs must be FP32 NumPy arrays")
        return {name: np.asarray(array, dtype=np.float32)}
    with np.load(options.inputs, allow_pickle=False) as archive:
        result = {}
        for name in names:
            array = archive[name]
            if array.dtype != np.float32:
                raise ValueError(f"matched input {name} must remain FP32")
            result[name] = np.asarray(array, dtype=np.float32)
    return result


def validate_shapes(options: Options, arrays: Arrays) -> None:
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("input contains nonfinite values")
    if options.component == "parakeet_encoder":
        x = arrays["features"]
        if x.ndim != 2 or x.shape[0] not in (1024, 3000):
            raise ValueError(
                "matched Parakeet rows use 1024 or 3000 unpadded mel frames"
            )
    elif options.component == "pocket_flow":
        batch = arrays["c"].shape[0]
        expected: dict[str, tuple[int, ...]] = {
            "c": (batch, 1024),
            "s": (batch, 1),
            "t": (batch, 1),
            "x": (batch, 32),
        }
        if batch not in (1, 32) or any(
            arrays[k].shape != v for k, v in expected.items()
        ):
            raise ValueError(
                "matched flow rows require batches 1 and 32 and native c/s/t/x dimensions"
            )
    elif options.component == "pocket_backbone":
        if arrays["embedded"].shape != (options.prompt + options.decode, 1024):
            raise ValueError(
                "embedded inputs must contain the exact prefill and decode vectors"
            )
    elif options.component == "pocket_mimi":
        x = arrays["latent"]
        if x.ndim != 2 or x.shape[0] not in (8, 32) or x.shape[1] != 32:
            raise ValueError(
                "matched initial-state Mimi rows use 8 or 32 raw latent frames"
            )
    else:
        shape = FluxShape.model_validate_json(
            (options.checkpoint / "config.json").read_bytes()
        )
        expected = {
            "image": (256, shape.in_channels),
            "text": (32, shape.joint_attention_dim),
            "timestep": (1,),
            "image_positions": (256, 4),
            "text_positions": (32, 4),
        }
        if any(arrays[k].shape != v for k, v in expected.items()):
            raise ValueError(
                "matched denoisers require all 25 blocks, 256 image tokens and 32 text tokens"
            )


def checkpoint_files(checkpoint: Path) -> tuple[Path, ...]:
    files = (
        (checkpoint,)
        if checkpoint.is_file()
        else tuple(sorted(checkpoint.glob("*.safetensors")))
    )
    if not files or any(path.suffix != ".safetensors" for path in files):
        raise ValueError("a native floating safetensors checkpoint is required")
    return files


def checkpoint_metadata(checkpoint: Path, *, payload_hashes: bool) -> dict[str, object]:
    records: dict[str, object] = {}
    dtypes: Counter[str] = Counter()
    for path in checkpoint_files(checkpoint):
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 0 < length <= 64 * 1024**2:
                raise ValueError("invalid safetensors header size")
            header = stream.read(length)
        tensors = _HEADER.validate_json(header)
        for tensor in tensors.values():
            if isinstance(tensor, TensorHeader):
                dtypes[tensor.dtype] += 1
                if tensor.dtype not in ("F32", "BF16", "F16", "I64", "I32", "BOOL"):
                    raise ValueError(
                        f"unsupported native checkpoint dtype: {tensor.dtype}"
                    )
        item: dict[str, object] = {
            "bytes": path.stat().st_size,
            "header_sha256": hashlib.sha256(header).hexdigest(),
        }
        if payload_hashes:
            item["sha256"] = file_hash(path)
        records[str(path)] = item
    return {
        "files": records,
        "native_tensor_dtypes": dict(dtypes),
        "full_payload_hashes": payload_hashes,
    }


def array(tensor: Tensor) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    return np.array(tensor.detach().float().cpu().numpy(), dtype=np.float32, copy=True)


def factory(module: str, name: str) -> Factory:
    return cast(Factory, getattr(importlib.import_module(module), name))


def load_prefix(archive: Archive, prefix: str, module: Module) -> int:
    names = archive.keys()
    state = {
        name.removeprefix(prefix): archive.get_tensor(name)
        for name in names
        if name.startswith(prefix)
    }
    if not state or any(
        str(value.dtype) != "torch.bfloat16" for value in state.values()
    ):
        raise ValueError("Pocket controls require the original native BF16 tensors")
    module.load_state_dict(state, strict=True)
    return len(state)


def stamp_pocket_state_names(model: Module) -> None:
    stamp = cast(
        Callable[[object], None],
        importlib.import_module("pocket_tts.models.tts_model").stamp_state_names,
    )
    stamp(SimpleNamespace(flow_lm=model, mimi=model))


def parakeet(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    transformer = importlib.import_module("transformers")
    cls = cast(PretrainedFactory, transformer.ParakeetForTDT)
    model = (
        cls.from_pretrained(
            options.checkpoint,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation=options.attention,
        )
        .float()
        .cpu()
        .eval()
    )
    encoder = cast(EncoderModel, model).encoder
    features = torch.from_numpy(arrays["features"]).unsqueeze(0)

    def forward() -> Arrays:
        result = cast(EncoderOutput, encoder(features, attention_mask=None))
        return {"encoder_hidden_states": array(result.last_hidden_state.squeeze(0))}

    return Adapter(
        forward,
        None,
        None,
        (model,),
        ("transformers.models.parakeet.modeling_parakeet",),
        f"Transformers ParakeetForTDT.encoder, {options.attention}",
        "Complete 24-layer Conformer encoder on one unpadded sequence, including learned relative-position attention bias",
        "Feature extraction, encoder projector, TDT predictor/joint and token-duration decoding",
        "No persistent inference state",
    )


def pocket_flow(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    model = (
        factory("pocket_tts.modules.mlp", "SimpleMLPAdaLN")(
            32, 512, 32, 1024, 6, num_time_conds=2
        )
        .float()
        .cpu()
        .eval()
    )
    opener = cast(SafeOpen, importlib.import_module("safetensors").safe_open)
    with opener(str(options.checkpoint), framework="pt", device="cpu") as archive:
        count = load_prefix(archive, "flow_lm.flow_net.", model)
    if count != 68:
        raise ValueError("flow head must load every one of its 68 native tensors")
    values = tuple(torch.from_numpy(arrays[name]) for name in ("c", "s", "t", "x"))

    def forward() -> Arrays:
        return {"velocity": array(cast(Tensor, model(*values)))}

    return Adapter(
        forward,
        None,
        None,
        (model,),
        ("pocket_tts.modules.mlp",),
        "Official PocketTTS SimpleMLPAdaLN forward",
        "Complete two-time-conditioned six-block flow head, including saved frequency values and correction 1 variance normalization",
        "Autoregressive backbone, flow solver, tokenization and waveform decoder",
        "No persistent inference state",
        count,
    )


def pocket_backbone(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    current = int(importlib.metadata.version("pocket-tts").split(".")[0]) >= 4
    transformer_module = (
        "pocket_tts.modules.transformer"
        if current
        else "pocket_tts.modules.mimi_transformer"
    )
    arguments: dict[str, object] = {
        "dim_feedforward": 4096,
        "max_period": 10000.0,
    }
    if not current:
        arguments["kind"] = "flow_lm"
    model = (
        factory(transformer_module, "StreamingTransformer")(1024, 16, 6, **arguments)
        .float()
        .cpu()
        .eval()
    )
    norm = factory("torch.nn", "LayerNorm")(1024, eps=1e-5).float().cpu().eval()
    opener = cast(SafeOpen, importlib.import_module("safetensors").safe_open)
    with opener(str(options.checkpoint), framework="pt", device="cpu") as archive:
        count = load_prefix(archive, "flow_lm.transformer.", model)
        count += load_prefix(archive, "flow_lm.out_norm.", norm)
    if count != 50:
        raise ValueError("backbone must load every one of its 50 native tensors")
    if current:
        stamp_pocket_state_names(model)
    state_module = importlib.import_module("pocket_tts.modules.stateful_module")
    initialize = cast(InitializeStates, state_module.init_states)
    increment = cast(IncrementSteps, state_module.increment_steps)
    values = torch.from_numpy(arrays["embedded"]).unsqueeze(0)
    state: State = {}

    def reset() -> None:
        nonlocal state
        state = initialize(model, batch_size=1, sequence_length=options.cache)

    def forward() -> Arrays:
        encoded = cast(Tensor, model(values[:, : options.prompt], state))
        result = array(cast(Tensor, norm(encoded)).squeeze(0))
        increment(model, state, increment=options.prompt)
        return {"prefill": result}

    def decode(step: int) -> Arrays:
        position = options.prompt + step
        encoded = cast(Tensor, model(values[:, position : position + 1], state))
        result = array(cast(Tensor, norm(encoded)).squeeze(0))
        increment(model, state, increment=1)
        return {"decode": result}

    return Adapter(
        forward,
        reset,
        decode,
        (model, norm),
        (
            transformer_module,
            "pocket_tts.modules.transformer",
            "pocket_tts.modules.rope",
            "pocket_tts.modules.stateful_module",
            "pocket_tts.models.tts_model",
        ),
        "Official PocketTTS flow backbone StreamingTransformer plus official LayerNorm boundary wrapper",
        "Complete six-layer embedded-input causal backbone plus out_norm, prefill followed by persistent-cache decode",
        "Token embedding, latent input projection, EOS head, flow head, solver and Mimi",
        "Fresh primary cache per trial, allocated outside timing; primary offset increments are inside timed calls",
        count,
    )


def pocket_mimi(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    current = int(importlib.metadata.version("pocket-tts").split(".")[0]) >= 4
    transformer_module = (
        "pocket_tts.modules.transformer"
        if current
        else "pocket_tts.modules.mimi_transformer"
    )
    seanet: dict[str, object] = {
        "channels": 1,
        "dimension": 512,
        "n_filters": 64,
        "n_residual_layers": 1,
        "ratios": [6, 5, 4],
        "kernel_size": 7,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "dilation_base": 2,
        "pad_mode": "constant",
        "compress": 2,
    }
    transformer: dict[str, object] = {
        "input_dimension": 512,
        "output_dimensions": (512,),
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 2,
        "layer_scale": 0.01,
        "context": 250,
        "max_period": 10000.0,
        "dim_feedforward": 2048,
    }
    projected = factory(transformer_module, "ProjectedTransformer")
    arguments: dict[str, object] = {}
    if current:
        arguments.update(inner_dim=None, outer_dim=None)
    model = (
        factory("pocket_tts.models.mimi", "MimiModel")(
            encoder=factory("pocket_tts.modules.seanet", "SEANetEncoder")(**seanet),
            decoder=factory("pocket_tts.modules.seanet", "SEANetDecoder")(**seanet),
            quantizer=factory("pocket_tts.modules.dummy_quantizer", "DummyQuantizer")(
                32, 512
            ),
            frame_rate=12.5,
            encoder_frame_rate=200.0,
            sample_rate=24000,
            channels=1,
            encoder_transformer=projected(**transformer),
            decoder_transformer=projected(**transformer),
            **arguments,
        )
        .float()
        .cpu()
        .eval()
    )
    opener = cast(SafeOpen, importlib.import_module("safetensors").safe_open)
    with opener(str(options.checkpoint), framework="pt", device="cpu") as archive:
        count = sum(
            load_prefix(
                archive, "mimi." + name + ".", cast(Module, getattr(model, name))
            )
            for name in ("quantizer", "upsample", "decoder_transformer", "decoder")
        )
        mean = archive.get_tensor("flow_lm.emb_mean").float()
        std = archive.get_tensor("flow_lm.emb_std").float()
    if count != 44:
        raise ValueError("Mimi must strictly load all 44 executed decoder tensors")
    if current:
        stamp_pocket_state_names(model)
    initialize = cast(
        InitializeStates,
        importlib.import_module("pocket_tts.modules.stateful_module").init_states,
    )
    values = torch.from_numpy(arrays["latent"]).unsqueeze(0)
    mimi = cast(MimiModel, model)
    state: State = {}

    def reset() -> None:
        nonlocal state
        state = initialize(model, batch_size=1, sequence_length=1000)

    def forward() -> Arrays:
        latent = values * std + mean
        if current:
            result = mimi.decode_from_latent(latent, state)
        else:
            projected_latent = cast(Tensor, mimi.quantizer(latent.transpose(1, 2)))
            result = mimi.decode_from_latent(projected_latent, state)
        return {"waveform": array(result.squeeze(0))}

    return Adapter(
        forward,
        reset,
        None,
        (model,),
        (
            "pocket_tts.models.mimi",
            "pocket_tts.models.tts_model",
            "pocket_tts.modules.seanet",
            transformer_module,
            "pocket_tts.modules.dummy_quantizer",
            "pocket_tts.modules.resample",
            "pocket_tts.modules.conv",
            "pocket_tts.modules.stateful_module",
            "pocket_tts.modules.rope",
            "pocket_tts.modules.layer_scale",
        ),
        "Official PocketTTS MimiModel.decode_from_latent with matched latent affine/projection wrapper",
        "Complete raw 32-channel latent to initial-state waveform component, including both transformer layers and SEANet",
        "Text, autoregressive backbone, flow solver and streaming across chunks",
        "Zero history, convolution partials and empty attention cache recreated outside timing for each independent chunk",
        count + 2,
    )


def flux(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    cls = cast(
        PretrainedFactory, importlib.import_module("diffusers").Flux2Transformer2DModel
    )
    model = (
        cls.from_pretrained(
            options.checkpoint, torch_dtype=torch.float32, local_files_only=True
        )
        .float()
        .cpu()
        .eval()
    )
    values = {name: torch.from_numpy(value) for name, value in arrays.items()}

    def forward() -> Arrays:
        result = cast(
            tuple[Tensor, ...],
            model(
                hidden_states=values["image"].unsqueeze(0),
                encoder_hidden_states=values["text"].unsqueeze(0),
                timestep=values["timestep"],
                img_ids=values["image_positions"],
                txt_ids=values["text_positions"],
                return_dict=False,
            ),
        )
        return {"denoised": array(result[0].squeeze(0))}

    return Adapter(
        forward,
        None,
        None,
        (model,),
        (
            "diffusers.models.transformers.transformer_flux2",
            "diffusers.models.normalization",
            "diffusers.models.embeddings",
        ),
        "Official Diffusers Flux2Transformer2DModel, default CPU scaled_dot_product_attention",
        "Complete 25-block denoiser forward with 256 packed image tokens and 32 text embeddings",
        "Text encoder, VAE and diffusion sampling loop",
        "No persistent inference state",
    )


def load_adapter(options: Options, arrays: Arrays, torch: TorchAPI) -> Adapter:
    if options.component == "parakeet_encoder":
        return parakeet(options, arrays, torch)
    if options.component == "pocket_flow":
        return pocket_flow(options, arrays, torch)
    if options.component == "pocket_backbone":
        return pocket_backbone(options, arrays, torch)
    if options.component == "pocket_mimi":
        return pocket_mimi(options, arrays, torch)
    return flux(options, arrays, torch)


def load_reference(options: Options) -> Arrays:
    names: tuple[str, ...]
    if options.component == "pocket_backbone":
        names = ("prefill", "decode")
    elif options.component in ("flux2_denoiser", "bonsai_image_denoiser"):
        names = ("denoised",)
    else:
        name = {
            "parakeet_encoder": "encoder_hidden_states",
            "pocket_flow": "velocity",
            "pocket_mimi": "waveform",
        }[options.component]
        return {
            name: np.asarray(
                np.load(options.reference, allow_pickle=False), dtype=np.float32
            )
        }
    with np.load(options.reference, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float32) for name in names}


def compare(
    actual: Arrays, expected: Arrays, options: Options
) -> dict[str, dict[str, object]]:
    if actual.keys() != expected.keys():
        raise ValueError("reference output names differ")
    result: dict[str, dict[str, object]] = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or not np.isfinite(value).all():
            raise ValueError(
                f"reference shape or finite output check failed for {name}"
            )
        difference = value.astype(np.float64) - reference.astype(np.float64)
        matches = bool(
            np.allclose(value, reference, rtol=options.rtol, atol=options.atol)
        )
        result[name] = {
            "shape": list(value.shape),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(difference * difference))),
            "allclose": matches,
            "rtol": options.rtol,
            "atol": options.atol,
        }
    return result


def trial(adapter: Adapter, options: Options) -> tuple[dict[str, list[float]], Arrays]:
    if adapter.reset is not None:
        adapter.reset()
    timings: dict[str, list[float]] = {}
    started = time.perf_counter() if not options.validate_only else 0.0
    outputs = adapter.forward()
    if not options.validate_only:
        phase = "prefill" if adapter.decode is not None else "forward"
        timings[phase] = [time.perf_counter() - started]
    if adapter.decode is not None:
        steps = []
        for step in range(options.decode):
            started = time.perf_counter() if not options.validate_only else 0.0
            decoded = adapter.decode(step)
            if not options.validate_only:
                timings.setdefault("decode", []).append(time.perf_counter() - started)
            steps.append(decoded["decode"])
        outputs["decode"] = np.concatenate(steps, axis=0)
    return timings, outputs


def run(options: Options, arrays: Arrays) -> dict[str, object]:
    torch = cast(TorchAPI, importlib.import_module("torch"))
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(options.interop_threads)
    torch.set_default_device("cpu")
    torch.set_float32_matmul_precision("highest")
    adapter = load_adapter(options, arrays, torch)
    for module in adapter.modules:
        for name, parameter in module.named_parameters():
            if (
                parameter.device.type != "cpu"
                or str(parameter.dtype) != "torch.float32"
            ):
                raise ValueError(f"expected FP32 CPU parameters: {name}")
    sources = {}
    for name in adapter.source_modules:
        module_path = importlib.import_module(name).__file__
        if module_path is None:
            raise ValueError(f"official module has no source path: {name}")
        sources[name] = file_hash(Path(module_path))
    expected = load_reference(options)
    big = options.component in (
        "parakeet_encoder",
        "flux2_denoiser",
        "bonsai_image_denoiser",
    )
    repeats = options.repeats if options.repeats is not None else 3 if big else 9
    iterations = 1 if options.validate_only else options.warmups + repeats
    warmups: dict[str, list[float]] = {}
    samples: dict[str, list[float]] = {}
    output_hashes: list[dict[str, str]] = []
    outputs: Arrays = {}
    checks: dict[str, dict[str, object]] = {}
    fidelity_pass = True
    with torch.inference_mode():
        for index in range(iterations):
            times, outputs = trial(adapter, options)
            checks = compare(outputs, expected, options)
            fidelity_pass &= all(value["allclose"] is True for value in checks.values())
            target = warmups if index < options.warmups else samples
            for phase, values in times.items():
                target.setdefault(phase, []).extend(values)
            output_hashes.append(
                {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in outputs.items()
                }
            )
            print(f"{options.component} trial {index + 1} complete", flush=True)
    output_path = options.output.with_name(options.output.name + "-outputs.npz")
    if "prefill" in outputs:
        np.savez(output_path, prefill=outputs["prefill"], decode=outputs["decode"])
    elif "encoder_hidden_states" in outputs:
        np.savez(output_path, encoder_hidden_states=outputs["encoder_hidden_states"])
    elif "velocity" in outputs:
        np.savez(output_path, velocity=outputs["velocity"])
    elif "waveform" in outputs:
        np.savez(output_path, waveform=outputs["waveform"])
    else:
        np.savez(output_path, denoised=outputs["denoised"])
    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy")}
    for name in ("transformers", "diffusers", "pocket-tts"):
        if any(name.replace("-", "_") in source for source in adapter.source_modules):
            packages[name] = importlib.metadata.version(name)
    return {
        "status": (
            "fidelity_warning"
            if not fidelity_pass
            else "validated" if options.validate_only else "measured"
        ),
        "reference_allclose": fidelity_pass,
        "component": options.component,
        "implementation": adapter.implementation,
        "performance_claim": "Named eager official-module CPU control; not a fastest-possible PyTorch claim",
        "scope": adapter.scope,
        "exclusions": adapter.exclusions,
        "timing_scope": "Preloaded FP32 CPU model with fixed prepared tensor inputs, inference_mode, and independent FP32 NumPy output copies. Excludes loading, weight conversion and reference checks.",
        "cache_reset_scope": adapter.reset_scope,
        "activation_dtype": "fp32",
        "weight_conversion": "Exact FP32 expansion of native floating checkpoint values; no converted checkpoint files",
        "executed_native_tensor_count": adapter.native_tensor_count,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_trials": 0 if options.validate_only else options.warmups,
        "measured_trials": 0 if options.validate_only else repeats,
        "warmup_seconds": warmups,
        "samples_seconds": samples,
        "median_seconds": {
            phase: statistics.median(values) for phase, values in samples.items()
        },
        "reference_comparison": checks,
        "output_sha256": output_hashes,
        "source_sha256": sources,
        "versions": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=(
            "parakeet_encoder",
            "pocket_flow",
            "pocket_backbone",
            "pocket_mimi",
            "flux2_denoiser",
            "bonsai_image_denoiser",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--decode", type=int, default=4)
    parser.add_argument("--cache", type=int, default=512)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    options = Options.model_validate(vars(parser.parse_args()))
    configure_environment()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = load_inputs(options)
    validate_shapes(options, arrays)
    inputs_before = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    metadata_path = options.reference.with_suffix(".json")
    reference_metadata = (
        ReferenceMetadata.model_validate_json(metadata_path.read_bytes())
        if metadata_path.exists()
        else ReferenceMetadata()
    )
    expected_inputs = reference_metadata.input_sha256
    if isinstance(expected_inputs, dict) and expected_inputs != inputs_before:
        raise ValueError("cached reference input hashes differ")
    if isinstance(expected_inputs, str) and (
        len(inputs_before) != 1 or expected_inputs != next(iter(inputs_before.values()))
    ):
        raise ValueError("cached reference input hash differs")
    if reference_metadata.config_sha256 is not None and (
        reference_metadata.config_sha256
        != file_hash(options.checkpoint / "config.json")
    ):
        raise ValueError("cached reference checkpoint config differs")
    result: dict[str, object] = {
        "component": options.component,
        "checkpoint": str(options.checkpoint),
        "inputs": str(options.inputs),
        "reference": str(options.reference),
        "input_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "input_sha256": inputs_before,
        "input_file_sha256": file_hash(options.inputs),
        "input_combined_sha256": hashlib.sha256(
            b"".join(value.tobytes() for value in arrays.values())
        ).hexdigest(),
        "reference_file_sha256": file_hash(options.reference),
        "cached_reference_metadata": reference_metadata.model_dump(mode="json"),
        "runner_sha256": file_hash(Path(__file__)),
        "interface_sha256": file_hash(
            Path(__file__).with_name("benchmark_torch_components_dataclasses.py")
        ),
        "host": platform.platform(),
        "options": options.model_dump(mode="json"),
    }
    if options.plan_only:
        result["status"] = "plan_only_no_model_loaded"
    else:
        result.update(run(options, arrays))
    if inputs_before != {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in arrays.items()
    }:
        raise ValueError("the control mutated matched caller inputs")
    fingerprint = checkpoint_metadata(
        options.checkpoint, payload_hashes=not options.plan_only
    )
    expected_checkpoint = reference_metadata.checkpoint_sha256
    if expected_checkpoint is not None and not options.plan_only:
        files = cast(dict[str, dict[str, object]], fingerprint["files"])
        if (
            len(files) != 1
            or next(iter(files.values())).get("sha256") != expected_checkpoint
        ):
            raise ValueError("cached reference checkpoint payload hash differs")
    result["checkpoint_metadata"] = fingerprint
    suffix = (
        "-plan.json"
        if options.plan_only
        else "-validation.json"
        if options.validate_only
        else "-results.json"
    )
    options.output.with_name(options.output.name + suffix).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
