"""Benchmark local Transformers decoders on CPU with a fixed teacher-token stream.

Run schema/discover without Torch installed. Model loading and token preparation
are outside timing. Every timed forward includes an owned FP32 CPU logits copy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    model_type: str
    vocab_size: int | None = None
    architectures: list[str] = Field(default_factory=list)
    text_config: CheckpointConfig | None = None
    quantization_config: dict[str, object] | None = None
    is_bitnet_config: bool = False


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checkpoint: Path
    tokens_file: Path
    output: Path
    prompt: int = Field(default=16, ge=1)
    decode: int = Field(default=128, ge=1)
    cache: int = Field(default=512, ge=1)
    threads: int = Field(default=20, ge=1)
    interop_threads: int = Field(default=1, ge=1)
    warmups: int = Field(default=2, ge=0)
    repeats: int = Field(default=3, ge=1)
    attention: Literal["auto", "sdpa", "eager"] = "auto"
    cache_implementation: Literal["dynamic", "static"] = "dynamic"
    save_full_logits: bool = False
    reference_logits: Path | None = None
    atol: float = Field(default=2e-4, ge=0)
    rtol: float = Field(default=2e-5, ge=0)

    @model_validator(mode="after")
    def check_capacity(self) -> Self:
        if self.prompt + self.decode > self.cache:
            raise ValueError("cache must fit the prompt and every decode input")
        if (
            not self.output.expanduser()
            .resolve()
            .is_relative_to(Path.home() / ".trillim")
        ):
            raise ValueError("output must be beneath ~/.trillim")
        return self


class Device(Protocol):
    type: str


class Tensor(Protocol):
    dtype: object
    device: Device

    def __getitem__(self, key: object) -> Tensor: ...

    def is_floating_point(self) -> bool: ...

    def detach(self) -> Tensor: ...

    def float(self) -> Tensor: ...

    def cpu(self) -> Tensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


class ModelOutput(Protocol):
    logits: Tensor
    past_key_values: object


class Model(Protocol):
    config: object

    def eval(self) -> Self: ...

    def float(self) -> Self: ...

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]: ...

    def named_buffers(self) -> Iterator[tuple[str, Tensor]]: ...

    def __call__(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: object | None,
        use_cache: bool,
        logits_to_keep: int,
        return_dict: bool,
    ) -> ModelOutput: ...


class ModelFactory(Protocol):
    def from_pretrained(self, checkpoint: Path, **kwargs: object) -> Model: ...


class StaticCacheFactory(Protocol):
    def __call__(self, *, config: object, max_cache_len: int) -> object: ...


class Mxfp4Factory(Protocol):
    def __call__(self, *, dequantize: bool) -> object: ...


class TorchAPI(Protocol):
    __version__: str
    float32: object
    bfloat16: object
    long: object

    def set_num_threads(self, count: int) -> None: ...

    def set_num_interop_threads(self, count: int) -> None: ...

    def set_default_device(self, device: str) -> None: ...

    def set_float32_matmul_precision(self, precision: str) -> None: ...

    def get_num_threads(self) -> int: ...

    def get_num_interop_threads(self) -> int: ...

    def tensor(self, data: list[list[int]], *, dtype: object) -> Tensor: ...

    def inference_mode(self) -> AbstractContextManager[object]: ...


_FACTORIES = {
    "qwen3": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "llama": "AutoModelForCausalLM",
    "bitnet": "AutoModelForCausalLM",
    "gpt_oss": "AutoModelForCausalLM",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_text": "Gemma4ForCausalLM",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def read_config(checkpoint: Path) -> CheckpointConfig:
    return CheckpointConfig.model_validate_json(
        (checkpoint / "config.json").read_bytes()
    )


def checkpoint_identity(checkpoint: Path) -> list[dict[str, object]]:
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise ValueError("this benchmark requires native local safetensors weights")
    unique_files: dict[tuple[int, int], Path] = {}
    for path in files:
        stat = path.stat()
        unique_files[(stat.st_dev, stat.st_ino)] = path
    if sum(path.stat().st_size for path in unique_files.values()) > 32_000_000_000:
        raise ValueError("checkpoint exceeds the 32 GB model-size limit")
    identities: list[dict[str, object]] = []
    for path in unique_files.values():
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"checkpoint file changed while hashing: {path}")
        identities.append({"path": str(path), "bytes": after.st_size, "sha256": digest})
    return identities


def support(config: CheckpointConfig) -> tuple[bool, str]:
    if config.model_type not in _FACTORIES:
        return False, "No matched text-decoder adapter for this model_type"
    quantization = config.quantization_config
    if quantization is not None:
        method = quantization.get("quant_method")
        if method not in {"mxfp4", "bitnet"}:
            return False, f"Unsupported checkpoint quantization: {method!r}"
        if method == "mxfp4" and config.model_type != "gpt_oss":
            return False, "MXFP4 adapter is limited to GPT-oss"
        if method == "bitnet" and (
            quantization.get("linear_class") not in {"bitlinear", "autobitlinear"}
            or quantization.get("quantization_mode") not in {"offline", "online"}
        ):
            return (
                False,
                "BitNet metadata must specify linear_class and quantization_mode; use the documented normalized Llama checkpoint",
            )
    elif config.is_bitnet_config:
        return False, "BitNet checkpoint is missing explicit quantization metadata"
    return True, "FP32 CPU decoder; native floating values expanded exactly in RAM"


def discover(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.expanduser().rglob("config.json")):
        try:
            config = read_config(path.parent)
            supported, reason = support(config)
            rows.append(
                {
                    "checkpoint": str(path.parent),
                    "model_type": config.model_type,
                    "factory": _FACTORIES.get(config.model_type),
                    "supported_adapter": supported,
                    "status": "not_execution_validated",
                    "reason": reason,
                    "quantization": config.quantization_config,
                    "config_sha256": sha256(path),
                }
            )
        except (ValueError, OSError) as error:
            rows.append({"checkpoint": str(path.parent), "error": str(error)})
    return rows


def read_tokens(path: Path, count: int, vocab_size: int | None) -> list[int]:
    content = path.read_text().strip()
    if content.startswith("["):
        tokens = TypeAdapter(list[int]).validate_json(content, strict=True)
    else:
        tokens = [int(token) for token in content.split()]
    if len(tokens) < count:
        raise ValueError(f"teacher stream has {len(tokens)} tokens; need {count}")
    selected = tokens[:count]
    if any(
        token < 0 or vocab_size is not None and token >= vocab_size
        for token in selected
    ):
        raise ValueError(
            "teacher stream has token IDs outside the checkpoint vocabulary"
        )
    return selected


def audit_model(model: Model, torch: TorchAPI) -> dict[str, object]:
    dtypes: Counter[str] = Counter()
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} is on {tensor.device.type}; CPU is required")
        dtypes[str(tensor.dtype)] += 1
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"{name} has {tensor.dtype}; FP32 execution is required")
    return {"parameter_and_buffer_dtype_counts": dict(dtypes), "all_devices_cpu": True}


def audit_cache(past: object, torch: TorchAPI) -> dict[str, object]:
    layers = cast(list[object], getattr(past, "layers", []))
    dtypes: Counter[str] = Counter()
    for layer in layers:
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value: object = getattr(layer, name, None)
            values = list(value.values()) if isinstance(value, dict) else [value]
            for candidate in values:
                if candidate is None:
                    continue
                tensor = cast(Tensor, candidate)
                if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ValueError(f"cache {name} must be FP32 on CPU")
                dtypes[str(tensor.dtype)] += 1
    if not dtypes:
        raise ValueError("cache tensor audit found no KV or recurrent-state tensors")
    return {"layer_count": len(layers), "tensor_dtype_counts": dict(dtypes)}


def attention_policy(
    options: Options, config: CheckpointConfig
) -> str | dict[str, str]:
    requested = options.attention
    if config.model_type == "gpt_oss":
        if requested == "sdpa":
            raise ValueError(
                "GPT-oss attention sinks require eager CPU attention in this Transformers version"
            )
        return "eager"
    selected = "sdpa" if requested == "auto" else requested
    if config.model_type == "gemma4":
        return {
            "text_config": selected,
            "vision_config": "eager",
            "audio_config": "eager",
        }
    return selected


def numerical_comparison(
    actual: NDArray[np.float32], reference_path: Path, options: Options
) -> dict[str, object]:
    reference = np.load(reference_path, allow_pickle=False)
    if not isinstance(reference, np.ndarray):
        reference.close()
        raise TypeError("reference logits must be an uncompressed .npy array")
    if reference.ndim == 3 and reference.shape[1] == 1:
        reference = reference[:, 0, :]
    if reference.ndim != 2 or reference.shape[1] != actual.shape[1]:
        raise ValueError("reference logits require shape [steps, vocabulary]")
    count = min(len(actual), len(reference))
    if count == 0:
        raise ValueError("reference logits are empty")
    rows: list[dict[str, object]] = []
    for step in range(count):
        expected = reference[step].astype(np.float64)
        difference = actual[step].astype(np.float64) - expected
        within = np.abs(difference) <= options.atol + options.rtol * np.abs(expected)
        rows.append(
            {
                "step": step,
                "finite": bool(np.isfinite(expected).all()),
                "max_absolute_error": float(np.max(np.abs(difference))),
                "rms_error": float(np.sqrt(np.mean(difference * difference))),
                "outside_tolerance": int(np.count_nonzero(~within)),
                "allclose": bool(within.all()),
                "greedy_match": bool(np.argmax(actual[step]) == np.argmax(expected)),
            }
        )
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "checked_steps": count,
        "atol": options.atol,
        "rtol": options.rtol,
        "comparisons": rows,
        "allclose": all(row["allclose"] is True for row in rows),
        "greedy_match": all(row["greedy_match"] is True for row in rows),
    }


def benchmark(options: Options) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path.home() / "Trillim/models/.hf-cache"))
    os.environ.setdefault(
        "TORCH_HOME", str(Path.home() / ".trillim/cpu-compare-20261113/torch-cache")
    )
    checkpoint = options.checkpoint.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = read_config(checkpoint)
    supported, reason = support(config)
    if not supported:
        raise ValueError(reason)
    text_config = config.text_config or config
    tokens = read_tokens(
        options.tokens_file, options.prompt + options.decode, text_config.vocab_size
    )
    weights = checkpoint_identity(checkpoint)
    torch = cast(TorchAPI, importlib.import_module("torch"))
    transformers = importlib.import_module("transformers")
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(options.interop_threads)
    torch.set_default_device("cpu")
    torch.set_float32_matmul_precision("highest")
    factory = cast(ModelFactory, getattr(transformers, _FACTORIES[config.model_type]))
    load_kwargs: dict[str, object] = {
        "dtype": torch.float32,
        "attn_implementation": attention_policy(options, config),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if config.model_type == "gpt_oss":
        # The CPU MXFP4 loader expands exact FP4 values through BF16 before FP32.
        load_kwargs["dtype"] = torch.bfloat16
        mxfp4 = cast(Mxfp4Factory, transformers.Mxfp4Config)
        load_kwargs["quantization_config"] = mxfp4(dequantize=True)
    model = factory.from_pretrained(checkpoint, **load_kwargs).float().eval()
    dtype_audit = audit_model(model, torch)
    architecture = importlib.import_module(type(model).__module__)
    architecture_path = Path(architecture.__file__) if architecture.__file__ else None
    architecture_sha256 = sha256(architecture_path) if architecture_path else None
    inputs = [torch.tensor([tokens[: options.prompt]], dtype=torch.long)]
    positions = [torch.tensor([list(range(options.prompt))], dtype=torch.long)]
    for step in range(options.decode):
        position = options.prompt + step
        inputs.append(torch.tensor([[tokens[position]]], dtype=torch.long))
        positions.append(torch.tensor([[position]], dtype=torch.long))
    static_cache = cast(StaticCacheFactory, transformers.StaticCache)
    sessions: list[dict[str, object]] = []
    saved_logits: list[NDArray[np.float32]] = []
    cache_types: set[str] = set()
    all_ids: list[list[int]] = []
    cache_audits: list[dict[str, object]] = []
    prefill_times: list[float] = []
    measured_decode_times: list[float] = []
    with torch.inference_mode():
        for trial in range(options.warmups + options.repeats):
            past: object | None = None
            if options.cache_implementation == "static":
                past = static_cache(config=model.config, max_cache_len=options.cache)
            latencies: list[float] = []
            predictions: list[int] = []
            for step, (input_ids, position_ids) in enumerate(
                zip(inputs, positions, strict=True)
            ):
                started = time.perf_counter_ns()
                output = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                last = output.logits[0, -1, :].detach().float().cpu().numpy().copy()
                elapsed = (time.perf_counter_ns() - started) * 1e-9
                past = output.past_key_values
                if past is None:
                    raise ValueError("model did not return a decoder cache")
                cache_types.add(type(past).__name__)
                if not np.isfinite(last).all():
                    raise ValueError(f"nonfinite logits in trial {trial}, step {step}")
                latencies.append(elapsed)
                predictions.append(int(np.argmax(last)))
                if trial == options.warmups and (step < 5 or options.save_full_logits):
                    saved_logits.append(last)
                del output
            if trial >= options.warmups:
                cache_audits.append(audit_cache(past, torch))
                decode_times = latencies[1:]
                prefill_times.append(latencies[0])
                measured_decode_times.extend(decode_times)
                all_ids.append(predictions)
                sessions.append(
                    {
                        "session": trial - options.warmups,
                        "prefill_seconds": latencies[0],
                        "decode_seconds": decode_times,
                        "decode_tokens_per_second_median": 1
                        / statistics.median(decode_times),
                        "decode_tokens_per_second_aggregate": len(decode_times)
                        / sum(decode_times),
                        "predicted_token_ids": predictions,
                    }
                )
            print(
                f"{'warmup' if trial < options.warmups else 'measured'} session {trial} complete",
                flush=True,
            )
            del past
    logits = np.stack(saved_logits)
    logits_path = artifact(output_path, "-logits.npy")
    np.save(logits_path, logits)
    validation: dict[str, object] = {
        "logits_path": str(logits_path),
        "logits_shape": list(logits.shape),
        "logits_sha256": sha256(logits_path),
        "steps_saved": "all" if options.save_full_logits else "first five calls",
        "teacher_forced": True,
        "predicted_token_ids": all_ids,
        "predictions_identical_between_sessions": all(
            ids == all_ids[0] for ids in all_ids
        ),
        "independent_numerical_reference": "not provided",
    }
    if options.reference_logits is not None:
        validation["independent_numerical_reference"] = numerical_comparison(
            logits, options.reference_logits, options
        )
    validation_path = artifact(output_path, "-validation.json")
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    result: dict[str, object] = {
        "engine": "PyTorch Transformers CPU",
        "options": options.model_dump(mode="json"),
        "checkpoint": str(checkpoint),
        "config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_weights": weights,
        "model_type": config.model_type,
        "model_class": type(model).__name__,
        "architecture_source": str(architecture_path) if architecture_path else None,
        "architecture_sha256": architecture_sha256,
        "benchmark_source_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": str(transformers.__version__),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "attention_requested": options.attention,
        "attention_selected": getattr(
            getattr(model.config, "text_config", model.config),
            "_attn_implementation",
            None,
        ),
        "cache_implementation": options.cache_implementation,
        "cache_types": sorted(cache_types),
        "cache_audits": cache_audits,
        "cache_capacity_semantics": "static allocation bound"
        if options.cache_implementation == "static"
        else "logical bound; dynamic caches grow with context",
        "dtype_audit": dtype_audit,
        "activation_and_cache_dtype": "float32",
        "weight_representation": "native checkpoint values expanded to FP32 in RAM; no converted checkpoint saved",
        "teacher_file_sha256": sha256(options.tokens_file),
        "teacher_token_ids": tokens,
        "teacher_token_ids_sha256": hashlib.sha256(
            np.asarray(tokens, dtype="<i8").tobytes()
        ).hexdigest(),
        "vocabulary_size": int(logits.shape[1]),
        "vocabulary_padding": "none added by benchmark; full model output vocabulary retained",
        "warmup_sessions": options.warmups,
        "measured_sessions": options.repeats,
        "prefill_sample_count": options.repeats,
        "decode_sample_count": options.repeats * options.decode,
        "prefill_seconds_median": statistics.median(prefill_times),
        "prefill_tokens_per_second_median": options.prompt
        / statistics.median(prefill_times),
        "decode_tokens_per_second_median": 1 / statistics.median(measured_decode_times),
        "decode_tokens_per_second_aggregate": len(measured_decode_times)
        / sum(measured_decode_times),
        "timing_scope": "model forward, last-token logits selection, and owned FP32 CPU logits copy; CPU is synchronous",
        "excluded_from_timing": [
            "loading",
            "tokenization",
            "input tensor construction",
            "sampling/argmax",
            "validation",
            "file writes",
            "explicit static cache construction",
        ],
        "sessions": sessions,
        "validation_path": str(validation_path),
    }
    result_path = artifact(output_path, "-results.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema")
    discovery = commands.add_parser("discover")
    discovery.add_argument(
        "--models-root", type=Path, default=Path.home() / "Trillim/models"
    )
    discovery.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--tokens-file", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    for name, default in (
        ("prompt", 16),
        ("decode", 128),
        ("cache", 512),
        ("threads", 20),
        ("interop-threads", 1),
        ("warmups", 2),
        ("repeats", 3),
    ):
        run.add_argument("--" + name, type=int, default=default)
    run.add_argument("--attention", choices=("auto", "sdpa", "eager"), default="auto")
    run.add_argument(
        "--cache-implementation", choices=("dynamic", "static"), default="dynamic"
    )
    run.add_argument("--save-full-logits", action="store_true")
    run.add_argument("--reference-logits", type=Path)
    run.add_argument("--atol", type=float, default=2e-4)
    run.add_argument("--rtol", type=float, default=2e-5)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if command == "schema":
        print(json.dumps(Options.model_json_schema(), indent=2))
    elif command == "discover":
        rows = discover(arguments["models_root"])
        text = json.dumps(rows, indent=2) + "\n"
        output = arguments["output"]
        if output is not None:
            if (
                not output.expanduser()
                .resolve()
                .is_relative_to(Path.home() / ".trillim")
            ):
                raise ValueError("discovery output must be beneath ~/.trillim")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        print(text, end="")
    else:
        result = benchmark(Options.model_validate(arguments))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
