"""Model directory validation and metadata extraction for image generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    canonicalize_model_config as _canonicalize_model_config,
)
from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    ModelRuntimeConfig,
)
from trillim.errors import ModelValidationError

_IMAGE_ARCHITECTURE_NAME = "Flux2Transformer2DModel"
_IMAGE_ARCHITECTURE_KEY = _IMAGE_ARCHITECTURE_NAME.lower()
_IMAGE_RUNTIME_ARTIFACTS = ("qmodel.tensors", "qmodel.index", "rope.cache")
_DEFAULT_EOS_TOKEN = 151645


@dataclass(frozen=True, slots=True)
class _ImageArchitectureInfo:
    arch_type: ArchitectureType
    activation: ActivationType
    has_attn_sub_norm: bool
    has_ffn_sub_norm: bool
    has_qkv_bias: bool = False


_IMAGE_ARCH_REGISTRY: dict[str, _ImageArchitectureInfo] = {
    _IMAGE_ARCHITECTURE_KEY: _ImageArchitectureInfo(
        arch_type=ArchitectureType.BONSAI_IMAGE,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
}
_ACTIVATION_MAP = {
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
}


def validate_image_model_dir(model_dir: str | Path) -> ModelRuntimeConfig:
    """Validate a Bonsai Image model directory and extract runtime metadata."""
    path = _resolve_directory(
        model_dir,
        label="Image model directory",
        symlink_message="Image model bundle must not use symlinks",
    )
    _validate_model_bundle_metadata(path)
    config_path = path / "config.json"
    _raise_if_symlink(config_path, "Image model bundle must not use symlinks")
    if not config_path.is_file():
        raise ModelValidationError(f"config.json not found in {path}")
    config_payload = _load_json(config_path)
    if not isinstance(config_payload, dict):
        raise ModelValidationError(f"config.json must be a JSON object in {path}")
    config = _canonicalize_model_config(config_payload)
    arch_info = _resolve_arch_info(config)
    _require_runtime_artifacts(path)
    dimensions = _extract_dimensions(config)
    return ModelRuntimeConfig(
        name=path.name,
        path=path,
        arch_type=arch_info.arch_type,
        activation=_resolve_activation(config, arch_info),
        hidden_dim=dimensions["hidden_dim"],
        intermediate_dim=dimensions["intermediate_dim"],
        num_layers=dimensions["num_layers"],
        num_heads=dimensions["num_heads"],
        num_kv_heads=dimensions["num_kv_heads"],
        vocab_size=dimensions["vocab_size"],
        head_dim=dimensions["head_dim"],
        max_position_embeddings=dimensions["max_position_embeddings"],
        norm_eps=float(config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))),
        rope_theta=_resolve_rope_theta(config),
        eos_tokens=tuple(_collect_eos_tokens(config)),
        has_qkv_bias=bool(config.get("attention_bias", arch_info.has_qkv_bias)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        has_attn_sub_norm=arch_info.has_attn_sub_norm,
        has_ffn_sub_norm=arch_info.has_ffn_sub_norm,
    )


def is_image_model_dir(model_dir: str | Path) -> bool:
    """Return whether ``model_dir`` declares a supported image architecture."""
    try:
        config_payload = _load_json(Path(model_dir) / "config.json")
    except ModelValidationError:
        return False
    if not isinstance(config_payload, dict):
        return False
    config = _canonicalize_model_config(config_payload)
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "unknown"
    return str(arch_name).lower() in _IMAGE_ARCH_REGISTRY


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"Could not read JSON from {path}") from exc


def _validate_model_bundle_metadata(model_dir: Path) -> None:
    config_path = model_dir / "trillim_config.json"
    _raise_if_symlink(config_path, "Image model bundle must not use symlinks")
    if not config_path.is_file():
        raise ModelValidationError(
            f"Image model bundle metadata is missing or unsupported in {model_dir}"
        )
    payload = _load_json(config_path)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CURRENT_FORMAT_VERSION
    ):
        raise ModelValidationError(
            f"Image model bundle metadata is missing or unsupported in {model_dir}"
        )


def _resolve_arch_info(config: dict) -> _ImageArchitectureInfo:
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "unknown"
    try:
        return _IMAGE_ARCH_REGISTRY[str(arch_name).lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported image model architecture: {arch_name}"
        ) from exc


def _require_runtime_artifacts(model_dir: Path) -> None:
    for filename in _IMAGE_RUNTIME_ARTIFACTS:
        artifact_path = model_dir / filename
        _raise_if_symlink(artifact_path, "Image model bundle must not use symlinks")
        if not artifact_path.is_file():
            raise ModelValidationError(f"{filename} not found in {model_dir}")


def _extract_dimensions(config: dict) -> dict[str, int]:
    hidden_dim = _require_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate_dim = _require_positive_int(
        config.get("intermediate_size"),
        "intermediate_size",
    )
    num_heads = _require_positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )
    num_kv_heads = _require_positive_int(
        config.get("num_key_value_heads", num_heads),
        "num_key_value_heads",
    )
    max_position_embeddings = _require_positive_int(
        config.get("max_position_embeddings", 4096),
        "max_position_embeddings",
    )
    vocab_size = _require_positive_int(config.get("vocab_size", 1), "vocab_size")
    head_dim = int(config.get("head_dim", hidden_dim // num_heads))
    if head_dim <= 0:
        raise ModelValidationError("head_dim must be a positive integer")
    return {
        "hidden_dim": _align_to_128(hidden_dim),
        "intermediate_dim": _align_to_128(intermediate_dim),
        "num_layers": _require_positive_int(config.get("num_hidden_layers"), "num_hidden_layers"),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
    }


def _align_to_128(value: int) -> int:
    return ((value + 127) // 128) * 128


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ModelValidationError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ModelValidationError(f"{field_name} must be a positive integer")
    return number


def _resolve_activation(
    config: dict,
    arch_info: _ImageArchitectureInfo,
) -> ActivationType:
    hidden_act = config.get("hidden_act")
    if hidden_act is None:
        return arch_info.activation
    try:
        return _ACTIVATION_MAP[str(hidden_act).lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported image activation function: {hidden_act}"
        ) from exc


def _resolve_rope_theta(config: dict) -> float:
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        else:
            rope_theta = 10000.0
    try:
        return float(rope_theta)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("rope_theta must be numeric") from exc


def _collect_eos_tokens(config: dict) -> list[int]:
    eos_raw = config.get("eos_token_id", _DEFAULT_EOS_TOKEN)
    if isinstance(eos_raw, list):
        eos_tokens = [int(token_id) for token_id in eos_raw]
    else:
        eos_tokens = [int(eos_raw)]
    deduped: list[int] = []
    seen: set[int] = set()
    for token_id in eos_tokens:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    if not deduped:
        raise ModelValidationError("No EOS tokens could be determined for the image model")
    return deduped


def _resolve_directory(
    directory: str | Path,
    *,
    label: str,
    symlink_message: str,
) -> Path:
    original = str(directory)
    path = Path(os.path.abspath(Path(directory).expanduser()))
    _raise_if_symlink(path, symlink_message)
    if not path.exists():
        raise ModelValidationError(f"{label} does not exist: {original}")
    if not path.is_dir():
        raise ModelValidationError(f"{label} is not a directory: {path}")
    return path


def _raise_if_symlink(path: Path, message: str) -> None:
    if path.is_symlink():
        raise ModelValidationError(f"{message}: {path}")
