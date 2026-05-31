"""Shared bundle metadata helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CURRENT_FORMAT_VERSION = 4


def canonicalize_model_config(config: dict) -> dict:
    """Flatten ``text_config`` into the top-level model payload when present."""
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or not text_config:
        return dict(config)
    normalized = _merge_nested_dicts(
        {
            key: value
            for key, value in config.items()
            if key != "text_config"
        },
        text_config,
    )
    normalized["architectures"] = config.get(
        "architectures",
        text_config.get("architectures", normalized.get("architectures", [])),
    )
    return normalized


def compute_base_model_config_hash(model_dir: str | Path) -> str:
    """Compute the canonical base-model compatibility hash."""
    root = Path(model_dir)
    config_path = root / "config.json"
    transformer_config_path = root / "transformer" / "config.json"
    if not config_path.is_file() and transformer_config_path.is_file():
        return _compute_bonsai_image_config_hash(transformer_config_path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model config from {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a JSON object in {config_path}")
    config = canonicalize_model_config(raw)
    return _compute_text_config_hash(config)


def _compute_text_config_hash(config: dict) -> str:
    num_heads = _require_positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )
    num_kv_heads = _require_positive_int(
        config.get("num_key_value_heads", num_heads),
        "num_key_value_heads",
    )
    identity = {
        "architectures": config.get("architectures", []),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": config.get("vocab_size"),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _compute_bonsai_image_config_hash(config_path: Path) -> str:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model config from {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a JSON object in {config_path}")
    if raw.get("_class_name") != "Flux2Transformer2DModel":
        root_config_path = config_path.parent.parent / "config.json"
        raise ValueError(f"Could not read model config from {root_config_path}")
    num_heads = _require_positive_int(
        raw.get("num_attention_heads"),
        "num_attention_heads",
    )
    head_dim = _require_positive_int(
        raw.get("attention_head_dim"),
        "attention_head_dim",
    )
    identity = {
        "architecture": "bonsai_image",
        "class_name": raw.get("_class_name"),
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "hidden_size": num_heads * head_dim,
        "num_layers": _require_positive_int(raw.get("num_layers"), "num_layers"),
        "num_single_layers": _require_positive_int(
            raw.get("num_single_layers"),
            "num_single_layers",
        ),
        "mlp_ratio": raw.get("mlp_ratio", 3.0),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _merge_nested_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        current_value = merged.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dicts(current_value, value)
            continue
        merged[key] = value
    return merged


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number
