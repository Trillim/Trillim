"""Public local quantization entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trillim import _model_store

from ._config import load_model_config
from ._manifest import (
    determine_language_model_only,
    resolve_quantize_binary,
    run_adapter_quantizer,
    run_model_quantizer,
    validate_adapter_source,
)
from ._output import (
    build_staging_dir,
    copy_adapter_support_files,
    copy_model_support_files,
    mark_staging_complete,
    prepare_output_target,
    publish_staging_dir,
    write_adapter_metadata,
    write_model_metadata,
)
from ._quantization import normalize_quantization


@dataclass(frozen=True, slots=True)
class QuantizeResult:
    bundle_path: Path
    bundle_type: str
    used_language_model_only: bool


def quantize(
    model_dir: str | Path,
    adapter_dir: str | Path | None = None,
    *,
    quantization: str = "auto",
) -> QuantizeResult:
    quantization_target = normalize_quantization(quantization)
    source_model_dir = _normalize_source_dir(model_dir, label="Model directory")
    source_adapter_dir = (
        None
        if adapter_dir is None
        else _normalize_source_dir(adapter_dir, label="Adapter directory")
    )
    if source_adapter_dir is not None and source_adapter_dir == source_model_dir:
        raise ValueError("Adapter directory must be different from model directory")
    if source_adapter_dir is not None and quantization_target.value != "auto":
        raise ValueError("--quantization only applies when quantizing a base model")

    config = load_model_config(source_model_dir)
    language_model_only = determine_language_model_only(source_model_dir, config)
    if language_model_only:
        print(
            "Warning: This checkpoint uses a partially supported multimodal layout. "
            "Trillim will quantize only the language-model tensors for text inference."
        )

    binary_path = resolve_quantize_binary()
    if source_adapter_dir is not None:
        validate_adapter_source(source_adapter_dir, config)
        target = prepare_output_target(source_adapter_dir)
        staging_dir = build_staging_dir(target)
        run_adapter_quantizer(
            binary_path,
            source_model_dir,
            config,
            adapter_dir=source_adapter_dir,
            output_dir=staging_dir,
            language_model_only=language_model_only,
        )
        copy_adapter_support_files(source_adapter_dir, staging_dir)
        write_adapter_metadata(
            staging_dir,
            config=config,
            adapter_dir=source_adapter_dir,
            model_dir=source_model_dir,
        )
        mark_staging_complete(staging_dir)
        publish_staging_dir(target)
        print(f"Quantized adapter ready at: {target}")
        return QuantizeResult(
            bundle_path=target,
            bundle_type="adapter",
            used_language_model_only=language_model_only,
        )

    target = prepare_output_target(source_model_dir)
    staging_dir = build_staging_dir(target)
    run_model_quantizer(
        binary_path,
        source_model_dir,
        config,
        output_dir=staging_dir,
        language_model_only=language_model_only,
        quantization=quantization_target.value,
    )
    copy_model_support_files(source_model_dir, staging_dir)
    write_model_metadata(
        staging_dir,
        config=config,
        model_dir=source_model_dir,
        quantization=quantization_target.value,
    )
    mark_staging_complete(staging_dir)
    publish_staging_dir(target)
    print(f"Quantized model ready at: {target}")
    return QuantizeResult(
        bundle_path=target,
        bundle_type="model",
        used_language_model_only=language_model_only,
    )


def _normalize_source_dir(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    try:
        resolved.relative_to(_model_store.MODELS_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{label} must not be inside {_model_store.MODELS_ROOT}")
