"""Public image generation component API."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from pathlib import Path

from trillim import _model_store
from trillim.components import Component
from trillim.components.llm._config import ArchitectureType, ModelRuntimeConfig
from trillim.components.llm._model_dir import validate_model_dir
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ModelValidationError


class Image(Component):
    """Bonsai Image component owning one image model runtime."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        trust_remote_code: bool = False,
        _model_validator=validate_model_dir,
    ) -> None:
        if not model_dir:
            raise ValueError("model_dir is required")
        self._model_dir = _normalize_store_id("model_dir", model_dir)
        self._trust_remote_code = trust_remote_code
        self._model_validator = _model_validator
        self._owner_loop: AbstractEventLoop | None = None
        self._model: ModelRuntimeConfig | None = None
        self._started = False

    async def start(self) -> None:
        """Validate the configured image model bundle."""
        self._require_owner_loop()
        if self._started:
            return
        model = self._model_validator(self._model_dir)
        if model.arch_type != ArchitectureType.BONSAI_IMAGE:
            raise ModelValidationError(
                "image generation requires a Bonsai Image model"
            )
        self._model = model
        self._started = True

    async def stop(self) -> None:
        """Clear in-memory image model state."""
        self._require_owner_loop()
        self._model = None
        self._started = False

    def generate(
        self,
        prompt: str,
        output_path: str | Path,
        *,
        steps: int = 4,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
    ) -> Path:
        """Generate one image from text and write it to ``output_path``."""
        self._require_owner_loop()
        self._require_started()
        _normalize_prompt(prompt)
        _normalize_output_path(output_path)
        _validate_positive_int("steps", steps)
        _validate_positive_int("width", width)
        _validate_positive_int("height", height)
        if seed is not None and isinstance(seed, bool):
            raise InvalidRequestError("seed must be an integer")
        raise NotImplementedError(
            "Bonsai Image generation runtime is not implemented yet"
        )

    def _require_owner_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise ComponentLifecycleError(
                "Image is bound to one event loop; create a new Image per thread/event loop"
            )

    def _require_started(self) -> None:
        if not self._started:
            raise ComponentLifecycleError("Image component is not running")


def _normalize_store_id(field_name: str, value: str | Path) -> Path:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    try:
        return _model_store.resolve_existing_store_id(
            normalized,
            error_type=InvalidRequestError,
        )
    except InvalidRequestError as exc:
        raise InvalidRequestError(f"{field_name}: {exc}") from exc


def _normalize_prompt(prompt: str) -> str:
    normalized = str(prompt).strip()
    if not normalized:
        raise InvalidRequestError("prompt must not be empty")
    return normalized


def _normalize_output_path(output_path: str | Path) -> Path:
    normalized = Path(output_path).expanduser()
    if not str(output_path).strip():
        raise InvalidRequestError("output_path must not be empty")
    if normalized.exists() and normalized.is_dir():
        raise InvalidRequestError("output_path must be a file path")
    if normalized.suffix.lower() != ".png":
        raise InvalidRequestError("output_path must end in .png")
    return normalized


def _validate_positive_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidRequestError(f"{field_name} must be a positive integer")
