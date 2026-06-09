"""Public image generation component API."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter

from trillim import _model_store
from trillim.components import Component
from trillim.components.image._engine import ImageEngine
from trillim.components.image._model_dir import validate_image_model_dir
from trillim.components.image._router import build_router
from trillim.components.llm._config import ArchitectureType, ModelRuntimeConfig
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ModelValidationError


class Image(Component):
    """Bonsai Image component owning one image model runtime."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        num_threads: int = 0,
        trust_remote_code: bool = False,
        _model_validator=validate_image_model_dir,
        _tokenizer_loader=None,
        _engine_factory=None,
    ) -> None:
        if not model_dir:
            raise ValueError("model_dir is required")
        self._model_dir = _normalize_store_id("model_dir", model_dir)
        self._num_threads = num_threads
        self._trust_remote_code = trust_remote_code
        self._model_validator = _model_validator
        self._tokenizer_loader = load_image_tokenizer if _tokenizer_loader is None else _tokenizer_loader
        self._engine_factory = ImageEngine if _engine_factory is None else _engine_factory
        self._owner_loop: AbstractEventLoop | None = None
        self._model: ModelRuntimeConfig | None = None
        self._engine: ImageEngine | None = None
        self._started = False

    def router(self) -> APIRouter:
        """Return the FastAPI router for this image component."""
        return build_router(self)

    async def start(self) -> None:
        """Validate the configured image model bundle and start its worker."""
        self._require_owner_loop()
        if self._started:
            return
        model = self._model_validator(self._model_dir)
        if model.arch_type != ArchitectureType.BONSAI_IMAGE:
            raise ModelValidationError(
                "image generation requires a Bonsai Image model"
            )
        tokenizer = self._tokenizer_loader(
            self._model_dir / "tokenizer",
            trust_remote_code=self._trust_remote_code,
        )
        engine = self._engine_factory(
            model,
            tokenizer,
            num_threads=self._num_threads,
        )
        await engine.start()
        self._model = model
        self._engine = engine
        self._started = True

    async def stop(self) -> None:
        """Clear in-memory image model state."""
        self._require_owner_loop()
        engine = self._engine
        self._engine = None
        self._model = None
        self._started = False
        if engine is not None:
            await engine.stop()

    async def generate(
        self,
        prompt: str,
        output_path: str | Path,
        *,
        steps: int = 4,
        seed: int | None = None,
        width: int = 320,
        height: int = 240,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Generate one image from text and write it to ``output_path``."""
        self._require_owner_loop()
        self._require_started()
        prompt = _normalize_prompt(prompt)
        output = _normalize_output_path(output_path)
        _validate_positive_int("steps", steps)
        _validate_positive_int("width", width)
        _validate_positive_int("height", height)
        if width > 4096 or height > 4096:
            raise InvalidRequestError("width and height must be <= 4096")
        if width % 16 != 0 or height % 16 != 0:
            raise InvalidRequestError("width and height must be multiples of 16")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise InvalidRequestError("seed must be a non-negative integer")
        engine = self._engine
        if engine is None:
            raise ComponentLifecycleError("Image component is not running")
        return await engine.generate(
            prompt,
            output,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
            progress_callback=progress_callback,
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


def load_image_tokenizer(model_dir: Path, *, trust_remote_code: bool):
    """Load the tokenizer used by a Bonsai Image bundle."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ModelValidationError("transformers is required to load tokenizers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ModelValidationError(
            f"Could not load tokenizer from {model_dir}"
        ) from exc
    if not callable(tokenizer):
        raise ModelValidationError(
            f"Tokenizer loaded from {model_dir} is not callable"
        )
    return tokenizer
