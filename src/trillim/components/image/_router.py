"""HTTP router for the image generation component."""

from __future__ import annotations

import asyncio
import base64
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from trillim.components.llm._engine import EngineError
from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    InvalidRequestError,
    ModelValidationError,
    ProgressTimeoutError,
)

REQUEST_BODY_LIMIT_BYTES = 64 * 1024


def build_router(image) -> APIRouter:
    """Build the HTTP router for an image component instance."""
    router = APIRouter()
    request_lock = asyncio.Lock()

    @router.post("/v1/images/generations")
    async def image_generations(request: Request):
        if request_lock.locked():
            raise _as_http_error(
                AdmissionRejectedError("Image is already handling a request")
            )
        await request_lock.acquire()
        try:
            payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
            prompt = _required_str(payload, "prompt")
            width, height = _resolve_size(payload)
            steps = _positive_int(payload.get("steps", 4), "steps")
            seed = _optional_seed(payload.get("seed"))
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "image.png"
                await image.generate(
                    prompt,
                    output,
                    steps=steps,
                    seed=seed,
                    width=width,
                    height=height,
                )
                b64_png = base64.b64encode(output.read_bytes()).decode("ascii")
            return {"created": int(time.time()), "data": [{"b64_json": b64_png}]}
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            request_lock.release()

    return router


async def _read_json_body(request: Request, limit: int) -> dict[str, Any]:
    body = await request.body()
    if len(body) > limit:
        raise InvalidRequestError(f"request body exceeds the {limit} byte limit")
    try:
        payload = await request.json()
    except Exception as exc:
        raise InvalidRequestError("request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("request body must be a JSON object")
    return payload


def _required_str(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(f"{field_name} must be a non-empty string")
    return value


def _resolve_size(payload: dict[str, Any]) -> tuple[int, int]:
    if "size" in payload:
        size = payload["size"]
        if not isinstance(size, str):
            raise InvalidRequestError("size must use WIDTHxHEIGHT format")
        normalized = size.lower()
        if "x" not in normalized:
            raise InvalidRequestError("size must use WIDTHxHEIGHT format")
        width_text, height_text = normalized.split("x", 1)
        try:
            return (
                _positive_int(int(width_text), "width"),
                _positive_int(int(height_text), "height"),
            )
        except ValueError as exc:
            raise InvalidRequestError("size must use WIDTHxHEIGHT format") from exc
    return (
        _positive_int(payload.get("width", 1024), "width"),
        _positive_int(payload.get("height", 1024), "height"),
    )


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidRequestError(f"{field_name} must be a positive integer")
    return value


def _optional_seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidRequestError("seed must be a non-negative integer")
    return value


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, InvalidRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, (ComponentLifecycleError, ModelValidationError, EngineError)):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))
