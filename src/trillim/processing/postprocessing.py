"""Built-in output conversions, independent of runtime execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import JsonValue

from .processing_dataclasses import Audio, TensorOutputs
from .utils import decode_tokens, image_from_tensor


def _complete(outputs: TensorOutputs) -> None:
    truncated = outputs.get("truncated")
    if truncated is not None and bool(truncated.item()):
        raise ValueError(
            "model output was truncated; increase the TDL generation capacity or shorten the request"
        )


def text(
    outputs: TensorOutputs,
    *,
    assets: Mapping[str, Path],
    options: Mapping[str, JsonValue],
) -> str:
    """Decode generated tokens; the launch TDLs exclude prompt tokens from output."""
    _complete(outputs)
    return decode_tokens(
        outputs["tokens"],
        assets=assets["tokenizer"],
        skip_special_tokens=options.get("skip_special_tokens", True) is True,
    )


def audio(
    outputs: TensorOutputs,
    *,
    assets: Mapping[str, Path],
    options: Mapping[str, JsonValue],
) -> Audio:
    """Return a waveform and its model-declared sample rate."""
    _complete(outputs)
    rate = options.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("audio postprocessing requires a positive sample_rate option")
    samples = np.asarray(outputs["audio"], dtype=np.float32).reshape(-1)
    if not np.isfinite(samples).all():
        raise ValueError("audio output contains non-finite samples")
    return Audio(samples, rate)


def image(
    outputs: TensorOutputs,
    *,
    assets: Mapping[str, Path],
    options: Mapping[str, JsonValue],
) -> Image.Image:
    """Return a PIL RGB image."""
    return image_from_tensor(outputs["image"])


__all__ = ["audio", "image", "text"]
