"""Built-in preprocessors return the named inputs declared by model TDL."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from .processing_dataclasses import Audio, TensorInputs
from .processing_pydantic_models import _AudioRequest, _ImageRequest, _TextRequest
from .utils import _tokenizer, load_audio, log_mel, tokenize

_WHISPER_LOCK = RLock()


def text(
    value: object, *, assets: Mapping[str, Path], options: Mapping[str, JsonValue]
) -> TensorInputs:
    """Accept text or {text/messages, reset}; encode the complete supplied history."""
    request = _TextRequest.model_validate(
        {"text": value} if isinstance(value, str) else value
    )
    content = (
        request.text
        if request.text is not None
        else [message.model_dump() for message in request.messages or []]
    )
    tokens = tokenize(
        content, assets=assets["tokenizer"], chat=options.get("chat", True) is True
    )
    inputs: dict[str, NDArray[np.generic]] = {
        "tokens": tokens,
        "reset": np.asarray(request.reset, dtype=np.bool_),
    }
    if options.get("vision") is True:
        inputs.update(
            {
                "vision.patches": np.empty((0, 3, 2, 16, 16), dtype=np.float32),
                "vision.coordinates": np.empty((0, 2), dtype=np.int32),
                "vision.interpolation": np.empty((0, 2), dtype=np.float32),
                "vision.segments": np.empty((0,), dtype=np.int32),
                "vision.token_features": np.full(tokens.size, -1, dtype=np.int32),
                "positions": np.repeat(
                    np.arange(tokens.size, dtype=np.int32)[:, None], 3, axis=1
                ),
                "rope_delta": np.asarray(0, dtype=np.int32),
            }
        )
    return inputs


@lru_cache(maxsize=8)
def _whisper_features(directory: Path) -> object:
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(
        str(directory), local_files_only=True
    )


def whisper(
    value: object, *, assets: Mapping[str, Path], options: Mapping[str, JsonValue]
) -> TensorInputs:
    """Accept an audio path or {path, language, task}; one Whisper window per call."""
    from transformers import WhisperFeatureExtractor

    request = (
        value
        if isinstance(value, Audio)
        else _AudioRequest.model_validate(
            {"path": value} if isinstance(value, str) else value
        )
    )
    audio = load_audio(request if isinstance(request, Audio) else request.path)
    if audio.samples.size > 30 * audio.sample_rate:
        raise ValueError(
            "Whisper input exceeds 30 seconds; split longer audio into utterances"
        )
    features = cast(WhisperFeatureExtractor, _whisper_features(assets["features"]))
    mel = np.asarray(
        features(audio.samples, sampling_rate=audio.sample_rate, return_tensors="np")[
            "input_features"
        ],
        dtype=np.float32,
    )[0]
    tokenizer = _tokenizer(assets["tokenizer"])
    prefix = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    with _WHISPER_LOCK:
        prefix.extend(
            token
            for _, token in tokenizer.get_decoder_prompt_ids(
                language="en" if isinstance(request, Audio) else request.language,
                task="transcribe" if isinstance(request, Audio) else request.task,
                no_timestamps=True,
            )
        )
    return {
        "mel": mel,
        "tokens": np.asarray(prefix, dtype=np.int32),
        "reset": np.asarray(True),
    }


def parakeet(
    value: object, *, assets: Mapping[str, Path], options: Mapping[str, JsonValue]
) -> TensorInputs:
    """Accept an audio path, {path}, or Audio; produce unpadded Parakeet features."""
    request = (
        value
        if isinstance(value, Audio)
        else _AudioRequest.model_validate(
            {"path": value} if isinstance(value, str) else value
        )
    )
    audio = load_audio(request if isinstance(request, Audio) else request.path)
    return {"mel": log_mel(audio)}


def pocket_tts(
    value: object, *, assets: Mapping[str, Path], options: Mapping[str, JsonValue]
) -> TensorInputs:
    """Accept a text string and use PocketTTS's bundled SentencePiece tokenizer."""
    request = _TextRequest.model_validate(
        {"text": value} if isinstance(value, str) else value
    )
    if request.text is None or not request.text.strip():
        raise ValueError("PocketTTS requires nonempty plain text")
    prompt = " ".join(request.text.split())
    prompt = prompt[0].upper() + prompt[1:]
    if prompt[-1] not in ".!?…":
        prompt += "."
    return {"tokens": tokenize(prompt, assets=assets["tokenizer"], chat=False)}


def image(
    value: object, *, assets: Mapping[str, Path], options: Mapping[str, JsonValue]
) -> TensorInputs:
    """Prepare FLUX.2 Klein prompt, four-axis positions and an Euler sigma schedule."""
    request = _ImageRequest.model_validate(
        {"prompt": value} if isinstance(value, str) else value
    )
    tokens = tokenize(request.prompt, assets=assets["tokenizer"], enable_thinking=False)
    if tokens.size > 512:
        raise ValueError("image prompt exceeds 512 tokens")
    rows, columns = request.height // 16, request.width // 16
    text_coordinates = np.zeros((tokens.size, 4), dtype=np.int32)
    text_coordinates[:, 3] = np.arange(tokens.size, dtype=np.int32)
    image_coordinates = np.zeros((rows * columns, 4), dtype=np.int32)
    image_coordinates[:, 1] = np.repeat(np.arange(rows, dtype=np.int32), columns)
    image_coordinates[:, 2] = np.tile(np.arange(columns, dtype=np.int32), rows)
    length = rows * columns
    long_schedule = 0.00016927 * length + 0.45666666
    short_schedule = 8.73809524e-05 * length + 1.89833333
    shift = (
        long_schedule
        if length > 4300
        else long_schedule
        + (request.steps - 200) * (long_schedule - short_schedule) / 190
    )
    schedule = np.linspace(1, 1 / request.steps, request.steps, dtype=np.float64)
    schedule = np.exp(shift) / (np.exp(shift) + (1 / schedule - 1))
    sigmas = np.concatenate((schedule, [0])).astype(np.float32)
    return {
        "tokens": tokens,
        "coordinates": np.concatenate((text_coordinates, image_coordinates)),
        "sigmas": sigmas,
        "height": np.asarray(request.height, dtype=np.int32),
        "width": np.asarray(request.width, dtype=np.int32),
        "seed": np.asarray(request.seed, dtype=np.uint64),
    }


__all__ = ["image", "parakeet", "pocket_tts", "text", "whisper"]
