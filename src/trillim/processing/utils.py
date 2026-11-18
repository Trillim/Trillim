"""Reusable text, audio and image conversions; no model execution."""

from __future__ import annotations

from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from .processing_dataclasses import Audio


class _Tokenizer(Protocol):
    chat_template: str | dict[str, str] | None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...
    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
        return_dict: bool = False,
    ) -> list[int]: ...
    def convert_tokens_to_ids(self, token: str) -> int: ...
    def get_decoder_prompt_ids(
        self, *, language: str, task: str, no_timestamps: bool
    ) -> list[tuple[int, int]]: ...


@lru_cache(maxsize=8)
def _tokenizer(assets: Path) -> _Tokenizer:
    from transformers import AutoTokenizer

    return cast(
        _Tokenizer,
        AutoTokenizer.from_pretrained(
            str(assets), local_files_only=True, trust_remote_code=False
        ),
    )


@lru_cache(maxsize=8)
def _sentencepiece(assets: Path) -> object:
    from sentencepiece import SentencePieceProcessor

    return SentencePieceProcessor(model_file=str(assets / "tokenizer.model"))


def tokenize(
    text: str | list[dict[str, str]],
    *,
    assets: str | Path,
    chat: bool = True,
    enable_thinking: bool | None = None,
) -> NDArray[np.int32]:
    """Tokenize locally bundled assets, applying their chat template when requested."""
    directory = Path(assets).resolve()
    if (directory / "tokenizer.model").is_file() and not any(
        (directory / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        from sentencepiece import SentencePieceProcessor

        if not isinstance(text, str):
            raise ValueError("this SentencePiece tokenizer accepts plain text")
        tokenizer = cast(SentencePieceProcessor, _sentencepiece(directory))
        return np.asarray(tokenizer.encode(text, out_type=int), dtype=np.int32)
    tokenizer_hf = _tokenizer(directory)
    if chat and tokenizer_hf.chat_template:
        messages = (
            [{"role": "user", "content": text}] if isinstance(text, str) else text
        )
        if enable_thinking is None:
            ids = tokenizer_hf.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        else:
            ids = tokenizer_hf.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_dict=False,
            )
    else:
        if not isinstance(text, str):
            raise ValueError("messages require a tokenizer with a chat template")
        ids = tokenizer_hf.encode(text, add_special_tokens=True)
    return np.asarray(ids, dtype=np.int32)


def decode_tokens(
    tokens: NDArray[np.generic], *, assets: str | Path, skip_special_tokens: bool = True
) -> str:
    """Decode one model's generated token sequence."""
    directory = Path(assets).resolve()
    if tokens.dtype.kind not in "iu" or tokens.ndim != 1:
        raise ValueError("token output must be a rank-one integer array")
    ids = [int(value) for value in tokens]
    if (directory / "tokenizer.model").is_file() and not any(
        (directory / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        from sentencepiece import SentencePieceProcessor

        return str(cast(SentencePieceProcessor, _sentencepiece(directory)).decode(ids))
    return _tokenizer(directory).decode(
        ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False
    )


def load_audio(value: str | Path | Audio, *, sample_rate: int = 16000) -> Audio:
    """Load or resample mono audio; file channels are averaged before resampling."""
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    if isinstance(value, Audio):
        samples, source_rate = value.samples, value.sample_rate
    else:
        samples, source_rate = cast(
            tuple[NDArray[np.float32], int], sf.read(str(value), dtype="float32")
        )
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if (
        samples.ndim != 1
        or not samples.size
        or source_rate <= 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("audio must contain finite samples and a positive sample rate")
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        samples = np.asarray(
            resample_poly(samples, sample_rate // divisor, source_rate // divisor),
            dtype=np.float32,
        )
    return Audio(np.ascontiguousarray(samples), sample_rate)


def log_mel(
    audio: Audio,
    *,
    mel_bins: int = 128,
    n_fft: int = 512,
    window_length: int = 400,
    hop_length: int = 160,
    preemphasis: float = 0.97,
) -> NDArray[np.float32]:
    """NeMo-style log-mel features with sample-standard-deviation normalization."""
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    if not 0 < window_length <= n_fft or hop_length <= 0 or mel_bins <= 0:
        raise ValueError("invalid mel feature dimensions")
    waveform = audio.samples.copy()
    if waveform.size < n_fft:
        raise ValueError("audio is too short for the configured FFT")
    waveform[1:] -= preemphasis * audio.samples[:-1]
    window = window_function(window_length, periodic=False, frame_length=n_fft)
    bank = mel_filter_bank(
        n_fft // 2 + 1,
        mel_bins,
        0.0,
        audio.sample_rate / 2,
        audio.sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    power = spectrogram(
        waveform,
        window,
        n_fft,
        hop_length,
        fft_length=n_fft,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )
    frames = waveform.size // hop_length
    mel = np.log(
        np.asarray(bank.T @ power[:, :frames], dtype=np.float32) + np.float32(2**-24)
    )
    if frames < 2:
        raise ValueError("audio must produce at least two feature frames")
    return np.asarray(
        (mel - mel.mean(axis=1, keepdims=True))
        / (mel.std(axis=1, ddof=1, keepdims=True) + 1e-5),
        dtype=np.float32,
    )


def image_from_tensor(tensor: NDArray[np.generic]) -> Image.Image:
    """Convert a channel-first RGB tensor in [-1, 1] to a PIL image."""
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not np.isfinite(tensor).all():
        raise ValueError("image output must be finite [3, height, width] RGB")
    pixels = np.rint(np.clip((tensor.astype(np.float32) + 1) * 127.5, 0, 255)).astype(
        np.uint8
    )
    return Image.fromarray(pixels.transpose(1, 2, 0))


__all__ = ["decode_tokens", "image_from_tensor", "load_audio", "log_mel", "tokenize"]
