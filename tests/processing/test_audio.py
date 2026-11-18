from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from trillim.processing import Audio, postprocessing, preprocessing
from trillim.processing.utils import load_audio, log_mel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("rate", [44100, 16000, 48000])
def test_load_real_stereo_speech(
    audio_reference: dict[str, NDArray[np.float32]], rate: int
) -> None:
    expected = audio_reference[f"mono{rate}"]
    for source in (
        FIXTURES / "speech.flac",
        str(FIXTURES / "speech.flac"),
        Audio(audio_reference["stereo44100"], 44100),
    ):
        audio = load_audio(source, sample_rate=rate)
        assert audio.sample_rate == rate and audio.samples.dtype == np.float32
        assert audio.samples.ndim == 1 and audio.samples.flags.c_contiguous
        np.testing.assert_allclose(audio.samples, expected, rtol=1e-6, atol=1e-7)


def test_load_mono_audio_preserves_samples(
    audio_reference: dict[str, NDArray[np.float32]],
) -> None:
    original = audio_reference["mono16000"]
    # A channel view is not contiguous; callers must still get contiguous mono audio.
    for samples in (original, audio_reference["stereo44100"][:, 0]):
        before = samples.copy()
        audio = load_audio(Audio(samples, 16000))
        assert audio.samples.flags.c_contiguous
        np.testing.assert_array_equal(audio.samples, before)
        np.testing.assert_array_equal(samples, before)


@pytest.mark.parametrize(
    "damage", ["empty", "nan", "inf", "rank", "source-rate", "target-rate"]
)
def test_load_audio_rejects_invalid_waveforms(
    audio_reference: dict[str, NDArray[np.float32]], damage: str
) -> None:
    samples = audio_reference["mono16000"].copy()
    if damage == "empty":
        samples = samples[:0]
    elif damage == "nan":
        samples[0] = np.nan
    elif damage == "inf":
        samples[0] = np.inf
    elif damage == "rank":
        samples = samples.reshape(1, 1, -1)
    with pytest.raises(ValueError, match="finite samples|positive"):
        load_audio(
            Audio(samples, 0 if damage == "source-rate" else 16000),
            sample_rate=0 if damage == "target-rate" else 16000,
        )


@pytest.mark.parametrize("configuration", ["default", "alternate", "short"])
def test_log_mel_matches_nemo_reference(
    audio_reference: dict[str, NDArray[np.float32]], configuration: str
) -> None:
    samples = audio_reference["mono16000"].copy()
    before = samples.copy()
    if configuration == "alternate":
        features = log_mel(
            Audio(samples, 16000),
            mel_bins=64,
            n_fft=256,
            window_length=200,
            hop_length=80,
            preemphasis=0.95,
        )
        expected = audio_reference["parakeet64"]
    else:
        if configuration == "short":
            samples = samples[:320]
        features = log_mel(Audio(samples, 16000))
        expected = audio_reference[
            "parakeet_short" if configuration == "short" else "parakeet128"
        ]
    assert features.dtype == np.float32
    # Different FFT implementations round differently before log and normalization.
    np.testing.assert_allclose(features, expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(samples, before[: samples.size])


@pytest.mark.parametrize(
    "parameters",
    [{"window_length": 0}, {"window_length": 513}, {"hop_length": 0}, {"mel_bins": 0}],
)
def test_log_mel_invalid_dimensions(
    audio_reference: dict[str, NDArray[np.float32]], parameters: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="invalid mel feature dimensions"):
        log_mel(Audio(audio_reference["mono16000"], 16000), **parameters)


@pytest.mark.parametrize("damage", ["too-short", "nan", "inf", "rank", "rate"])
def test_log_mel_invalid_audio(
    audio_reference: dict[str, NDArray[np.float32]], damage: str
) -> None:
    samples = audio_reference["mono16000"].copy()
    if damage == "too-short":
        samples = samples[:159]
    elif damage in {"nan", "inf"}:
        samples[0] = np.nan if damage == "nan" else np.inf
    elif damage == "rank":
        samples = samples.reshape(1, -1)
    with pytest.raises(ValueError, match="finite|sample rate|feature frames|mono"):
        log_mel(Audio(samples, 0 if damage == "rate" else 16000))


@pytest.mark.parametrize("kind", ["path", "request", "audio"])
def test_parakeet_real_speech(
    audio_reference: dict[str, NDArray[np.float32]], kind: str
) -> None:
    value: object = str(FIXTURES / "speech.flac")
    if kind == "request":
        value = {"path": value}
    elif kind == "audio":
        value = Audio(audio_reference["mono16000"], 16000)
    inputs = preprocessing.parakeet(value, assets={}, options={})
    assert set(inputs) == {"mel"}
    np.testing.assert_allclose(
        np.asarray(inputs["mel"], dtype=np.float32),
        audio_reference["parakeet128"],
        atol=2e-4,
        rtol=2e-4,
    )


def test_whisper_matches_openai_reference_and_language_prefixes(
    assets: dict[str, Path],
    expected: dict[str, list[int] | list[float] | str],
    audio_reference: dict[str, NDArray[np.float32]],
) -> None:
    arguments = {"tokenizer": assets["whisper"], "features": assets["whisper"]}
    # Language and task from one call must not leak into the next.
    for language, task in (
        ("fr", "translate"),
        ("fr", "transcribe"),
        ("en", "transcribe"),
    ):
        inputs = cast(
            dict[str, NDArray[np.generic]],
            preprocessing.whisper(
                {
                    "path": str(FIXTURES / "speech.flac"),
                    "language": language,
                    "task": task,
                },
                assets=arguments,
                options={},
            ),
        )
        assert set(inputs) == {"mel", "tokens", "reset"}
        assert inputs["mel"].dtype == np.float32 and inputs["mel"].shape == (80, 3000)
        assert inputs["tokens"].dtype == np.int32
        np.testing.assert_allclose(
            np.asarray(inputs["mel"], dtype=np.float32),
            audio_reference["whisper80"],
            atol=2e-5,
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            inputs["tokens"], expected[f"whisper_{language}_{task}"]
        )
        assert inputs["reset"].shape == () and inputs["reset"].dtype == np.bool_
        assert inputs["reset"].item() is True
    for value in (
        Audio(audio_reference["mono16000"], 16000),
        str(FIXTURES / "speech.flac"),
    ):
        result = preprocessing.whisper(value, assets=arguments, options={})
        np.testing.assert_array_equal(
            result["tokens"], expected["whisper_en_transcribe"]
        )
        np.testing.assert_allclose(
            np.asarray(result["mel"], dtype=np.float32),
            audio_reference["whisper80"],
            atol=2e-5,
            rtol=1e-5,
        )


def test_whisper_window_limit(
    assets: dict[str, Path],
    audio_reference: dict[str, NDArray[np.float32]],
) -> None:
    samples = np.resize(audio_reference["mono16000"], 480001)
    arguments = {"tokenizer": assets["whisper"], "features": assets["whisper"]}
    inputs = preprocessing.whisper(
        Audio(samples[:480000], 16000), assets=arguments, options={}
    )
    mel = inputs["mel"]
    assert isinstance(mel, np.ndarray) and mel.shape == (80, 3000)
    with pytest.raises(ValueError, match="exceeds 30 seconds"):
        preprocessing.whisper(Audio(samples, 16000), assets=arguments, options={})


@pytest.mark.parametrize(
    "value", [{"task": "summarize"}, {"language": "not-a-language"}, {"extra": 1}]
)
def test_whisper_rejects_invalid_requests(
    assets: dict[str, Path], value: dict[str, str | int]
) -> None:
    with pytest.raises((ValueError, ValidationError)):
        preprocessing.whisper(
            {"path": str(FIXTURES / "speech.flac"), **value},
            assets={"tokenizer": assets["whisper"], "features": assets["whisper"]},
            options={},
        )


@pytest.mark.parametrize("batched", [False, True])
def test_audio_postprocessing(
    audio_reference: dict[str, NDArray[np.float32]],
    batched: bool,
) -> None:
    samples = audio_reference["mono48000"]
    output = samples[None, :] if batched else samples
    audio = postprocessing.audio(
        {"audio": output, "truncated": np.asarray(False)},
        assets={},
        options={"sample_rate": 48000},
    )
    assert audio.sample_rate == 48000 and audio.samples.dtype == np.float32
    np.testing.assert_array_equal(audio.samples, samples)
    with pytest.raises(ValueError, match="truncated"):
        postprocessing.audio(
            {"audio": output, "truncated": np.asarray(True)},
            assets={},
            options={"sample_rate": 48000},
        )


@pytest.mark.parametrize("rate", [None, 0, -1, True, 24000.0, "24000"])
def test_audio_postprocessing_requires_sample_rate(rate: object) -> None:
    from pydantic import JsonValue, TypeAdapter

    options = TypeAdapter(dict[str, JsonValue]).validate_python({"sample_rate": rate})
    with pytest.raises(ValueError, match="positive sample_rate"):
        postprocessing.audio(
            {"audio": np.asarray([0], dtype=np.float32)}, assets={}, options=options
        )


def test_audio_postprocessing_rejects_nonfinite(
    audio_reference: dict[str, NDArray[np.float32]],
) -> None:
    samples = audio_reference["mono48000"].copy()
    samples[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        postprocessing.audio(
            {"audio": samples}, assets={}, options={"sample_rate": 48000}
        )
