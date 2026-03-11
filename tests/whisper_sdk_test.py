# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the public Whisper SDK helpers."""

import asyncio
import io
import tempfile
import unittest
import wave

from trillim.server import Whisper


class _FakeWhisperEngine:
    def __init__(self):
        self.calls: list[tuple[bytes, str | None]] = []

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        self.calls.append((audio_bytes, language))
        return "decoded"


class _SlowWhisperEngine(_FakeWhisperEngine):
    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        await asyncio.sleep(0.05)
        return await super().transcribe(audio_bytes, language=language)


class WhisperSdkTests(unittest.IsolatedAsyncioTestCase):
    def _make_whisper(self, engine) -> Whisper:
        whisper = Whisper()
        whisper._engine = engine
        return whisper

    async def test_transcribe_bytes_uses_active_engine(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        result = await whisper.transcribe_bytes(b"audio", language="en")

        self.assertEqual(result, "decoded")
        self.assertEqual(engine.calls, [(b"audio", "en")])

    async def test_transcribe_wav_reads_file_bytes(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(b"RIFFdemo")
            handle.flush()

            result = await whisper.transcribe_wav(handle.name, language="fr")

        self.assertEqual(result, "decoded")
        self.assertEqual(engine.calls, [(b"RIFFdemo", "fr")])

    async def test_transcribe_array_encodes_valid_wav_from_frames_first_audio(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        result = await whisper.transcribe_array(
            [[0.25, -0.25], [0.5, -0.5], [0.0, 0.0]],
            sample_rate=44100,
            language="en",
        )

        self.assertEqual(result, "decoded")
        wav_bytes, language = engine.calls[-1]
        self.assertEqual(language, "en")
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 44100)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnframes(), 3)

    async def test_transcribe_array_accepts_channels_first_layout(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        await whisper.transcribe_array(
            [[0.25, 0.5, 0.75], [-0.25, -0.5, -0.75]],
            sample_rate=22050,
        )

        wav_bytes, _ = engine.calls[-1]
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 22050)
            self.assertEqual(wav_file.getnframes(), 3)

    async def test_transcribe_array_rejects_invalid_input(self):
        whisper = self._make_whisper(_FakeWhisperEngine())

        with self.assertRaisesRegex(ValueError, "sample_rate must be >= 1"):
            await whisper.transcribe_array([0.1, 0.2], sample_rate=0)

        with self.assertRaisesRegex(ValueError, "samples must not be empty"):
            await whisper.transcribe_array([], sample_rate=16000)

        with self.assertRaisesRegex(TypeError, "array-like sequence"):
            await whisper.transcribe_array(b"raw-bytes", sample_rate=16000)

    async def test_transcribe_methods_support_timeout(self):
        whisper = self._make_whisper(_SlowWhisperEngine())

        with self.assertRaisesRegex(TimeoutError, "Whisper transcription timed out"):
            await whisper.transcribe_bytes(b"audio", timeout=0.001)


if __name__ == "__main__":
    unittest.main()
