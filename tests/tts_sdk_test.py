# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the public TTS SDK helpers."""

import unittest

import trillim
from trillim import SentenceChunker
from trillim.server import TTS


class _FakeTTSEngine:
    DEFAULT_VOICE = "alba"

    def __init__(self):
        self.default_voice = "alba"
        self.sample_rate = 24000
        self.calls: list[tuple] = []
        self._voices = [
            {"voice_id": "alba", "name": "alba", "type": "predefined"},
            {"voice_id": "jean", "name": "jean", "type": "predefined"},
            {"voice_id": "custom", "name": "custom", "type": "custom"},
        ]

    def list_voices(self) -> list[dict]:
        self.calls.append(("list_voices",))
        return list(self._voices)

    async def register_voice(self, voice_id: str, audio_bytes: bytes) -> None:
        self.calls.append(("register_voice", voice_id, audio_bytes))
        self._voices.append({"voice_id": voice_id, "name": voice_id, "type": "custom"})

    async def delete_voice(self, voice_id: str) -> None:
        self.calls.append(("delete_voice", voice_id))
        self._voices = [voice for voice in self._voices if voice["voice_id"] != voice_id]
        if self.default_voice == voice_id:
            self.default_voice = self.DEFAULT_VOICE

    async def synthesize_stream(self, text: str, voice: str | None = None):
        self.calls.append(("synthesize_stream", text, voice))
        yield b"pcm-a"
        yield b"pcm-b"

    async def synthesize_full(self, text: str, voice: str | None = None) -> bytes:
        self.calls.append(("synthesize_full", text, voice))
        return b"RIFFdemo"


class TTSSdkTests(unittest.IsolatedAsyncioTestCase):
    def _make_tts(self) -> tuple[TTS, _FakeTTSEngine]:
        tts = TTS()
        engine = _FakeTTSEngine()
        tts._engine = engine
        tts._default_voice = engine.default_voice
        return tts, engine

    async def test_public_exports_include_sentence_chunker(self):
        self.assertIs(trillim.SentenceChunker, SentenceChunker)

    async def test_default_voice_getter_and_setter_use_component_api(self):
        tts, engine = self._make_tts()

        self.assertEqual(tts.default_voice, "alba")
        tts.default_voice = "jean"

        self.assertEqual(tts.default_voice, "jean")
        self.assertEqual(engine.default_voice, "jean")

    async def test_default_voice_can_be_set_before_start(self):
        tts = TTS()

        tts.default_voice = "jean"

        self.assertEqual(tts.default_voice, "jean")

    async def test_default_voice_rejects_empty_value(self):
        tts = TTS()

        with self.assertRaisesRegex(ValueError, "default_voice must not be empty"):
            tts.default_voice = ""

    async def test_default_voice_rejects_unknown_voice_when_started(self):
        tts, _ = self._make_tts()

        with self.assertRaisesRegex(ValueError, "Unknown voice"):
            tts.default_voice = "missing"

    async def test_sample_rate_and_list_voices_are_public(self):
        tts, engine = self._make_tts()

        self.assertEqual(tts.sample_rate, 24000)
        self.assertEqual(tts.list_voices(), engine.list_voices())

    async def test_register_and_delete_voice_use_public_wrappers(self):
        tts, engine = self._make_tts()
        tts.default_voice = "custom"

        await tts.register_voice("newvoice", b"wav-bytes")
        await tts.delete_voice("custom")

        self.assertIn(("register_voice", "newvoice", b"wav-bytes"), engine.calls)
        self.assertIn(("delete_voice", "custom"), engine.calls)
        self.assertEqual(tts.default_voice, "alba")

    async def test_synthesize_stream_and_wav_use_public_wrappers(self):
        tts, engine = self._make_tts()

        chunks = [chunk async for chunk in tts.synthesize_stream("hello", voice="jean")]
        wav_bytes = await tts.synthesize_wav("hello", voice="jean")

        self.assertEqual(chunks, [b"pcm-a", b"pcm-b"])
        self.assertEqual(wav_bytes, b"RIFFdemo")
        self.assertIn(("synthesize_stream", "hello", "jean"), engine.calls)
        self.assertIn(("synthesize_full", "hello", "jean"), engine.calls)

    async def test_synthesis_rejects_empty_input(self):
        tts, _ = self._make_tts()

        with self.assertRaisesRegex(ValueError, "input text is empty"):
            await tts.synthesize_wav("   ")

        with self.assertRaisesRegex(ValueError, "input text is empty"):
            [chunk async for chunk in tts.synthesize_stream("   ")]

    async def test_synthesis_requires_started_component(self):
        tts = TTS()

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            await tts.synthesize_wav("hello")

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            [chunk async for chunk in tts.synthesize_stream("hello")]

    async def test_tts_public_api_requires_started_component(self):
        tts = TTS()

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            _ = tts.sample_rate

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            tts.list_voices()

    async def test_sentence_chunker_is_top_level_and_usable(self):
        chunker = SentenceChunker()

        parts = chunker.feed("Hello world. Another sentence")
        remainder = chunker.flush()

        self.assertEqual(parts, ["Hello world."])
        self.assertEqual(remainder, "Another sentence")


if __name__ == "__main__":
    unittest.main()
