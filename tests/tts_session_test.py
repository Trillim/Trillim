# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for TTS session queueing, interruption, and flow control."""

import asyncio
import unittest
from collections import deque

from trillim import TTS


class _SessionEngine:
    def __init__(self, plans, gates=None):
        self.sample_rate = 24000
        self.speed = 1.0
        self.calls: list[tuple[str, str | None, float | None]] = []
        self._plans = plans
        self._gates = gates or {}
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ):
        self.calls.append((text, voice, speed))
        for index, chunk in enumerate(self._plans[text]):
            gate = self._gates.get((text, index))
            if gate is not None:
                await gate.wait()
            await asyncio.sleep(0)
            yield chunk


class TTSSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()

    def _make_tts(self, engine: _SessionEngine) -> TTS:
        tts = TTS()
        tts._engine = engine
        tts._loop = self.loop
        tts._active_session = None
        tts._queued_sessions = deque()
        return tts

    async def test_speak_queues_sessions_until_active_completes(self):
        release_first = asyncio.Event()
        engine = _SessionEngine(
            {
                "first": [b"f1", b"f2"],
                "second": [b"s1"],
            },
            gates={("first", 1): release_first},
        )
        tts = self._make_tts(engine)

        first = tts.speak("first")
        second = tts.speak("second")

        self.assertEqual(first.state, "running")
        self.assertEqual(second.state, "queued")
        first_iter = first.__aiter__()
        self.assertEqual(await anext(first_iter), b"f1")
        self.assertEqual(second.state, "queued")

        release_first.set()
        self.assertEqual([chunk async for chunk in first_iter], [b"f2"])
        self.assertEqual(await second.collect(), b"s1")
        self.assertEqual(
            engine.calls,
            [
                ("first", None, 1.0),
                ("second", None, 1.0),
            ],
        )

    async def test_speak_snapshots_explicit_speed_per_session(self):
        engine = _SessionEngine({"hello": [b"a"]})
        tts = self._make_tts(engine)

        session = tts.speak("hello", speed=1.5)

        self.assertEqual(await session.collect(), b"a")
        self.assertEqual(engine.calls, [("hello", None, 1.5)])

    async def test_pause_and_resume_gate_future_chunk_production(self):
        release_second = asyncio.Event()
        engine = _SessionEngine(
            {"hello": [b"a", b"b"]},
            gates={("hello", 1): release_second},
        )
        tts = self._make_tts(engine)
        session = tts.speak("hello")
        iterator = session.__aiter__()

        self.assertEqual(await anext(iterator), b"a")
        session.pause()
        await asyncio.sleep(0)
        self.assertEqual(session.state, "paused")

        next_chunk = asyncio.create_task(iterator.__anext__())
        release_second.set()
        await asyncio.sleep(0.01)
        self.assertFalse(next_chunk.done())

        session.resume()
        self.assertEqual(await next_chunk, b"b")
        await session.wait()
        self.assertEqual(session.state, "completed")

    async def test_interrupt_cancels_active_and_queued_sessions(self):
        block_active = asyncio.Event()
        engine = _SessionEngine(
            {
                "active": [b"a"],
                "queued": [b"q"],
                "replacement": [b"r"],
            },
            gates={("active", 0): block_active},
        )
        tts = self._make_tts(engine)

        active = tts.speak("active")
        queued = tts.speak("queued")
        replacement = tts.speak("replacement", interrupt=True)
        await asyncio.sleep(0)

        await active.wait()
        self.assertEqual(active.state, "cancelled")
        self.assertEqual(queued.state, "cancelled")
        self.assertEqual(await replacement.collect(), b"r")

    async def test_cancel_removes_queued_session(self):
        block_active = asyncio.Event()
        engine = _SessionEngine(
            {
                "active": [b"a"],
                "queued": [b"q"],
            },
            gates={("active", 0): block_active},
        )
        tts = self._make_tts(engine)

        active = tts.speak("active")
        queued = tts.speak("queued")
        queued.cancel()
        await asyncio.sleep(0)
        self.assertEqual(queued.state, "cancelled")

        block_active.set()
        self.assertEqual(await active.collect(), b"a")
        self.assertEqual(await queued.collect(), b"")

    async def test_session_timeout_marks_failure(self):
        never_release = asyncio.Event()
        engine = _SessionEngine(
            {"slow": [b"s"]},
            gates={("slow", 0): never_release},
        )
        tts = self._make_tts(engine)

        session = tts.speak("slow", timeout=0.01)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            await session.wait()
        self.assertEqual(session.state, "failed")

    async def test_stop_cancels_sessions_and_stops_engine(self):
        never_release = asyncio.Event()
        engine = _SessionEngine(
            {"slow": [b"s"]},
            gates={("slow", 0): never_release},
        )
        tts = self._make_tts(engine)

        session = tts.speak("slow")
        await tts.stop()

        self.assertTrue(session.done)
        self.assertEqual(session.state, "cancelled")
        self.assertIsNone(tts.engine)
        self.assertIsNone(tts._loop)
        self.assertTrue(engine.stopped)


if __name__ == "__main__":
    unittest.main()
