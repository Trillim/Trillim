"""Tests for TTSSession behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trillim.components.tts import TTSSession
from trillim.components.tts._session import _TTSSession, _create_tts_session
from trillim.components.tts.public import TTS
from trillim.errors import SessionBusyError, SessionClosedError
from tests.components.tts.support import make_started_tts


class TTSSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def _start_tts(self) -> TTS:
        tts, imports_patch, builtins_patch = make_started_tts()
        tts._spool_dir = self.spool_dir
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            await tts.start()
        return tts

    def _make_internal_session(self, owner) -> _TTSSession:
        return _create_tts_session(
            owner,
            text="hello world",
            voice="alba",
            voice_kind="predefined",
            voice_reference="alba",
            speed=1.0,
            cleanup_path=None,
            session_worker=SimpleNamespace(),
        )

    def test_public_session_cannot_be_constructed_or_subclassed(self):
        with self.assertRaisesRegex(TypeError, "cannot be constructed directly"):
            TTSSession()
        with self.assertRaisesRegex(TypeError, "cannot be subclassed publicly"):
            type("BadSession", (TTSSession,), {})

    def test_private_session_constructor_guards_reject_unauthorized_calls(self):
        with self.assertRaisesRegex(TypeError, "cannot be constructed directly"):
            _TTSSession()

        token = _create_tts_session.__globals__["_TTS_SESSION_OWNER_TOKEN"]

        unauthorized = object.__new__(_TTSSession)
        unauthorized._task = None
        with self.assertRaisesRegex(TypeError, "cannot be constructed directly"):
            _TTSSession.__init__(unauthorized, _owner_token=None)

        missing_fields = object.__new__(_TTSSession)
        missing_fields._task = None
        with self.assertRaisesRegex(TypeError, "cannot be constructed directly"):
            _TTSSession.__init__(missing_fields, _owner_token=token)

    async def test_collect_and_iteration_are_mutually_exclusive(self):
        tts = await self._start_tts()
        session = await tts.speak("one two three")
        iterator = session.__aiter__()
        with self.assertRaises(SessionBusyError):
            await session.collect()
        chunks = [chunk async for chunk in iterator]
        self.assertTrue(chunks)
        await tts.stop()

    async def test_double_iteration_is_single_consumer(self):
        tts = await self._start_tts()
        session = await tts.speak("one two three")
        iterator = session.__aiter__()
        with self.assertRaises(SessionBusyError):
            session.__aiter__()
        self.assertTrue([chunk async for chunk in iterator])
        await tts.stop()

    async def test_private_session_delegates_to_owner_and_cleans_up_once(self):
        calls: list[tuple[str, object]] = []

        class _Owner:
            async def _pause_session(self, session) -> None:
                calls.append(("pause", session))

            async def _resume_session(self, session) -> None:
                calls.append(("resume", session))

            async def _cancel_session(self, session) -> None:
                calls.append(("cancel", session))

            async def _set_session_speed(self, session, speed) -> None:
                session._speed = speed
                calls.append(("speed", speed))

        cleanup_path = self.spool_dir / "voice.state"
        cleanup_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_path.write_bytes(b"voice")
        session = _create_tts_session(
            _Owner(),
            text="hello world",
            voice="alba",
            voice_kind="predefined",
            voice_reference="alba",
            speed=1.0,
            cleanup_path=cleanup_path,
            session_worker=SimpleNamespace(),
        )

        self.assertEqual(session.voice, "alba")
        self.assertEqual(session.speed, 1.0)
        await session.pause()
        await session.resume()
        await session.close()
        await session.set_speed(1.5)
        await session._finish("completed")
        await session._finish("failed", RuntimeError("ignored"))

        self.assertFalse(cleanup_path.exists())
        self.assertEqual(session.state, "completed")
        self.assertEqual(session.speed, 1.5)
        self.assertEqual(
            calls,
            [("pause", session), ("resume", session), ("cancel", session), ("speed", 1.5)],
        )

    async def test_mark_owner_stopped_and_destructor_swallow_task_failures(self):
        session = self._make_internal_session(SimpleNamespace())
        session._task = asyncio.current_task()

        session._mark_owner_stopped()

        self.assertEqual(session.state, "owner_stopped")
        self.assertIsInstance(session._error, SessionClosedError)

        class _BadTask:
            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                raise RuntimeError("boom")

        session._task = _BadTask()
        session.__del__()
