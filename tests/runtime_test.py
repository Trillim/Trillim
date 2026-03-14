# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the Runtime lifecycle manager and sync bridges."""

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import threading
import unittest

from fastapi import APIRouter

from trillim import Runtime
import trillim.runtime as runtime_module
from trillim.server._component import Component


class LLM(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("llm.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("llm.stop")
        self.started = False

    async def chat(self, messages: list[dict]) -> str:
        self.calls.append(("llm.chat", tuple(m["content"] for m in messages)))
        return "reply"

    async def stream_chat(self, messages: list[dict]):
        self.calls.append(("llm.stream_chat", tuple(m["content"] for m in messages)))
        yield "event-1"
        yield "event-2"

    def session(self, messages: list[dict] | None = None):
        copied = [{"role": m["role"], "content": m["content"]} for m in messages or []]
        self.calls.append(("llm.session", tuple(m["content"] for m in copied)))
        return _ChatSession(self.calls, copied)

    @property
    def max_context_tokens(self) -> int:
        return 128

    def fail(self) -> None:
        raise ValueError("boom")


class Whisper(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("whisper.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("whisper.stop")
        self.started = False

    async def transcribe(self, audio_bytes: bytes) -> str:
        self.calls.append(("whisper.transcribe", audio_bytes))
        return audio_bytes.decode()


class TTS(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("tts.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("tts.stop")
        self.started = False

    def speak(self, text: str):
        self.calls.append(("tts.speak", text))
        return _TTSSession(self.calls, text)


class _TTSSession:
    _runtime_proxy = True

    def __init__(self, calls: list, text: str):
        self.calls = calls
        self._loop = asyncio.get_running_loop()
        self.state = "running"
        self.speed = 1.0
        self._chunks = [text.encode(), b"!"]

    def pause(self) -> None:
        self._schedule(self._pause)

    def _pause(self) -> None:
        self.calls.append("session.pause")
        self.state = "paused"

    def resume(self) -> None:
        self._schedule(self._resume)

    def _resume(self) -> None:
        self.calls.append("session.resume")
        self.state = "running"

    def stop(self) -> None:
        self._schedule(self._stop)

    def _stop(self) -> None:
        self.calls.append("session.stop")
        self.state = "cancelled"

    def set_speed(self, speed: float) -> None:
        self._schedule(self._set_speed, speed)

    def _set_speed(self, speed: float) -> None:
        self.calls.append(("session.set_speed", speed))
        self.speed = speed

    def _schedule(self, callback, *args) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            callback(*args)
            return
        self._loop.call_soon_threadsafe(callback, *args)

    async def collect(self) -> bytes:
        self.calls.append("session.collect")
        return b"".join(self._chunks)

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _ChatSession:
    _runtime_proxy = True

    def __init__(self, calls: list, messages: list[dict]):
        self.calls = calls
        self._messages = list(messages)

    @property
    def messages(self):
        return tuple({"role": m["role"], "content": m["content"]} for m in self._messages)

    @property
    def prompt_tokens(self) -> int:
        self.calls.append(("session.prompt_tokens", len(self._messages)))
        return len(self._messages)

    @property
    def max_context_tokens(self) -> int:
        return 128

    @property
    def remaining_context_tokens(self) -> int:
        self.calls.append(("session.remaining_context_tokens", len(self._messages)))
        return 128 - len(self._messages)

    def validate(self) -> int:
        self.calls.append(("session.validate", len(self._messages)))
        return len(self._messages)

    def add_user(self, content: str) -> None:
        self.calls.append(("session.add_user", content))
        self._messages.append({"role": "user", "content": content})

    def fail(self) -> None:
        raise ValueError("session boom")

    async def chat(self) -> str:
        self.calls.append(("session.chat", tuple(m["content"] for m in self._messages)))
        self._messages.append({"role": "assistant", "content": "session-reply"})
        return "session-reply"

    async def stream_chat(self):
        self.calls.append(("session.stream_chat", tuple(m["content"] for m in self._messages)))
        yield "session-event-1"
        yield "session-event-2"


class BrokenWhisper(Whisper):
    async def start(self) -> None:
        self.calls.append("whisper.start")
        raise RuntimeError("whisper start failed")


class BrokenStopLLM(LLM):
    async def stop(self) -> None:
        self.calls.append("llm.stop")
        raise RuntimeError("llm stop failed")


class RuntimeTests(unittest.TestCase):
    def test_runtime_requires_components(self):
        with self.assertRaisesRegex(ValueError, "Runtime requires at least one component"):
            Runtime()

    def test_runtime_starts_in_order_and_stops_in_reverse(self):
        calls: list = []
        runtime = Runtime(LLM(calls), Whisper(calls), TTS(calls))

        runtime.start()
        runtime.stop()

        self.assertEqual(
            calls,
            [
                "llm.start",
                "whisper.start",
                "tts.start",
                "tts.stop",
                "whisper.stop",
                "llm.stop",
            ],
        )

    def test_runtime_context_manager_and_sync_wrappers(self):
        calls: list = []
        messages = [{"role": "user", "content": "hello"}]
        llm = LLM(calls)
        whisper = Whisper(calls)
        tts = TTS(calls)

        with Runtime(llm, whisper, tts) as runtime:
            self.assertTrue(runtime.started)
            self.assertEqual(runtime.components, (llm, whisper, tts))
            self.assertIn("llm", dir(runtime))
            self.assertIsInstance(runtime.llm, runtime_module._RuntimeComponentProxy)
            self.assertIsInstance(runtime.whisper, runtime_module._RuntimeComponentProxy)
            self.assertIsInstance(runtime.tts, runtime_module._RuntimeComponentProxy)
            self.assertEqual(runtime.llm.chat(messages), "reply")
            self.assertEqual(list(runtime.llm.stream_chat(messages)), ["event-1", "event-2"])
            self.assertEqual(runtime.llm.max_context_tokens, 128)
            llm_session = runtime.llm.session(messages)
            self.assertIsInstance(llm_session, runtime_module._RuntimeObjectProxy)
            self.assertEqual(llm_session.messages, ({"role": "user", "content": "hello"},))
            self.assertEqual(llm_session.prompt_tokens, 1)
            self.assertEqual(llm_session.max_context_tokens, 128)
            self.assertEqual(llm_session.remaining_context_tokens, 127)
            self.assertEqual(llm_session.validate(), 1)
            llm_session.add_user("again")
            with self.assertRaisesRegex(ValueError, "session boom"):
                llm_session.fail()
            self.assertEqual(list(llm_session.stream_chat()), ["session-event-1", "session-event-2"])
            self.assertEqual(llm_session.chat(), "session-reply")
            self.assertEqual(runtime.whisper.transcribe(b"audio"), "audio")
            session = runtime.tts.speak("hello")
            self.assertIsInstance(session, runtime_module._RuntimeObjectProxy)
            self.assertEqual(session.state, "running")
            session.pause()
            self.assertEqual(session.state, "paused")
            session.resume()
            self.assertEqual(session.state, "running")
            session.set_speed(1.5)
            self.assertEqual(session.speed, 1.5)
            self.assertEqual(list(session), [b"hello", b"!"])
            self.assertEqual(session.collect(), b"hello!")
            session.stop()
            self.assertEqual(session.state, "cancelled")

        self.assertFalse(runtime.started)
        self.assertIn(("llm.chat", ("hello",)), calls)
        self.assertIn(("llm.stream_chat", ("hello",)), calls)
        self.assertIn(("llm.session", ("hello",)), calls)
        self.assertIn(("session.prompt_tokens", 1), calls)
        self.assertIn(("session.remaining_context_tokens", 1), calls)
        self.assertIn(("session.validate", 1), calls)
        self.assertIn(("session.add_user", "again"), calls)
        self.assertIn(("session.stream_chat", ("hello", "again")), calls)
        self.assertIn(("session.chat", ("hello", "again")), calls)
        self.assertIn(("whisper.transcribe", b"audio"), calls)
        self.assertIn(("tts.speak", "hello"), calls)
        self.assertIn("session.pause", calls)
        self.assertIn("session.resume", calls)
        self.assertIn(("session.set_speed", 1.5), calls)
        self.assertIn("session.collect", calls)
        self.assertIn("session.stop", calls)

    def test_runtime_start_is_idempotent(self):
        calls: list = []
        runtime = Runtime(LLM(calls), Whisper(calls))

        self.assertIs(runtime.start(), runtime)
        self.assertIs(runtime.start(), runtime)
        runtime.stop()

        self.assertEqual(calls, ["llm.start", "whisper.start", "whisper.stop", "llm.stop"])

    def test_runtime_rolls_back_started_components_when_start_fails(self):
        calls: list = []
        runtime = Runtime(LLM(calls), BrokenWhisper(calls), TTS(calls))

        with self.assertRaisesRegex(RuntimeError, "whisper start failed"):
            runtime.start()

        self.assertFalse(runtime.started)
        self.assertEqual(calls, ["llm.start", "whisper.start", "llm.stop"])

    def test_runtime_suppresses_rollback_stop_errors_after_start_failure(self):
        calls: list = []
        runtime = Runtime(BrokenStopLLM(calls), BrokenWhisper(calls))

        with self.assertRaisesRegex(RuntimeError, "whisper start failed"):
            runtime.start()

        self.assertFalse(runtime.started)
        self.assertEqual(calls, ["llm.start", "whisper.start", "llm.stop"])

    def test_runtime_rejects_duplicate_component_types(self):
        with self.assertRaisesRegex(ValueError, "Duplicate component type"):
            Runtime(LLM([]), LLM([]))

    def test_runtime_rejects_unknown_component_attribute(self):
        runtime = Runtime(LLM([]))

        with self.assertRaisesRegex(AttributeError, "has no attribute 'missing'"):
            runtime.missing

    def test_runtime_method_errors_propagate(self):
        runtime = Runtime(LLM([]))
        runtime.start()
        try:
            with self.assertRaisesRegex(ValueError, "boom"):
                runtime.llm.fail()
        finally:
            runtime.stop()

    def test_runtime_requires_start_before_component_use(self):
        runtime = Runtime(LLM([]))

        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime.llm.max_context_tokens

        coro = asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime._submit_to_loop(coro)
        coro.close()

        coro = asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime._submit_coroutine(coro)
        coro.close()

    def test_runtime_stop_is_safe_before_start_and_shutdown_loop_is_idempotent(self):
        runtime = Runtime(LLM([]))

        runtime.stop()
        runtime._shutdown_loop()

        self.assertFalse(runtime.started)
        self.assertIsNone(runtime._loop)
        self.assertIsNone(runtime._thread)

    def test_runtime_stop_propagates_component_stop_errors(self):
        calls: list = []
        runtime = Runtime(BrokenStopLLM(calls), Whisper(calls))
        runtime.start()

        with self.assertRaisesRegex(RuntimeError, "llm stop failed"):
            runtime.stop()

        self.assertFalse(runtime.started)
        self.assertEqual(calls, ["llm.start", "whisper.start", "whisper.stop", "llm.stop"])

    def test_runtime_stop_cancels_pending_loop_tasks(self):
        runtime = Runtime(LLM([]))
        runtime.start()
        try:
            future = runtime._submit_to_loop(asyncio.sleep(60))
        finally:
            runtime.stop()

        with self.assertRaises(FutureCancelledError):
            future.result()

    def test_sync_async_iterator_close_and_del_are_tolerant(self):
        calls: list = []
        runtime = Runtime(LLM(calls))
        runtime.start()
        iterator = runtime.llm.stream_chat([{"role": "user", "content": "hello"}])
        self.assertEqual(next(iterator), "event-1")
        runtime.stop()

        iterator.close()
        iterator.close()

        with self.assertRaises(StopIteration):
            next(iterator)

        raw_iterator = runtime_module._SyncAsyncIterator(runtime, _AsyncIteratorStub())
        raw_iterator.close = _raise_close
        raw_iterator.__del__()

    def test_runtime_object_proxy_rejects_non_async_iterables(self):
        proxy = runtime_module._RuntimeObjectProxy(Runtime(LLM([])), object())

        with self.assertRaisesRegex(TypeError, "is not iterable"):
            iter(proxy)

    def test_runtime_object_proxy_rechecks_iterability_on_runtime_loop(self):
        runtime = Runtime(LLM([]))
        runtime.start()
        try:
            proxy = runtime_module._RuntimeObjectProxy(
                runtime,
                _ThreadBoundAsyncIterable(threading.get_ident()),
            )

            with self.assertRaisesRegex(TypeError, "is not iterable"):
                iter(proxy)
        finally:
            runtime.stop()

    def test_invoke_attr_async_handles_noncallable_attributes(self):
        component = LLM([])
        runtime = Runtime(component)

        self.assertFalse(asyncio.run(runtime._invoke_attr_async(component, "started", (), {})))

        with self.assertRaisesRegex(TypeError, "'started' is not callable"):
            asyncio.run(runtime._invoke_attr_async(component, "started", (1,), {}))

    def test_invoke_managed_helper_requires_started(self):
        component = LLM([])
        runtime = Runtime(component)

        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime._invoke_managed_attr(component, "session", ([],), {})

        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime._invoke_managed_attr(object(), "__str__", (), {})

    def test_syncify_result_awaits_coroutines(self):
        runtime = Runtime(LLM([]))
        runtime.start()
        try:
            self.assertEqual(runtime._syncify_result(asyncio.sleep(0, result="done")), "done")
        finally:
            runtime.stop()


class _AsyncIteratorStub:
    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        return None


class _ThreadBoundAsyncIterable:
    def __init__(self, visible_thread_id: int):
        self._visible_thread_id = visible_thread_id

    def __getattribute__(self, name: str):
        if name == "__aiter__":
            visible_thread_id = object.__getattribute__(self, "_visible_thread_id")
            if threading.get_ident() != visible_thread_id:
                raise AttributeError(name)
            return object.__getattribute__(self, "_aiter_impl")
        return object.__getattribute__(self, name)

    def _aiter_impl(self):
        return _AsyncIteratorStub()


def _raise_close():
    raise ValueError("boom")


if __name__ == "__main__":
    unittest.main()
