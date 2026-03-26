"""Tests for the runtime sync facade."""

from __future__ import annotations

import asyncio
import concurrent.futures
import unittest

from trillim.components import Component
from trillim.runtime import Runtime, _RuntimeObjectProxy, _SyncAsyncIterator


class _Session:
    _runtime_proxy = True

    def __init__(self, calls: list):
        self.calls = calls
        self.state = "ready"

    async def ping(self) -> str:
        self.calls.append("session.ping")
        return "pong"

    async def __aiter__(self):
        self.calls.append("session.__aiter__")
        yield b"a"
        yield b"b"

    async def __aenter__(self):
        self.calls.append("session.__aenter__")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.calls.append("session.__aexit__")
        return False


class _EchoComponent(Component):
    def __init__(self, calls: list[str], name: str = "echo") -> None:
        self.calls = calls
        self._component_name = name
        self.started = False

    @property
    def component_name(self) -> str:
        return self._component_name

    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")
        self.started = False

    async def ping(self, value: str) -> str:
        self.calls.append(f"{self.component_name}.ping:{value}")
        return value.upper()

    async def stream(self):
        self.calls.append(f"{self.component_name}.stream")
        yield "alpha"
        yield "beta"

    def session(self):
        self.calls.append(f"{self.component_name}.session")
        return _Session(self.calls)

    def sync_session(self):
        self.calls.append(f"{self.component_name}.sync_session")
        return _SyncSession(self.calls)

    def bare_session(self):
        self.calls.append(f"{self.component_name}.bare_session")
        return _BareManaged()


class _BrokenComponent(_EchoComponent):
    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        raise RuntimeError("boom")


class _BrokenStopComponent(_EchoComponent):
    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")
        raise RuntimeError(f"{self.component_name} stop boom")


class _SyncSession:
    _runtime_proxy = True

    def __init__(self, calls: list[str]):
        self.calls = calls
        self.state = "sync-ready"

    def __enter__(self):
        self.calls.append("sync_session.__enter__")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append("sync_session.__exit__")
        return True


class _BareManaged:
    _runtime_proxy = True

    def __init__(self):
        self.state = "bare"


class RuntimeTests(unittest.TestCase):
    def test_runtime_requires_components(self):
        with self.assertRaisesRegex(ValueError, "at least one component"):
            Runtime()

    def test_runtime_rejects_duplicate_component_names(self):
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "Duplicate component name"):
            Runtime(_EchoComponent(calls), _EchoComponent(calls))

    def test_runtime_starts_and_stops_in_order(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls, "one"), _EchoComponent(calls, "two")) as runtime:
            self.assertTrue(runtime.started)
        self.assertEqual(
            calls,
            ["one.start", "two.start", "two.stop", "one.stop"],
        )

    def test_runtime_syncifies_async_methods_iterators_and_runtime_objects(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls)) as runtime:
            self.assertEqual(runtime.echo.ping("hello"), "HELLO")
            self.assertEqual(list(runtime.echo.stream()), ["alpha", "beta"])
            session = runtime.echo.session()
            self.assertIsInstance(session, _RuntimeObjectProxy)
            self.assertEqual(session.ping(), "pong")
            self.assertEqual(list(session), [b"a", b"b"])
            with runtime.echo.session() as managed:
                self.assertEqual(managed.state, "ready")
        self.assertIn("echo.session", calls)
        self.assertIn("session.ping", calls)
        self.assertIn("session.__aenter__", calls)
        self.assertIn("session.__aexit__", calls)

    def test_runtime_stops_started_components_if_startup_fails(self):
        calls: list[str] = []
        runtime = Runtime(_EchoComponent(calls, "one"), _BrokenComponent(calls, "two"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            runtime.start()
        self.assertEqual(calls, ["one.start", "two.start", "one.stop"])
        self.assertFalse(runtime.started)

    def test_runtime_exposes_components_dir_and_attribute_errors(self):
        calls: list[str] = []
        runtime = Runtime(_EchoComponent(calls, "echo"))

        self.assertEqual(runtime.components[0].component_name, "echo")
        self.assertIn("echo", dir(runtime))
        with self.assertRaisesRegex(AttributeError, "has no attribute 'missing'"):
            runtime.missing

    def test_runtime_start_stop_submit_and_shutdown_guards(self):
        calls: list[str] = []
        runtime = Runtime(_EchoComponent(calls))

        runtime._shutdown_loop()
        runtime.stop()
        self.assertFalse(runtime.started)

        self.assertIs(runtime.start(), runtime)
        self.assertIs(runtime.start(), runtime)
        runtime.stop()
        runtime.stop()

        submit_coro = asyncio.sleep(0)
        try:
            with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
                runtime._submit_to_loop(submit_coro)
        finally:
            submit_coro.close()

        syncify_coro = asyncio.sleep(0)
        try:
            with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
                runtime._submit_coroutine(syncify_coro)
        finally:
            syncify_coro.close()

    def test_runtime_iterator_close_and_destructor_swallow_errors(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls)) as runtime:
            iterator = runtime.echo.stream()
            self.assertEqual(next(iterator), "alpha")

        iterator.close()
        with self.assertRaises(StopIteration):
            next(iterator)

        async def _empty():
            if False:
                yield None

        class _BadRuntime:
            def _submit_coroutine(self, coro):
                coro.close()
                raise ValueError("boom")

        _SyncAsyncIterator(_BadRuntime(), _empty()).__del__()

    def test_runtime_object_proxy_supports_sync_contexts_non_iterables_and_awaitables(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls)) as runtime:
            with runtime.echo.sync_session() as managed:
                self.assertEqual(managed.state, "sync-ready")
            bare = runtime.echo.bare_session()
            self.assertIs(bare.__enter__(), bare)
            self.assertFalse(bare.__exit__(None, None, None))
            with self.assertRaisesRegex(TypeError, "not iterable"):
                list(bare)
            self.assertEqual(runtime._syncify_result(asyncio.sleep(0, result="done")), "done")
        self.assertIn("sync_session.__enter__", calls)
        self.assertIn("sync_session.__exit__", calls)

    def test_runtime_stop_propagates_stop_errors_and_cleans_up_pending_tasks(self):
        calls: list[str] = []
        runtime = Runtime(
            _EchoComponent(calls, "one"),
            _BrokenStopComponent(calls, "two"),
            _BrokenStopComponent(calls, "three"),
        )
        runtime.start()
        pending = runtime._submit_to_loop(asyncio.sleep(3600))

        with self.assertRaisesRegex(RuntimeError, "three stop boom"):
            runtime.stop()

        self.assertFalse(runtime.started)
        self.assertIsNone(runtime._loop)
        self.assertIsNone(runtime._thread)
        self.assertTrue(pending.done())
        with self.assertRaises((asyncio.CancelledError, concurrent.futures.CancelledError)):
            pending.result()

    def test_runtime_startup_cleanup_swallows_stop_failures(self):
        calls: list[str] = []
        runtime = Runtime(_BrokenStopComponent(calls, "one"), _BrokenComponent(calls, "two"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            runtime.start()

        self.assertEqual(calls, ["one.start", "two.start", "one.stop"])
        self.assertFalse(runtime.started)
