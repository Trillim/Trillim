"""Tests for STT temp-file normalization helpers."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.stt import _spool as stt_spool
from trillim.components.stt._config import OwnedAudioInput, SourceFileSnapshot
from trillim.components.stt._spool import (
    _copy_source_file_sync,
    copy_source_file,
    spool_audio_bytes,
    spool_request_stream,
)
from trillim.errors import InvalidRequestError
from tests.components.stt.support import list_spool_files


class STTSpoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_spool_request_stream_copies_chunks_without_buffering_the_whole_body(self):
        async def chunks():
            yield b"ab"
            yield b"cd"
            yield b""

        owned = await spool_request_stream(chunks(), spool_dir=self.spool_dir)
        self.assertEqual(owned.size_bytes, 4)
        self.assertEqual(owned.path.read_bytes(), b"abcd")

    async def test_spool_request_stream_rejects_oversize_payload_and_cleans_up(self):
        async def chunks():
            yield b"ab"
            yield b"cd"

        with patch("trillim.components.stt._spool.MAX_UPLOAD_BYTES", new=3):
            with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
                await spool_request_stream(chunks(), spool_dir=self.spool_dir)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_spool_request_stream_cleans_up_when_cancelled(self):
        async def chunks():
            yield b"ab"
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await spool_request_stream(chunks(), spool_dir=self.spool_dir)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_spool_audio_bytes_creates_owned_temp_file(self):
        owned = await spool_audio_bytes(b"payload", spool_dir=self.spool_dir)
        self.assertTrue(owned.path.exists())
        self.assertEqual(owned.path.read_bytes(), b"payload")

    async def test_spool_audio_bytes_cleans_up_temp_file_when_write_fails(self):
        with patch(
            "trillim.components.stt._spool.os.fdopen",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                await spool_audio_bytes(b"payload", spool_dir=self.spool_dir)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_copy_source_file_creates_owned_copy(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"source-bytes")
        owned = await copy_source_file(source, spool_dir=self.spool_dir)
        self.assertNotEqual(owned.path, source)
        self.assertEqual(owned.path.read_bytes(), b"source-bytes")

    async def test_copy_source_file_rejects_changed_metadata_and_cleans_up(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"source-bytes")
        with patch(
            "trillim.components.stt._spool.snapshot_source_file",
            side_effect=[
                SourceFileSnapshot(size_bytes=12, modified_ns=1),
                SourceFileSnapshot(size_bytes=12, modified_ns=2),
            ],
        ):
            with self.assertRaisesRegex(InvalidRequestError, "changed while it was being copied"):
                await copy_source_file(source, spool_dir=self.spool_dir)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_copy_source_file_cleans_up_owned_temp_when_cancelled(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abcdef")
        started = threading.Event()
        original_read = stt_spool._read_source_chunk

        def slow_read(source_handle):
            chunk = original_read(source_handle)
            if chunk and not started.is_set():
                started.set()
                time.sleep(0.05)
            return chunk

        with patch("trillim.components.stt._spool.SPOOL_CHUNK_SIZE_BYTES", 1), patch(
            "trillim.components.stt._spool._read_source_chunk",
            side_effect=slow_read,
        ):
            task = asyncio.create_task(copy_source_file(source, spool_dir=self.spool_dir))
            await asyncio.to_thread(started.wait)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_copy_source_file_cleans_up_completed_owned_copy_if_outer_task_is_cancelled(self):
        owned_path = self.spool_dir / "finished.audio"
        owned_path.parent.mkdir(parents=True, exist_ok=True)
        owned_path.write_bytes(b"abcdef")
        fake_task = object()

        async def first_cancel_then_return(awaitable):
            self.assertIs(awaitable, fake_task)
            if not getattr(first_cancel_then_return, "raised", False):
                first_cancel_then_return.raised = True
                raise asyncio.CancelledError()
            return OwnedAudioInput(owned_path, 6)

        def tracked_create_task(awaitable):
            awaitable.close()
            return fake_task

        with patch(
            "trillim.components.stt._spool.open_validated_source_file",
            return_value=123,
        ), patch(
            "trillim.components.stt._spool.asyncio.create_task",
            side_effect=tracked_create_task,
        ), patch(
            "trillim.components.stt._spool.asyncio.shield",
            side_effect=first_cancel_then_return,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await copy_source_file(Path("/tmp/source.wav"), spool_dir=self.spool_dir)

        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_copy_source_file_swallows_copy_task_failures_during_cancellation_cleanup(self):
        fake_task = object()

        async def cancel_then_fail(awaitable):
            self.assertIs(awaitable, fake_task)
            if not getattr(cancel_then_fail, "raised", False):
                cancel_then_fail.raised = True
                raise asyncio.CancelledError()
            raise RuntimeError("copy boom")

        def tracked_create_task(awaitable):
            awaitable.close()
            return fake_task

        with patch(
            "trillim.components.stt._spool.open_validated_source_file",
            return_value=123,
        ), patch(
            "trillim.components.stt._spool.asyncio.create_task",
            side_effect=tracked_create_task,
        ), patch(
            "trillim.components.stt._spool.asyncio.shield",
            side_effect=cancel_then_fail,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await copy_source_file(Path("/tmp/source.wav"), spool_dir=self.spool_dir)

    async def test_copy_source_file_swallows_real_thread_failures_during_cancellation_cleanup(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"source-bytes")
        started = threading.Event()

        def failing_copy(source_fd: int, spool_dir: Path, cancel_event) -> object:
            del source_fd, spool_dir
            started.set()
            while not cancel_event.is_set():
                time.sleep(0.01)
            raise RuntimeError("copy boom")

        with patch(
            "trillim.components.stt._spool._copy_source_file_sync",
            side_effect=failing_copy,
        ):
            task = asyncio.create_task(copy_source_file(source, spool_dir=self.spool_dir))
            await asyncio.to_thread(started.wait)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_copy_source_file_rejects_missing_source_before_copy(self):
        missing = Path(self._temp_dir.name) / "missing.wav"
        with self.assertRaisesRegex(InvalidRequestError, "does not exist"):
            await copy_source_file(missing, spool_dir=self.spool_dir)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_copy_source_file_uses_opened_descriptor_if_path_is_replaced(self):
        source = Path(self._temp_dir.name) / "source.wav"
        replacement = Path(self._temp_dir.name) / "replacement.wav"
        source.write_bytes(b"original")
        replacement.write_bytes(b"replacement")
        entered = threading.Event()
        proceed = threading.Event()
        original_copy = stt_spool._copy_source_file_sync

        def delayed_copy(source_fd: int, spool_dir: Path, cancel_event) -> object:
            entered.set()
            proceed.wait()
            return original_copy(source_fd, spool_dir, cancel_event)

        with patch(
            "trillim.components.stt._spool._copy_source_file_sync",
            side_effect=delayed_copy,
        ):
            task = asyncio.create_task(copy_source_file(source, spool_dir=self.spool_dir))
            await asyncio.to_thread(entered.wait)
            source.unlink()
            replacement.replace(source)
            proceed.set()
            owned = await task

        self.assertEqual(owned.path.read_bytes(), b"original")
        owned.path.unlink()

    async def test_copy_source_file_closes_source_fd_if_temp_creation_fails(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"original")
        source_fd = stt_spool.open_validated_source_file(source)
        original_close = os.close
        closed_fds: list[int] = []

        def tracked_close(fd: int) -> None:
            closed_fds.append(fd)
            original_close(fd)

        with patch(
            "trillim.components.stt._spool._create_owned_temp_file",
            side_effect=OSError("disk full"),
        ), patch(
            "trillim.components.stt._spool.os.close",
            side_effect=tracked_close,
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                stt_spool._copy_source_file_sync(source_fd, self.spool_dir, None)

        self.assertIn(source_fd, closed_fds)
        with self.assertRaises(OSError):
            os.fstat(source_fd)

    async def test_copy_source_file_sync_covers_oversize_and_temp_fd_cleanup(self):
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abcdef")
        source_fd = stt_spool.open_validated_source_file(source)

        with patch("trillim.components.stt._spool.MAX_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
                _copy_source_file_sync(source_fd, self.spool_dir, None)

        source_fd = stt_spool.open_validated_source_file(source)
        original_fdopen = os.fdopen
        original_close = os.close
        closed: list[int] = []

        def tracked_close(fd: int) -> None:
            closed.append(fd)
            original_close(fd)

        call_count = {"value": 0}

        def fail_second_fdopen(fd: int, mode: str, *args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 2:
                raise OSError("temp open boom")
            return original_fdopen(fd, mode, *args, **kwargs)

        with patch(
            "trillim.components.stt._spool.os.fdopen",
            side_effect=fail_second_fdopen,
        ), patch(
            "trillim.components.stt._spool.os.close",
            side_effect=tracked_close,
        ):
            with self.assertRaisesRegex(OSError, "temp open boom"):
                _copy_source_file_sync(source_fd, self.spool_dir, None)

        self.assertTrue(closed)
