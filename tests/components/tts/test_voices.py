"""Tests for the managed TTS voice store."""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trillim.components.tts._voices import (
    ManagedVoiceEntry,
    VoiceStore,
    VoiceStoreTamperedError,
    _copy_source_audio_sync,
    _storage_id_for_name,
    copy_source_audio,
    spool_request_voice_stream,
    spool_voice_state_bytes,
    spool_voice_bytes,
)
from trillim.errors import InvalidRequestError


class _UnsafeState:
    pass


def _valid_state_bytes() -> bytes:
    buffer = io.BytesIO()
    torch.save({"layer": {"cache": torch.tensor([1.0])}}, buffer)
    return buffer.getvalue()


class _Chunks:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class TTSVoiceStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"
        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_register_list_resolve_and_delete_custom_voice(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(audio_path: Path) -> bytes:
            self.assertEqual(audio_path.read_bytes(), b"voice")
            return _valid_state_bytes()

        try:
            self.assertEqual(
                await self.store.register_owned_upload(
                    name="custom",
                    upload=upload,
                    build_voice_state=fake_builder,
                ),
                "custom",
            )
        finally:
            upload.path.unlink(missing_ok=True)

        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])
        resolved = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        self.assertEqual(resolved.kind, "state_file")
        self.assertTrue(Path(resolved.reference).exists())
        Path(resolved.reference).unlink(missing_ok=True)

        self.assertEqual(await self.store.delete("custom"), "custom")
        self.assertEqual(await self.store.list_names(), ["alba", "marius"])

    async def test_duplicate_or_builtin_name_is_rejected(self):
        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await self.store.ensure_name_available("alba")

    async def test_delete_rejects_default_or_missing_voice(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()
        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)
        with self.assertRaisesRegex(InvalidRequestError, "default_voice"):
            await self.store.delete("custom", protected_name="custom")
        with self.assertRaises(KeyError):
            await self.store.delete("missing")

    async def test_delete_failure_keeps_manifest_and_state_consistent(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)

        with patch(
            "trillim.components.tts._voices.unlink_if_exists",
            side_effect=PermissionError("read only"),
        ):
            with self.assertRaisesRegex(PermissionError, "read only"):
                await self.store.delete("custom")
        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])
        resolved = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        Path(resolved.reference).unlink(missing_ok=True)

    async def test_register_owned_upload_rejects_malformed_worker_state_and_cleans_up_publish_failures(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        try:
            with self.assertRaisesRegex(RuntimeError, "malformed voice state"):
                await self.store.register_owned_upload(
                    name="custom",
                    upload=upload,
                    build_voice_state=lambda _path: asyncio.sleep(0, result=b"not-torch"),
                )
        finally:
            upload.path.unlink(missing_ok=True)

        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        try:
            with patch.object(
                self.store,
                "_write_manifest_locked",
                side_effect=OSError("manifest boom"),
            ):
                with self.assertRaisesRegex(OSError, "manifest boom"):
                    await self.store.register_owned_upload(
                        name="custom",
                        upload=upload,
                        build_voice_state=fake_builder,
                    )
        finally:
            upload.path.unlink(missing_ok=True)

        self.assertEqual(await self.store.list_names(), ["alba", "marius"])
        self.assertEqual(list(self.root.glob("*.state")), [])

    async def test_voice_store_limits_and_delete_rollback_are_enforced(self):
        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        try:
            with patch("trillim.components.tts._voices.MAX_CUSTOM_VOICES", 0):
                with self.assertRaisesRegex(InvalidRequestError, "already contains"):
                    await self.store.register_owned_upload(
                        name="custom",
                        upload=upload,
                        build_voice_state=fake_builder,
                    )
        finally:
            upload.path.unlink(missing_ok=True)

        state_bytes = _valid_state_bytes()
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        try:
            with patch(
                "trillim.components.tts._voices.MAX_TOTAL_CUSTOM_VOICE_BYTES",
                len(state_bytes) - 1,
            ):
                with self.assertRaisesRegex(InvalidRequestError, "storage exceeds"):
                    await self.store.register_owned_upload(
                        name="custom",
                        upload=upload,
                        build_voice_state=fake_builder,
                    )
        finally:
            upload.path.unlink(missing_ok=True)

        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)
        state_path = self.root / f"{_storage_id_for_name('custom')}.state"
        original_state = state_path.read_bytes()

        with patch.object(
            self.store,
            "_write_manifest_locked",
            side_effect=OSError("manifest boom"),
        ):
            with self.assertRaisesRegex(OSError, "manifest boom"):
                await self.store.delete("custom")

        self.assertEqual(state_path.read_bytes(), original_state)
        self.assertIn("custom", await self.store.list_names())

    async def test_spool_request_stream_and_copy_source_audio(self):
        owned = await spool_request_voice_stream(_Chunks([b"a", b"b"]), spool_dir=self.spool_dir)
        self.assertEqual(owned.size_bytes, 2)
        self.assertEqual(owned.path.read_bytes(), b"ab")
        owned.path.unlink(missing_ok=True)

        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"hello")
        copied = await copy_source_audio(source, spool_dir=self.spool_dir)
        self.assertEqual(copied.path.read_bytes(), b"hello")
        copied.path.unlink(missing_ok=True)

    async def test_copy_source_audio_rejects_empty_string_paths_before_touching_cwd(self):
        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            await copy_source_audio("", spool_dir=self.spool_dir)

    async def test_tampered_manifest_disables_custom_voice_functionality(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": "../escape",
                            "size_bytes": 4,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_manifest_with_path_like_voice_name_fails_closed(self):
        for name in ("../escape", "bad-name"):
            with self.subTest(name=name):
                state_bytes = _valid_state_bytes()
                storage_id = _storage_id_for_name(name)
                self.root.mkdir(parents=True, exist_ok=True)
                (self.root / f"{storage_id}.state").write_bytes(state_bytes)
                (self.root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "voices": [
                                {
                                    "name": name,
                                    "storage_id": storage_id,
                                    "size_bytes": len(state_bytes),
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
                    await self.store.list_names()
                self.store = VoiceStore(
                    self.root,
                    built_in_voice_names=("alba", "marius"),
                )
                for child in self.root.iterdir():
                    child.unlink()

    async def test_store_methods_reject_non_alphanumeric_voice_names(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fail_builder(_path: Path) -> bytes:
            raise AssertionError("build_voice_state should not be called")

        try:
            with self.assertRaisesRegex(
                InvalidRequestError,
                "must contain only letters and digits",
            ):
                await self.store.ensure_name_available("bad-name")
            with self.assertRaisesRegex(
                InvalidRequestError,
                "must contain only letters and digits",
            ):
                await self.store.register_owned_upload(
                    name="bad-name",
                    upload=upload,
                    build_voice_state=fail_builder,
                )
            with self.assertRaisesRegex(
                InvalidRequestError,
                "must contain only letters and digits",
            ):
                await self.store.delete("bad-name")
            with self.assertRaisesRegex(
                InvalidRequestError,
                "must contain only letters and digits",
            ):
                await self.store.resolve_for_session(
                    "bad-name",
                    spool_dir=self.spool_dir,
                )
        finally:
            upload.path.unlink(missing_ok=True)

    async def test_non_directory_voice_store_root_fails_closed(self):
        self.root.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_missing_manifest_with_leftover_files_fails_closed(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{_storage_id_for_name('custom')}.state").write_bytes(_valid_state_bytes())
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("custom")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_tampered_state_file_disables_custom_voice_resolution(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)

        initial = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        Path(initial.reference).unlink(missing_ok=True)

        buffer = io.BytesIO()
        torch.save({"bad": _UnsafeState()}, buffer)
        state_path = self.root / f"{_storage_id_for_name('custom')}.state"
        state_path.write_bytes(buffer.getvalue())

        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_manifest_is_revalidated_after_initial_successful_read(self):
        storage_id = _storage_id_for_name("custom")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{storage_id}.state").write_bytes(_valid_state_bytes())
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": storage_id,
                            "size_bytes": len(_valid_state_bytes()),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])

        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": "../escape",
                            "size_bytes": len(_valid_state_bytes()),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")

    async def test_manifest_shape_and_inventory_failures_disable_custom_voices(self):
        self.root.mkdir(parents=True, exist_ok=True)

        for payload in (
            b"not-json",
            json.dumps([]).encode("utf-8"),
            json.dumps({"voices": {}}).encode("utf-8"),
        ):
            with self.subTest(payload=payload):
                (self.root / "manifest.json").write_bytes(payload)
                with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
                    await self.store.list_names()
                self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))

        state_bytes = _valid_state_bytes()
        storage_id = _storage_id_for_name("custom")
        (self.root / f"{storage_id}.state").write_bytes(state_bytes)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "alba",
                            "storage_id": _storage_id_for_name("alba"),
                            "size_bytes": len(state_bytes),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        for child in self.root.iterdir():
            child.unlink()
        self.root.mkdir(exist_ok=True)
        extra_state = self.root / "extra.state"
        extra_state.write_bytes(state_bytes)
        (self.root / "manifest.json").write_text(json.dumps({"voices": []}), encoding="utf-8")
        with self.assertRaisesRegex(VoiceStoreTamperedError, "unexpected files"):
            await self.store.list_names()

    async def test_internal_validation_helpers_cover_state_and_symlink_failures(self):
        self.root.mkdir(parents=True, exist_ok=True)
        valid_state = _valid_state_bytes()
        state_path = self.root / "state.state"
        state_path.write_bytes(valid_state)

        with patch(
            "trillim.components.tts._voices.validate_voice_state_bytes",
            side_effect=InvalidRequestError("bad state"),
        ):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
                self.store._load_state_bytes_locked(
                    ManagedVoiceEntry(name="custom", storage_id="state", size_bytes=len(valid_state))
                )

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            self.store._load_state_bytes_locked(
                ManagedVoiceEntry(name="custom", storage_id="state", size_bytes=len(valid_state) - 1)
            )

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        state_path.write_bytes(b"x")
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            self.store._load_state_bytes_locked(
                ManagedVoiceEntry(name="custom", storage_id="state", size_bytes=1)
            )

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        with self.assertRaisesRegex(VoiceStoreTamperedError, "missing"):
            self.store._validate_manifest_state_file_locked(
                ManagedVoiceEntry(name="custom", storage_id="missing", size_bytes=1)
            )

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        directory_state = self.root / "dir.state"
        directory_state.mkdir()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
            self.store._validate_manifest_state_file_locked(
                ManagedVoiceEntry(name="custom", storage_id="dir", size_bytes=1)
            )

        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        target = Path(self._temp_dir.name) / "voice-target"
        target.mkdir()
        symlink_root = Path(self._temp_dir.name) / "voice-link"
        symlink_root.symlink_to(target)
        symlink_store = VoiceStore(symlink_root, built_in_voice_names=("alba", "marius"))
        with self.assertRaisesRegex(VoiceStoreTamperedError, "symlinks"):
            await symlink_store.list_names()

        link = Path(self._temp_dir.name) / "file-link"
        target_file = Path(self._temp_dir.name) / "target.txt"
        target_file.write_text("x", encoding="utf-8")
        link.symlink_to(target_file)
        with self.assertRaisesRegex(RuntimeError, "no symlinks"):
            self.store._raise_if_symlink(link, "no symlinks")

    async def test_spooling_helpers_cleanup_temp_files_on_write_failures(self):
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        source_fd = os.open(source, os.O_RDONLY)
        original_close = os.close
        closed: list[int] = []

        def tracked_close(fd: int) -> None:
            closed.append(fd)
            original_close(fd)

        with patch(
            "trillim.components.tts._voices.os.fdopen",
            side_effect=OSError("fdopen boom"),
        ):
            with self.assertRaisesRegex(OSError, "fdopen boom"):
                await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
            with self.assertRaisesRegex(OSError, "fdopen boom"):
                await spool_voice_state_bytes(b"voice", spool_dir=self.spool_dir)
        self.assertEqual(list(self.spool_dir.glob("*")), [])

        with patch(
            "trillim.components.tts._voices.os.fdopen",
            side_effect=OSError("fdopen boom"),
        ), patch(
            "trillim.components.tts._voices.os.close",
            side_effect=tracked_close,
        ):
            with self.assertRaisesRegex(OSError, "fdopen boom"):
                _copy_source_audio_sync(source_fd, self.spool_dir)

        self.assertIn(source_fd, closed)

    async def test_internal_manifest_entry_and_directory_error_branches(self):
        self.root.mkdir(parents=True, exist_ok=True)
        original_iterdir = Path.iterdir

        def failing_iterdir(path: Path):
            if path == self.root:
                raise OSError("boom")
            return original_iterdir(path)

        with patch("pathlib.Path.iterdir", autospec=True, side_effect=failing_iterdir):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
                await self.store.list_names()

        file_root = Path(self._temp_dir.name) / "file-root"
        file_root.write_text("x", encoding="utf-8")
        file_store = VoiceStore(file_root, built_in_voice_names=("alba", "marius"))
        with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
            file_store._ensure_store_root_locked()

        for item in (
            123,
            {"name": "custom"},
            {"name": " custom ", "storage_id": "id", "size_bytes": 1},
            {"name": "custom", "storage_id": "id", "size_bytes": 0},
        ):
            store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
            with self.subTest(item=item):
                with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
                    store._load_manifest_entry_locked(item)

        original_stat = Path.stat
        err_path = self.root / "err.state"
        err_path.write_bytes(b"x")

        def failing_stat(path: Path, *args, **kwargs):
            if path == err_path:
                raise OSError("boom")
            return original_stat(path, *args, **kwargs)

        store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        with patch("pathlib.Path.is_symlink", return_value=False), patch(
            "pathlib.Path.stat",
            autospec=True,
            side_effect=failing_stat,
        ):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
                store._validate_manifest_state_file_locked(
                    ManagedVoiceEntry(name="custom", storage_id="err", size_bytes=1)
                )

        store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        with patch("pathlib.Path.iterdir", autospec=True, side_effect=failing_iterdir):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "malformed"):
                store._validate_store_inventory_locked({})

        store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))
        first = second = None
        try:
            store._mark_store_tampered_locked("first")
        except VoiceStoreTamperedError as exc:
            first = exc
        try:
            store._mark_store_tampered_locked("second")
        except VoiceStoreTamperedError as exc:
            second = exc
        self.assertIs(first, second)

    async def test_untracked_state_files_fail_closed_with_cleanup_guidance(self):
        storage_id = _storage_id_for_name("custom")
        self.root.mkdir(parents=True, exist_ok=True)
        state_bytes = _valid_state_bytes()
        (self.root / f"{storage_id}.state").write_bytes(state_bytes)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": storage_id,
                            "size_bytes": len(state_bytes),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (self.root / f"{_storage_id_for_name('orphan')}.state").write_bytes(state_bytes)

        with self.assertRaisesRegex(
            VoiceStoreTamperedError,
            r"Delete stale \.state files",
        ):
            await self.store.list_names()
        with self.assertRaisesRegex(
            VoiceStoreTamperedError,
            r"Delete stale \.state files",
        ):
            await self.store.ensure_name_available("fresh")

    async def test_resolve_unknown_voice_and_delete_builtin_voice_are_rejected(self):
        with self.assertRaisesRegex(InvalidRequestError, "unknown voice"):
            await self.store.resolve_for_session("missing", spool_dir=self.spool_dir)
        with self.assertRaisesRegex(InvalidRequestError, "built in"):
            await self.store.delete("alba")

    async def test_spool_request_stream_and_copy_source_audio_enforce_size_and_empty_limits(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            await spool_request_voice_stream(_Chunks([b"", b""]), spool_dir=self.spool_dir)
        self.assertEqual(list(self.spool_dir.glob("*")), [])

        with patch("trillim.components.tts._voices.MAX_VOICE_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
                await spool_request_voice_stream(
                    _Chunks([b"ab", b"cd"]),
                    spool_dir=self.spool_dir,
                )
        self.assertEqual(list(self.spool_dir.glob("*")), [])

        source = Path(self._temp_dir.name) / "large-voice.wav"
        source.write_bytes(b"abcdef")
        with patch("trillim.components.tts._voices.MAX_VOICE_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
                await copy_source_audio(source, spool_dir=self.spool_dir)
        self.assertEqual(list(self.spool_dir.glob("*")), [])
