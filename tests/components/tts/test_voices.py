from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.tts._limits import VOICE_MANIFEST_NAME
from trillim.components.tts._validation import save_voice_state_safetensors
from trillim.components.tts._voices import (
    VOICE_STATE_SUFFIX,
    VoiceStoreTamperedError,
    delete_custom_voice,
    load_custom_voice_states,
    publish_custom_voice,
)
from trillim.errors import InvalidRequestError

from tests.components.tts.support import fake_voice_state


class VoicePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name) / "voices"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_load_custom_voice_states_loads_only_valid_safetensors(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(fake_voice_state(), state_path)
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        states = await load_custom_voice_states(
            self.root,
            built_in_voice_names=("alba",),
        )

        self.assertEqual(list(states), ["custom"])
        self.assertEqual(states["custom"]["module"]["cache"].tolist(), [1.0])

    async def test_load_custom_voice_states_skips_legacy_and_invalid_files_with_warning(self):
        self.root.mkdir(parents=True)
        (self.root / "legacy.state").write_bytes(b"legacy")
        bad_storage_id = _storage_id_for_name("bad")
        bad_path = self.root / f"{bad_storage_id}{VOICE_STATE_SUFFIX}"
        bad_path.write_bytes(b"not safetensors")
        manifest = {
            "voices": [
                {
                    "name": "bad",
                    "storage_id": bad_storage_id,
                    "size_bytes": bad_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("legacy", "\n".join(logs.output))
        self.assertIn("valid safetensors", "\n".join(logs.output))

    async def test_publish_and_delete_custom_voice_update_disk_manifest(self):
        name, state = await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=fake_voice_state(),
            existing_names={"alba"},
        )

        self.assertEqual(name, "custom")
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])
        manifest_path = self.root / VOICE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["voices"][0]["name"], "custom")
        state_path = self.root / f"{manifest['voices'][0]['storage_id']}{VOICE_STATE_SUFFIX}"
        self.assertTrue(state_path.exists())

        deleted = await delete_custom_voice(self.root, name="custom")
        self.assertEqual(deleted, "custom")
        self.assertFalse(state_path.exists())
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"voices": []},
        )

    async def test_publish_rejects_duplicate_names(self):
        await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=fake_voice_state(),
            existing_names={"alba"},
        )

        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await publish_custom_voice(
                self.root,
                name="custom",
                voice_state=fake_voice_state(),
                existing_names={"alba", "custom"},
            )

    async def test_publish_rolls_back_state_file_when_manifest_write_fails(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"

        with patch(
            "trillim.components.tts._voices.atomic_write_bytes",
            side_effect=OSError("manifest boom"),
        ):
            with self.assertRaisesRegex(OSError, "manifest boom"):
                await publish_custom_voice(
                    self.root,
                    name="custom",
                    voice_state=fake_voice_state(),
                    existing_names={"alba"},
                )

        self.assertFalse(state_path.exists())

    async def test_delete_rejects_symlinked_state_file_for_write_safety(self):
        await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=fake_voice_state(),
            existing_names={"alba"},
        )
        state_path = self.root / f"{_storage_id_for_name('custom')}{VOICE_STATE_SUFFIX}"
        target_path = self.root / "target.safetensors"
        state_path.rename(target_path)
        state_path.symlink_to(target_path)

        with self.assertLogs("trillim.components.tts._voices", level="WARNING"):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "symlinks"):
                await delete_custom_voice(self.root, name="custom")

    async def test_malformed_manifest_is_skipped_with_warning(self):
        self.root.mkdir(parents=True)
        (self.root / VOICE_MANIFEST_NAME).write_text("{", encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("manifest is malformed", "\n".join(logs.output))

    async def test_malformed_manifest_entries_are_skipped_with_warning(self):
        self.root.mkdir(parents=True)
        manifest = {
            "voices": [
                "not an entry",
                {"name": "bad-name", "storage_id": "x", "size_bytes": 1},
                {"name": "alba", "storage_id": _storage_id_for_name("alba"), "size_bytes": 1},
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("malformed custom TTS voice manifest entry", "\n".join(logs.output))

    async def test_missing_manifest_with_legacy_and_unexpected_files_warns_and_skips(self):
        self.root.mkdir(parents=True)
        (self.root / "legacy.state").write_bytes(b"legacy")
        (self.root / "unexpected.safetensors").write_bytes(b"not tracked")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        output = "\n".join(logs.output)
        self.assertEqual(states, {})
        self.assertIn("legacy", output)
        self.assertIn("manifest is missing", output)

    async def test_inventory_mismatch_warns_and_keeps_valid_manifest_voices(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(fake_voice_state(), state_path)
        (self.root / "orphan.safetensors").write_bytes(b"orphan")
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(list(states), ["custom"])
        self.assertIn("unexpected TTS voice store files", "\n".join(logs.output))

    async def test_manifest_size_mismatch_skips_voice(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(fake_voice_state(), state_path)
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size + 1,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("size does not match manifest", "\n".join(logs.output))


def _storage_id_for_name(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
