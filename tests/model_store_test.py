# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for model store and HuggingFace integration helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import trillim.model_store as model_store


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _Repo:
    def __init__(
        self,
        repo_id: str,
        *,
        siblings: list[str] | None = None,
        tags: list[str] | None = None,
        downloads: int | None = None,
        last_modified: datetime | None = None,
    ) -> None:
        self.id = repo_id
        self.siblings = [SimpleNamespace(rfilename=name) for name in (siblings or [])]
        self.tags = tags or []
        self.downloads = downloads
        self.last_modified = last_modified


class ModelStoreTests(unittest.TestCase):
    def test_looks_like_hf_id_accepts_expected_shapes(self):
        self.assertTrue(model_store._looks_like_hf_id("Org/Model-1"))
        self.assertFalse(model_store._looks_like_hf_id("/tmp/model"))
        self.assertFalse(model_store._looks_like_hf_id("./model"))
        self.assertFalse(model_store._looks_like_hf_id("~/model"))
        self.assertFalse(model_store._looks_like_hf_id("not-an-id"))

    def test_resolve_model_dir_handles_directories_hf_ids_and_passthrough(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            direct_dir = temp_path / "direct"
            direct_dir.mkdir()
            expanded_dir = temp_path / "expanded"
            expanded_dir.mkdir()
            local_dir = temp_path / "models" / "Org" / "Model"
            local_dir.mkdir(parents=True)

            with patch.object(model_store, "MODELS_DIR", temp_path / "models"):
                self.assertEqual(model_store.resolve_model_dir(str(direct_dir)), str(direct_dir))
                with patch("trillim.model_store.os.path.expanduser", return_value=str(expanded_dir)):
                    self.assertEqual(model_store.resolve_model_dir("~/expanded"), str(expanded_dir))
                self.assertEqual(model_store.resolve_model_dir("Org/Model"), str(local_dir))
                with self.assertRaisesRegex(RuntimeError, "not found locally"):
                    model_store.resolve_model_dir("Org/Missing")
                self.assertEqual(model_store.resolve_model_dir("raw-value"), "raw-value")

    def test_validate_trillim_config_handles_missing_invalid_and_warning_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.assertIsNone(model_store.validate_trillim_config(temp_path))

            bad_path = temp_path / "trillim_config.json"
            bad_path.write_text("{", encoding="utf-8")
            with patch("builtins.print") as print_mock:
                self.assertIsNone(model_store.validate_trillim_config(temp_path))
            self.assertIn("Warning: Could not read trillim_config.json", print_mock.call_args.args[0])

            bad_path.write_text(
                json.dumps({"format_version": 99, "platforms": ["arm64"]}),
                encoding="utf-8",
            )
            with (
                patch("trillim.model_store.platform.machine", return_value="x86_64"),
                patch("builtins.print") as print_mock,
            ):
                config = model_store.validate_trillim_config(temp_path)

            self.assertEqual(config["format_version"], 99)
            messages = [call.args[0] for call in print_mock.call_args_list]
            self.assertTrue(any("newer than supported version" in message for message in messages))
            self.assertTrue(any("but your system is x86_64" in message for message in messages))

    def test_validate_adapter_model_compat_rejects_old_and_mismatched_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter_dir = temp_path / "adapter"
            adapter_dir.mkdir()
            model_dir = temp_path / "model"
            model_dir.mkdir()
            utils_module = _module("trillim.utils", compute_base_model_hash=lambda path: "current-hash")

            self.assertIsNone(model_store.validate_adapter_model_compat(str(adapter_dir), str(model_dir)))

            config_path = adapter_dir / "trillim_config.json"
            config_path.write_text("{", encoding="utf-8")
            self.assertIsNone(model_store.validate_adapter_model_compat(str(adapter_dir), str(model_dir)))

            config_path.write_text(json.dumps({"format_version": 2}), encoding="utf-8")
            with patch.dict("sys.modules", {"trillim.utils": utils_module}):
                with self.assertRaisesRegex(model_store.AdapterCompatError, "older format"):
                    model_store.validate_adapter_model_compat(str(adapter_dir), str(model_dir))

            config_path.write_text(
                json.dumps(
                    {
                        "format_version": 3,
                        "base_model_config_hash": "other-hash",
                        "source_model": "Org/BaseModel",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("sys.modules", {"trillim.utils": utils_module}):
                with self.assertRaisesRegex(model_store.AdapterCompatError, "Adapter/model mismatch"):
                    model_store.validate_adapter_model_compat(str(adapter_dir), str(model_dir))

            config_path.write_text(
                json.dumps(
                    {
                        "format_version": 3,
                        "base_model_config_hash": "current-hash",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("sys.modules", {"trillim.utils": utils_module}):
                self.assertIsNone(
                    model_store.validate_adapter_model_compat(str(adapter_dir), str(model_dir))
                )

    def test_pull_model_returns_existing_copy_without_downloading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            local_dir = models_dir / "Org" / "Model"
            local_dir.mkdir(parents=True)

            with (
                patch.object(model_store, "MODELS_DIR", models_dir),
                patch("builtins.print") as print_mock,
            ):
                resolved = model_store.pull_model("Org/Model")

            self.assertEqual(resolved, local_dir)
            self.assertEqual(print_mock.call_args_list[0].args[0], f"Model 'Org/Model' already exists at {local_dir}")

    def test_pull_model_downloads_and_validates_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            calls: list[tuple[str, str, str | None]] = []

            def snapshot_download(model_id, *, local_dir, revision):
                calls.append((model_id, local_dir, revision))
                Path(local_dir).mkdir(parents=True, exist_ok=True)

            hub_module = _module("huggingface_hub", snapshot_download=snapshot_download)

            with (
                patch.object(model_store, "MODELS_DIR", models_dir),
                patch("trillim.model_store.validate_trillim_config") as validate_mock,
                patch.dict("sys.modules", {"huggingface_hub": hub_module}),
            ):
                resolved = model_store.pull_model("Org/Model", revision="main", force=True)

            self.assertEqual(resolved, models_dir / "Org" / "Model")
            self.assertEqual(calls, [("Org/Model", str(models_dir / "Org" / "Model"), "main")])
            validate_mock.assert_called_once_with(models_dir / "Org" / "Model")

    def test_pull_model_maps_huggingface_errors_and_reraises_unknown_ones(self):
        class RepositoryNotFoundError(Exception):
            pass

        class GatedRepoError(Exception):
            pass

        class OtherError(Exception):
            pass

        def _raising(exception):
            def _snapshot_download(*args, **kwargs):
                raise exception

            return _snapshot_download

        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"

            with patch.object(model_store, "MODELS_DIR", models_dir):
                with patch.dict("sys.modules", {"huggingface_hub": _module("huggingface_hub", snapshot_download=_raising(RepositoryNotFoundError()))}):
                    with self.assertRaisesRegex(RuntimeError, "Repository 'Org/Model' not found"):
                        model_store.pull_model("Org/Model", force=True)

                with patch.dict("sys.modules", {"huggingface_hub": _module("huggingface_hub", snapshot_download=_raising(GatedRepoError()))}):
                    with self.assertRaisesRegex(RuntimeError, "Authenticate with: hf auth login"):
                        model_store.pull_model("Org/Model", force=True)

                with patch.dict("sys.modules", {"huggingface_hub": _module("huggingface_hub", snapshot_download=_raising(OtherError("boom")))}):
                    with self.assertRaisesRegex(OtherError, "boom"):
                        model_store.pull_model("Org/Model", force=True)

    def test_local_model_ids_and_scan_models_dir_classify_models_and_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            model_dir = models_dir / "Org" / "BaseModel"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "qmodel.tensors").write_bytes(b"x" * 2048)
            (model_dir / "trillim_config.json").write_text(
                json.dumps({"architecture": "llama", "source_model": "Org/Base", "quantization": "ternary"}),
                encoding="utf-8",
            )

            adapter_dir = models_dir / "Org" / "Adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "qmodel.lora").write_bytes(b"x" * 1024)
            (adapter_dir / "trillim_config.json").write_text(
                json.dumps({"base_model_config_hash": "hash-model"}),
                encoding="utf-8",
            )

            invalid_model = models_dir / "Org" / "BrokenModel"
            invalid_model.mkdir(parents=True)
            (invalid_model / "config.json").write_text("{}", encoding="utf-8")
            (invalid_model / "trillim_config.json").write_text("{", encoding="utf-8")

            utils_module = _module("trillim.utils", compute_base_model_hash=lambda path: "hash-model")

            with (
                patch.object(model_store, "MODELS_DIR", models_dir),
                patch.dict("sys.modules", {"trillim.utils": utils_module}),
            ):
                self.assertEqual(model_store._local_model_ids(), {"Org/BaseModel", "Org/Adapter", "Org/BrokenModel"})
                models, adapters = model_store._scan_models_dir()

            self.assertEqual(
                [entry["model_id"] for entry in models],
                ["Org/BaseModel", "Org/BrokenModel"],
            )
            self.assertEqual([entry["model_id"] for entry in adapters], ["Org/Adapter"])
            self.assertEqual(models[0]["size_human"], "2.0 KB")
            self.assertEqual(adapters[0]["size_human"], "1.0 KB")
            self.assertEqual(models[0]["base_model_config_hash"], "hash-model")

    def test_local_model_ids_and_scan_models_dir_handle_missing_roots_and_non_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "missing"
            with patch.object(model_store, "MODELS_DIR", missing_root):
                self.assertEqual(model_store._local_model_ids(), set())
                self.assertEqual(model_store._scan_models_dir(), ([], []))

            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()
            (models_dir / "README.txt").write_text("skip", encoding="utf-8")
            org_dir = models_dir / "Org"
            org_dir.mkdir()
            (org_dir / "note.txt").write_text("skip", encoding="utf-8")
            empty_model = org_dir / "Empty"
            empty_model.mkdir()
            adapter_dir = org_dir / "AdapterOnly"
            adapter_dir.mkdir()
            (adapter_dir / "trillim_config.json").write_text("{}", encoding="utf-8")

            with patch.object(model_store, "MODELS_DIR", models_dir):
                self.assertEqual(model_store._local_model_ids(), {"Org/Empty", "Org/AdapterOnly"})
                models, adapters = model_store._scan_models_dir()

            self.assertEqual(models, [])
            self.assertEqual(adapters[0]["size_human"], "-")

    def test_list_models_and_list_adapters_delegate_to_scan_results(self):
        with patch("trillim.model_store._scan_models_dir", return_value=([{"model_id": "m"}], [{"model_id": "a"}])):
            self.assertEqual(model_store.list_models(), [{"model_id": "m"}])
            self.assertEqual(model_store.list_adapters(), [{"model_id": "a"}])

    def test_list_available_models_formats_results_and_maps_errors(self):
        repos = [
            _Repo(
                "Trillim/Base",
                siblings=["qmodel.tensors"],
                tags=["base_model:meta/llama", "base_model:quantized:ignore"],
                downloads=10,
                last_modified=datetime(2026, 3, 14),
            ),
            _Repo(
                "Trillim/Adapter",
                siblings=["qmodel.lora"],
                tags=["base_model:adapter:ignore"],
                downloads=None,
            ),
        ]

        hub_module = _module("huggingface_hub", list_models=lambda author, full=True: repos)

        with (
            patch("trillim.model_store._local_model_ids", return_value={"Trillim/Base"}),
            patch.dict("sys.modules", {"huggingface_hub": hub_module}),
        ):
            entries = model_store.list_available_models()

        self.assertEqual(
            entries,
            [
                {
                    "model_id": "Trillim/Base",
                    "type": "model",
                    "downloads": 10,
                    "last_modified": "2026-03-14",
                    "base_model": "meta/llama",
                    "local": True,
                },
                {
                    "model_id": "Trillim/Adapter",
                    "type": "adapter",
                    "downloads": 0,
                    "last_modified": "",
                    "base_model": "",
                    "local": False,
                },
            ],
        )

        with patch.dict("sys.modules", {"huggingface_hub": _module("huggingface_hub")}):
            with self.assertRaisesRegex(RuntimeError, "huggingface_hub is required"):
                model_store.list_available_models()

        class ConnectionError(Exception):
            pass

        class OtherError(Exception):
            pass

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": _module("huggingface_hub", list_models=lambda author, full=True: (_ for _ in ()).throw(ConnectionError("offline")))},
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to fetch models from HuggingFace: offline"):
                model_store.list_available_models()

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": _module("huggingface_hub", list_models=lambda author, full=True: (_ for _ in ()).throw(OtherError("boom")))},
        ):
            with self.assertRaisesRegex(OtherError, "boom"):
                model_store.list_available_models()

    def test_human_size_formats_multiple_units(self):
        self.assertEqual(model_store._human_size(512), "512 B")
        self.assertEqual(model_store._human_size(2048), "2.0 KB")
        self.assertEqual(model_store._human_size(5 * 1024 * 1024), "5.0 MB")
        self.assertEqual(model_store._human_size(3 * 1024**4), "3.0 TB")
        self.assertEqual(model_store._human_size(2 * 1024**5), "2.0 PB")


if __name__ == "__main__":
    unittest.main()
