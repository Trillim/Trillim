"""Tests for model-store ID parsing and resolution helpers."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path, PureWindowsPath
import unittest

from trillim import _model_store
from tests.components.llm.support import patched_model_store


class _ModelStoreError(RuntimeError):
    pass


class ModelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self.root = self._stack.enter_context(patched_model_store())

    def test_store_namespace_root_and_path_resolution(self):
        self.assertEqual(
            _model_store.store_namespace_root("Trillim"),
            _model_store.DOWNLOADED_ROOT,
        )
        self.assertEqual(
            _model_store.store_namespace_root("Local"),
            _model_store.LOCAL_ROOT,
        )
        self.assertEqual(
            _model_store.store_path_for_id("Trillim/demo"),
            _model_store.DOWNLOADED_ROOT / "demo",
        )
        self.assertEqual(
            _model_store.store_path_for_id("Local/demo"),
            _model_store.LOCAL_ROOT / "demo",
        )
        with self.assertRaisesRegex(AssertionError, "Unsupported namespace"):
            _model_store.store_namespace_root("Other")

    def test_parse_store_id_accepts_paths_and_custom_error_types(self):
        self.assertEqual(
            _model_store.parse_store_id(PureWindowsPath("Trillim\\demo")),
            ("Trillim", "demo"),
        )
        self.assertEqual(
            _model_store.parse_store_id(Path("Local/demo")),
            ("Local", "demo"),
        )

        with self.assertRaisesRegex(_ModelStoreError, "form Trillim/<name> or Local/<name>"):
            _model_store.parse_store_id("bad", error_type=_ModelStoreError)

    def test_parse_store_id_rejects_invalid_shapes(self):
        for value in (
            "",
            "Trillim/",
            "Local/..",
            "Trillim/demo/extra",
            "Trillim/demo name",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "form Trillim/<name> or Local/<name>"):
                    _model_store.parse_store_id(value)

    def test_resolve_existing_store_id_reports_missing_directories(self):
        existing = _model_store.DOWNLOADED_ROOT / "demo"
        existing.mkdir(parents=True, exist_ok=True)
        self.assertEqual(
            _model_store.resolve_existing_store_id("Trillim/demo"),
            existing,
        )

        with self.assertRaisesRegex(
            _ModelStoreError,
            "was not found locally",
        ):
            _model_store.resolve_existing_store_id(
                "Local/missing",
                error_type=_ModelStoreError,
            )
