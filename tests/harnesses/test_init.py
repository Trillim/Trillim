"""Tests for harness package exports."""

import unittest

import trillim.harnesses as harness_exports


class HarnessExportTests(unittest.TestCase):
    def test_harness_exports_are_not_public(self):
        self.assertFalse(hasattr(harness_exports, "DefaultHarness"))
        self.assertFalse(hasattr(harness_exports, "Harness"))
        self.assertEqual(harness_exports.__all__, [])

    def test_public_harness_imports_fail(self):
        with self.assertRaises(ImportError):
            exec("from trillim.harnesses import DefaultHarness", {})
        with self.assertRaises(ModuleNotFoundError):
            exec("from trillim.harnesses.default import DefaultHarness", {})
