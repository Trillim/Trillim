"""Tests for local search harness exports."""

import unittest

import trillim.harnesses.search as search_harness_exports


class SearchHarnessExportTests(unittest.TestCase):
    def test_search_harness_export_is_not_available(self):
        self.assertFalse(hasattr(search_harness_exports, "SearchHarness"))
        self.assertEqual(search_harness_exports.__all__, [])

    def test_public_search_harness_imports_fail(self):
        with self.assertRaises(ImportError):
            exec("from trillim.harnesses.search import SearchHarness", {})
        with self.assertRaises(ModuleNotFoundError):
            exec("from trillim.harnesses.search.harness import SearchHarness", {})
