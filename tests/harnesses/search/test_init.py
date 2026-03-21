"""Tests for local search harness exports."""

import unittest

from trillim.harnesses.search import SearchHarness


class SearchHarnessExportTests(unittest.TestCase):
    def test_search_harness_export_is_available(self):
        self.assertEqual(SearchHarness.__name__, "SearchHarness")
