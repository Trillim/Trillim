"""Tests for search metrics aggregation."""

import unittest

from trillim.harnesses.search.metrics import SearchMetrics


class SearchMetricsTests(unittest.TestCase):
    def test_record_generation_tracks_only_final_turn_usage(self):
        metrics = SearchMetrics()

        metrics.record_generation(prompt_tokens=10, completion_tokens=3, cached_tokens=0)
        metrics.record_generation(prompt_tokens=25, completion_tokens=5, cached_tokens=7)

        self.assertEqual(metrics.prompt_tokens, 25)
        self.assertEqual(metrics.completion_tokens, 5)
        self.assertEqual(metrics.cached_tokens, 7)
        self.assertEqual(metrics.total_tokens, 30)
