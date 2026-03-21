"""Tests for the DDGS provider."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from trillim.harnesses.search._ddgs import DDGSSearchProvider
from trillim.harnesses.search.provider import SearchError


class SearchDDGSTests(unittest.TestCase):
    @patch("ddgs.DDGS")
    def test_ddgs_provider_normalizes_results(self, ddgs_cls):
        ddgs_cls.return_value.text.return_value = [
            {
                "title": "Example",
                "href": "https://example.com",
                "body": "Body",
            },
            {
                "title": "Missing URL",
                "href": "",
                "body": "ignored",
            },
        ]

        results = DDGSSearchProvider().search("cats", max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Example")
        self.assertEqual(results[0].url, "https://example.com")

    @patch("ddgs.DDGS")
    def test_ddgs_provider_wraps_backend_errors(self, ddgs_cls):
        ddgs_cls.return_value.text.side_effect = RuntimeError("boom")

        with self.assertRaisesRegex(SearchError, "DDGS search failed"):
            DDGSSearchProvider().search("cats", max_results=5)
