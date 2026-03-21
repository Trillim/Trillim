"""Tests for search-provider helpers."""

import unittest

from trillim.harnesses.search.provider import (
    SearchError,
    coerce_search_result,
    extract_search_query,
    normalize_provider_name,
    resolve_search_token_budget,
    validate_harness_name,
)


class SearchProviderHelperTests(unittest.TestCase):
    def test_extract_search_query_normalizes_whitespace(self):
        self.assertEqual(
            extract_search_query("hello <search>  cats   and dogs </search>"),
            "cats and dogs",
        )
        self.assertIsNone(extract_search_query("hello world"))

    def test_extract_search_query_rejects_empty_queries(self):
        with self.assertRaisesRegex(SearchError, "must not be empty"):
            extract_search_query("<search>   </search>")

    def test_provider_and_harness_names_are_normalized(self):
        self.assertEqual(normalize_provider_name("BRAVE_SEARCH"), "brave")
        self.assertEqual(normalize_provider_name("duckduckgo"), "ddgs")
        self.assertEqual(validate_harness_name("SEARCH"), "search")
        with self.assertRaisesRegex(ValueError, "Unknown search provider"):
            normalize_provider_name("bing")
        with self.assertRaisesRegex(ValueError, "Unknown harness"):
            validate_harness_name("tool")

    def test_search_budget_is_clamped_to_quarter_context(self):
        self.assertEqual(
            resolve_search_token_budget(2048, max_context_tokens=1024),
            256,
        )

    def test_coerce_search_result_drops_missing_urls(self):
        self.assertIsNone(coerce_search_result(title="x", url="", snippet="y"))
        result = coerce_search_result(
            title="  Example   Title  ",
            url="https://example.com",
            snippet="  short   body ",
        )
        self.assertEqual(result.title, "Example Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.snippet, "short body")
