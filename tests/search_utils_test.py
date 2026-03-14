# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for tokenizer-aware search truncation helpers."""

import unittest

from trillim.harnesses._search_utils import (
    BraveSearchProvider,
    SearchClient,
    SearchProvider,
    SearchResult,
    truncate_to_token_budget,
)


def _word_token_count(text: str) -> int:
    return len(text.split())


class _ProviderStub(SearchProvider):
    def search(self, query: str) -> list[SearchResult]:
        del query
        return []


class _CapturingProvider:
    def __init__(self):
        self.received_count_tokens = None

    def search(self, query: str) -> list[SearchResult]:
        del query
        return [SearchResult(title="Result", href="https://example.com", body="body")]

    def format_for_prompt(
        self,
        query: str,
        results: list[SearchResult],
        *,
        token_budget: int,
        fetch_and_extract,
        count_tokens=None,
    ) -> str:
        del query, results, token_budget, fetch_and_extract
        self.received_count_tokens = count_tokens
        return "formatted"


class SearchUtilsTests(unittest.TestCase):
    def test_truncate_to_token_budget_falls_back_to_char_budget_without_counter(self):
        result = truncate_to_token_budget("abcdefghij", "query", token_budget=2)
        self.assertEqual(result, "abcdefgh")

    def test_truncate_to_token_budget_uses_exact_counter_for_single_paragraph(self):
        result = truncate_to_token_budget(
            "alpha beta gamma delta",
            "alpha",
            token_budget=2,
            count_tokens=_word_token_count,
        )
        self.assertEqual(result.strip(), "alpha beta")
        self.assertLessEqual(_word_token_count(result), 2)

    def test_truncate_to_token_budget_selects_relevant_paragraph_with_exact_counter(self):
        result = truncate_to_token_budget(
            "dogs bark loudly\n\ncats purr softly",
            "cats",
            token_budget=3,
            count_tokens=_word_token_count,
        )
        self.assertEqual(result, "cats purr softly")

    def test_base_provider_format_for_prompt_hard_caps_exact_budget(self):
        provider = _ProviderStub(token_budget=6)
        output = provider.format_for_prompt(
            "alpha delta",
            [
                SearchResult(title="First", href="https://1", body="alpha beta gamma"),
                SearchResult(title="Second", href="https://2", body="delta epsilon zeta"),
            ],
            token_budget=6,
            fetch_and_extract=lambda url: None,
            count_tokens=_word_token_count,
        )
        self.assertIn("[1] First", output)
        self.assertLessEqual(_word_token_count(output), 6)

    def test_brave_provider_applies_exact_local_cap_when_counter_present(self):
        provider = BraveSearchProvider(token_budget=5)
        output = provider.format_for_prompt(
            "query",
            [
                SearchResult(title="One", href="https://1", body="alpha beta gamma delta"),
                SearchResult(title="Two", href="https://2", body="epsilon zeta eta theta"),
            ],
            token_budget=5,
            fetch_and_extract=lambda url: None,
            count_tokens=_word_token_count,
        )
        self.assertLessEqual(_word_token_count(output), 5)

    def test_search_client_passes_token_counter_to_provider(self):
        client = SearchClient(provider_name="brave", count_tokens=_word_token_count)
        provider = _CapturingProvider()
        client.provider = provider

        output = client._search_sync("cats")

        self.assertEqual(output, "formatted")
        self.assertIs(provider.received_count_tokens, _word_token_count)
