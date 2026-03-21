"""Tests for the search client."""

from __future__ import annotations

import unittest

from trillim.harnesses.search.client import SearchClient
from trillim.harnesses.search.provider import SearchResult


class _ProviderStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int):
        self.calls.append((query, max_results))
        return [SearchResult(title="Example", url="https://example.com", snippet="snippet")]


class SearchClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_client_normalizes_query_and_limits_result_count(self):
        client = SearchClient(provider_name="ddgs", token_budget=32, max_results=99)
        provider = _ProviderStub()
        client._provider = provider
        client._fetcher = lambda *_args, **_kwargs: "fetched body"

        content = await client.search("  cats   and   dogs ")

        self.assertEqual(provider.calls, [("cats and dogs", 5)])
        self.assertIn("Example", content)
        self.assertIn("fetched body", content)

    def test_client_normalizes_provider_aliases(self):
        client = SearchClient(provider_name="BRAVE_SEARCH", token_budget=32)
        self.assertEqual(client._provider_name, "brave")
