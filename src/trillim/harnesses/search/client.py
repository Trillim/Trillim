"""Search-provider selection and bounded search execution."""

from __future__ import annotations

import asyncio

from trillim.harnesses.search._brave import BraveSearchProvider
from trillim.harnesses.search._ddgs import DDGSSearchProvider
from trillim.harnesses.search.fetch import build_search_context
from trillim.harnesses.search.provider import (
    MAX_SEARCH_RESULTS,
    SearchProvider,
    normalize_provider_name,
    validate_search_query,
)


class SearchClient:
    """Search pipeline over a user-selected provider."""

    def __init__(
        self,
        *,
        provider_name: str,
        token_budget: int,
        max_results: int = MAX_SEARCH_RESULTS,
        fetcher=None,
    ) -> None:
        self._provider_name = normalize_provider_name(provider_name)
        self._token_budget = token_budget
        self._max_results = min(max(1, max_results), MAX_SEARCH_RESULTS)
        self._fetcher = fetcher
        self._provider = self._build_provider()

    async def search(self, query: str) -> str:
        """Run the search pipeline on a worker thread."""
        validated = validate_search_query(query)
        return await asyncio.to_thread(self._search_sync, validated)

    def _search_sync(self, query: str) -> str:
        results = self._provider.search(query, max_results=self._max_results)
        return build_search_context(
            query,
            results,
            token_budget=self._token_budget,
            fetcher=self._fetcher,
        )

    def _build_provider(self) -> SearchProvider:
        if self._provider_name == "ddgs":
            return DDGSSearchProvider()
        return BraveSearchProvider(token_budget=self._token_budget)
