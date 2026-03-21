"""Internal search provider contracts and query helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from trillim.components.llm._limits import MAX_MESSAGE_CHARS

DEFAULT_SEARCH_TOKEN_BUDGET = 1024
FALLBACK_SEARCH_FAILURE_MESSAGE = "Search failed, answer from memory"
MAX_SEARCH_ITERATIONS = 3
MAX_SEARCH_RESULTS = 5
SEARCH_CONTENT_CHARS_PER_TOKEN = 4

_SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_HARNESS_NAMES = frozenset({"default", "search"})
_SEARCH_PROVIDER_ALIASES = {
    "brave": "brave",
    "brave_search": "brave",
    "ddgs": "ddgs",
    "duckduckgo": "ddgs",
}


class SearchError(RuntimeError):
    """Raised when search results could not be produced."""


class SearchAuthenticationError(SearchError):
    """Raised when the selected provider cannot authenticate."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Normalized provider search hit."""

    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    """Protocol for concrete search providers."""

    name: str

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return normalized search results."""


def extract_search_query(text: str) -> str | None:
    """Extract and normalize a model-emitted ``<search>`` query."""
    match = _SEARCH_TAG_RE.search(text)
    if match is None:
        return None
    return validate_search_query(match.group(1))


def validate_search_query(query: str) -> str:
    """Normalize and bound a search query before provider use."""
    normalized = " ".join(query.split())
    if not normalized:
        raise SearchError("search query must not be empty")
    if len(normalized) > MAX_MESSAGE_CHARS:
        normalized = normalized[:MAX_MESSAGE_CHARS]
    return normalized


def normalize_provider_name(name: str) -> str:
    """Normalize a provider name or alias."""
    normalized = name.strip().lower()
    try:
        return _SEARCH_PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_SEARCH_PROVIDER_ALIASES))
        raise ValueError(
            f"Unknown search provider {name!r}. Available: {available}"
        ) from exc


def validate_harness_name(name: str) -> str:
    """Normalize and validate the requested harness name."""
    normalized = name.strip().lower()
    if normalized not in _HARNESS_NAMES:
        available = ", ".join(sorted(_HARNESS_NAMES))
        raise ValueError(f"Unknown harness {name!r}. Available: {available}")
    return normalized


def resolve_search_token_budget(
    requested_budget: int,
    *,
    max_context_tokens: int,
) -> int:
    """Clamp the configured search budget to the active model's quarter-context cap."""
    if requested_budget < 1:
        raise ValueError("search_token_budget must be at least 1")
    return min(requested_budget, max(1, max_context_tokens // 4))


def coerce_search_result(*, title: str, url: str, snippet: str = "") -> SearchResult | None:
    """Normalize one provider result or drop it when it lacks a URL."""
    normalized_url = url.strip()
    if not normalized_url:
        return None
    normalized_title = " ".join(title.split())[:MAX_MESSAGE_CHARS]
    normalized_snippet = " ".join(snippet.split())[:MAX_MESSAGE_CHARS]
    return SearchResult(
        title=normalized_title,
        url=normalized_url,
        snippet=normalized_snippet,
    )
