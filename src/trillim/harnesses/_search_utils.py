# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Search utilities — tag extraction, smart truncation, and search providers."""

import asyncio
import json
import os
import re
import urllib.parse
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Search tag extraction
# ---------------------------------------------------------------------------

_SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


class SearchError(Exception):
    """Raised when web search fails (network error, no results, etc.)."""


def extract_search_query(text: str) -> str | None:
    """Extract the query from a <search>query</search> tag, or None if absent."""
    match = _SEARCH_TAG_RE.search(text)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Smart truncation
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "about",
        "up",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "any",
        "if",
    }
)


def _truncate_at_sentence(text: str, budget: int) -> str:
    """Truncate text at the last sentence boundary within budget chars."""
    if len(text) <= budget:
        return text
    snippet = text[:budget]
    # Find last sentence-ending punctuation
    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in ".!?":
            return snippet[: i + 1]
    # No sentence boundary found — hard truncate
    return snippet


def truncate_to_budget(text: str, query: str, budget: int = 2000) -> str:
    """Select the most query-relevant paragraphs within a character budget.

    1. Split into paragraphs
    2. Score by keyword overlap with query
    3. Greedily select top-scoring paragraphs until budget
    4. Re-sort selected by original position (preserve reading order)
    """
    # Split into paragraphs
    paragraphs = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped:
                paragraphs.append(stripped)

    if not paragraphs:
        return ""

    # Single paragraph — just truncate at sentence boundary
    if len(paragraphs) == 1:
        return _truncate_at_sentence(paragraphs[0], budget)

    # Build keyword set from query
    query_words = {w for w in re.findall(r"\w+", query.lower()) if w not in _STOPWORDS}

    # Score each paragraph by keyword overlap
    scored = []
    for idx, para in enumerate(paragraphs):
        para_words = set(re.findall(r"\w+", para.lower()))
        score = len(query_words & para_words)
        scored.append((-score, idx, para))

    # Sort by score descending, then by original position for ties
    scored.sort()

    # Greedily select until budget
    selected = []
    remaining = budget
    for score, idx, para in scored:
        if len(para) <= remaining:
            selected.append((idx, para))
            remaining -= len(para) + 2  # account for \n\n join
        elif not selected:
            # First paragraph exceeds budget — truncate at sentence boundary
            selected.append((idx, _truncate_at_sentence(para, budget)))
            break

    # Re-sort by original position
    selected.sort()
    return "\n\n".join(para for _, para in selected)


# ---------------------------------------------------------------------------
# Search providers + extraction pipeline
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single search hit from a provider."""

    title: str
    href: str
    body: str = ""


class SearchProvider(ABC):
    """Provider interface for search backends."""

    def __init__(self, max_results: int = 3, api_key: str | None = None):
        self.max_results = max_results
        self.api_key = api_key

    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        """Return provider-native search hits normalized to SearchResult."""


class DDGSSearchProvider(SearchProvider):
    """DuckDuckGo provider via ddgs."""

    def search(self, query: str) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            # Suppress C-level stderr from primp's impersonation warnings
            fd = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                raw_results = DDGS().text(query, max_results=self.max_results)
            finally:
                os.dup2(fd, 2)
                os.close(fd)
        except Exception as exc:
            raise SearchError(f"Search unavailable: {exc}") from exc

        if not raw_results:
            return []

        entries: list[SearchResult] = []
        for item in raw_results:
            title = item.get("title", "").strip()
            href = item.get("href", "").strip()
            body = item.get("body", "").strip()
            if not href:
                continue
            entries.append(SearchResult(title=title, href=href, body=body))
        return entries


class BraveSearchProvider(SearchProvider):
    """Brave provider via official Web Search API."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str) -> list[SearchResult]:
        api_key = self.api_key or os.environ.get("SEARCH_API_KEY")
        if not api_key:
            raise SearchError(
                "Brave search requires SEARCH_API_KEY in the environment."
            )

        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": max(1, min(self.max_results, 20)),
                "extra_snippets": "true",
                "safesearch": "moderate",
            }
        )
        req = urllib.request.Request(
            f"{self._ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
                "User-Agent": "Trillim/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            if exc.code in (401, 403):
                raise SearchError(
                    "Brave authentication failed. Check SEARCH_API_KEY."
                ) from exc
            raise SearchError(
                f"Brave search API error ({exc.code}). {detail}".strip()
            ) from exc
        except Exception as exc:
            raise SearchError(f"Search unavailable: {exc}") from exc

        web_obj = payload.get("web", {})
        raw_results = web_obj.get("results", []) if isinstance(web_obj, dict) else []
        entries: list[SearchResult] = []
        for item in raw_results:
            title = str(item.get("title", "")).strip()
            href = str(item.get("url", "")).strip()
            body = str(item.get("description", "")).strip()
            extras = item.get("extra_snippets", [])
            if isinstance(extras, list) and extras:
                snippets = [str(x).strip() for x in extras if x]
                body = "\n".join([body, *snippets]).strip()
            if not href:
                continue
            entries.append(SearchResult(title=title, href=href, body=body))
        return entries


def get_search_provider(
    name: str,
    *,
    max_results: int = 3,
    api_key: str | None = None,
) -> SearchProvider:
    """Factory for concrete search providers."""
    name_normalized = name.strip().lower()
    providers: dict[str, type[SearchProvider]] = {
        "ddgs": DDGSSearchProvider,
        "brave": BraveSearchProvider,
    }
    if name_normalized not in providers:
        available = ", ".join(sorted(providers))
        raise ValueError(
            f"Unknown search provider {name!r}. Available: {available}"
        )
    return providers[name_normalized](max_results=max_results, api_key=api_key)


class SearchClient:
    """Search + extraction + truncation pipeline over a provider."""

    def __init__(
        self,
        provider_name: str = "ddgs",
        max_results: int = 3,
        char_budget: int = 2000,
        api_key: str | None = None,
    ):
        self.provider = get_search_provider(
            provider_name, max_results=max_results, api_key=api_key
        )
        self.char_budget = char_budget

    async def search(self, query: str) -> str:
        """Run search pipeline in a thread (providers + urllib are sync)."""
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> str:
        results = self.provider.search(query)
        if not results:
            raise SearchError("No search results found.")

        # If query mentions a year, boost results that reference it
        year_match = re.search(r"\b(\d{4})\b", query)
        if year_match:
            year = year_match.group(1)
            results.sort(
                key=lambda r: year not in (r.title + r.body),
            )

        # Fetch and extract full page content for each result
        entries: list[tuple[str, str]] = []
        for r in results:
            title = r.title
            href = r.href
            body = r.body
            page_text = self._fetch_and_extract(href)
            if page_text:
                body = page_text
            if body:
                entries.append((title, body))

        if not entries:
            raise SearchError("No relevant search results found.")

        # Smart truncate each result and format
        per_result = self.char_budget // len(entries)
        parts: list[str] = []
        for i, (title, body) in enumerate(entries, 1):
            truncated = truncate_to_budget(body, query, budget=per_result)
            parts.append(f"[{i}] {title}\n{truncated}")

        output = "\n\n".join(parts)
        # Hard limit — shouldn't normally trigger given per-result budgets
        if len(output) > self.char_budget:
            output = _truncate_at_sentence(output, self.char_budget)
        return output

    def _fetch_and_extract(self, url: str) -> str | None:
        """Fetch a URL and extract main text with trafilatura."""
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read(200_000).decode("utf-8", errors="ignore")
            import trafilatura

            text = trafilatura.extract(
                html,
                no_fallback=True,
                include_tables=False,
                include_comments=False,
            )
            return text
        except Exception:
            return None
