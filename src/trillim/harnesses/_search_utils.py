# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Search utilities — tag extraction, smart truncation, and search providers."""

import asyncio
import gzip
import json
import os
import re
import sys
import urllib.parse
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Search tag extraction
# ---------------------------------------------------------------------------

_SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_CHARS_PER_TOKEN = 4 # Temporary conversion factor


class SearchError(Exception):
    """Raised when web search fails (network error, no results, etc.)."""


def extract_search_query(text: str) -> str | None:
    """Extract the query from a <search>query</search> tag, or None if absent."""
    match = _SEARCH_TAG_RE.search(text)
    return match.group(1).strip() if match else None


def _token_budget_to_char_budget(token_budget: int) -> int:
    """Fallback conversion when no tokenizer-aware counter is available."""
    return max(1, token_budget) * _CHARS_PER_TOKEN


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


def _truncate_to_exact_token_budget(
    text: str,
    token_budget: int,
    count_tokens: Callable[[str], int],
) -> str:
    """Truncate text to the exact token budget, preferring sentence boundaries."""
    if not text:
        return ""

    budget = max(1, token_budget)
    if count_tokens(text) <= budget:
        return text

    sentence_ends = [idx + 1 for idx, ch in enumerate(text) if ch in ".!?"]
    best_sentence = ""
    left = 0
    right = len(sentence_ends) - 1
    while left <= right:
        mid = (left + right) // 2
        candidate = text[:sentence_ends[mid]]
        if count_tokens(candidate) <= budget:
            best_sentence = candidate
            left = mid + 1
        else:
            right = mid - 1
    if best_sentence:
        return best_sentence

    # Sentence itself is very long
    best_len = 0
    left = 1
    right = len(text)
    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= budget:
            best_len = mid
            left = mid + 1
        else:
            right = mid - 1
    return text[:best_len]


def truncate_to_token_budget(
    text: str,
    query: str,
    token_budget: int = 1024,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Select the most query-relevant paragraphs within a token budget.

    1. Split into paragraphs
    2. Score by keyword overlap with query
    3. Greedily select top-scoring paragraphs until budget
    4. Re-sort selected by original position (preserve reading order)

    When ``count_tokens`` is provided, paragraph selection and truncation are
    exact with respect to the active model tokenizer. Otherwise, the fallback
    char heuristic is used.
    """
    char_budget = _token_budget_to_char_budget(token_budget)
    exact_budget = max(1, token_budget)

    def _count(text_value: str) -> int:
        assert count_tokens is not None
        return count_tokens(text_value)

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
        if count_tokens is not None:
            return _truncate_to_exact_token_budget(
                paragraphs[0],
                exact_budget,
                _count,
            )
        return _truncate_at_sentence(paragraphs[0], char_budget)

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
    if count_tokens is None:
        remaining = char_budget
        for score, idx, para in scored:
            if len(para) <= remaining:
                selected.append((idx, para))
                remaining -= len(para) + 2  # account for \n\n join
            elif not selected:
                # First paragraph exceeds budget — truncate at sentence boundary
                selected.append((idx, _truncate_at_sentence(para, char_budget)))
                break
    else:
        token_count_cache: dict[str, int] = {}

        def _cached_count(text_value: str) -> int:
            cached = token_count_cache.get(text_value)
            if cached is None:
                cached = _count(text_value)
                token_count_cache[text_value] = cached
            return cached

        for score, idx, para in scored:
            candidate = selected + [(idx, para)]
            candidate.sort()
            candidate_text = "\n\n".join(candidate_para for _, candidate_para in candidate)
            if _cached_count(candidate_text) <= exact_budget:
                selected = candidate
            elif not selected:
                selected.append(
                    (
                        idx,
                        _truncate_to_exact_token_budget(
                            para,
                            exact_budget,
                            _cached_count,
                        ),
                    )
                )
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

    def __init__(
        self,
        max_results: int = 3,
        token_budget: int = 1024,
        api_key: str | None = None,
    ):
        self.max_results = max_results
        self.token_budget = token_budget
        self.api_key = api_key

    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        """Return provider-native search hits normalized to SearchResult."""

    def format_for_prompt(
        self,
        query: str,
        results: list[SearchResult],
        *,
        token_budget: int,
        fetch_and_extract: Callable[[str], str | None],
        count_tokens: Callable[[str], int] | None = None,
    ) -> str:
        """Provider-owned post-processing into model-ready text.

        Default behavior is tuned for web result providers (e.g., DDGS):
        relevance sort, fetch full page text, then truncate to budget. When
        ``count_tokens`` is available, the final prompt budget is exact.
        """
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
            page_text = fetch_and_extract(href)
            if page_text:
                body = page_text
            if body:
                entries.append((title, body))

        if not entries:
            raise SearchError("No relevant search results found.")

        # Smart truncate each result and format
        per_result = max(1, token_budget // len(entries))
        parts: list[str] = []
        for i, (title, body) in enumerate(entries, 1):
            prefix = f"[{i}] {title}\n"
            body_budget = per_result
            if count_tokens is not None:
                body_budget = max(1, per_result - count_tokens(prefix))
            truncated = truncate_to_token_budget(
                body,
                query,
                token_budget=body_budget,
                count_tokens=count_tokens,
            )
            parts.append(f"{prefix}{truncated}")

        output = "\n\n".join(parts)
        if count_tokens is not None:
            if count_tokens(output) > max(1, token_budget):
                output = _truncate_to_exact_token_budget(
                    output,
                    token_budget,
                    count_tokens,
                )
        else:
            # Hard limit by approximate tokens.
            char_budget = _token_budget_to_char_budget(token_budget)
            if len(output) > char_budget:
                output = _truncate_at_sentence(output, char_budget)
        return output


class DDGSSearchProvider(SearchProvider):
    """DuckDuckGo provider via ddgs."""
    def __init__(
        self,
        max_results: int = 3,
        token_budget: int = 1024,
        api_key: str | None = None,
    ):
        super().__init__(max_results, token_budget, api_key)
        print(
            "Note: ddgs/primp may print an impersonation warning; this is harmless and can be ignored.",
            file=sys.stderr,
        )

    def search(self, query: str) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            raw_results = DDGS().text(query, max_results=self.max_results)
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
    """Brave provider via official LLM Context API."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"

    def search(self, query: str) -> list[SearchResult]:
        api_key = self.api_key or os.environ.get("SEARCH_API_KEY")
        if not api_key:
            raise SearchError(
                "Brave search requires SEARCH_API_KEY in the environment."
            )

        params = urllib.parse.urlencode(
            {
                "q": query,
                # LLM context supports up to 50 result URLs.
                "count": max(1, min(self.max_results, 50)),
                "maximum_number_of_urls": max(1, min(self.max_results, 50)),
                "maximum_number_of_tokens": max(1, self.token_budget),
                "context_threshold_mode": "balanced",
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
                raw = resp.read()
                content_encoding = str(resp.headers.get("Content-Encoding", "")).lower()
                if "gzip" in content_encoding:
                    raw = gzip.decompress(raw)
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
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

        grounding = payload.get("grounding", {})
        sources = payload.get("sources", {})

        raw_results: list[dict] = []
        if isinstance(grounding, dict):
            generic = grounding.get("generic", [])
            if isinstance(generic, list):
                raw_results.extend(x for x in generic if isinstance(x, dict))

            poi = grounding.get("poi")
            if isinstance(poi, dict):
                raw_results.append(poi)

            map_items = grounding.get("map", [])
            if isinstance(map_items, list):
                raw_results.extend(x for x in map_items if isinstance(x, dict))

        entries: list[SearchResult] = []
        for item in raw_results:
            href = str(item.get("url", "")).strip()
            source_meta = sources.get(href, {}) if isinstance(sources, dict) else {}
            title = str(item.get("title", "") or source_meta.get("title", "")).strip()
            snippets = item.get("snippets", [])
            body = ""
            if isinstance(snippets, list) and snippets:
                body = "\n".join(str(x).strip() for x in snippets if x).strip()
            if not title:
                title = href
            if not href:
                continue
            entries.append(SearchResult(title=title, href=href, body=body))
        return entries

    def format_for_prompt(
        self,
        query: str,
        results: list[SearchResult],
        *,
        token_budget: int,
        fetch_and_extract: Callable[[str], str | None],
        count_tokens: Callable[[str], int] | None = None,
    ) -> str:
        """Brave LLM Context is already curated; pass grounding snippets through."""
        del query, fetch_and_extract # Intentionally unused coz Brave auto grounds results
        if not results:
            raise SearchError("No search results found.")

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.title or r.href
            body = r.body.strip()
            if body:
                parts.append(f"[{i}] {title}\n{body}")
            elif r.href:
                parts.append(f"[{i}] {title}\n{r.href}")

        output = "\n\n".join(parts).strip()
        if not output:
            raise SearchError("No relevant search results found.")
        if count_tokens is not None and count_tokens(output) > max(1, token_budget):
            output = _truncate_to_exact_token_budget(
                output,
                token_budget,
                count_tokens,
            )
        return output


def get_search_provider(
    name: str,
    *,
    max_results: int = 3,
    token_budget: int = 1024,
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
    return providers[name_normalized](
        max_results=max_results,
        token_budget=token_budget,
        api_key=api_key,
    )


class SearchClient:
    """Search pipeline over a provider."""

    def __init__(
        self,
        provider_name: str = "ddgs",
        max_results: int = 3,
        token_budget: int = 1024,
        api_key: str | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ):
        self.provider = get_search_provider(
            provider_name,
            max_results=max_results,
            token_budget=token_budget,
            api_key=api_key,
        )
        self.token_budget = token_budget
        self.count_tokens = count_tokens

    async def search(self, query: str) -> str:
        """Run search pipeline in a thread (providers + urllib are sync)."""
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> str:
        results = self.provider.search(query)
        return self.provider.format_for_prompt(
            query,
            results,
            token_budget=self.token_budget,
            fetch_and_extract=self._fetch_and_extract,
            count_tokens=self.count_tokens,
        )

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
