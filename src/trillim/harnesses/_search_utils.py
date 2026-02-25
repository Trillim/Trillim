# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Search utilities — tag extraction, smart truncation, DuckDuckGo search."""

import asyncio
import os
import re
import urllib.request

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
# DuckDuckGo search
# ---------------------------------------------------------------------------


class DuckDuckGoSearch:
    """DuckDuckGo search with trafilatura extraction and smart truncation."""

    def __init__(self, max_results: int = 3, char_budget: int = 2000):
        self.max_results = max_results
        self.char_budget = char_budget

    async def search(self, query: str) -> str:
        """Run search in a thread (ddgs and urllib are sync)."""
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> str:
        from ddgs import DDGS

        try:
            # Suppress C-level stderr from primp's impersonation warnings
            fd = os.dup(2)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            try:
                results = DDGS().text(query, max_results=self.max_results)
            finally:
                os.dup2(fd, 2)
                os.close(fd)
        except Exception as exc:
            raise SearchError(f"Search unavailable: {exc}") from exc

        if not results:
            raise SearchError("No search results found.")

        # If query mentions a year, boost results that reference it
        year_match = re.search(r"\b(\d{4})\b", query)
        if year_match:
            year = year_match.group(1)
            results.sort(
                key=lambda r: year not in r.get("title", "") + r.get("body", ""),
            )

        # Fetch and extract full page content for each result
        entries: list[tuple[str, str]] = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")  # DDGS snippet as fallback

            if href:
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
