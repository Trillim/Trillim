"""Tests for search result fetching and truncation."""

from __future__ import annotations

import socket
import sys
import types
import urllib.error
import urllib.request
import unittest
from unittest.mock import patch

from trillim.harnesses.search.fetch import (
    _fetch_and_extract,
    _open_request,
    _SafeRedirectHandler,
    _truncate_at_sentence,
    build_search_context,
    is_safe_url,
    resolves_to_safe_addresses,
    truncate_to_token_budget,
)
from trillim.harnesses.search.provider import SearchError, SearchResult


class _FetchResponseStub:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _limit: int):
        return self._payload


class SearchFetchTests(unittest.TestCase):
    def test_is_safe_url_rejects_local_targets(self):
        self.assertFalse(is_safe_url("file:///tmp/x"))
        self.assertFalse(is_safe_url("http://localhost/test"))
        self.assertFalse(is_safe_url("https://sub.localhost/test"))
        self.assertFalse(is_safe_url("https://127.0.0.1/test"))
        self.assertFalse(is_safe_url("https:///missing-host"))
        self.assertTrue(is_safe_url("https://example.com/test"))

    def test_build_search_context_limits_results_and_uses_fetch_bodies(self):
        results = [
            SearchResult(title=f"Title {index}", url=f"https://example.com/{index}")
            for index in range(6)
        ]
        calls: list[str] = []

        def fetcher(url: str, *, timeout: float, max_bytes: int) -> str:
            calls.append(url)
            self.assertEqual(timeout, 5.0)
            self.assertEqual(max_bytes, 200_000)
            return f"body from {url}"

        content = build_search_context(
            "cats",
            results,
            token_budget=40,
            fetcher=fetcher,
        )

        self.assertEqual(len(calls), 5)
        self.assertIn("[1] Title 0", content)
        self.assertIn("body from https://example.com/0", content)

    def test_build_search_context_raises_when_no_fetchable_results_exist(self):
        with self.assertRaisesRegex(SearchError, "search returned no results"):
            build_search_context("cats", [], token_budget=40)
        with self.assertRaisesRegex(SearchError, "no relevant fetchable results"):
            build_search_context(
                "cats",
                [SearchResult(title="Bad", url="file:///tmp/x", snippet="")],
                token_budget=40,
                fetcher=lambda *_args, **_kwargs: None,
            )

    def test_build_search_context_falls_back_to_snippets_and_skips_empty_truncations(self):
        result = SearchResult(title="", url="https://example.com", snippet="snippet body")

        content = build_search_context(
            "cats",
            [result],
            token_budget=20,
            fetcher=lambda *_args, **_kwargs: None,
        )
        self.assertIn("[1] https://example.com", content)
        self.assertIn("snippet body", content)

        with patch(
            "trillim.harnesses.search.fetch.truncate_to_token_budget",
            return_value="",
        ):
            with self.assertRaisesRegex(SearchError, "no relevant fetchable results"):
                build_search_context(
                    "cats",
                    [SearchResult(title="Title", url="https://example.com", snippet="snippet")],
                    token_budget=8,
                    fetcher=lambda *_args, **_kwargs: "body",
                )

        content = build_search_context(
            "cats",
            [
                SearchResult(title="Skip", url="https://example.com/skip", snippet=""),
                SearchResult(title="Keep", url="https://example.com/keep", snippet="kept snippet"),
            ],
            token_budget=12,
            fetcher=lambda url, **_kwargs: None if url.endswith("/skip") else "",
        )
        self.assertIn("[1] Keep", content)

    def test_build_search_context_skips_individual_fetch_failures_when_other_results_work(self):
        results = [
            SearchResult(title="Broken", url="https://example.com/bad", snippet=""),
            SearchResult(title="Working", url="https://example.com/good", snippet="good snippet"),
        ]

        def fetcher(url: str, *, timeout: float, max_bytes: int) -> str:
            del timeout, max_bytes
            if url.endswith("/bad"):
                raise SearchError("upstream timeout")
            return "good body"

        content = build_search_context(
            "cats",
            results,
            token_budget=20,
            fetcher=fetcher,
        )

        self.assertIn("[1] Working", content)
        self.assertIn("good body", content)
        self.assertNotIn("Broken", content)

    def test_truncate_to_token_budget_uses_rough_character_budget(self):
        text = "cats " * 100
        truncated = truncate_to_token_budget(text, "cats", token_budget=8)

        self.assertLessEqual(len(truncated), 32)

    def test_resolves_to_safe_addresses_and_truncation_helpers_cover_edge_cases(self):
        self.assertFalse(resolves_to_safe_addresses("https:///missing-host"))
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            side_effect=OSError("boom"),
        ):
            self.assertFalse(resolves_to_safe_addresses("https://example.com"))
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[],
        ):
            self.assertFalse(resolves_to_safe_addresses("https://example.com"))
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("bad-ip", 443))],
        ):
            self.assertFalse(resolves_to_safe_addresses("https://example.com"))
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            self.assertTrue(resolves_to_safe_addresses("https://example.com"))
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 4443))],
        ):
            self.assertTrue(resolves_to_safe_addresses("https://example.com:4443"))

        self.assertEqual(truncate_to_token_budget(" \n\n ", "cats", token_budget=4), "")
        self.assertEqual(
            truncate_to_token_budget("Only cats matter.", "cats", token_budget=8),
            "Only cats matter.",
        )
        selected = truncate_to_token_budget(
            "dogs only\n\ncats are great\n\ncats and dogs together",
            "cats dogs",
            token_budget=8,
        )
        self.assertIn("cats and dogs together", selected)
        self.assertEqual(
            truncate_to_token_budget(
                "short pick\n\n" + ("z" * 120),
                "short",
                token_budget=4,
            ),
            "short pick",
        )
        self.assertEqual(
            truncate_to_token_budget("x" * 200, "cats", token_budget=2),
            "x" * 8,
        )

        self.assertEqual(_truncate_at_sentence("Hello. Goodbye.", 7), "Hello.")
        self.assertEqual(_truncate_at_sentence("abcdef", 3), "abc")

    def test_truncate_to_token_budget_prefers_truncated_best_match_when_nothing_else_fits(self):
        long_match = "cats " + ("x" * 200)
        fallback = "dogs only"

        selected = truncate_to_token_budget(
            f"{long_match}\n\n{fallback}",
            "cats",
            token_budget=4,
        )

        self.assertEqual(selected, long_match[:16])

    def test_truncate_to_token_budget_breaks_after_budget_runs_out_with_prior_selection(self):
        selected = truncate_to_token_budget(
            "cats short\n\n" + ("cats " + ("x" * 60)),
            "cats",
            token_budget=5,
        )

        self.assertEqual(selected, "cats short")

    def test_truncate_to_token_budget_keeps_multiple_ranked_paragraphs_when_all_fit(self):
        selected = truncate_to_token_budget(
            "cats alpha\n\ncats beta",
            "cats",
            token_budget=20,
        )

        self.assertEqual(selected, "cats alpha\n\ncats beta")

    def test_fetch_and_extract_reads_html_and_uses_trafilatura(self):
        fake_module = types.SimpleNamespace(
            extract=lambda html, **_kwargs: f"parsed:{html[:5]}"
        )
        with patch.dict(sys.modules, {"trafilatura": fake_module}):
            with patch(
                "trillim.harnesses.search.fetch.socket.getaddrinfo",
                return_value=[
                    (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        6,
                        "",
                        ("93.184.216.34", 443),
                    )
                ],
            ):
                with patch(
                    "trillim.harnesses.search.fetch._open_request",
                    return_value=_FetchResponseStub(b"<html>example</html>"),
                ):
                    text = _fetch_and_extract(
                        "https://example.com",
                        timeout=5.0,
                        max_bytes=200_000,
                    )

        self.assertEqual(text, "parsed:<html")

    def test_fetch_and_extract_returns_none_on_errors(self):
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        ):
            with patch(
                "trillim.harnesses.search.fetch._open_request",
                side_effect=TimeoutError("boom"),
            ):
                self.assertIsNone(
                    _fetch_and_extract(
                        "https://example.com",
                        timeout=5.0,
                        max_bytes=200_000,
                    )
                )

    def test_fetch_and_extract_rejects_hostnames_resolving_to_private_addresses(self):
        with patch(
            "trillim.harnesses.search.fetch.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("127.0.0.1", 443),
                )
            ],
        ):
            with patch("trillim.harnesses.search.fetch._open_request") as open_request:
                self.assertIsNone(
                    _fetch_and_extract(
                        "https://example.com",
                        timeout=5.0,
                        max_bytes=200_000,
                    )
                )

        open_request.assert_not_called()

    def test_safe_redirect_handler_rejects_unsafe_redirect_targets(self):
        handler = _SafeRedirectHandler()
        request = urllib.request.Request("https://example.com/start")

        with patch(
            "trillim.harnesses.search.fetch.resolves_to_safe_addresses",
            return_value=False,
        ):
            with self.assertRaises(urllib.error.URLError):
                handler.redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {"Location": "https://redirected.example"},
                    "https://redirected.example",
                )

    def test_safe_redirect_handler_and_open_request_allow_safe_paths(self):
        handler = _SafeRedirectHandler()
        request = urllib.request.Request("https://example.com/start")

        with patch(
            "trillim.harnesses.search.fetch.resolves_to_safe_addresses",
            return_value=True,
        ), patch(
            "urllib.request.HTTPRedirectHandler.redirect_request",
            return_value="redirected",
        ) as redirect_request:
            result = handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {"Location": "/next"},
                "/next",
            )

        self.assertEqual(result, "redirected")
        redirect_request.assert_called_once()

        opener = types.SimpleNamespace(open=lambda request, timeout: (request.full_url, timeout))
        with patch("urllib.request.build_opener", return_value=opener) as build_opener:
            opened = _open_request(urllib.request.Request("https://example.com"), timeout=1.5)

        self.assertEqual(opened, ("https://example.com", 1.5))
        self.assertTrue(build_opener.called)
