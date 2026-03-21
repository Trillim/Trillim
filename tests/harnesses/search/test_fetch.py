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
    _SafeRedirectHandler,
    build_search_context,
    is_safe_url,
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
        self.assertFalse(is_safe_url("https://127.0.0.1/test"))
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
        with self.assertRaisesRegex(SearchError, "no relevant fetchable results"):
            build_search_context(
                "cats",
                [SearchResult(title="Bad", url="file:///tmp/x", snippet="")],
                token_budget=40,
                fetcher=lambda *_args, **_kwargs: None,
            )

    def test_truncate_to_token_budget_uses_rough_character_budget(self):
        text = "cats " * 100
        truncated = truncate_to_token_budget(text, "cats", token_budget=8)

        self.assertLessEqual(len(truncated), 32)

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
