"""Tests for the Brave provider."""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch
import urllib.error

from trillim.harnesses.search._brave import BraveSearchProvider
from trillim.harnesses.search.provider import SearchAuthenticationError


class _ResponseStub:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


class BraveProviderTests(unittest.TestCase):
    def test_brave_provider_requires_search_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                SearchAuthenticationError,
                "SEARCH_API_KEY",
            ):
                BraveSearchProvider(token_budget=32).search("cats", max_results=5)

    def test_brave_provider_reports_wrong_keys(self):
        error = urllib.error.HTTPError(
            "https://example.com",
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )
        with patch.dict(os.environ, {"SEARCH_API_KEY": "bad-key"}, clear=True):
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(
                    SearchAuthenticationError,
                    "wrong SEARCH_API_KEY",
                ):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

    def test_brave_provider_parses_results(self):
        payload = json.dumps(
            {
                "grounding": {
                    "generic": [
                        {
                            "url": "https://example.com",
                            "title": "Example",
                            "snippets": ["One", "Two"],
                        }
                    ]
                },
                "sources": {},
            }
        ).encode("utf-8")
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload),
            ):
                results = BraveSearchProvider(token_budget=32).search(
                    "cats",
                    max_results=5,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Example")
        self.assertEqual(results[0].url, "https://example.com")
        self.assertEqual(results[0].snippet, "One Two")
