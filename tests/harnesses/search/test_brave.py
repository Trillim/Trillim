"""Tests for the Brave provider."""

from __future__ import annotations

import gzip
import io
import json
import os
import unittest
from unittest.mock import patch
import urllib.error

from trillim.harnesses.search._brave import BraveSearchProvider
from trillim.harnesses.search.provider import SearchAuthenticationError, SearchError


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

    def test_brave_provider_handles_gzip_and_source_title_fallbacks(self):
        payload = gzip.compress(
            json.dumps(
                {
                    "grounding": {
                        "generic": [
                            {
                                "url": "https://example.com/generic",
                                "title": "",
                                "snippets": ["One", "Two"],
                            }
                        ],
                        "poi": {
                            "url": "https://example.com/poi",
                            "title": "Point",
                            "snippets": [],
                        },
                        "map": [
                            {
                                "url": "https://example.com/map",
                                "title": "Mapped",
                                "snippets": ["Map snippet"],
                            }
                        ],
                    },
                    "sources": {
                        "https://example.com/generic": {"title": "Fallback Title"},
                    },
                }
            ).encode("utf-8")
        )
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload, headers={"Content-Encoding": "gzip"}),
            ):
                results = BraveSearchProvider(token_budget=0).search("cats", max_results=5)

        self.assertEqual([result.title for result in results], ["Fallback Title", "Point", "Mapped"])
        self.assertEqual(results[0].snippet, "One Two")

    def test_brave_provider_wraps_backend_failures_and_unusable_payloads(self):
        payload = json.dumps({"grounding": {}, "sources": {}}).encode("utf-8")
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            error = urllib.error.HTTPError(
                "https://example.com",
                500,
                "server error",
                hdrs=None,
                fp=io.BytesIO(b"{}"),
            )
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(SearchError, "HTTP 500"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

            with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(SearchError, "Brave search failed: boom"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(b"{not json"),
            ):
                with self.assertRaisesRegex(SearchError, "invalid JSON"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload),
            ):
                with self.assertRaisesRegex(SearchError, "no usable results"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

    def test_brave_provider_wraps_corrupt_gzip_payloads(self):
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(b"not-gzip", headers={"Content-Encoding": "gzip"}),
            ):
                with self.assertRaisesRegex(SearchError, "Brave search failed"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

    def test_brave_provider_ignores_non_mapping_source_metadata(self):
        payload = json.dumps(
            {
                "grounding": {
                    "generic": [
                        {
                            "url": "https://example.com",
                            "title": "",
                            "snippets": [],
                        }
                    ]
                },
                "sources": {
                    "https://example.com": "not-a-dict",
                },
            }
        ).encode("utf-8")

        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload),
            ):
                results = BraveSearchProvider(token_budget=32).search("cats", max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "https://example.com")

    def test_brave_provider_handles_nonstandard_grounding_shapes_and_empty_results(self):
        payload = json.dumps(
            {
                "grounding": {
                    "generic": {"bad": "shape"},
                    "poi": [],
                    "map": {"also": "bad"},
                },
                "sources": [],
            }
        ).encode("utf-8")
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload),
            ):
                with self.assertRaisesRegex(SearchError, "no usable results"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

        payload = json.dumps({"grounding": [], "sources": {}}).encode("utf-8")
        with patch.dict(os.environ, {"SEARCH_API_KEY": "good-key"}, clear=True):
            with patch(
                "urllib.request.urlopen",
                return_value=_ResponseStub(payload),
            ):
                with self.assertRaisesRegex(SearchError, "no usable results"):
                    BraveSearchProvider(token_budget=32).search("cats", max_results=5)

        payload = json.dumps(
            {
                "grounding": {
                    "generic": [
                        {
                            "url": "",
                            "title": "skip",
                            "snippets": [],
                        },
                        {
                            "url": "https://example.com",
                            "title": "Example",
                            "snippets": "not-a-list",
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
                results = BraveSearchProvider(token_budget=32).search("cats", max_results=5)

        self.assertEqual(results[0].snippet, "")
