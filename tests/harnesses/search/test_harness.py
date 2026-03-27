"""Tests for the search harness."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from trillim import _model_store
from trillim.components.llm import ChatDoneEvent
from trillim.components.llm.public import LLM
from trillim.errors import SessionExhaustedError
from trillim.harnesses.search.provider import (
    FALLBACK_SEARCH_FAILURE_MESSAGE,
    SearchAuthenticationError,
    SearchError,
    SearchResult,
)
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
)
from trillim.harnesses.search._harness import _SearchHarness


class _SuccessfulSearch:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[str] = []

    async def search(self, query: str) -> str:
        self.calls.append(query)
        return self.content


class _FailingSearch:
    async def search(self, query: str) -> str:
        raise SearchError(query)


class _AuthFailingSearch:
    async def search(self, query: str) -> str:
        raise SearchAuthenticationError("Brave search failed: wrong SEARCH_API_KEY")


class SearchHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())
        _model_store.store_path_for_id("Trillim/fake").mkdir(parents=True, exist_ok=True)

    def _make_llm(self, *, responses, search_token_budget: int = 32) -> LLM:
        return LLM(
            "Trillim/fake",
            harness_name="search",
            search_provider="ddgs",
            search_token_budget=search_token_budget,
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=responses),
        )

    async def test_search_harness_appends_search_history_and_reports_final_usage(self):
        llm = self._make_llm(responses=["<search>cats</search>", "answer"])
        await llm.start()
        llm._harness._search = _SuccessfulSearch("curated cat results")

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        done = events[-1]
        self.assertIsInstance(done, ChatDoneEvent)
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "Find cats"},
                {"role": "assistant", "content": "<search>cats</search>"},
                {"role": "search", "content": "curated cat results"},
                {"role": "assistant", "content": "answer"},
            ),
        )
        self.assertEqual(
            done.usage.completion_tokens,
            len("answer"),
        )
        self.assertEqual(
            done.usage.prompt_tokens + done.usage.completion_tokens,
            llm._engine.cached_token_count,
        )
        self.assertEqual(done.usage.total_tokens, llm._engine.cached_token_count)
        self.assertEqual(done.usage.cached_tokens, llm._engine.last_cache_hit)
        await llm.stop()

    async def test_search_harness_uses_fallback_message_when_search_fails(self):
        llm = self._make_llm(responses=["<search>cats</search>", "answer"])
        await llm.start()
        llm._harness._search = _FailingSearch()

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            result = await session.chat(max_tokens=8)

        self.assertEqual(result, "answer")
        self.assertEqual(session.messages[2]["role"], "search")
        self.assertEqual(session.messages[2]["content"], FALLBACK_SEARCH_FAILURE_MESSAGE)
        await llm.stop()

    async def test_search_harness_auth_failures_leave_session_reusable(self):
        llm = self._make_llm(responses=["<search>cats</search>", "recovered"])
        await llm.start()
        llm._harness._search = _AuthFailingSearch()
        session = llm.open_session([{"role": "user", "content": "Find cats"}])

        with self.assertRaisesRegex(RuntimeError, "wrong SEARCH_API_KEY"):
            await session.chat(max_tokens=8)

        self.assertEqual(session.state, "open")
        self.assertEqual(
            session.messages,
            ({"role": "user", "content": "Find cats"},),
        )
        self.assertEqual(await session.chat(max_tokens=8), "recovered")
        await llm.stop()

    async def test_search_harness_trims_search_content_to_token_budget(self):
        llm = self._make_llm(
            responses=["<search>cats</search>", "answer"],
            search_token_budget=4,
        )
        await llm.start()
        llm._harness._search = _SuccessfulSearch("abcdefghijklmnop")

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            await session.chat(max_tokens=8)

        self.assertLessEqual(
            len(llm._tokenizer.encode(session.messages[2]["content"], add_special_tokens=False)),
            4,
        )
        await llm.stop()

    async def test_search_harness_exhausts_when_follow_up_prompt_crosses_session_limit(self):
        llm = self._make_llm(responses=["<search>cats</search>", "unused"])
        await llm.start()
        llm._harness._search = _SuccessfulSearch("x" * 48)
        session = llm.open_session([{"role": "user", "content": "Find cats"}])

        with patch("trillim.components.llm._session.SESSION_TOKEN_LIMIT", 80):
            with self.assertRaises(SessionExhaustedError):
                await session.chat(max_tokens=8)

        self.assertEqual(session.state, "exhausted")
        self.assertEqual(
            session.messages,
            ({"role": "user", "content": "Find cats"},),
        )
        await llm.stop()

    async def test_search_harness_without_tags_returns_buffered_text(self):
        llm = self._make_llm(responses=["hello"])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "final_text", "done"])
        self.assertEqual(events[-1].text, "hello")
        await llm.stop()

    async def test_search_harness_uses_engine_reported_usage(self):
        llm = self._make_llm(responses=["hello"])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
            prompt_tokens = len(session._prepare_generation())
            llm._engine.kv_positions = [prompt_tokens + 1]
            events = [event async for event in session.stream_chat(max_tokens=8)]

        done = events[-1]
        self.assertEqual(done.usage.prompt_tokens, prompt_tokens)
        self.assertEqual(done.usage.completion_tokens, 1)
        self.assertEqual(done.usage.total_tokens, prompt_tokens + 1)
        await llm.stop()

    async def test_search_harness_uses_real_search_client_pipeline(self):
        llm = self._make_llm(responses=["<search>cats</search>", "answer"])
        await llm.start()

        with patch(
            "trillim.harnesses.search.client.DDGSSearchProvider.search",
            return_value=[
                SearchResult(
                    title="Cats",
                    url="https://example.com/cats",
                    snippet="snippet",
                )
            ],
        ) as provider_search, patch(
            "trillim.harnesses.search.client.build_search_context",
            return_value="curated cat results",
        ) as build_context:
            async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
                result = await session.chat(max_tokens=8)

        self.assertEqual(result, "answer")
        provider_search.assert_called_once_with("cats", max_results=5)
        build_context.assert_called_once()
        self.assertEqual(session.messages[2]["role"], "search")
        self.assertEqual(session.messages[2]["content"], "curated cat results")
        await llm.stop()

    async def test_search_harness_emits_only_final_event_for_empty_buffered_text(self):
        llm = self._make_llm(responses=[""])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say nothing"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertEqual([event.type for event in events], ["final_text", "done"])
        self.assertEqual(events[-1].text, "")
        await llm.stop()

    async def test_search_harness_streams_final_generation_after_max_iterations(self):
        llm = self._make_llm(
            responses=["<search>cats</search>", "<search>dogs</search>", "done"],
        )
        await llm.start()
        llm._harness._search = _SuccessfulSearch("curated search results")

        async with llm.open_session([{"role": "user", "content": "Search twice"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertEqual(session.messages[1]["content"], "<search>cats</search>")
        self.assertEqual(session.messages[2]["role"], "search")
        self.assertEqual(session.messages[3]["content"], "<search>dogs</search>")
        self.assertEqual(session.messages[-1]["content"], "done")
        self.assertEqual(events[-1].text, "done")
        await llm.stop()

    async def test_search_harness_skips_empty_decode_chunks_in_final_streaming_phase(self):
        class _Engine:
            def __init__(self) -> None:
                self.tokenizer = FakeTokenizer()
                self.last_prompt_tokens = 3
                self.last_completion_tokens = 1
                self.last_cache_hit = 0

            async def generate(self, token_ids, **sampling):
                del token_ids, sampling
                yield 1
                yield 2

        class _Session:
            def __init__(self) -> None:
                self.messages = ({"role": "user", "content": "Find cats"},)
                self.committed = None

            def _prepare_generation(self, messages=None):
                del messages
                return [1, 2, 3]

            def _commit_messages(self, messages):
                self.committed = tuple(messages)

        harness = _SearchHarness(_Engine(), search_provider="ddgs", search_token_budget=16)
        harness._search = _SuccessfulSearch("curated cat results")
        session = _Session()

        with patch.object(
            harness,
            "_generate_buffered",
            return_value="<search>cats</search>",
        ), patch(
            "trillim.harnesses.search._harness.IncrementalDecoder.decode",
            side_effect=["", "k"],
        ):
            events = [event async for event in harness.stream_events(session, max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "final_text"])
        self.assertEqual(events[-1].text, "k")
        self.assertEqual(session.committed[-1]["content"], "k")

    async def test_search_harness_skips_empty_decode_chunks_in_final_generation(self):
        llm = self._make_llm(responses=["ab"])
        await llm.start()

        with patch(
            "trillim.harnesses.search._harness.IncrementalDecoder.decode",
            side_effect=["", "b"],
        ):
            async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
                events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "final_text", "done"])
        self.assertEqual(events[-1].text, "b")
        self.assertEqual(session.messages[-1]["content"], "b")
        await llm.stop()
