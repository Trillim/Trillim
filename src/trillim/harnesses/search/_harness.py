"""Private search harness for models that emit ``<search>...</search>`` tags."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from trillim.components.llm._events import ChatEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.components.llm._session import _ChatSession
from trillim.harnesses._base import _Harness
from trillim.harnesses.search.client import SearchClient
from trillim.harnesses.search.metrics import SearchMetrics
from trillim.harnesses.search.provider import (
    FALLBACK_SEARCH_FAILURE_MESSAGE,
    MAX_SEARCH_ITERATIONS,
    SearchAuthenticationError,
    SearchError,
    extract_search_query,
)


class _SearchHarness(_Harness):
    """Run a bounded search loop before streaming the final answer."""

    def __init__(
        self,
        engine,
        *,
        search_provider: str,
        search_token_budget: int,
        _search_client_factory=SearchClient,
    ) -> None:
        super().__init__(engine)
        self._search = _search_client_factory(
            provider_name=search_provider,
            token_budget=search_token_budget,
        )
        self._search_token_budget = search_token_budget

    async def stream_events(
        self,
        session: _ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Run buffered search iterations, then stream the final answer."""
        self._reset_usage()
        metrics = SearchMetrics()
        working_messages = [message.copy() for message in session.messages]
        for _ in range(MAX_SEARCH_ITERATIONS - 1):
            token_ids = session._prepare_generation(messages=working_messages)
            full_text = await self._generate_buffered(
                token_ids,
                **sampling,
            )
            query = extract_search_query(full_text)
            if query is None:
                metrics.record_generation(
                    prompt_tokens=self._engine.last_prompt_tokens,
                    completion_tokens=self._engine.last_completion_tokens,
                    cached_tokens=self._engine.last_cache_hit,
                )
                self._apply_metrics(metrics)
                working_messages.append({"role": "assistant", "content": full_text})
                session._commit_messages(working_messages)
                if full_text:
                    yield ChatTokenEvent(text=full_text)
                yield ChatFinalTextEvent(text=full_text)
                return
            try:
                search_content = await self._search.search(query)
            except SearchAuthenticationError:
                self._apply_metrics(metrics)
                raise
            except SearchError:
                search_content = FALLBACK_SEARCH_FAILURE_MESSAGE
            if search_content != FALLBACK_SEARCH_FAILURE_MESSAGE:
                search_content = self._trim_search_content(search_content)
            working_messages.append({"role": "assistant", "content": full_text})
            working_messages.append({"role": "search", "content": search_content})

        token_ids = session._prepare_generation(messages=working_messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self._engine.generate(token_ids=token_ids, **sampling):
            chunk = decoder.decode(token_id)
            if not chunk:
                continue
            full_text += chunk
            yield ChatTokenEvent(text=chunk)
        metrics.record_generation(
            prompt_tokens=self._engine.last_prompt_tokens,
            completion_tokens=self._engine.last_completion_tokens,
            cached_tokens=self._engine.last_cache_hit,
        )
        self._apply_metrics(metrics)
        working_messages.append({"role": "assistant", "content": full_text})
        session._commit_messages(working_messages)
        yield ChatFinalTextEvent(text=full_text)

    async def _generate_buffered(
        self,
        token_ids: list[int],
        **sampling: Any,
    ) -> str:
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self._engine.generate(token_ids=token_ids, **sampling):
            full_text += decoder.decode(token_id)
        return full_text

    def _trim_search_content(self, content: str) -> str:
        token_ids = list(
            self.tokenizer.encode(
                content,
                add_special_tokens=False,
            )
        )
        if len(token_ids) <= self._search_token_budget:
            return content
        return self.tokenizer.decode(
            token_ids[: self._search_token_budget],
            skip_special_tokens=True,
        ).strip()

    def _apply_metrics(self, metrics: SearchMetrics) -> None:
        self._prompt_tokens = metrics.prompt_tokens
        self._completion_tokens = metrics.completion_tokens
        self._cached_tokens = metrics.cached_tokens
