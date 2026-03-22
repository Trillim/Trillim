"""Private default harness for direct single-pass LLM generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from trillim.components.llm._events import ChatEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.components.llm._session import _ChatSession
from trillim.harnesses._base import _Harness


class _DefaultHarness(_Harness):
    """Run one direct generation with no tool orchestration."""

    async def stream_events(
        self,
        session: _ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Stream token and final-text events for one generation."""
        self._reset_usage()
        token_ids = session._prepare_generation()
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self._engine.generate(token_ids=token_ids, **sampling):
            chunk = decoder.decode(token_id)
            if not chunk:
                continue
            full_text += chunk
            yield ChatTokenEvent(text=chunk)
        session._commit_assistant_turn(full_text)
        self._apply_engine_usage()
        yield ChatFinalTextEvent(text=full_text)
