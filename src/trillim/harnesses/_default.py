# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""DefaultHarness — passthrough harness with no tool use."""

from collections.abc import AsyncIterator
from typing import Any

from trillim.events import ChatFinalTextEvent, ChatTokenEvent, ChatEvent
from trillim.token_utils import IncrementalDecoder
from ._base import Harness


class DefaultHarness(Harness):
    """Passthrough harness — single generation, no tool calls."""

    async def stream_events(
        self,
        session,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Stream tokens directly from the engine."""
        self._last_completion_tokens = 0
        token_ids, prompt_str = session._prepare_reply()
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        generated_token_ids: list[int] = []
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            self._last_completion_tokens += 1
            generated_token_ids.append(token_id)
            chunk = decoder.decode(token_id)
            full_text += chunk
            yield ChatTokenEvent(text=chunk)

        session._finalize_assistant(full_text, generated_token_ids)
        yield ChatFinalTextEvent(text=full_text)
