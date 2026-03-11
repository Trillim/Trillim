# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Harness ABC — abstract base for inference harnesses that steer multi-step execution."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trillim.engine import InferenceEngine
from trillim.events import ChatEvent, ChatTokenEvent


class Harness(abc.ABC):
    """Abstract base for inference harnesses that steer multi-step execution.

    Subclasses implement stream_events() for full orchestration.
    """

    DEBUG: ClassVar[bool] = False

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self._last_completion_tokens = 0

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def arch_config(self):
        return self.engine.arch_config

    async def run(self, messages: list[dict], **sampling: Any) -> AsyncIterator[str]:
        """Compatibility text stream built from structured chat events."""
        async for event in self.stream_events(messages, **sampling):
            if isinstance(event, ChatTokenEvent):
                yield event.text

    @abc.abstractmethod
    async def stream_events(
        self,
        messages: list[dict],
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Structured orchestration loop for app-facing streaming APIs."""
        ...
        yield  # type: ignore  # abstract async generator

    def _update_cache(self, messages: list[dict]) -> None:
        """Update the engine's cached_prompt_str for KV cache reuse."""
        tokenizer = self.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        if has_template:
            self.engine._cached_prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

    def _prepare_tokens(self, messages: list[dict]) -> tuple[list[int], str | None]:
        """Render messages via chat template and encode to token IDs.

        Returns (token_ids, prompt_str). prompt_str is for string-level KV
        cache matching, or None if no chat template.
        """
        tokenizer = self.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        if has_template:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            return token_ids, prompt_str
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"
            token_ids = tokenizer.encode(prompt)
            return token_ids, None
