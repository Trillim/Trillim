# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Backend prompt-cache planning and commit helpers for inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSnapshot:
    """Exact prompt state that may be used for backend cache planning."""

    token_ids: tuple[int, ...]
    prompt_str: str | None = None

    @classmethod
    def create(
        cls,
        token_ids: Sequence[int],
        prompt_str: str | None = None,
    ) -> PromptSnapshot:
        return cls(tuple(token_ids), prompt_str)


@dataclass(frozen=True)
class CachePlan:
    """How the next request should interact with the backend cache."""

    request: PromptSnapshot
    delta_tokens: tuple[int, ...]
    reset_flag: int
    cache_hit: int


class PromptCacheManager:
    """Own the prompt prefix that the inference subprocess cache represents."""

    def __init__(self) -> None:
        self.clear()

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self._token_ids

    @property
    def prompt_str(self) -> str | None:
        return self._prompt_str

    @property
    def last_cache_hit(self) -> int:
        return self._last_cache_hit

    def clear(self) -> None:
        self._token_ids: tuple[int, ...] = ()
        self._prompt_str: str | None = None
        self._last_cache_hit = 0

    def restore(
        self,
        snapshot: PromptSnapshot | None,
        *,
        last_cache_hit: int = 0,
    ) -> None:
        if snapshot is None:
            self.clear()
            return
        self._token_ids = snapshot.token_ids
        self._prompt_str = snapshot.prompt_str
        self._last_cache_hit = last_cache_hit

    def plan(
        self,
        request: PromptSnapshot,
        *,
        encode_suffix: Callable[[str], Sequence[int]],
    ) -> CachePlan:
        if request.prompt_str is not None:
            return self._plan_string_request(request, encode_suffix=encode_suffix)
        return self._plan_token_request(request)

    def _plan_string_request(
        self,
        request: PromptSnapshot,
        *,
        encode_suffix: Callable[[str], Sequence[int]],
    ) -> CachePlan:
        # Chat-style requests only reuse cache when the backend can prove the
        # exact cached prompt prefix. Falling back to token-prefix matching here
        # risks reusing shared template headers from a different conversation.
        if self._prompt_str is None or not request.prompt_str.startswith(self._prompt_str):
            return CachePlan(
                request=request,
                delta_tokens=request.token_ids,
                reset_flag=1,
                cache_hit=0,
            )

        suffix = request.prompt_str[len(self._prompt_str):]
        return CachePlan(
            request=request,
            delta_tokens=tuple(encode_suffix(suffix)),
            reset_flag=0,
            cache_hit=len(self._token_ids),
        )

    def _plan_token_request(self, request: PromptSnapshot) -> CachePlan:
        match_len = 0
        limit = min(len(request.token_ids), len(self._token_ids))
        while match_len < limit and request.token_ids[match_len] == self._token_ids[match_len]:
            match_len += 1

        if match_len > 0 and match_len == len(self._token_ids):
            delta_tokens = request.token_ids[match_len:]
            reset_flag = 0
        else:
            delta_tokens = request.token_ids
            reset_flag = 1
            match_len = 0

        return CachePlan(
            request=request,
            delta_tokens=delta_tokens,
            reset_flag=reset_flag,
            cache_hit=match_len,
        )

    def commit_generation(
        self,
        plan: CachePlan,
        *,
        generated_token_ids: Sequence[int],
        kv_position: int,
    ) -> None:
        combined = plan.request.token_ids + tuple(generated_token_ids)
        self._token_ids = combined[:kv_position]
        self._last_cache_hit = plan.cache_hit
        if plan.request.prompt_str is not None and kv_position == len(plan.request.token_ids):
            self._prompt_str = plan.request.prompt_str
        else:
            self._prompt_str = None

    def finalize_prompt(self, snapshot: PromptSnapshot) -> bool:
        if snapshot.token_ids != self._token_ids:
            self._prompt_str = None
            return False
        self._prompt_str = snapshot.prompt_str
        return True
