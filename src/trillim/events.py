# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Public SDK event types for structured chat streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(slots=True, frozen=True)
class ChatUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int


@dataclass(slots=True, frozen=True)
class ChatSearchStartedEvent:
    query: str
    type: Literal["search_started"] = "search_started"


@dataclass(slots=True, frozen=True)
class ChatSearchResultEvent:
    query: str
    content: str
    available: bool = True
    type: Literal["search_result"] = "search_result"


@dataclass(slots=True, frozen=True)
class ChatTokenEvent:
    text: str
    type: Literal["token"] = "token"


@dataclass(slots=True, frozen=True)
class ChatFinalTextEvent:
    text: str
    type: Literal["final_text"] = "final_text"


@dataclass(slots=True, frozen=True)
class ChatDoneEvent:
    text: str
    usage: ChatUsage
    type: Literal["done"] = "done"


ChatEvent: TypeAlias = (
    ChatSearchStartedEvent
    | ChatSearchResultEvent
    | ChatTokenEvent
    | ChatFinalTextEvent
    | ChatDoneEvent
)
