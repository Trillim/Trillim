# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Public ChatSession export for multi-turn LLM conversations.

Use ``llm.session(...)`` to create instances.
"""

from .server._llm import ChatSession

__all__ = ["ChatSession"]
