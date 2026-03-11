# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Public SDK exceptions."""


class ContextOverflowError(ValueError):
    """Raised when a prompt exceeds the active model context window."""

    def __init__(
        self,
        token_count: int,
        max_context_tokens: int,
        detail: str | None = None,
    ) -> None:
        self.token_count = token_count
        self.max_context_tokens = max_context_tokens
        message = detail or (
            f"Prompt length ({token_count} tokens) exceeds context window "
            f"({max_context_tokens})."
        )
        super().__init__(message)
