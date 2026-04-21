"""Incremental token decoding helpers."""

from __future__ import annotations


_MAX_PENDING_TOKENS = 32


class IncrementalDecoder:
    """Decode token IDs one at a time while preserving spacing."""

    def __init__(self, tokenizer) -> None:
        """Create a decoder for one generation stream."""
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted_text = ""

    def decode(self, token_id: int) -> str:
        """Decode a token ID after any incomplete byte sequence has settled."""
        self._token_ids.append(token_id)
        decoded = self._decode_tokens(self._token_ids)
        if not decoded.startswith(self._emitted_text):
            return ""
        text = decoded[len(self._emitted_text) :]
        if text.endswith("\ufffd"):
            text = text.rstrip("\ufffd")
        self._emitted_text += text
        self._compact_pending_tokens()
        return text

    def reset(self) -> None:
        """Reset decoder state for a new generation."""
        self._token_ids.clear()
        self._emitted_text = ""

    def _compact_pending_tokens(self) -> None:
        overflow = len(self._token_ids) - _MAX_PENDING_TOKENS
        if overflow <= 0:
            return
        remaining = self._token_ids[overflow:]
        remaining_decoded = self._decode_tokens(remaining)
        suffix = self._emitted_suffix_for(remaining_decoded)
        if suffix is None:
            return
        self._token_ids = remaining
        self._emitted_text = suffix

    def _emitted_suffix_for(self, decoded: str) -> str | None:
        if not self._emitted_text or not decoded:
            return ""
        start = max(0, len(self._emitted_text) - len(decoded))
        for index in range(start, len(self._emitted_text)):
            suffix = self._emitted_text[index:]
            if decoded.startswith(suffix):
                return suffix
        return None

    def _decode_tokens(self, token_ids: list[int]) -> str:
        try:
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
