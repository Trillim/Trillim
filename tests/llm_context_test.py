# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the public LLM context inspection API."""

from types import SimpleNamespace
import unittest

import trillim
from trillim import ContextOverflowError
from trillim.server import LLM
from trillim.server._models import ServerState


class _FakeHarness:
    def __init__(self, token_count: int):
        self._token_count = token_count
        self.seen_messages = None

    def _prepare_tokens(self, messages):
        self.seen_messages = messages
        return list(range(self._token_count)), "prompt"


def _make_llm(token_count: int, max_context_tokens: int = 32) -> tuple[LLM, _FakeHarness]:
    llm = LLM("models/fake")
    llm.state = ServerState.RUNNING
    llm.engine = SimpleNamespace(
        arch_config=SimpleNamespace(max_position_embeddings=max_context_tokens)
    )
    harness = _FakeHarness(token_count)
    llm.harness = harness
    return llm, harness


class LLMContextTests(unittest.TestCase):
    def test_context_overflow_error_is_exported(self):
        self.assertIs(trillim.ContextOverflowError, ContextOverflowError)

    def test_count_tokens_uses_active_harness(self):
        llm, harness = _make_llm(token_count=7)
        messages = [{"role": "user", "content": "hello"}]

        token_count = llm.count_tokens(messages)

        self.assertEqual(token_count, 7)
        self.assertEqual(harness.seen_messages, messages)

    def test_max_context_tokens_reads_model_config(self):
        llm, _ = _make_llm(token_count=1, max_context_tokens=4096)

        self.assertEqual(llm.max_context_tokens, 4096)

    def test_validate_context_returns_token_count_when_within_limit(self):
        llm, _ = _make_llm(token_count=12, max_context_tokens=32)
        messages = [{"role": "user", "content": "safe"}]

        token_count = llm.validate_context(messages)

        self.assertEqual(token_count, 12)

    def test_validate_context_raises_typed_overflow_error(self):
        llm, _ = _make_llm(token_count=32, max_context_tokens=32)

        with self.assertRaises(ContextOverflowError) as ctx:
            llm.validate_context([{"role": "user", "content": "too long"}])

        self.assertEqual(ctx.exception.token_count, 32)
        self.assertEqual(ctx.exception.max_context_tokens, 32)
        self.assertIn("exceeds context window", str(ctx.exception))

    def test_public_context_api_requires_started_llm(self):
        llm = LLM("models/fake")

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.count_tokens([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            _ = llm.max_context_tokens


if __name__ == "__main__":
    unittest.main()
