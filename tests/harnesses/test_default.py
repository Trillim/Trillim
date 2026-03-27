"""Tests for the default harness."""

from __future__ import annotations

from pathlib import Path
import unittest

from trillim.components.llm._config import SamplingDefaults
from trillim.harnesses._default import _DefaultHarness
from tests.components.llm.support import FakeEngine, FakeTokenizer, make_runtime_model


class _SessionStub:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.final_text = None

    def _prepare_generation(self):
        return list(self.token_ids)

    def _commit_assistant_turn(self, text: str):
        self.final_text = text


class _SparseTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        mapping = {
            (1,): "",
            (1, 2): "b",
            (2,): "b",
        }
        return mapping[tuple(token_ids)]


class _SparseEngine:
    def __init__(self) -> None:
        self.tokenizer = _SparseTokenizer()
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_cache_hit = 0

    async def generate(self, token_ids, **sampling):
        del sampling
        self.last_prompt_tokens = len(token_ids)
        yield 1
        yield 2
        self.last_completion_tokens = 1


class DefaultHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_harness_streams_tokens_and_final_text(self):
        engine = FakeEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            responses=["ok"],
        )
        harness = _DefaultHarness(engine)
        session = _SessionStub([1, 2, 3])

        events = [event async for event in harness.stream_events(session, max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "token", "final_text"])
        self.assertEqual(events[-1].text, "ok")
        self.assertEqual(harness.prompt_tokens, 3)
        self.assertEqual(harness.completion_tokens, 2)
        self.assertEqual(harness.cached_tokens, 0)
        self.assertEqual(session.final_text, "ok")

    async def test_default_harness_uses_engine_reported_usage(self):
        engine = FakeEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            responses=["ok"],
            kv_positions=[4],
        )
        harness = _DefaultHarness(engine)
        session = _SessionStub([1, 2, 3])

        events = [event async for event in harness.stream_events(session, max_tokens=8)]

        self.assertEqual(events[-1].text, "ok")
        self.assertEqual(harness.prompt_tokens, 3)
        self.assertEqual(harness.completion_tokens, 1)

    async def test_default_harness_skips_empty_decode_chunks(self):
        harness = _DefaultHarness(_SparseEngine())
        session = _SessionStub([1, 2, 3])

        events = [event async for event in harness.stream_events(session, max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "final_text"])
        self.assertEqual(events[-1].text, "b")
        self.assertEqual(session.final_text, "b")
