# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the interactive CLI chat helpers."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trillim import ChatSession
from trillim.events import ChatSearchResultEvent, ChatSearchStartedEvent, ChatTokenEvent
import trillim.inference as inference


class _Tokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True):
        return list(range(len(text)))


class _Engine:
    def __init__(self, *, max_context_tokens: int):
        self.model_dir = "models/fake"
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self.tokenizer = _Tokenizer()
        self.reset_calls = 0

    def reset_prompt_cache(self) -> None:
        self.reset_calls += 1


class _Harness:
    def __init__(self, *, max_context_tokens: int):
        self.engine = _Engine(max_context_tokens=max_context_tokens)
        self.arch_config = self.engine.arch_config


class _LLM:
    def __init__(self, *, max_context_tokens: int):
        self.model_name = "fake"
        self.engine = _Engine(max_context_tokens=max_context_tokens)
        self.harness = _Harness(max_context_tokens=max_context_tokens)
        self.harness.engine = self.engine
        self._session_generation = 0

    def _require_started(self):
        return self.engine, self.harness

    def _chat_sampling(self, **sampling):
        return sampling

    @property
    def max_context_tokens(self) -> int:
        return self.engine.arch_config.max_position_embeddings

    def session(self, messages: list[dict] | None = None):
        copied = [{"role": m["role"], "content": m["content"]} for m in messages or []]
        return ChatSession(self, copied)


class _SessionStub:
    def __init__(self):
        self.sampling = None

    async def stream_chat(self, **sampling):
        self.sampling = sampling
        yield ChatSearchStartedEvent(query="cats")
        yield ChatSearchResultEvent(
            query="cats",
            content="curated result",
            available=True,
        )
        yield ChatSearchResultEvent(
            query="cats",
            content="Search unavailable",
            available=False,
        )
        yield ChatTokenEvent(text="O")
        yield ChatTokenEvent(text="K")


class InferenceLoopTests(unittest.TestCase):
    def test_run_chat_loop_uses_chat_sessions_and_filters_sampling_params(self):
        loop = asyncio.new_event_loop()
        llm = _LLM(max_context_tokens=128)
        streamed: list[tuple[type, tuple[dict, ...], dict]] = []

        async def fake_stream_response(chat, sampling_params):
            streamed.append((type(chat), chat.messages, dict(sampling_params)))

        try:
            with (
                patch("trillim.inference._make_key_bindings", return_value=object()),
                patch(
                    "trillim.inference.better_input",
                    side_effect=["hello", "/new", "again", "q"],
                ),
                patch("trillim.inference._stream_response", new=fake_stream_response),
                patch("builtins.print"),
            ):
                inference._run_chat_loop(
                    loop,
                    llm,
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "rep_penalty_lookback": 64,
                    },
                )
        finally:
            loop.close()

        self.assertEqual(
            streamed,
            [
                (
                    ChatSession,
                    ({"role": "user", "content": "hello"},),
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                    },
                ),
                (
                    ChatSession,
                    ({"role": "user", "content": "again"},),
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                    },
                ),
            ],
        )
        self.assertEqual(llm.engine.reset_calls, 1)

    def test_run_chat_loop_resets_to_latest_message_and_skips_oversized_input(self):
        loop = asyncio.new_event_loop()
        llm = _LLM(max_context_tokens=30)
        streamed: list[tuple[dict, ...]] = []

        async def fake_stream_response(chat, sampling_params):
            streamed.append(chat.messages)
            chat._append_message("assistant", "ok")

        try:
            with (
                patch("trillim.inference._make_key_bindings", return_value=object()),
                patch(
                    "trillim.inference.better_input",
                    side_effect=[
                        "hi",
                        "yo",
                        "this message is definitely too long",
                        "q",
                    ],
                ),
                patch("trillim.inference._stream_response", new=fake_stream_response),
                patch("builtins.print") as print_mock,
            ):
                inference._run_chat_loop(loop, llm, {})
        finally:
            loop.close()

        self.assertEqual(
            streamed,
            [
                ({"role": "user", "content": "hi"},),
                ({"role": "user", "content": "yo"},),
            ],
        )
        self.assertEqual(llm.engine.reset_calls, 2)
        printed = [call.args[0] for call in print_mock.call_args_list if call.args]
        self.assertEqual(
            printed.count("Context window full (30 tokens). Starting new conversation."),
            2,
        )
        self.assertIn(
            "Last message exceeds the context window (30 tokens). Shorten it and try again.",
            printed,
        )


class InferenceStreamingTests(unittest.TestCase):
    def test_stream_response_prints_status_markers_and_tokens(self):
        session = _SessionStub()

        with patch("builtins.print") as print_mock:
            asyncio.run(
                inference._stream_response(
                    session,
                    {"temperature": 0.4, "max_tokens": 32},
                )
            )

        self.assertEqual(session.sampling, {"temperature": 0.4, "max_tokens": 32})
        self.assertEqual(
            print_mock.call_args_list,
            [
                unittest.mock.call("[Searching: cats]", flush=True),
                unittest.mock.call("[Synthesizing...]", flush=True),
                unittest.mock.call("[Search unavailable]", flush=True),
                unittest.mock.call("O", end="", flush=True),
                unittest.mock.call("K", end="", flush=True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
