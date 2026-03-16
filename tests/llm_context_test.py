# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for ChatSession prompt metrics and validation."""

from types import SimpleNamespace
import unittest

import trillim
from trillim import ChatSession, ContextOverflowError
from trillim.harnesses._default import DefaultHarness
from trillim.server import LLM
from trillim.server._models import ServerState


class _TrackingTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)


class _PlainTokenizer(_TrackingTokenizer):
    chat_template = None


class _RewritingTokenizer(_TrackingTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not add_generation_prompt and len(messages) > 1:
            return f"rewritten::{rendered}"
        return rendered


class _EotTemplateTokenizer(_TrackingTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}<|eot_id|>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _PatternMergingTokenizer(_TrackingTokenizer):
    def __init__(self, patterns: list[str]):
        super().__init__()
        self._patterns = sorted(
            enumerate(patterns),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        self._reverse = {1000 + index: pattern for index, pattern in enumerate(patterns)}

    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            for pattern_index, pattern in self._patterns:
                if text.startswith(pattern, index):
                    token_ids.append(1000 + pattern_index)
                    index += len(pattern)
                    break
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids

    def decode(self, token_ids, skip_special_tokens: bool = True):
        chars: list[str] = []
        for token_id in token_ids:
            pattern = self._reverse.get(token_id)
            if pattern is not None:
                chars.append(pattern)
            else:
                chars.append(chr(token_id))
        return "".join(chars)


class _PlainPatternMergingTokenizer(_PatternMergingTokenizer):
    chat_template = None


class _EncodeOnlyTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(ch) for ch in text]


class _LegacyDecodeTokenizer(_TrackingTokenizer):
    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class _BrokenDecodeTokenizer(_TrackingTokenizer):
    def decode(self, token_ids, skip_special_tokens: bool = True):
        raise RuntimeError("decode failed")


class _TypeErrorThenBrokenDecodeTokenizer(_TrackingTokenizer):
    def decode(self, token_ids, skip_special_tokens: bool = True):
        if skip_special_tokens is False:
            raise TypeError("legacy signature")
        raise RuntimeError("decode still failed")


class _WindowRankingTokenizer(_TrackingTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        if text == "c!":
            return [900]
        if text == "bc!":
            return [901]
        if text == "abc!":
            return [902]
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == 900:
                pieces.append("c!")
            elif token_id == 901:
                pieces.append("bc!")
            elif token_id == 902:
                pieces.append("abc!")
            else:
                pieces.append(chr(token_id))
        return "".join(pieces)


class _FullPromptOnlyDecodeTokenizer(_TrackingTokenizer):
    def __init__(self, prompt_text: str):
        super().__init__()
        self._prompt_text = prompt_text

    def decode(self, token_ids, skip_special_tokens: bool = True):
        text = "".join(chr(token_id) for token_id in token_ids)
        if text == self._prompt_text:
            return text
        raise RuntimeError("tail decode unavailable")


class _PrefixCachingEngine:
    def __init__(self, responses: list[str], tokenizer, *, max_context_tokens: int = 128):
        self._responses = list(responses)
        self.tokenizer = tokenizer
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self._cached_token_ids: list[int] = []
        self._last_cache_hit = 0
        self.generate_calls: list[dict] = []

    @property
    def cached_token_ids(self) -> list[int]:
        return list(self._cached_token_ids)

    @property
    def last_cache_hit(self) -> int:
        return self._last_cache_hit

    def reset_prompt_cache(self) -> None:
        self._cached_token_ids = []
        self._last_cache_hit = 0

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        request_tokens = list(kwargs["token_ids"])
        cached_tokens = self._cached_token_ids
        if cached_tokens and request_tokens[: len(cached_tokens)] == cached_tokens:
            self._last_cache_hit = len(cached_tokens)
        else:
            self._last_cache_hit = 0
        response = self._responses.pop(0)
        generated_tokens = [ord(ch) for ch in response]
        self._cached_token_ids = request_tokens + generated_tokens
        for token_id in generated_tokens:
            yield token_id


def _make_llm(
    tokenizer=None,
    *,
    responses: list[str] | None = None,
    max_context_tokens: int = 128,
) -> tuple[LLM, _TrackingTokenizer]:
    tokenizer = tokenizer or _TrackingTokenizer()
    llm = LLM("models/fake")
    llm.state = ServerState.RUNNING
    llm.engine = _PrefixCachingEngine(
        responses or [],
        tokenizer,
        max_context_tokens=max_context_tokens,
    )
    llm.harness = DefaultHarness(llm.engine)
    return llm, tokenizer


class ChatSessionMetricTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_overflow_error_is_exported(self):
        self.assertIs(trillim.ContextOverflowError, ContextOverflowError)

    async def test_session_returns_public_chat_session_type(self):
        llm, _ = _make_llm()
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertIs(type(session), ChatSession)

    async def test_session_memoizes_prepared_prompt_tokens_until_messages_change(self):
        llm, tokenizer = _make_llm()
        session = llm.session([{"role": "system", "content": "rules"}])
        session.add_user("hello")

        prompt_tokens = session.prompt_tokens
        encode_calls_after_first_prepare = list(tokenizer.encode_calls)

        self.assertEqual(session.prompt_tokens, prompt_tokens)
        self.assertEqual(session.validate(), prompt_tokens)
        self.assertEqual(session.max_context_tokens, 128)
        self.assertEqual(session.remaining_context_tokens, 128 - prompt_tokens)
        self.assertEqual(tokenizer.encode_calls, encode_calls_after_first_prepare)
        self.assertEqual(
            session.messages,
            (
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hello"},
            ),
        )

    async def test_session_preserves_exact_assistant_tokens_and_appends_new_suffixes(self):
        llm, tokenizer = _make_llm(responses=["ok", "again"], max_context_tokens=2048)
        session = llm.session([{"role": "user", "content": "hello"}])
        initial_prompt = "<user>hello</user><assistant>"

        self.assertEqual(
            session.prompt_tokens,
            len(initial_prompt),
        )
        self.assertIn(("<user>hello</user>", False), tokenizer.encode_calls)
        self.assertTrue(
            any(text.endswith("<assistant>") for text, _ in tokenizer.encode_calls)
        )
        self.assertNotIn((initial_prompt, False), tokenizer.encode_calls)

        self.assertEqual(await session.chat(), "ok")
        encode_count_after_reply = len(tokenizer.encode_calls)
        session.add_user("next")
        next_prompt = "<user>hello</user><assistant>ok</assistant><user>next</user><assistant>"

        self.assertEqual(
            session.prompt_tokens,
            len(next_prompt),
        )
        new_call_texts = [text for text, _ in tokenizer.encode_calls[encode_count_after_reply:]]
        self.assertTrue(any(text.endswith("<user>next</user>") for text in new_call_texts))
        self.assertTrue(any(text.endswith("<assistant>") for text in new_call_texts))
        self.assertNotIn(next_prompt, new_call_texts)
        self.assertEqual(
            llm.engine.generate_calls[0]["token_ids"],
            [ord(ch) for ch in initial_prompt],
        )

    async def test_session_uses_plain_prompt_fallback_without_chat_template(self):
        llm, tokenizer = _make_llm(tokenizer=_PlainTokenizer())
        session = llm.session()
        session.add_system("rules")
        session.add_user("hello")

        self.assertEqual(
            session._render_prompt(add_generation_prompt=False),
            "system: rules\nuser: hello",
        )
        self.assertEqual(
            session.prompt_tokens,
            len("system: rules\nuser: hello\nassistant: "),
        )
        self.assertEqual(tokenizer.encode_calls[0], ("system: rules", True))
        self.assertTrue(
            any(
                text.endswith("\nuser: hello") and not add_special_tokens
                for text, add_special_tokens in tokenizer.encode_calls
            )
        )
        self.assertTrue(
            any(
                text.endswith("\nassistant: ") and not add_special_tokens
                for text, add_special_tokens in tokenizer.encode_calls
            )
        )
        self.assertNotIn(
            ("system: rules\nuser: hello\nassistant: ", True),
            tokenizer.encode_calls[1:],
        )

    async def test_session_does_not_mutate_backend_cache_before_generation(self):
        llm, _ = _make_llm(responses=["ok"])
        llm.engine._cached_token_ids = [9, 9, 9]
        llm.engine._last_cache_hit = 3
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(llm.engine.cached_token_ids, [9, 9, 9])
        _ = session.prompt_tokens
        self.assertEqual(llm.engine.cached_token_ids, [9, 9, 9])

        await session.chat()
        self.assertNotEqual(llm.engine.cached_token_ids, [9, 9, 9])

    async def test_session_helper_short_circuits_cover_empty_suffix_and_missing_decode(self):
        llm, _ = _make_llm(tokenizer=_EncodeOnlyTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])
        plain_llm, _ = _make_llm()
        plain_session = plain_llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(session._encode_suffix(""), [])
        self.assertIsNone(session._decode_prompt_tokens([1, 2, 3]))
        self.assertEqual(session._shared_prefix_len([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(
            plain_session._materialize_append_only_tokens(
                "same",
                base_prompt_str="same",
                base_token_ids=[ord("s"), ord("a"), ord("m"), ord("e")],
            ),
            [ord("s"), ord("a"), ord("m"), ord("e")],
        )

    async def test_session_heals_from_smallest_tail_window_that_preserves_cache(self):
        llm, tokenizer = _make_llm(
            tokenizer=_PatternMergingTokenizer(["abc!", "ab"]),
        )
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(
            session._materialize_append_only_tokens(
                "abc!",
                base_prompt_str="abc",
                base_token_ids=[1001, ord("c")],
            ),
            [1001, ord("c"), ord("!")],
        )
        self.assertIn(("c!", False), tokenizer.encode_calls)
        self.assertNotIn(("abc!", False), tokenizer.encode_calls)

    async def test_session_keeps_best_partial_healing_candidate(self):
        llm, tokenizer = _make_llm(tokenizer=_WindowRankingTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(
            session._materialize_append_only_tokens(
                "abc!",
                base_prompt_str="abc",
                base_token_ids=[ord("a"), ord("b"), ord("c")],
            ),
            [ord("a"), ord("b"), 900],
        )
        self.assertIn(("bc!", False), tokenizer.encode_calls)
        self.assertIn(("abc!", False), tokenizer.encode_calls)

    async def test_session_falls_back_to_full_reencode_when_tail_healing_is_unavailable(self):
        base_prompt = "a" * 33
        llm, tokenizer = _make_llm(tokenizer=_FullPromptOnlyDecodeTokenizer(base_prompt))
        session = llm.session([{"role": "user", "content": "hello"}])
        prompt = base_prompt + "!"

        self.assertEqual(
            session._materialize_append_only_tokens(
                prompt,
                base_prompt_str=base_prompt,
                base_token_ids=[ord("a")] * 33,
            ),
            [ord("a")] * 33 + [ord("!")],
        )
        self.assertIn((prompt, False), tokenizer.encode_calls)

    async def test_session_decode_prompt_tokens_handles_legacy_and_broken_decoders(self):
        llm, _ = _make_llm(tokenizer=_LegacyDecodeTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])
        self.assertEqual(session._decode_prompt_tokens([65, 66]), "AB")

        llm, _ = _make_llm(tokenizer=_BrokenDecodeTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])
        self.assertIsNone(session._decode_prompt_tokens([65, 66]))

        llm, _ = _make_llm(tokenizer=_TypeErrorThenBrokenDecodeTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])
        self.assertIsNone(session._decode_prompt_tokens([65, 66]))

    async def test_session_validate_raises_typed_overflow_error(self):
        llm, _ = _make_llm(max_context_tokens=8)
        session = llm.session([{"role": "user", "content": "too long"}])

        with self.assertRaises(ContextOverflowError) as ctx:
            session.validate()

        self.assertEqual(ctx.exception.max_context_tokens, 8)
        self.assertIn("exceeds context window", str(ctx.exception))

    async def test_session_requires_turn_ready_state(self):
        llm, _ = _make_llm(responses=["ok", "again"])
        empty = llm.session()

        with self.assertRaisesRegex(ValueError, "no messages"):
            _ = empty.prompt_tokens

        session = llm.session([{"role": "user", "content": "hello"}])
        self.assertEqual(await session.chat(), "ok")

        with self.assertRaisesRegex(ValueError, "assistant reply"):
            session.validate()

        session.add_user("again")
        self.assertEqual(await session.chat(), "again")

    async def test_session_rejects_stale_model_changes(self):
        llm, _ = _make_llm()
        session = llm.session([{"role": "user", "content": "hello"}])

        replacement = _PrefixCachingEngine(["new"], _TrackingTokenizer())
        llm.engine = replacement
        llm.harness = DefaultHarness(replacement)
        llm._session_generation += 1

        with self.assertRaisesRegex(RuntimeError, "stale"):
            _ = session.messages

    async def test_session_allows_template_rewrites_after_assistant_turns(self):
        llm, tokenizer = _make_llm(
            tokenizer=_RewritingTokenizer(),
            responses=["ok", "again"],
        )
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(await session.chat(), "ok")
        session.add_user("again")

        self.assertEqual(await session.chat(), "again")
        self.assertTrue(
            any(
                text.startswith(
                    "rewritten::<user>hello</user><assistant>ok</assistant>"
                )
                for text, _ in tokenizer.encode_calls
            )
        )

    async def test_finalize_assistant_requires_a_prepared_turn(self):
        llm, _ = _make_llm(responses=["unused"])
        session = llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            session._finalize_assistant("oops")

    async def test_eot_template_sessions_report_cache_hits_on_follow_up_turns(self):
        llm, _ = _make_llm(
            tokenizer=_EotTemplateTokenizer(),
            responses=["ok", "again"],
            max_context_tokens=2048,
        )
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(await session.chat(), "ok")
        session.add_user("next")
        _, usage = await session._collect_chat()

        self.assertEqual(
            llm.engine.generate_calls[1]["token_ids"],
            [ord(ch) for ch in "<user>hello<|eot_id|><assistant>ok<|eot_id|><user>next<|eot_id|><assistant>"],
        )
        self.assertGreater(usage.cached_tokens, 0)

    async def test_finalize_assistant_preserves_exact_cached_prefix_without_template_terminator(self):
        llm, _ = _make_llm(
            tokenizer=_EotTemplateTokenizer(),
            responses=[],
            max_context_tokens=2048,
        )
        session = llm.session([{"role": "user", "content": "hello"}])
        session._prepare_reply()

        cached_prompt = "<user>hello<|eot_id|><assistant>ok"
        llm.engine._cached_token_ids = [ord(ch) for ch in cached_prompt]
        session._finalize_assistant("ok")

        self.assertEqual(session._committed_prompt_str, cached_prompt)
        self.assertEqual(session._committed_token_ids, [ord(ch) for ch in cached_prompt])

        session.add_user("next")
        self.assertEqual(
            session._committed_token_ids[: len(cached_prompt)],
            [ord(ch) for ch in cached_prompt],
        )

    async def test_stateless_session_restore_recovers_exact_assistant_prefix(self):
        tokenizer = _PlainPatternMergingTokenizer([" ok"])
        llm, _ = _make_llm(
            tokenizer=tokenizer,
            responses=["ok", "again"],
            max_context_tokens=2048,
        )

        first_messages = [{"role": "user", "content": "hello"}]
        first_text, first_usage = await llm._collect_chat(first_messages)
        second_messages = first_messages + [
            {"role": "assistant", "content": first_text},
            {"role": "user", "content": "next"},
        ]
        second_text, second_usage = await llm._collect_chat(second_messages)

        self.assertEqual(first_usage.cached_tokens, 0)
        self.assertEqual(second_text, "again")
        self.assertGreater(second_usage.cached_tokens, 0)
        self.assertNotIn(
            "user: hello\nassistant: ok\nuser: next\nassistant: ",
            [text for text, _ in tokenizer.encode_calls],
        )


if __name__ == "__main__":
    unittest.main()
