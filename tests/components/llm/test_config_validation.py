from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trillim.components.llm._config import SamplingDefaults, load_sampling_defaults
from trillim.components.llm._limits import MAX_OUTPUT_TOKENS
from trillim.components.llm._validation import (
    validate_chat_template_kwargs,
    validate_chat_request,
    validate_messages,
    validate_sampling_options,
    validate_swap_request,
    validate_user_message,
)
from trillim.errors import InvalidRequestError


class SamplingDefaultsTests(unittest.TestCase):
    def test_load_sampling_defaults_uses_valid_generation_config_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "generation_config.json").write_text(
                json.dumps(
                    {
                        "temperature": "0.2",
                        "top_k": "7",
                        "top_p": 0.8,
                        "repetition_penalty": 1.05,
                        "rep_penalty_lookback": 12,
                        "max_new_tokens": MAX_OUTPUT_TOKENS + 100,
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_sampling_defaults(path)

        self.assertEqual(defaults.temperature, 0.2)
        self.assertEqual(defaults.top_k, 7)
        self.assertEqual(defaults.top_p, 0.8)
        self.assertEqual(defaults.repetition_penalty, 1.05)
        self.assertEqual(defaults.rep_penalty_lookback, 12)
        self.assertEqual(defaults.max_tokens, MAX_OUTPUT_TOKENS)

    def test_load_sampling_defaults_falls_back_for_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "generation_config.json").write_text(
                json.dumps(
                    {
                        "temperature": True,
                        "top_k": False,
                        "top_p": "bad",
                        "max_new_tokens": True,
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_sampling_defaults(path)

        self.assertEqual(defaults, SamplingDefaults())


class LLMValidationTests(unittest.TestCase):
    def test_validate_chat_request_accepts_basic_request(self):
        request = validate_chat_request(
            {
                "model": "active",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "stream": True,
            },
            active_model_name="active",
        )

        self.assertTrue(request.stream)
        self.assertEqual(request.messages[-1].content, "hello")
        self.assertEqual(request.chat_template_kwargs.to_kwargs(), {})

    def test_validate_chat_request_accepts_only_supported_chat_template_kwargs(self):
        request = validate_chat_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            active_model_name="active",
        )

        self.assertEqual(
            request.chat_template_kwargs.to_kwargs(),
            {"enable_thinking": False},
        )

        invalid_payloads = (
            {"chat_template_kwargs": {"unknown": False}},
            {"chat_template_kwargs": {"enable_thinking": "false"}},
            {"chat_template_kwargs": []},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidRequestError):
                    validate_chat_request(
                        {
                            "messages": [{"role": "user", "content": "hello"}],
                            **payload,
                        },
                        active_model_name="active",
                    )

    def test_validate_chat_template_kwargs_rejects_malformed_sdk_kwargs(self):
        self.assertEqual(
            validate_chat_template_kwargs({"enable_thinking": False}).to_kwargs(),
            {"enable_thinking": False},
        )
        self.assertEqual(validate_chat_template_kwargs(None).to_kwargs(), {})

        with self.assertRaises(InvalidRequestError):
            validate_chat_template_kwargs([])

    def test_validate_chat_request_rejects_wrong_model_and_non_user_final_turn(self):
        with self.assertRaisesRegex(InvalidRequestError, "does not match"):
            validate_chat_request(
                {"model": "other", "messages": [{"role": "user", "content": "hi"}]},
                active_model_name="active",
            )
        with self.assertRaisesRegex(InvalidRequestError, "already contains"):
            validate_chat_request(
                {"messages": [{"role": "assistant", "content": "done"}]},
                active_model_name="active",
            )

    def test_validate_messages_checks_empty_content_and_assistant_turn(self):
        self.assertEqual(
            validate_user_message("hello"),
            "hello",
        )
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_user_message("")
        with self.assertRaisesRegex(InvalidRequestError, "already contains"):
            validate_messages(
                [{"role": "assistant", "content": "done"}],
                require_user_turn=True,
                allow_empty=False,
            )

    def test_validate_sampling_options_rejects_unknown_and_out_of_range_fields(self):
        self.assertEqual(validate_sampling_options(top_k=1).top_k, 1)
        with self.assertRaisesRegex(InvalidRequestError, "Extra inputs"):
            validate_sampling_options(unknown=True)
        with self.assertRaisesRegex(InvalidRequestError, "less than or equal"):
            validate_sampling_options(top_p=2)

    def test_validate_swap_request_normalizes_search_options(self):
        request = validate_swap_request(
            {
                "model_dir": "Local/example",
                "num_threads": 0,
                "harness_name": " Search ",
                "search_provider": " DuckDuckGo ",
                "search_token_budget": 5,
            }
        )

        self.assertEqual(request.harness_name, " Search ")
        self.assertEqual(request.search_provider, " DuckDuckGo ")

        with self.assertRaisesRegex(InvalidRequestError, "Unknown harness"):
            validate_swap_request(
                {"model_dir": "Local/example", "harness_name": "unknown"}
            )
