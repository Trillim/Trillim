"""Tests for LLM package exports."""

import unittest

from trillim.components.llm import (
    ChatDoneEvent,
    ChatEvent,
    ChatFinalTextEvent,
    ChatSession,
    ChatTokenEvent,
    ChatUsage,
    LLM,
)
from trillim.components.llm import __all__ as llm_exports


class LLMInitTests(unittest.TestCase):
    def test_package_exports_public_llm_types(self):
        self.assertEqual(
            llm_exports,
            [
                "ChatDoneEvent",
                "ChatEvent",
                "ChatFinalTextEvent",
                "ChatSession",
                "ChatTokenEvent",
                "ChatUsage",
                "LLM",
            ],
        )
        self.assertIsNotNone(ChatDoneEvent)
        self.assertIsNotNone(ChatEvent)
        self.assertIsNotNone(ChatFinalTextEvent)
        self.assertIsNotNone(ChatSession)
        self.assertIsNotNone(ChatTokenEvent)
        self.assertIsNotNone(ChatUsage)
        self.assertIsNotNone(LLM)
