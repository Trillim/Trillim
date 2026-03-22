"""Tests for public LLM package exports."""

import unittest

import trillim.components.llm as llm_exports
from trillim.components.llm import (
    ChatDoneEvent,
    ChatEvent,
    ChatFinalTextEvent,
    ChatSession,
    ChatTokenEvent,
    ChatUsage,
    LLM,
    ModelInfo,
)


class LLMExportTests(unittest.TestCase):
    def test_llm_package_exports_are_available(self):
        self.assertIsNotNone(LLM)
        self.assertIsNotNone(ChatSession)
        self.assertIsNotNone(ChatUsage)
        self.assertIsNotNone(ChatTokenEvent)
        self.assertIsNotNone(ChatFinalTextEvent)
        self.assertIsNotNone(ChatDoneEvent)
        self.assertIsNotNone(ChatEvent)
        self.assertIsNotNone(ModelInfo)
        self.assertTrue(hasattr(llm_exports, "ChatSession"))
        self.assertIn("ChatSession", llm_exports.__all__)

    def test_chat_session_is_importable_from_runtime_package(self):
        namespace: dict[str, object] = {}
        exec("from trillim.components.llm import ChatSession", namespace)
        self.assertIs(namespace["ChatSession"], ChatSession)
