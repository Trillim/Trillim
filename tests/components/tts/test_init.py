"""Tests for TTS package exports."""

import unittest

from trillim.components.tts import TTS, TTSSession
from trillim.components.tts import __all__ as tts_exports


class TTSInitTests(unittest.TestCase):
    def test_package_exports_tts_and_tts_session(self):
        self.assertEqual(tts_exports, ["TTSSession", "TTS"])
        self.assertIsNotNone(TTS)
        self.assertIsNotNone(TTSSession)
