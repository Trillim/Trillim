"""Tests for TTS text segmentation."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trillim.components.tts._limits import (
    HARD_TEXT_SEGMENT_CAP,
    MIN_USEFUL_TTS_TOKENS,
    TARGET_TTS_TOKENS,
)
from trillim.components.tts._segmenter import (
    _hard_split_unit,
    _iter_paragraph_segments,
    _slice_long_word,
    _split_with,
    count_tts_tokens,
    iter_text_segments,
    load_pocket_tts_tokenizer,
)
from tests.components.tts.support import FakeTokenizer


class TTSSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_count_tts_tokens_uses_tokenizer_shape(self):
        self.assertEqual(count_tts_tokens("one two three", self.tokenizer), 3)

    def test_iter_text_segments_prefers_paragraph_and_sentence_boundaries(self):
        text = (
            "one two three four five six seven eight nine ten. "
            "eleven twelve thirteen.\n\n"
            "alpha beta gamma delta epsilon."
        )
        segments = list(iter_text_segments(text, self.tokenizer))
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(segment.strip() == segment for segment in segments))
        self.assertIn("alpha beta gamma delta epsilon.", segments[-1])

    def test_iter_text_segments_replaces_too_long_non_whitespace_tokens(self):
        word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        segments = list(iter_text_segments(f"alpha {word} omega", self.tokenizer))
        self.assertEqual(segments, ["alpha too-long-word-skipped omega"])
        self.assertTrue(all("x" * 51 not in segment for segment in segments))
        self.assertTrue(
            all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments)
        )

    def test_segmenter_internal_split_helpers_cover_edge_cases(self):
        self.assertEqual(_split_with(__import__("re").compile(r"\n+"), " one \n\n two \n"), ["one", "two"])
        long_word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        self.assertEqual(
            _slice_long_word(long_word),
            ["x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )
        self.assertEqual(_hard_split_unit("   ", self.tokenizer), [])
        self.assertEqual(
            _hard_split_unit(f"alpha {long_word}", self.tokenizer),
            ["alpha", "x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )

    def test_segmenter_paragraph_branches_cover_hard_caps_and_token_budget(self):
        paragraph = " ".join(f"word{i}" for i in range(10)) + ". " + " ".join(
            f"more{i}" for i in range(11)
        )
        segments = list(_iter_paragraph_segments(paragraph, self.tokenizer))
        self.assertEqual(segments, [paragraph])
        self.assertEqual(count_tts_tokens(segments[0], self.tokenizer), 21)

        with patch(
            "trillim.components.tts._segmenter._hard_split_unit",
            return_value=["a" * 300, "b" * 300],
        ):
            segments = list(_iter_paragraph_segments("ignored", self.tokenizer))
        self.assertEqual(segments, ["a" * 300, "b" * 300])

        self.assertEqual(list(_iter_paragraph_segments(" \n ", self.tokenizer)), [])

    def test_iter_paragraph_segments_splits_once_combined_tokens_exceed_budget(self):
        first_tokens = max(MIN_USEFUL_TTS_TOKENS, TARGET_TTS_TOKENS // 2)
        if first_tokens >= TARGET_TTS_TOKENS:
            self.skipTest("current limits make token-budget paragraph splitting unreachable")
        second_tokens = TARGET_TTS_TOKENS - first_tokens + 1
        first = " ".join(["a"] * first_tokens) + "."
        second = " ".join(["b"] * second_tokens) + "."
        combined = f"{first} {second}"
        if len(combined) >= HARD_TEXT_SEGMENT_CAP:
            self.skipTest("current limits trigger the hard text cap before the token budget")

        segments = list(_iter_paragraph_segments(combined, self.tokenizer))

        self.assertEqual(segments, [first, second])
        self.assertLess(len(combined), HARD_TEXT_SEGMENT_CAP)
        self.assertLessEqual(count_tts_tokens(segments[0], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertLessEqual(count_tts_tokens(segments[1], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertGreater(count_tts_tokens(combined, self.tokenizer), TARGET_TTS_TOKENS)

    def test_hard_split_unit_covers_first_word_overflow_and_reset_without_reslicing(self):
        long_word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        capped_word = "y" * HARD_TEXT_SEGMENT_CAP

        self.assertEqual(
            _hard_split_unit(long_word, self.tokenizer),
            ["x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )
        self.assertEqual(
            _hard_split_unit(f"{capped_word} z", self.tokenizer),
            [capped_word, "z"],
        )

    def test_hard_split_unit_splits_on_token_budget_without_hard_cap_overflow(self):
        words = ["a"] * (TARGET_TTS_TOKENS + 1)
        text = " ".join(words)
        if len(text) >= HARD_TEXT_SEGMENT_CAP:
            self.skipTest("current limits trigger the hard text cap before the token budget")

        pieces = _hard_split_unit(text, self.tokenizer)

        self.assertEqual(len(pieces), 2)
        self.assertEqual(count_tts_tokens(pieces[0], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertEqual(count_tts_tokens(pieces[1], self.tokenizer), 1)
        self.assertLess(len(text), HARD_TEXT_SEGMENT_CAP)

    def test_iter_paragraph_segments_skips_blank_pieces_after_hard_split(self):
        with patch(
            "trillim.components.tts._segmenter._hard_split_unit",
            return_value=["   ", "alpha"],
        ):
            segments = list(_iter_paragraph_segments("ignored", self.tokenizer))

        self.assertEqual(segments, ["alpha"])

    def test_load_pocket_tts_tokenizer_uses_lookup_configuration(self):
        captured: list[tuple[int, str]] = []

        class _SentencePieceTokenizer:
            def __init__(self, n_bins: int, tokenizer_path: str) -> None:
                captured.append((n_bins, tokenizer_path))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_file = root / "pocket_tts" / "models" / "tts_model.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text("# stub\n", encoding="utf-8")
            fake_modules = {
                "pocket_tts.conditioners.text": SimpleNamespace(
                    SentencePieceTokenizer=_SentencePieceTokenizer
                ),
                "pocket_tts.default_parameters": SimpleNamespace(DEFAULT_VARIANT="demo"),
                "pocket_tts.models": SimpleNamespace(tts_model=SimpleNamespace(__file__=str(model_file))),
                "pocket_tts.utils.config": SimpleNamespace(
                    load_config=lambda path: SimpleNamespace(
                        flow_lm=SimpleNamespace(
                            lookup_table=SimpleNamespace(
                                n_bins=7,
                                tokenizer_path=path.parent / "demo.model",
                            )
                        )
                    )
                ),
            }
            with patch.dict(sys.modules, fake_modules):
                tokenizer = load_pocket_tts_tokenizer()

        self.assertIsInstance(tokenizer, _SentencePieceTokenizer)
        self.assertEqual(captured, [(7, str(model_file.parents[1] / "config" / "demo.model"))])
