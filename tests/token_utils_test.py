# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for incremental token decoding helpers."""

import unittest

from trillim.token_utils import IncrementalDecoder


class _TokenizerStub:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        mapping = {
            (1,): "Hello",
            (2,): "world",
            (3,): "again",
            (1, 2): "Hello world",
            (2, 3): "world again",
        }
        return mapping[tuple(token_ids)]


class IncrementalDecoderTests(unittest.TestCase):
    def test_decode_uses_pair_decode_after_first_token(self):
        decoder = IncrementalDecoder(_TokenizerStub())

        self.assertEqual(decoder.decode(1), "Hello")
        self.assertEqual(decoder.decode(2), " world")
        self.assertEqual(decoder.prev_token, 2)

    def test_reset_clears_previous_token_state(self):
        decoder = IncrementalDecoder(_TokenizerStub())
        decoder.decode(2)

        decoder.reset()

        self.assertIsNone(decoder.prev_token)
        self.assertEqual(decoder.decode(3), "again")


if __name__ == "__main__":
    unittest.main()
