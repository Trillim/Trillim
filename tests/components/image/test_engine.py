from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trillim.components.image._engine import (
    _build_image_init_block,
    _build_image_request_block,
    _encode_prompt_tokens,
    _parse_progress_status,
    _write_png,
)
from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    ModelRuntimeConfig,
)


def _model() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        name="image",
        path=Path("/tmp/image"),
        arch_type=ArchitectureType.BONSAI_IMAGE,
        activation=ActivationType.SILU,
        hidden_dim=3072,
        intermediate_dim=9216,
        num_layers=25,
        num_heads=24,
        num_kv_heads=24,
        vocab_size=1,
        head_dim=128,
        max_position_embeddings=4096,
        norm_eps=1e-6,
        rope_theta=2000.0,
        eos_tokens=(151645,),
        has_qkv_bias=False,
        tie_word_embeddings=False,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    )


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("template", messages, kwargs))
        return "<chat>bonsai</chat>"

    def __call__(self, prompts, **kwargs):
        self.calls.append(("tokenize", prompts, kwargs))
        if kwargs["padding"] is False:
            return {"input_ids": [[11, 12, 13]]}
        return {
            "input_ids": [[11, 12, 13] + [0] * 29],
            "attention_mask": [[1, 1, 1] + [0] * 29],
        }


class ImageEngineHelperTests(unittest.TestCase):
    def test_build_image_protocol_blocks(self):
        self.assertEqual(_build_image_init_block(_model(), 4), "2\narch_type=7\nnum_threads=4\n")
        block = _build_image_request_block(
            output_path=Path("/tmp/out\nignored.rgb"),
            token_ids=[1, 2, 3],
            attention_mask=[1, 1, 0],
            steps=5,
            seed=9,
            width=64,
            height=32,
        )

        self.assertEqual(
            block,
            "7\noutput_path=/tmp/out\ntokens=1,2,3\nattention_mask=1,1,0\nsteps=5\nseed=9\nwidth=64\nheight=32\n",
        )

    def test_encode_prompt_tokens_matches_bucketed_chat_template_path(self):
        tokenizer = RecordingTokenizer()

        token_ids, attention_mask = _encode_prompt_tokens(tokenizer, "bonsai")

        self.assertEqual(token_ids[:4], [11, 12, 13, 0])
        self.assertEqual(attention_mask[:4], [1, 1, 1, 0])
        self.assertEqual(len(token_ids), 32)
        self.assertEqual(len(attention_mask), 32)
        self.assertEqual(tokenizer.calls[0][0], "template")
        self.assertEqual(tokenizer.calls[1][2]["padding"], False)
        self.assertEqual(tokenizer.calls[2][2]["padding"], "max_length")
        self.assertEqual(tokenizer.calls[2][2]["max_length"], 32)

    def test_parse_progress_status(self):
        self.assertEqual(_parse_progress_status("progress 2 4"), (2, 4))
        self.assertEqual(_parse_progress_status("progress 9 4"), (4, 4))
        self.assertIsNone(_parse_progress_status("ok"))
        self.assertIsNone(_parse_progress_status("progress nope 4"))

    def test_write_png_outputs_valid_png_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "out.png"
            _write_png(path, 2, 1, bytes([255, 0, 0, 0, 255, 0]))

            payload = path.read_bytes()

        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", payload)
        self.assertIn(b"IDAT", payload)
        self.assertIn(b"IEND", payload)
