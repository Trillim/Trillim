from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from trillim import _model_store
from trillim._app import build_app
from trillim.components.image import Image
from tests.support import write_llm_bundle


class RecordingTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


class RecordingImageEngine:
    def __init__(self, model, tokenizer, *, num_threads=0):
        self.model = model
        self.tokenizer = tokenizer
        self.num_threads = num_threads
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def generate(self, _prompt, output, **_kwargs):
        progress_callback = _kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(1, 2)
            progress_callback(2, 2)
        output = Path(output)
        output.write_bytes(b"png")
        return output


class ImagePublicTests(unittest.TestCase):
    def test_image_component_validates_and_uses_engine(self):
        async def run() -> tuple[Path, list[tuple[int, int]]]:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                progress = []
                write_llm_bundle(
                    root / "Local" / "image",
                    architecture="Flux2Transformer2DModel",
                    config_overrides={
                        "hidden_size": 3072,
                        "intermediate_size": 9216,
                        "num_hidden_layers": 25,
                        "num_attention_heads": 24,
                        "num_key_value_heads": 24,
                        "head_dim": 128,
                        "vocab_size": 1,
                        "eos_token_id": 151645,
                    },
                )
                with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                    image = Image(
                        "Local/image",
                        num_threads=2,
                        _tokenizer_loader=lambda *_args, **_kwargs: RecordingTokenizer(),
                        _engine_factory=RecordingImageEngine,
                    )
                    await image.start()
                    output = await image.generate(
                        "a bonsai",
                        root / "out.png",
                        progress_callback=lambda done, total: progress.append(
                            (done, total)
                        ),
                    )
                    await image.stop()
                    return output, progress

        output, progress = asyncio.run(run())

        self.assertEqual(output.name, "out.png")
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_image_router_generates_base64_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_llm_bundle(
                root / "Local" / "image",
                architecture="Flux2Transformer2DModel",
                config_overrides={
                    "hidden_size": 3072,
                    "intermediate_size": 9216,
                    "num_hidden_layers": 25,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 24,
                    "head_dim": 128,
                    "vocab_size": 1,
                    "eos_token_id": 151645,
                },
            )
            with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                image = Image(
                    "Local/image",
                    _tokenizer_loader=lambda *_args, **_kwargs: RecordingTokenizer(),
                    _engine_factory=RecordingImageEngine,
                )
                app = build_app([image])

                with TestClient(app) as client:
                    response = client.post(
                        "/v1/images/generations",
                        json={"prompt": "a bonsai", "size": "32X16", "steps": 1},
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("created", payload)
        self.assertEqual(base64.b64decode(payload["data"][0]["b64_json"]), b"png")

    def test_image_generate_rejects_non_multiple_of_16_dimensions(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                write_llm_bundle(
                    root / "Local" / "image",
                    architecture="Flux2Transformer2DModel",
                    config_overrides={
                        "hidden_size": 3072,
                        "intermediate_size": 9216,
                        "num_hidden_layers": 25,
                        "num_attention_heads": 24,
                        "num_key_value_heads": 24,
                        "head_dim": 128,
                        "vocab_size": 1,
                        "eos_token_id": 151645,
                    },
                )
                with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                    image = Image(
                        "Local/image",
                        _tokenizer_loader=lambda *_args, **_kwargs: RecordingTokenizer(),
                        _engine_factory=RecordingImageEngine,
                    )
                    await image.start()
                    try:
                        with self.assertRaisesRegex(ValueError, "multiples of 16"):
                            await image.generate("a bonsai", root / "out.png", width=17)
                    finally:
                        await image.stop()

        asyncio.run(run())
