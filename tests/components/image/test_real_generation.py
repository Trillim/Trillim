from __future__ import annotations

import asyncio
import struct
import tempfile
import unittest
from pathlib import Path

from trillim import _model_store
from trillim.components.image import Image
from tests.support import requires_integration


BONSAI_IMAGE_MODEL_ID = "Local/Bonsai-Image-Ternary-4B-TRNQ"
BONSAI_IMAGE_MODEL_DIR = _model_store.store_path_for_id(BONSAI_IMAGE_MODEL_ID)


@requires_integration
@unittest.skipUnless(
    BONSAI_IMAGE_MODEL_DIR.is_dir(),
    f"{BONSAI_IMAGE_MODEL_ID} must be installed in the Trillim model store",
)
class RealBonsaiImageGenerationTests(unittest.TestCase):
    def test_generate_writes_png_with_requested_dimensions(self):
        async def run(output_path: Path) -> None:
            image = Image(BONSAI_IMAGE_MODEL_ID)
            await image.start()
            try:
                result = await image.generate(
                    "a tiny bonsai on a desk",
                    output_path,
                    width=16,
                    height=16,
                    steps=1,
                    seed=123,
                )
            finally:
                await image.stop()
            self.assertEqual(result, output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bonsai.png"
            asyncio.run(run(output_path))
            payload = output_path.read_bytes()

        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", payload[16:24])
        self.assertEqual((width, height), (16, 16))
