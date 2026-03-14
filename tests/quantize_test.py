# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for quantize-time adapter validation."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors.numpy import save_file

from trillim.quantize import _validate_adapter_dims


class QuantizeValidationTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_layers=1,
            hidden_dim_orig=250,
            hidden_dim=256,
            intermediate_dim_orig=300,
            intermediate_dim=384,
        )

    def _write_adapter(self, root: str, *, rank: int, a_shape: tuple[int, int], b_shape: tuple[int, int]) -> str:
        adapter_dir = Path(root) / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({
                "r": rank,
                "lora_alpha": rank * 2,
                "target_modules": ["q_proj", "gate_proj"],
            }),
            encoding="utf-8",
        )
        save_file(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros(a_shape, dtype=np.float32),
                "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": np.zeros(b_shape, dtype=np.float32),
            },
            str(adapter_dir / "adapter_model.safetensors"),
        )
        return str(adapter_dir)

    def test_validate_adapter_dims_accepts_matching_rank_axes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = self._write_adapter(
                temp_dir,
                rank=4,
                a_shape=(4, 250),
                b_shape=(300, 4),
            )

            _validate_adapter_dims(adapter_dir, self._config())

    def test_validate_adapter_dims_rejects_lora_a_rank_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = self._write_adapter(
                temp_dir,
                rank=4,
                a_shape=(5, 250),
                b_shape=(300, 4),
            )

            with self.assertRaisesRegex(ValueError, r"lora_A.*r=4"):
                _validate_adapter_dims(adapter_dir, self._config())

    def test_validate_adapter_dims_rejects_lora_b_rank_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = self._write_adapter(
                temp_dir,
                rank=4,
                a_shape=(4, 250),
                b_shape=(300, 5),
            )

            with self.assertRaisesRegex(ValueError, r"lora_B.*r=4"):
                _validate_adapter_dims(adapter_dir, self._config())
