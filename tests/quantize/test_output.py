from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim.components.llm._config import ArchitectureType
from trillim.quantize import _output
from trillim.quantize._config import load_model_config
from trillim.quantize._output import (
    build_staging_dir,
    copy_adapter_support_files,
    copy_model_support_files,
    mark_staging_complete,
    prepare_output_target,
    publish_staging_dir,
    recover_publish_state,
    write_adapter_metadata,
    write_model_metadata,
    _collect_remote_code_files,
    _parse_remote_code_module_path,
    _quantization_name,
)

from tests.quantize.test_config_manifest import _write_config


def _write_bonsai_image_config(model_dir: Path) -> None:
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2Transformer2DModel",
                "_name_or_path": "black-forest-labs/FLUX.2-klein-4B",
                "attention_head_dim": 128,
                "mlp_ratio": 3.0,
                "num_attention_heads": 24,
                "num_layers": 5,
                "num_single_layers": 20,
            }
        ),
        encoding="utf-8",
    )


class QuantizeOutputTests(unittest.TestCase):
    def test_publish_recover_and_prepare_output_target_use_managed_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            local_root = root / "Local"
            with patch.object(_model_store, "LOCAL_ROOT", local_root):
                with patch.object(_output, "_should_prompt_for_overwrite", return_value=False):
                    target = prepare_output_target(source)
                    self.assertEqual(target, local_root / "source-TRNQ")

                    staging = build_staging_dir(target)
                    (staging / "payload.txt").write_text("new", encoding="utf-8")
                    mark_staging_complete(staging)
                    publish_staging_dir(target)
                    self.assertEqual((target / "payload.txt").read_text(encoding="utf-8"), "new")

                    replacement = build_staging_dir(target)
                    (replacement / "payload.txt").write_text("replacement", encoding="utf-8")
                    mark_staging_complete(replacement)
                    publish_staging_dir(target)
                    self.assertEqual(
                        (target / "payload.txt").read_text(encoding="utf-8"),
                        "replacement",
                    )

                    stale_target = local_root / "stale"
                    stale_staging = local_root / "stale-new"
                    stale_staging.mkdir()
                    recover_publish_state(stale_target)
                    self.assertFalse(stale_staging.exists())

                    deduped = prepare_output_target(source)
                    self.assertEqual(deduped, local_root / "source-TRNQ-2")

    def test_copy_model_support_files_copies_allowlist_and_remote_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            _write_config(model_dir)
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "LocalTokenizer"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_local.py").write_text(
                "from .helper import Helper\nclass LocalTokenizer: pass\n",
                encoding="utf-8",
            )
            (model_dir / "helper.py").write_text("class Helper: pass\n", encoding="utf-8")
            (model_dir / "ignore.bin").write_bytes(b"ignored")

            copy_model_support_files(model_dir, output_dir)

            tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                tokenizer_config["auto_map"]["AutoTokenizer"],
                ["tokenization_local.LocalTokenizer", None],
            )
            self.assertTrue((output_dir / "tokenization_local.py").is_file())
            self.assertTrue((output_dir / "helper.py").is_file())
            self.assertFalse((output_dir / "ignore.bin").exists())

    def test_copy_model_support_files_preserves_bonsai_image_layout_without_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            _write_bonsai_image_config(model_dir)
            (model_dir / "README.md").write_text("Bonsai Image binary 1-bit\n", encoding="utf-8")
            (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer").mkdir()
            (model_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model_dir / "scheduler").mkdir()
            (model_dir / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "text_encoder").mkdir(exist_ok=True)
            (model_dir / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "text_encoder" / "model.safetensors").write_bytes(b"skip")
            (model_dir / "vae").mkdir()
            (model_dir / "vae" / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"skip")
            config = load_model_config(model_dir)

            copy_model_support_files(model_dir, output_dir, config=config)

            self.assertTrue((output_dir / "README.md").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())
            runtime_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_config["architectures"], ["Flux2Transformer2DModel"])
            self.assertEqual(runtime_config["hidden_size"], 3072)
            self.assertEqual(runtime_config["intermediate_size"], 9216)
            self.assertTrue((output_dir / "transformer" / "config.json").is_file())
            self.assertTrue((output_dir / "tokenizer" / "tokenizer.json").is_file())
            self.assertTrue((output_dir / "scheduler" / "scheduler_config.json").is_file())
            self.assertTrue((output_dir / "text_encoder" / "config.json").is_file())
            self.assertTrue((output_dir / "vae" / "config.json").is_file())
            self.assertFalse((output_dir / "text_encoder" / "model.safetensors").exists())
            self.assertFalse((output_dir / "vae" / "diffusion_pytorch_model.safetensors").exists())

    def test_copy_adapter_support_files_sanitizes_inherited_tokenizer_loader_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "adapter"
            output_dir = root / "out"
            (adapter_dir / "__pycache__").mkdir(parents=True)
            (adapter_dir / "nested").mkdir()
            (adapter_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "BaseTokenizer",
                        "auto_map": {"Other": "x"},
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "adapter.Config"}}),
                encoding="utf-8",
            )
            (adapter_dir / "nested" / "keep.txt").write_text("keep", encoding="utf-8")
            (adapter_dir / "qmodel.lora").write_bytes(b"skip")
            (adapter_dir / "__pycache__" / "skip.pyc").write_bytes(b"skip")

            copy_adapter_support_files(adapter_dir, output_dir)

            tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("tokenizer_class", tokenizer_config)
            self.assertEqual(tokenizer_config["auto_map"], {"Other": "x"})
            self.assertEqual(
                (output_dir / "nested" / "keep.txt").read_text(encoding="utf-8"),
                "keep",
            )
            self.assertFalse((output_dir / "qmodel.lora").exists())
            self.assertFalse((output_dir / "__pycache__").exists())

    def test_write_model_and_adapter_metadata_use_real_config_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_out = root / "model-out"
            adapter_out = root / "adapter-out"
            model_dir.mkdir()
            adapter_dir.mkdir()
            _write_config(model_dir, _name_or_path="source-model")
            config = load_model_config(model_dir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base-model"}),
                encoding="utf-8",
            )
            (adapter_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "adapter.Config"}}),
                encoding="utf-8",
            )
            (adapter_dir / "adapter.py").write_text("class Config: pass\n", encoding="utf-8")

            write_model_metadata(model_out, config=config, model_dir=model_dir)
            write_adapter_metadata(
                adapter_out,
                config=config,
                adapter_dir=adapter_dir,
                model_dir=model_dir,
            )

            model_payload = json.loads(
                (model_out / "trillim_config.json").read_text(encoding="utf-8")
            )
            adapter_payload = json.loads(
                (adapter_out / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model_payload["type"], "model")
            self.assertEqual(model_payload["source_model"], "source-model")
            self.assertEqual(adapter_payload["type"], "lora_adapter")
            self.assertEqual(adapter_payload["source_model"], "base-model")
            self.assertTrue(adapter_payload["remote_code"])

    def test_write_model_metadata_supports_bonsai_image_transformer_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "image-model"
            output_dir = root / "out"
            model_dir.mkdir()
            _write_bonsai_image_config(model_dir)
            (model_dir / "README.md").write_text("Bonsai Image binary 1-bit\n", encoding="utf-8")
            config = load_model_config(model_dir)
            self.assertIsNotNone(config.image_quantization)
            self.assertEqual(config.image_quantization.name, "binary-image")

            write_model_metadata(output_dir, config=config, model_dir=model_dir)

            payload = json.loads(
                (output_dir / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["quantization"], "binary-image")
            self.assertEqual(payload["architecture"], "bonsai_image")
            self.assertEqual(
                payload["source_model"],
                "black-forest-labs/FLUX.2-klein-4B",
            )
            self.assertEqual(len(payload["base_model_config_hash"]), 64)

    def test_write_model_metadata_reports_bonsai_image_quantization_flavor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary_dir = root / "Bonsai-Image-Binary-4B"
            ternary_dir = root / "Bonsai-Image-Ternary-4B"
            binary_out = root / "binary-out"
            ternary_out = root / "ternary-out"
            binary_dir.mkdir()
            ternary_dir.mkdir()
            _write_bonsai_image_config(binary_dir)
            _write_bonsai_image_config(ternary_dir)
            (binary_dir / "README.md").write_text("Bonsai Image binary 1-bit\n", encoding="utf-8")
            (ternary_dir / "README.md").write_text("Bonsai Image ternary\n", encoding="utf-8")

            binary_config = load_model_config(binary_dir)
            ternary_config = load_model_config(ternary_dir)
            write_model_metadata(binary_out, config=binary_config, model_dir=binary_dir)
            write_model_metadata(ternary_out, config=ternary_config, model_dir=ternary_dir)

            binary_payload = json.loads((binary_out / "trillim_config.json").read_text(encoding="utf-8"))
            ternary_payload = json.loads((ternary_out / "trillim_config.json").read_text(encoding="utf-8"))
            self.assertEqual(binary_payload["quantization"], "binary-image")
            self.assertEqual(ternary_payload["quantization"], "grouped-ternary-image")

    def test_remote_code_reference_validation_and_quantization_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "modeling_local.Config"}}),
                encoding="utf-8",
            )
            (model_dir / "modeling_local.py").write_text(
                "from .layers import Layer\nclass Config: pass\n",
                encoding="utf-8",
            )
            (model_dir / "layers.py").write_text("class Layer: pass\n", encoding="utf-8")

            self.assertEqual(
                _collect_remote_code_files(model_dir),
                [Path("modeling_local.py"), Path("layers.py")],
            )

        with self.assertRaisesRegex(ValueError, "External remote-code"):
            _parse_remote_code_module_path("other--repo.module.Class")
        self.assertEqual(_quantization_name(ArchitectureType.BONSAI), "binary")
        self.assertEqual(
            _quantization_name(ArchitectureType.BONSAI_TERNARY),
            "grouped-ternary",
        )
        self.assertEqual(_quantization_name(ArchitectureType.BONSAI_IMAGE), "bf16-image")
        self.assertEqual(_quantization_name(ArchitectureType.LLAMA), "ternary")
