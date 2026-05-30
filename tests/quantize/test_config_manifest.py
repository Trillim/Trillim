from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.llm._config import ArchitectureType
from trillim.quantize._config import load_model_config
from trillim.quantize._manifest import (
    ACTION_BF16_RAW,
    ACTION_Q1_0_128,
    ACTION_REPACK_TERNARY,
    ACTION_TERNARY_QUANTIZE,
    DTYPE_BF16,
    SECTION_IMAGE_TEXT_ENCODER,
    SECTION_IMAGE_TRANSFORMER,
    SECTION_IMAGE_VAE,
    get_sharded_files,
    get_tensor_metadata,
    build_manifest,
    determine_language_model_only,
    run_model_quantizer,
    run_adapter_quantizer,
    validate_adapter_source,
    resolve_quantize_binary,
    _safetensors_dtype_code,
    _quantized_tensor_action,
)
from tests.support import requires_bundle_test


def _write_config(path: Path, **overrides) -> None:
    payload = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 129,
        "intermediate_size": 257,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 100,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
        "rope_theta": 10000,
    }
    payload.update(overrides)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    header: dict[str, dict[str, object]] = {}
    body = bytearray()
    for key, (dtype, shape, data) in tensors.items():
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        body.extend(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(body))


def _read_manifest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    offset = 0

    def unpack(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, offset)
        offset += size
        return value[0] if len(value) == 1 else value

    shard_paths = []
    for _ in range(unpack("<H")):
        name_len = unpack("<H")
        shard_paths.append(Path(data[offset:offset + name_len].decode("utf-8")))
        offset += name_len

    tensor_entries = []
    for _ in range(unpack("<I")):
        values = unpack("<BBIIIIHQQBHQQ")
        tensor_entries.append(
            {
                "action": values[0],
                "dtype": values[1],
                "row": values[2],
                "col": values[3],
                "padded_row": values[4],
                "padded_col": values[5],
                "shard_idx": values[6],
                "data_offset": values[7],
                "data_size": values[8],
                "has_scale": values[9],
                "scale_shard_idx": values[10],
                "scale_offset": values[11],
                "scale_size": values[12],
            }
        )

    sections = []
    for _ in range(unpack("<I")):
        sections.append(unpack("<BII"))

    return {"shard_paths": shard_paths, "tensor_entries": tensor_entries, "sections": sections}


def _read_image_index(path: Path) -> list[dict[str, object]]:
    data = path.read_bytes()
    offset = 0

    def unpack(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, offset)
        offset += size
        return value[0] if len(value) == 1 else value

    self_magic = data[offset:offset + 4]
    offset += 4
    if self_magic != b"TRIX":
        raise ValueError(f"bad image index magic: {self_magic!r}")
    if unpack("<I") != 1:
        raise ValueError("bad image index version")

    entries = []
    for _ in range(unpack("<I")):
        section = unpack("<B")
        name_len = unpack("<H")
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        row = unpack("<I")
        col = unpack("<I")
        padded_row = unpack("<I")
        padded_col = unpack("<I")
        qmodel_offset = unpack("<Q")
        data_size = unpack("<Q")
        entries.append(
            {
                "section": section,
                "name": name,
                "row": row,
                "col": col,
                "padded_row": padded_row,
                "padded_col": padded_col,
                "qmodel_offset": qmodel_offset,
                "data_size": data_size,
            }
        )
    return entries


class QuantizeConfigManifestTests(unittest.TestCase):
    def test_load_model_config_extracts_and_aligns_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            _write_config(model_dir, _name_or_path="source")

            config = load_model_config(model_dir)

        self.assertEqual(config.arch_type, ArchitectureType.LLAMA)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.intermediate_dim, 384)
        self.assertEqual(config.hidden_dim_orig, 129)
        self.assertEqual(config.intermediate_dim_orig, 257)
        self.assertEqual(config.source_model, "source")

    def test_load_model_config_handles_yarn_bonsai_and_bitnet_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            _write_config(
                model_dir,
                rope_scaling={
                    "rope_type": "yarn",
                    "factor": 2,
                    "original_max_position_embeddings": 1024,
                    "beta_slow": 1,
                    "beta_fast": 32,
                },
                rope_parameters={"partial_rotary_factor": 0.5},
            )
            config = load_model_config(model_dir)
            self.assertEqual(config.yarn_factor, 2.0)
            self.assertEqual(config.original_max_position_embeddings, 1024)
            self.assertEqual(config.partial_rotary_factor, 0.5)

            bonsai_dir = root / "bonsai"
            bonsai_dir.mkdir()
            _write_config(bonsai_dir, architectures=["Qwen3ForCausalLM"])
            (bonsai_dir / "README.md").write_text("ternary checkpoint\n", encoding="utf-8")
            bonsai = load_model_config(bonsai_dir)
            self.assertEqual(bonsai.arch_type, ArchitectureType.BONSAI_TERNARY)

            bitnet_dir = root / "bitnet"
            bitnet_dir.mkdir()
            _write_config(bitnet_dir, architectures=["BitNetForCausalLM"], hidden_act="relu2")
            _write_safetensors(
                bitnet_dir / "model.safetensors",
                {
                    "model.layers.0.self_attn.inner_attn_ln.weight": ("F32", [128], b"\0" * 512),
                    "model.layers.0.mlp.ffn_layernorm.weight": ("F32", [128], b"\0" * 512),
                },
            )
            bitnet = load_model_config(bitnet_dir)
            self.assertIn("self_attn.inner_attn_ln", bitnet.arch_info.component_order)
            self.assertIn("mlp.ffn_layernorm", bitnet.arch_info.component_order)

    def test_load_model_config_recognizes_bonsai_image_transformer_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            transformer_dir = model_dir / "transformer"
            transformer_dir.mkdir()
            (transformer_dir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "Flux2Transformer2DModel",
                        "_name_or_path": "black-forest-labs/FLUX.2-klein-4B",
                        "attention_head_dim": 128,
                        "eps": 1e-6,
                        "mlp_ratio": 3.0,
                        "num_attention_heads": 24,
                        "num_layers": 5,
                        "num_single_layers": 20,
                        "rope_theta": 2000,
                    }
                ),
                encoding="utf-8",
            )

            config = load_model_config(model_dir)

        self.assertEqual(config.arch_type, ArchitectureType.BONSAI_IMAGE)
        self.assertEqual(config.hidden_dim, 3072)
        self.assertEqual(config.intermediate_dim, 9216)
        self.assertEqual(config.num_layers, 25)
        self.assertEqual(config.num_heads, 24)
        self.assertEqual(config.num_kv_heads, 24)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.rope_theta, 2000.0)
        self.assertEqual(config.source_model, "black-forest-labs/FLUX.2-klein-4B")

    def test_load_model_config_rejects_missing_or_invalid_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_model_config(Path(temp_dir))

            path = Path(temp_dir)
            (path / "config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_model_config(path)

    def test_safetensors_metadata_and_shard_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            _write_safetensors(
                model_dir / "model.safetensors",
                {"tensor": ("F32", [2, 2], b"\0" * 16)},
            )

            self.assertEqual(get_tensor_metadata(model_dir / "model.safetensors"), [{"key": "tensor", "start": 0, "shape": [2, 2]}])
            self.assertEqual(get_sharded_files(model_dir), ([model_dir / "model.safetensors"], {}))

            (model_dir / "model.safetensors").unlink()
            (model_dir / "shard-a.safetensors").write_bytes(b"")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "shard-a.safetensors"}}),
                encoding="utf-8",
            )
            shard_files, weight_map = get_sharded_files(model_dir)
            self.assertEqual(shard_files, [model_dir / "shard-a.safetensors"])
            self.assertEqual(weight_map, {"a": model_dir / "shard-a.safetensors"})

            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid sharded"):
                get_sharded_files(model_dir)

    def test_quantized_tensor_action_and_binary_resolution(self):
        self.assertEqual(
            _quantized_tensor_action("F32", ArchitectureType.LLAMA),
            ACTION_TERNARY_QUANTIZE,
        )
        self.assertEqual(
            _quantized_tensor_action("I8", ArchitectureType.LLAMA),
            ACTION_REPACK_TERNARY,
        )
        self.assertEqual(
            _quantized_tensor_action("F32", ArchitectureType.BONSAI),
            ACTION_Q1_0_128,
        )
        with self.assertRaisesRegex(ValueError, "Unknown safetensors dtype"):
            _safetensors_dtype_code("BAD")

    def test_build_manifest_writes_model_tensor_entries_and_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            output_dir.mkdir()
            _write_config(model_dir, tie_word_embeddings=False)
            _write_safetensors(
                model_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F32", [100, 128], b"\0" * 100 * 128 * 4),
                    "model.norm.weight": ("F32", [128], b"\0" * 128 * 4),
                    "model.layers.0.self_attn.q_proj.weight": (
                        "F32",
                        [128, 128],
                        b"\0" * 128 * 128 * 4,
                    ),
                    "lm_head.weight": ("F32", [100, 128], b"\0" * 100 * 128 * 4),
                },
            )
            config = load_model_config(model_dir)

            manifest_path = build_manifest(model_dir, config, output_dir=output_dir)

            with manifest_path.open("rb") as handle:
                shard_count = struct.unpack("<H", handle.read(2))[0]
                shard_name_len = struct.unpack("<H", handle.read(2))[0]
                shard_name = handle.read(shard_name_len).decode("utf-8")
                tensor_count = struct.unpack("<I", handle.read(4))[0]
            self.assertEqual(shard_count, 1)
            self.assertEqual(Path(shard_name), model_dir / "model.safetensors")
            self.assertEqual(tensor_count, 4)

    def test_build_manifest_writes_bonsai_image_component_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            transformer_dir = model_dir / "transformer"
            text_encoder_dir = model_dir / "text_encoder"
            vae_dir = model_dir / "vae"
            transformer_dir.mkdir(parents=True)
            text_encoder_dir.mkdir()
            vae_dir.mkdir()
            output_dir.mkdir()
            (transformer_dir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "Flux2Transformer2DModel",
                        "attention_head_dim": 128,
                        "mlp_ratio": 3.0,
                        "num_attention_heads": 24,
                        "num_layers": 5,
                        "num_single_layers": 20,
                    }
                ),
                encoding="utf-8",
            )
            _write_safetensors(
                text_encoder_dir / "model.safetensors",
                {"model.embed_tokens.weight": ("BF16", [2, 3], b"\0" * 12)},
            )
            _write_safetensors(
                transformer_dir / "diffusion_pytorch_model.safetensors",
                {
                    "proj_out.weight": ("BF16", [4, 3], b"\0" * 24),
                    "context_embedder.weight": ("BF16", [3, 2], b"\0" * 12),
                },
            )
            _write_safetensors(
                vae_dir / "diffusion_pytorch_model.safetensors",
                {
                    "bn.num_batches_tracked": ("I64", [], b"\0" * 8),
                    "decoder.conv_in.weight": ("BF16", [2, 2, 1, 1], b"\0" * 8),
                },
            )
            config = load_model_config(model_dir)

            manifest_path = build_manifest(model_dir, config, output_dir=output_dir)
            manifest = _read_manifest(manifest_path)
            image_index = _read_image_index(output_dir / "qmodel.index")

            self.assertEqual(len(manifest["tensor_entries"]), 4)
            self.assertEqual(
                manifest["sections"],
                [
                    (SECTION_IMAGE_TEXT_ENCODER, 0, 1),
                    (SECTION_IMAGE_TRANSFORMER, 1, 2),
                    (SECTION_IMAGE_VAE, 3, 1),
                ],
            )
            for entry in manifest["tensor_entries"]:
                self.assertEqual(entry["action"], ACTION_BF16_RAW)
                self.assertEqual(entry["dtype"], DTYPE_BF16)
            self.assertEqual(
                [entry["name"] for entry in image_index],
                [
                    "model.embed_tokens.weight",
                    "context_embedder.weight",
                    "proj_out.weight",
                    "decoder.conv_in.weight",
                ],
            )
            self.assertEqual([entry["section"] for entry in image_index], [4, 5, 5, 6])
            self.assertEqual([entry["qmodel_offset"] for entry in image_index], [31, 43, 55, 79])
            self.assertEqual([entry["data_size"] for entry in image_index], [12, 12, 24, 8])

    def test_build_manifest_validates_supported_tensors_and_language_model_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            output_dir.mkdir()
            _write_config(model_dir)
            _write_safetensors(
                model_dir / "model.safetensors",
                {"unsupported.weight": ("F32", [1], b"\0" * 4)},
            )
            config = load_model_config(model_dir)

            with self.assertRaisesRegex(ValueError, "unsupported"):
                build_manifest(model_dir, config, output_dir=output_dir)

            _write_config(model_dir, architectures=["Qwen3_5ForConditionalGeneration"])
            config = load_model_config(model_dir)
            _write_safetensors(
                model_dir / "model.safetensors",
                {"model.visual.patch.weight": ("F32", [1], b"\0" * 4)},
            )
            self.assertTrue(determine_language_model_only(model_dir, config))
            with self.assertRaisesRegex(ValueError, "text-only"):
                build_manifest(model_dir, config, output_dir=output_dir)
            manifest_path = build_manifest(
                model_dir,
                config,
                output_dir=output_dir,
                language_model_only=True,
            )
            self.assertTrue(manifest_path.is_file())

    def test_validate_adapter_source_checks_real_adapter_metadata_and_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            _write_config(model_dir)
            config = load_model_config(model_dir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"r": 2, "target_modules": ["q_proj"], "lora_alpha": 4}),
                encoding="utf-8",
            )
            _write_safetensors(
                adapter_dir / "adapter_model.safetensors",
                {
                    "model.layers.0.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        [2, 129],
                        b"\0" * 2 * 129 * 4,
                    ),
                    "model.layers.0.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        [128, 2],
                        b"\0" * 128 * 2 * 4,
                    ),
                },
            )

            validate_adapter_source(adapter_dir, config)

            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"r": 2, "target_modules": ["bad_proj"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                validate_adapter_source(adapter_dir, config)

    def test_run_model_quantizer_invokes_binary_and_cleans_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            output_dir.mkdir()
            _write_config(model_dir)
            _write_safetensors(
                model_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F32", [100, 128], b"\0" * 100 * 128 * 4),
                },
            )
            config = load_model_config(model_dir)

            with patch("subprocess.run") as run:
                run.return_value.returncode = 0
                run_model_quantizer(
                    Path("/bin/quantizer"),
                    model_dir,
                    config,
                    output_dir=output_dir,
                    language_model_only=False,
                )

            command = run.call_args.args[0]
            self.assertIn("--manifest", command)
            self.assertIn("--rope-output", command)
            self.assertFalse((output_dir / ".quantize_manifest.bin").exists())

    def test_run_adapter_quantizer_invokes_binary_and_cleans_temp_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            output_dir = root / "out"
            model_dir.mkdir()
            adapter_dir.mkdir()
            output_dir.mkdir()
            _write_config(model_dir)
            config = load_model_config(model_dir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"r": 2, "target_modules": ["q_proj"]}),
                encoding="utf-8",
            )
            _write_safetensors(
                adapter_dir / "adapter_model.safetensors",
                {
                    "model.layers.0.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        [2, 129],
                        b"\0" * 2 * 129 * 4,
                    ),
                    "model.layers.0.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        [128, 2],
                        b"\0" * 128 * 2 * 4,
                    ),
                },
            )

            with patch("subprocess.run") as run:
                run.return_value.returncode = 0
                run_adapter_quantizer(
                    Path("/bin/quantizer"),
                    model_dir,
                    config,
                    adapter_dir=adapter_dir,
                    output_dir=output_dir,
                    language_model_only=False,
                )

            command = run.call_args.args[0]
            self.assertIn("--lora-output", command)
            self.assertIn("--lora-rank", command)
            self.assertFalse((output_dir / ".quantize_manifest.bin").exists())


@requires_bundle_test
class QuantizeBundleTests(unittest.TestCase):
    def test_resolve_quantize_binary_finds_bundled_binary(self):
        self.assertTrue(resolve_quantize_binary().is_file())
