from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim import cli
from trillim._bundle_metadata import CURRENT_FORMAT_VERSION
from tests.support import write_llm_bundle, write_lora_bundle


BONSAI_MODEL_ID = "Trillim/Bonsai-1.7B-TRNQ"
BONSAI_MODEL_DIR = _model_store.store_path_for_id(BONSAI_MODEL_ID)


class CLITests(unittest.TestCase):
    def test_parser_and_main_help_paths(self):
        parser = cli.build_parser()

        self.assertEqual(parser.parse_args(["list"]).command, "list")
        self.assertEqual(parser.parse_args(["doctor"]).command, "doctor")
        self.assertFalse(parser.parse_args(["doctor"]).deep)
        self.assertTrue(parser.parse_args(["doctor", "--deep"]).deep)
        self.assertEqual(parser.parse_args(["chat", "Trillim/model"]).command, "chat")
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = cli.main([])

        self.assertEqual(code, 1)
        self.assertIn("Trillim", stdout.getvalue())

    def test_pull_id_validation_and_platform_normalization(self):
        self.assertEqual(
            cli._validate_pull_id(" Trillim/BitNet-TRNQ "), "Trillim/BitNet-TRNQ"
        )
        self.assertEqual(cli._normalize_platform_name("ARM64"), "aarch64")
        self.assertEqual(cli._normalize_platform_name("amd64"), "x86_64")
        self.assertEqual(cli._normalize_platform_name("riscv64"), "riscv64")

        with self.assertRaisesRegex(RuntimeError, "only supports"):
            cli._validate_pull_id("Local/model")
        with self.assertRaisesRegex(RuntimeError, "Model IDs"):
            cli._validate_pull_id("bad/model")

    def test_warn_on_trillim_config_reports_bad_and_future_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing"
            missing.mkdir()
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertEqual(stdout.getvalue(), "")

            (missing / "trillim_config.json").write_text("{", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertIn("Could not read", stdout.getvalue())

            (missing / "trillim_config.json").write_text("[]", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertIn("Could not interpret", stdout.getvalue())

            (missing / "trillim_config.json").write_text(
                json.dumps(
                    {
                        "format_version": CURRENT_FORMAT_VERSION + 1,
                        "platforms": ["definitely-not-this-platform"],
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            output = stdout.getvalue()
            self.assertIn("newer than supported", output)
            self.assertIn("lists platforms", output)

    def test_remote_code_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = write_llm_bundle(root / "Trillim" / "remote")
            payload = json.loads(
                (bundle / "trillim_config.json").read_text(encoding="utf-8")
            )
            payload["remote_code"] = True
            (bundle / "trillim_config.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            with patch.object(_model_store, "DOWNLOADED_ROOT", root / "Trillim"):
                with self.assertRaisesRegex(RuntimeError, "trust_remote_code"):
                    cli._require_remote_code_opt_in(
                        "Trillim/remote",
                        label="Model",
                        trust_remote_code=False,
                    )
                cli._require_remote_code_opt_in(
                    "Trillim/remote",
                    label="Model",
                    trust_remote_code=True,
                )

    def test_local_bundle_listing_and_list_command_use_real_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloaded_root = root / "Trillim"
            local_root = root / "Local"
            model_dir = write_llm_bundle(downloaded_root / "model")
            write_lora_bundle(local_root / "adapter", model_dir=model_dir)
            (downloaded_root / "invalid").mkdir(parents=True)

            with (
                patch.object(_model_store, "DOWNLOADED_ROOT", downloaded_root),
                patch.object(_model_store, "LOCAL_ROOT", local_root),
            ):
                downloaded = cli._iter_local_bundles("Trillim")
                local = cli._iter_local_bundles("Local")
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = cli.main(["list"])

        self.assertEqual(code, 0)
        self.assertEqual([bundle.model_id for bundle in downloaded], ["Trillim/model"])
        self.assertEqual([bundle.model_id for bundle in local], ["Local/adapter"])
        output = stdout.getvalue()
        self.assertIn("Downloaded", output)
        self.assertIn("Trillim/model", output)
        self.assertIn("Local/adapter", output)

    def test_downloaded_statuses_marks_stale_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloaded_root = root / "Trillim"
            write_llm_bundle(downloaded_root / "valid")
            stale = downloaded_root / "stale"
            stale.mkdir()
            (stale / "trillim_config.json").write_text(
                json.dumps({"format_version": CURRENT_FORMAT_VERSION + 1}),
                encoding="utf-8",
            )
            bad = downloaded_root / "bad-json"
            bad.mkdir()
            (bad / "trillim_config.json").write_text("{", encoding="utf-8")

            with patch.object(_model_store, "DOWNLOADED_ROOT", downloaded_root):
                statuses = cli._downloaded_statuses()

        self.assertEqual(statuses["Trillim/valid"], "local")
        self.assertEqual(statuses["Trillim/stale"], "stale")
        self.assertNotIn("Trillim/bad-json", statuses)

    def test_table_printers_cover_empty_and_populated_output(self):
        bundles = [
            cli._LocalBundle(
                model_id="Trillim/model",
                entry_type="model",
                size_bytes=1024,
                size_human="1 KB",
            )
        ]
        entries = [
            {
                "model_id": "Trillim/model",
                "type": "model",
                "downloads": 7,
                "last_modified": "2026-01-02",
                "base_model": "base",
                "status": "local",
                "local": True,
            }
        ]

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            cli._print_local_table("Empty", [])
            cli._print_local_table("Bundles", bundles)
            cli._print_available_table("Remote", entries)

        output = stdout.getvalue()
        self.assertIn("(none)", output)
        self.assertIn("Trillim/model", output)
        self.assertIn("base", output)

    def test_pull_existing_model_does_not_download_without_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_dir = root / "Trillim" / "model"
            local_dir.mkdir(parents=True)
            with patch.object(_model_store, "DOWNLOADED_ROOT", root / "Trillim"):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    result = cli._pull_model(
                        "Trillim/model", revision=None, force=False
                    )

        self.assertEqual(result, local_dir)
        self.assertIn("already exists", stdout.getvalue())

    def test_main_reports_runtime_errors(self):
        with patch.object(
            cli, "_run_models_command", side_effect=RuntimeError("offline")
        ):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = cli.main(["models"])

        self.assertEqual(code, 1)
        self.assertIn("offline", stderr.getvalue())

    def test_project_version_uses_pyproject_fallback(self):
        expected = cli.tomllib.loads(
            Path("pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        with patch.object(
            cli.importlib_metadata,
            "version",
            side_effect=cli.importlib_metadata.PackageNotFoundError,
        ):
            self.assertEqual(cli._project_version(), expected)

    def test_platform_tag_reports_supported_and_unsupported_platforms(self):
        with (
            patch.object(cli.platform, "system", return_value="Darwin"),
            patch.object(cli.platform, "machine", return_value="arm64"),
        ):
            self.assertEqual(cli._platform_tag(), "macosx_11_0_arm64")

        with (
            patch.object(cli.platform, "system", return_value="FreeBSD"),
            patch.object(cli.platform, "machine", return_value="riscv64"),
        ):
            self.assertEqual(cli._platform_tag(), "unsupported (FreeBSD riscv64)")

    def test_binary_status_reports_missing_executable_and_windows_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "trillim-inference"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)

            with patch.object(cli, "_binary_path", return_value=executable):
                status = cli._binary_status("trillim-inference")

            self.assertEqual(status.path, executable)
            self.assertTrue(status.exists)
            self.assertTrue(status.executable)

            missing_exe = root / "trillim-quantize.exe"
            fallback = root / "trillim-quantize"
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")

            with (
                patch.object(cli, "_binary_path", return_value=missing_exe),
                patch.object(cli.os, "name", "nt"),
            ):
                status = cli._binary_status("trillim-quantize")

            self.assertEqual(status.path, fallback)
            self.assertTrue(status.exists)
            self.assertFalse(status.executable)

    def test_voice_dependency_statuses_uses_shallow_availability_by_default(self):
        def find_spec(name):
            if name == "numpy":
                return object()
            return None

        with patch.object(cli.importlib_util, "find_spec", side_effect=find_spec):
            statuses = cli._voice_dependency_statuses()

        self.assertEqual(
            statuses,
            {
                "faster_whisper": False,
                "numpy": True,
                "soundfile": False,
                "pocket_tts": False,
            },
        )

    def test_voice_dependency_statuses_deep_imports_modules(self):
        def import_module(name):
            if name == "numpy":
                return object()
            raise ModuleNotFoundError(name)

        with patch.object(cli.importlib, "import_module", side_effect=import_module):
            statuses = cli._voice_dependency_statuses(deep=True)

        self.assertEqual(
            statuses,
            {
                "faster_whisper": False,
                "numpy": True,
                "soundfile": False,
                "pocket_tts": False,
            },
        )

    def test_doctor_command_reports_local_diagnostics(self):
        downloaded = [
            cli._LocalBundle(
                model_id="Trillim/model",
                entry_type="model",
                size_bytes=1024,
                size_human="1 KB",
            )
        ]
        local: list[cli._LocalBundle] = []

        def iter_bundles(namespace):
            return downloaded if namespace == "Trillim" else local

        voice_statuses = patch.object(
            cli,
            "_voice_dependency_statuses",
            return_value={
                "faster_whisper": False,
                "numpy": True,
                "soundfile": False,
                "pocket_tts": True,
            },
        )

        with (
            patch.object(cli, "_project_version", return_value="1.2.3"),
            patch.object(cli, "_platform_tag", return_value="macosx_11_0_arm64"),
            patch.object(
                cli,
                "_binary_status",
                side_effect=[
                    cli._BinaryStatus(
                        name="trillim-inference",
                        path=Path("/tmp/trillim-inference"),
                        exists=True,
                        executable=True,
                    ),
                    cli._BinaryStatus(
                        name="trillim-quantize",
                        path=Path("/tmp/trillim-quantize"),
                        exists=True,
                        executable=False,
                    ),
                ],
            ),
            voice_statuses as dependency_statuses,
            patch.object(cli, "_iter_local_bundles", side_effect=iter_bundles),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            code = cli.main(["doctor"])

        self.assertEqual(code, 0)
        dependency_statuses.assert_called_once_with(deep=False)
        output = stdout.getvalue()
        self.assertIn("Trillim Doctor\n\nSystem", output)
        self.assertIn("System\n  Package      1.2.3", output)
        self.assertIn("  Platform     macosx_11_0_arm64", output)
        self.assertIn("  Python       ", output)
        self.assertIn("  Executable   ", output)
        self.assertIn("  Model store  ", output)
        self.assertIn("Binaries\n  Name               Status", output)
        self.assertIn("trillim-inference  ready", output)
        self.assertIn("trillim-quantize   not executable", output)
        self.assertIn("Trillim/model  model  1 KB", output)
        self.assertIn("Local models\n  (none)", output)
        self.assertIn("numpy           available", output)
        self.assertIn("faster_whisper  missing", output)
        self.assertIn("Quantization\n  Available  no", output)
        self.assertNotIn("\033[", output)

    def test_doctor_command_deep_imports_voice_dependencies(self):
        with (
            patch.object(cli, "_project_version", return_value="1.2.3"),
            patch.object(cli, "_platform_tag", return_value="macosx_11_0_arm64"),
            patch.object(
                cli,
                "_binary_status",
                side_effect=[
                    cli._BinaryStatus(
                        name="trillim-inference",
                        path=Path("/tmp/trillim-inference"),
                        exists=False,
                        executable=False,
                    ),
                    cli._BinaryStatus(
                        name="trillim-quantize",
                        path=Path("/tmp/trillim-quantize"),
                        exists=False,
                        executable=False,
                    ),
                ],
            ),
            patch.object(
                cli,
                "_voice_dependency_statuses",
                return_value={
                    "faster_whisper": False,
                    "numpy": True,
                    "soundfile": False,
                    "pocket_tts": True,
                },
            ) as dependency_statuses,
            patch.object(cli, "_iter_local_bundles", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            code = cli.main(["doctor", "--deep"])

        self.assertEqual(code, 0)
        dependency_statuses.assert_called_once_with(deep=True)
        output = stdout.getvalue()
        self.assertIn("numpy           importable", output)
        self.assertIn("faster_whisper  import failed", output)

    def test_doctor_binary_status_labels(self):
        ready = cli._BinaryStatus(
            name="tool",
            path=Path("/tmp/tool"),
            exists=True,
            executable=True,
        )
        not_executable = cli._BinaryStatus(
            name="tool",
            path=Path("/tmp/tool"),
            exists=True,
            executable=False,
        )
        missing = cli._BinaryStatus(
            name="tool",
            path=Path("/tmp/tool"),
            exists=False,
            executable=False,
        )

        self.assertEqual(cli._doctor_binary_status(ready), "ready")
        self.assertEqual(cli._doctor_binary_status(not_executable), "not executable")
        self.assertEqual(cli._doctor_binary_status(missing), "missing")
        self.assertEqual(cli._doctor_binary_status_cell(ready).color, "success")
        self.assertEqual(
            cli._doctor_binary_status_cell(not_executable).color, "warning"
        )
        self.assertEqual(cli._doctor_binary_status_cell(missing).color, "error")

    def test_doctor_status_cells_classify_health(self):
        self.assertEqual(
            cli._doctor_dependency_status(True),
            cli._DoctorCell(text="available", color="success"),
        )
        self.assertEqual(
            cli._doctor_dependency_status(False),
            cli._DoctorCell(text="missing", color="warning"),
        )
        self.assertEqual(
            cli._doctor_dependency_status(True, deep=True),
            cli._DoctorCell(text="importable", color="success"),
        )
        self.assertEqual(
            cli._doctor_dependency_status(False, deep=True),
            cli._DoctorCell(text="import failed", color="warning"),
        )
        self.assertEqual(
            cli._doctor_bool_status(True),
            cli._DoctorCell(text="yes", color="success"),
        )
        self.assertEqual(
            cli._doctor_bool_status(False),
            cli._DoctorCell(text="no", color="warning"),
        )

    def test_doctor_color_text_respects_terminal_color_settings(self):
        class TTY:
            @staticmethod
            def isatty():
                return True

        with (
            patch.object(cli.sys, "stdout", TTY()),
            patch.dict(cli.os.environ, {"TERM": "xterm-256color"}, clear=True),
        ):
            self.assertEqual(
                cli._doctor_color_text("ready", "success"),
                "\033[32mready\033[0m",
            )

        with (
            patch.object(cli.sys, "stdout", TTY()),
            patch.dict(cli.os.environ, {"NO_COLOR": "1"}, clear=True),
        ):
            self.assertEqual(cli._doctor_color_text("ready", "success"), "ready")

        with (
            patch.object(cli.sys, "stdout", TTY()),
            patch.dict(cli.os.environ, {"TERM": "dumb"}, clear=True),
        ):
            self.assertEqual(cli._doctor_color_text("ready", "success"), "ready")

    def test_doctor_table_colors_status_cells(self):
        with (
            patch.object(cli, "_doctor_color_enabled", return_value=True),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            cli._print_doctor_table(
                ("Name", "Status"),
                [
                    ("ok", cli._doctor_cell("ready", "success")),
                    ("bad", cli._doctor_cell("missing", "error")),
                ],
            )

        output = stdout.getvalue()
        self.assertIn("\033[32mready\033[0m", output)
        self.assertIn("\033[31mmissing\033[0m", output)

    def test_voice_dependency_preflight_passes_with_voice_extra(self):
        cli._preflight_voice_dependencies()

    @unittest.skipUnless(
        BONSAI_MODEL_DIR.is_dir(),
        f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
    )
    def test_chat_command_starts_real_bonsai_runtime_and_quits(self):
        with patch.object(cli, "better_input", side_effect=["/new", "q"]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = cli._run_chat(BONSAI_MODEL_ID, None)

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Model: Trillim/Bonsai-1.7B-TRNQ", output)
        self.assertIn("Conversation reset.", output)
