"""Tests for developer wheel-building helpers."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import call, patch
import zipfile

from scripts import build_wheels


@contextmanager
def _without_repo_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(build_wheels.DARKNET_ENV, None)
        os.environ.pop(build_wheels.DARKQUANT_ENV, None)
        yield


def _write_executable(path: Path, text: str = "binary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class BuildWheelHelperTests(unittest.TestCase):
    def test_clean_bin_dir_removes_generated_files_but_keeps_gitignore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "_bin"
            (bin_dir / "nested").mkdir(parents=True)
            (bin_dir / "old").write_text("stale", encoding="utf-8")
            (bin_dir / "nested" / "old").write_text("stale", encoding="utf-8")
            (bin_dir / ".gitignore").write_text("keep\n", encoding="utf-8")

            with patch.object(build_wheels, "BIN_DIR", bin_dir):
                build_wheels.clean_bin_dir()

            self.assertTrue(bin_dir.is_dir())
            self.assertFalse((bin_dir / "old").exists())
            self.assertFalse((bin_dir / "nested").exists())
            self.assertEqual((bin_dir / ".gitignore").read_text(encoding="utf-8"), "keep\n")

    def test_repo_root_uses_explicit_env_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = Path(temp_dir) / "custom-darknet"
            (override / "executables").mkdir(parents=True)

            with patch.dict(os.environ, {build_wheels.DARKNET_ENV: str(override)}):
                resolved = build_wheels._resolve_repo_root(
                    "DarkNet",
                    build_wheels.DARKNET_ENV,
                    root=Path(temp_dir) / "Trillim" / "main",
                )

            self.assertEqual(resolved, override.resolve())

    def test_repo_root_reports_invalid_explicit_env_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = Path(temp_dir) / "missing-darknet"

            with patch.dict(os.environ, {build_wheels.DARKNET_ENV: str(override)}):
                with self.assertRaisesRegex(RuntimeError, build_wheels.DARKNET_ENV):
                    build_wheels._resolve_repo_root(
                        "DarkNet",
                        build_wheels.DARKNET_ENV,
                        root=Path(temp_dir) / "Trillim" / "main",
                    )

    def test_repo_root_uses_mirrored_worktree_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_repo_env():
            workspace = Path(temp_dir)
            root = workspace / "Trillim" / "models" / "qwen3"
            root.mkdir(parents=True)
            (root / ".git").write_text(
                f"gitdir: {workspace / 'Trillim.git' / 'worktrees' / 'qwen3'}\n",
                encoding="utf-8",
            )
            expected = workspace / "DarkNet" / "models" / "qwen3"
            (expected / "executables").mkdir(parents=True)

            resolved = build_wheels._resolve_repo_root(
                "DarkNet",
                build_wheels.DARKNET_ENV,
                root=root,
            )

            self.assertEqual(resolved, expected.resolve())

    def test_repo_root_uses_legacy_sibling_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_repo_env():
            workspace = Path(temp_dir)
            root = workspace / "Trillim"
            root.mkdir()
            expected = workspace / "DarkQuant"
            (expected / "executables").mkdir(parents=True)

            resolved = build_wheels._resolve_repo_root(
                "DarkQuant",
                build_wheels.DARKQUANT_ENV,
                root=root,
            )

            self.assertEqual(resolved, expected.resolve())

    def test_repo_root_error_lists_attempted_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_repo_env():
            root = Path(temp_dir) / "Trillim"
            root.mkdir()

            with self.assertRaisesRegex(RuntimeError, "Tried:") as raised:
                build_wheels._resolve_repo_root(
                    "DarkNet",
                    build_wheels.DARKNET_ENV,
                    root=root,
                )

            self.assertIn(str(root.parent / "DarkNet"), str(raised.exception))

    def test_git_common_dir_ignores_missing_or_unrecognized_git_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            self.assertIsNone(build_wheels._git_common_dir(root))

            (root / ".git").write_text("not a gitdir file\n", encoding="utf-8")
            self.assertIsNone(build_wheels._git_common_dir(root))

    def test_git_common_dir_ignores_unreadable_git_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            (root / ".git").write_text("gitdir: ../repo.git/worktrees/main\n", encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=OSError):
                self.assertIsNone(build_wheels._git_common_dir(root))

    def test_git_common_dir_accepts_relative_gitdir_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo" / "main"
            root.mkdir(parents=True)
            (root / ".git").write_text("gitdir: ../../repo.git/worktrees/main\n", encoding="utf-8")

            self.assertEqual(
                build_wheels._git_common_dir(root),
                (Path(temp_dir) / "repo.git").resolve(),
            )

    def test_git_common_dir_accepts_non_worktree_gitdir_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            (root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")

            self.assertEqual(
                build_wheels._git_common_dir(root),
                (Path(temp_dir) / "repo.git").resolve(),
            )

    def test_mirrored_worktree_candidate_ignores_regular_git_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            (root / ".git").mkdir(parents=True)

            self.assertIsNone(build_wheels._mirrored_worktree_candidate(root, "DarkNet"))

    def test_mirrored_worktree_candidate_ignores_unrelated_common_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "Trillim" / "main"
            root.mkdir(parents=True)
            (root / ".git").write_text(
                f"gitdir: {workspace / 'Other.git' / 'worktrees' / 'main'}\n",
                encoding="utf-8",
            )

            self.assertIsNone(build_wheels._mirrored_worktree_candidate(root, "DarkNet"))

    def test_mirrored_worktree_candidate_ignores_bare_repo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            root = workspace / "Trillim"
            root.mkdir()
            (root / ".git").write_text(
                f"gitdir: {workspace / 'Trillim.git' / 'worktrees' / 'main'}\n",
                encoding="utf-8",
            )

            self.assertIsNone(build_wheels._mirrored_worktree_candidate(root, "DarkNet"))

    def test_unique_paths_normalizes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(
                build_wheels._unique_paths([root / "a", root / "." / "a"]),
                [(root / "a").resolve()],
            )

    def test_copy_binaries_uses_flat_fallback_only_when_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_repo_env():
            workspace = Path(temp_dir)
            root = workspace / "Trillim" / "main"
            root.mkdir(parents=True)
            (root / ".git").write_text(
                f"gitdir: {workspace / 'Trillim.git' / 'worktrees' / 'main'}\n",
                encoding="utf-8",
            )
            bin_dir = root / "src" / "trillim" / "_bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / ".gitkeep").write_text("", encoding="utf-8")

            darknet = workspace / "DarkNet" / "main"
            darkquant = workspace / "DarkQuant" / "main"
            _write_executable(
                darknet / "executables" / "macos-arm64" / "trillim-inference",
                "inference",
            )
            _write_executable(darkquant / "executables" / "trillim-quantize", "quantize")

            with (
                patch.object(build_wheels, "ROOT", root),
                patch.object(build_wheels, "BIN_DIR", bin_dir),
            ):
                build_wheels.copy_binaries("macos-arm64", allow_flat=True)

                self.assertEqual((bin_dir / "trillim-inference").read_text(), "inference")
                self.assertEqual((bin_dir / "trillim-quantize").read_text(), "quantize")
                self.assertFalse((bin_dir / ".gitkeep").exists())

                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    build_wheels.copy_binaries("macos-arm64")

            self.assertIn("executables/macos-arm64/trillim-quantize", stderr.getvalue())

    def test_copy_binaries_reports_missing_repo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir, _without_repo_env():
            root = Path(temp_dir) / "Trillim"
            root.mkdir()
            stderr = io.StringIO()

            with (
                patch.object(build_wheels, "ROOT", root),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                build_wheels.copy_binaries("macos-arm64")

            self.assertIn("could not find DarkNet executables", stderr.getvalue())

    def test_copy_local_binaries_cleans_then_allows_flat_sources(self):
        with (
            patch.object(build_wheels, "clean_bin_dir") as clean_bin_dir,
            patch.object(build_wheels, "copy_binaries") as copy_binaries,
        ):
            build_wheels.copy_local_binaries("linux-arm64")

        clean_bin_dir.assert_called_once_with()
        copy_binaries.assert_called_once_with("linux-arm64", allow_flat=True)

    def test_build_wheel_returns_newest_wheel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            dist = root / "dist"
            dist.mkdir(parents=True)
            older = dist / "trillim-0.1.0-py3-none-any.whl"
            newer = dist / "trillim-0.2.0-py3-none-any.whl"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            with (
                patch.object(build_wheels, "ROOT", root),
                patch.object(build_wheels, "DIST_DIR", dist),
                patch.object(
                    build_wheels.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stderr=""),
                ) as run,
            ):
                result = build_wheels.build_wheel()

            self.assertEqual(result, newer)
            run.assert_called_once_with(
                ["uv", "build", "--wheel"],
                cwd=root,
                capture_output=True,
                text=True,
            )

    def test_build_wheel_reports_uv_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = io.StringIO()
            with (
                patch.object(build_wheels, "ROOT", Path(temp_dir)),
                patch.object(
                    build_wheels.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1, stderr="boom"),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                build_wheels.build_wheel()

            self.assertIn("uv build failed", stderr.getvalue())

    def test_build_wheel_reports_missing_wheel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dist = root / "dist"
            dist.mkdir()
            stderr = io.StringIO()

            with (
                patch.object(build_wheels, "ROOT", root),
                patch.object(build_wheels, "DIST_DIR", dist),
                patch.object(
                    build_wheels.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stderr=""),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                build_wheels.build_wheel()

            self.assertIn("no wheel found", stderr.getvalue())

    def test_retag_wheel_updates_filename_and_wheel_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dist = Path(temp_dir)
            wheel = dist / "trillim-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "trillim-0.1.0.dist-info/WHEEL",
                    "Wheel-Version: 1.0\nTag: py3-none-any\n",
                )
                archive.writestr(
                    "trillim-0.1.0-py3-none-any.data/data.txt",
                    "payload",
                )

            with patch.object(build_wheels, "DIST_DIR", dist):
                result = build_wheels.retag_wheel(wheel, "macosx_11_0_arm64")

            self.assertEqual(result.name, "trillim-0.1.0-py3-none-macosx_11_0_arm64.whl")
            self.assertFalse(wheel.exists())
            with zipfile.ZipFile(result) as archive:
                wheel_metadata = archive.read("trillim-0.1.0.dist-info/WHEEL").decode()
                self.assertIn("Tag: py3-none-macosx_11_0_arm64", wheel_metadata)
                self.assertIn(
                    "trillim-0.1.0-py3-none-macosx_11_0_arm64.data/data.txt",
                    archive.namelist(),
                )

    def test_build_platform_cleans_before_and_after_build(self):
        wheel = Path("trillim-0.1.0-py3-none-any.whl")
        tagged = Path("trillim-0.1.0-py3-none-macosx_11_0_arm64.whl")

        with (
            patch.object(build_wheels, "clean_bin_dir") as clean_bin_dir,
            patch.object(build_wheels, "copy_binaries") as copy_binaries,
            patch.object(build_wheels, "build_wheel", return_value=wheel) as build_wheel,
            patch.object(build_wheels, "retag_wheel", return_value=tagged) as retag_wheel,
        ):
            result = build_wheels.build_platform("macos-arm64")

        self.assertEqual(result, tagged)
        self.assertEqual(clean_bin_dir.mock_calls, [call(), call()])
        copy_binaries.assert_called_once_with("macos-arm64")
        build_wheel.assert_called_once_with()
        retag_wheel.assert_called_once_with(wheel, "macosx_11_0_arm64")

    def test_main_builds_requested_platform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dist = Path(temp_dir) / "dist"
            wheel = dist / "requested.whl"
            stdout = io.StringIO()

            with (
                patch.object(build_wheels, "DIST_DIR", dist),
                patch.object(build_wheels.sys, "argv", ["build_wheels.py", "macos-arm64"]),
                patch.object(build_wheels, "build_platform", return_value=wheel) as build_platform,
                redirect_stdout(stdout),
            ):
                build_wheels.main()

            build_platform.assert_called_once_with("macos-arm64")
            self.assertIn("Built 1 wheel(s)", stdout.getvalue())
            self.assertIn("requested.whl", stdout.getvalue())

    def test_main_builds_all_platforms_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dist = Path(temp_dir) / "dist"
            platforms = {
                "linux-arm64": {"tag": "linux"},
                "macos-arm64": {"tag": "macos"},
            }

            with (
                patch.object(build_wheels, "DIST_DIR", dist),
                patch.object(build_wheels, "PLATFORMS", platforms),
                patch.object(build_wheels.sys, "argv", ["build_wheels.py"]),
                patch.object(
                    build_wheels,
                    "build_platform",
                    side_effect=lambda platform: dist / f"{platform}.whl",
                ) as build_platform,
                redirect_stdout(io.StringIO()),
            ):
                build_wheels.main()

            self.assertEqual(
                build_platform.mock_calls,
                [call("linux-arm64"), call("macos-arm64")],
            )
