# Copyright (c) 2026 Trillim. Proprietary and confidential. See LICENSE.
"""Build platform-specific wheels for all 6 supported platforms.

WARNING: This script is NOT for general use. It requires access to the private
DarkNet and DarkQuant repositories with pre-compiled binaries for all target
platforms. Only Trillim developers building distribution wheels should run this.

For each platform:
1. Cleans src/trillim/_bin/
2. Copies the correct binaries from the DarkNet and DarkQuant worktrees
3. Builds a wheel with `uv build --wheel`
4. Retags the wheel with the correct platform tag
5. Moves the tagged wheel to dist/

Usage:
    uv run scripts/build_wheels.py                 # Build all 6 platforms
    uv run scripts/build_wheels.py linux-x86_64    # Build one platform

Set TRILLIM_DARKNET_ROOT or TRILLIM_DARKQUANT_ROOT to override worktree
autodiscovery.
"""
from __future__ import annotations

import argparse
from copy import copy
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "src" / "trillim" / "_bin"
DIST_DIR = ROOT / "dist"
DARKNET_ENV = "TRILLIM_DARKNET_ROOT"
DARKQUANT_ENV = "TRILLIM_DARKQUANT_ROOT"
PRESERVED_BIN_FILES = {".gitignore"}

BINARY_SOURCES = {
    "trillim-inference": {
        "repo_name": "DarkNet",
        "env_var": DARKNET_ENV,
        "source_name": "trillim-inference",
    },
    "trillim-quantize": {
        "repo_name": "DarkQuant",
        "env_var": DARKQUANT_ENV,
        "source_name": "trillim-quantize",
    },
}

PLATFORMS: dict[str, dict] = {
    "linux-x86_64": {
        "tag": "manylinux_2_27_x86_64.manylinux2014_x86_64",
        "exe_suffix": "",
    },
    "linux-arm64": {
        "tag": "manylinux_2_27_aarch64.manylinux2014_aarch64",
        "exe_suffix": "",
    },
    "macos-x86_64": {
        "tag": "macosx_11_0_x86_64",
        "exe_suffix": "",
    },
    "macos-arm64": {
        "tag": "macosx_11_0_arm64",
        "exe_suffix": "",
    },
    "win-x86_64": {
        "tag": "win_amd64",
        "exe_suffix": ".exe",
    },
    "win-arm64": {
        "tag": "win_arm64",
        "exe_suffix": ".exe",
    },
}


def clean_bin_dir() -> None:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    for child in BIN_DIR.iterdir():
        if child.name in PRESERVED_BIN_FILES:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = _normalize_path(path)
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def _git_common_dir(root: Path) -> Path | None:
    git_path = root / ".git"
    if git_path.is_dir():
        return _normalize_path(git_path)
    if not git_path.is_file():
        return None

    try:
        text = git_path.read_text(encoding="utf-8")
    except OSError:
        return None

    first_line = text.splitlines()[0] if text.splitlines() else ""
    prefix = "gitdir:"
    if not first_line.startswith(prefix):
        return None

    git_dir = Path(first_line.removeprefix(prefix).strip()).expanduser()
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    git_dir = _normalize_path(git_dir)
    if git_dir.parent.name == "worktrees":
        return git_dir.parent.parent
    return git_dir


def _mirrored_worktree_candidate(root: Path, repo_name: str) -> Path | None:
    common_dir = _git_common_dir(root)
    if common_dir is None or common_dir.name == ".git" or not common_dir.name.endswith(".git"):
        return None

    repo_container = common_dir.with_name(common_dir.name.removesuffix(".git"))
    try:
        worktree_relative = _normalize_path(root).relative_to(_normalize_path(repo_container))
    except ValueError:
        return None
    if not worktree_relative.parts:
        return None
    return common_dir.parent / repo_name / worktree_relative


def _repo_root_candidates(repo_name: str, env_var: str, *, root: Path | None = None) -> list[Path]:
    project_root = root or ROOT
    override = os.environ.get(env_var)
    if override:
        return [_normalize_path(Path(override))]

    candidates: list[Path] = []
    mirrored = _mirrored_worktree_candidate(project_root, repo_name)
    if mirrored is not None:
        candidates.append(mirrored)
    candidates.append(project_root.parent / repo_name)
    return _unique_paths(candidates)


def _resolve_repo_root(repo_name: str, env_var: str, *, root: Path | None = None) -> Path:
    candidates = _repo_root_candidates(repo_name, env_var, root=root)
    for candidate in candidates:
        if (candidate / "executables").is_dir():
            return candidate

    tried = "\n".join(f"    - {candidate}" for candidate in candidates)
    raise RuntimeError(
        f"could not find {repo_name} executables. Set {env_var} to the {repo_name} "
        f"worktree root or use a mirrored worktree layout. Tried:\n{tried}"
    )


def _binary_candidates(
    repo: Path,
    platform: str,
    source_name: str,
    suffix: str,
    *,
    allow_flat: bool,
) -> list[Path]:
    filename = source_name + suffix
    candidates = [repo / "executables" / platform / filename]
    if allow_flat:
        candidates.append(repo / "executables" / filename)
    return candidates


def _resolve_binary(
    repo: Path,
    platform: str,
    source_name: str,
    suffix: str,
    *,
    allow_flat: bool,
) -> Path:
    candidates = _binary_candidates(
        repo,
        platform,
        source_name,
        suffix,
        allow_flat=allow_flat,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = "\n".join(f"    - {candidate}" for candidate in candidates)
    print(f"  ERROR: {source_name} not found. Tried:\n{tried}", file=sys.stderr)
    sys.exit(1)


def copy_binaries(platform: str, *, allow_flat: bool = False) -> None:
    info = PLATFORMS[platform]
    suffix: str = info["exe_suffix"]

    for name, binary_info in BINARY_SOURCES.items():
        try:
            repo = _resolve_repo_root(
                binary_info["repo_name"],
                binary_info["env_var"],
            )
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        src_name: str = binary_info["source_name"]
        src = _resolve_binary(
            repo,
            platform,
            src_name,
            suffix,
            allow_flat=allow_flat,
        )
        dst = BIN_DIR / (name + suffix)
        shutil.copy2(src, dst)
        dst.chmod(0o755)

    # Remove .gitkeep so force-include doesn't duplicate it
    gitkeep = BIN_DIR / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()

    print(f"  Copied binaries for {platform}")


def copy_local_binaries(platform: str) -> None:
    clean_bin_dir()
    copy_binaries(platform, allow_flat=True)


def build_wheel() -> Path:
    result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  uv build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    wheels = list(DIST_DIR.glob("trillim-*.whl"))
    if not wheels:
        print("  ERROR: no wheel found in dist/", file=sys.stderr)
        sys.exit(1)

    # Return the most recently created wheel
    return max(wheels, key=lambda p: p.stat().st_mtime)


def retag_wheel(wheel_path: Path, platform_tag: str) -> Path:
    """Rename wheel file and patch WHEEL metadata inside the zip."""
    name = wheel_path.name
    # Replace py3-none-any with py3-none-<platform_tag>
    new_name = name.replace("py3-none-any", f"py3-none-{platform_tag}")
    new_path = DIST_DIR / new_name

    with zipfile.ZipFile(wheel_path, "r") as zin:
        with zipfile.ZipFile(new_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                # Patch the WHEEL metadata
                if item.filename.endswith(".dist-info/WHEEL"):
                    text = data.decode("utf-8")
                    text = text.replace(
                        "Tag: py3-none-any",
                        f"Tag: py3-none-{platform_tag}",
                    )
                    data = text.encode("utf-8")

                # Patch the RECORD filename references
                new_filename = item.filename
                if "py3-none-any" in item.filename:
                    new_filename = item.filename.replace(
                        "py3-none-any", f"py3-none-{platform_tag}"
                    )

                # Preserve the original zip metadata so installed Unix binaries keep their mode bits.
                new_item = copy(item)
                new_item.filename = new_filename
                zout.writestr(new_item, data)

    # Remove the original any-tagged wheel
    wheel_path.unlink()

    print(f"  Tagged: {new_name}")
    return new_path


def build_platform(platform: str) -> Path:
    info = PLATFORMS[platform]
    print(f"\n[{platform}]")

    clean_bin_dir()
    copy_binaries(platform)
    wheel = build_wheel()
    tagged = retag_wheel(wheel, info["tag"])

    # Clean bin dir after build so binaries aren't left around
    clean_bin_dir()

    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build platform-specific wheels")
    parser.add_argument(
        "platforms",
        nargs="*",
        choices=list(PLATFORMS.keys()),
        help="Platforms to build (default: all)",
    )
    args = parser.parse_args()
    if not args.platforms:
        args.platforms = list(PLATFORMS.keys())

    DIST_DIR.mkdir(exist_ok=True)

    built: list[Path] = []
    for platform in args.platforms:
        built.append(build_platform(platform))

    print(f"\nBuilt {len(built)} wheel(s) in {DIST_DIR}/:")
    for w in built:
        print(f"  {w.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
