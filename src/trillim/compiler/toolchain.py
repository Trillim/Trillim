from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_ARCH_FLAGS = {
    "x86_64": ("-march=x86-64",),
    "amd64": ("-march=x86-64",),
    "aarch64": ("-march=armv8-a",),
    "arm64": ("-march=armv8-a",),
}


def _native_compilers(driver: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = [(driver,)]
    versioned: list[tuple[tuple[int, ...], str]] = []
    pattern = re.compile(rf"{re.escape(driver)}-(\d+(?:\.\d+)*)(?:\.exe)?")
    for directory in os.get_exec_path():
        try:
            for entry in Path(directory).iterdir():
                match = pattern.fullmatch(entry.name)
                if match:
                    version = tuple(int(part) for part in match[1].split("."))
                    versioned.append((version, str(entry)))
        except OSError:
            continue
    commands.extend(
        (path,) for _, path in sorted(versioned, key=lambda item: item[0], reverse=True)
    )
    return commands


def _compilers(*, cxx: bool = False) -> tuple[tuple[str, ...], ...]:
    variable = "CXX" if cxx else "CC"
    override = os.environ.get(variable)
    commands: list[tuple[str, ...]]
    if override is not None:
        arguments = shlex.split(override, posix=sys.platform != "win32")
        if sys.platform == "win32":
            arguments = [
                argument.removeprefix('"').removesuffix('"') for argument in arguments
            ]
        if not arguments:
            raise RuntimeError(f"{variable} must name a compiler")
        commands = [tuple(arguments)]
    else:
        gcc = _native_compilers("g++" if cxx else "gcc")
        clang = _native_compilers("clang++" if cxx else "clang")
        genuine_gcc: list[tuple[str, ...]] = []
        aliases: list[tuple[str, ...]] = []
        for command in gcc:
            executable = shutil.which(command[0])
            if executable is None:
                continue
            try:
                version = subprocess.check_output(
                    [executable, "--version"], stderr=subprocess.STDOUT, timeout=5
                )
            except (OSError, subprocess.SubprocessError):
                version = b""
            (aliases if b"clang" in version.lower() else genuine_gcc).append(command)
        commands = [("zig", "c++" if cxx else "cc"), *genuine_gcc, *clang, *aliases]
        if sys.platform == "win32":
            machine = platform.machine().lower()
            target = {
                "amd64": "x86_64",
                "x86_64": "x86_64",
                "aarch64": "aarch64",
                "arm64": "aarch64",
                "i386": "i686",
                "i686": "i686",
                "x86": "i686",
            }.get(machine)
            driver = "g++" if cxx else "gcc"
            if target is not None:
                commands.append((f"{target}-w64-mingw32-{driver}",))
            commands.append((f"mingw32-{driver}",))
    result: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        identity = (str(Path(executable).resolve()), *command[1:])
        if identity in seen:
            continue
        seen.add(identity)
        flags = (
            ("-mcpu=baseline",)
            if Path(executable).stem.lower() == "zig"
            else _ARCH_FLAGS.get(platform.machine().lower(), ())
        )
        result.append((executable, *command[1:], *flags))
    if not result:
        language = "C++" if cxx else "C"
        if override is not None:
            raise RuntimeError(
                f"no {language} compiler found for {variable}={override!r}"
            )
        raise RuntimeError(
            f"no {language} compiler found; install Zig, GCC, Clang or host MinGW"
        )
    return tuple(result)
