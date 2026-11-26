"""Set TRILLIM_TEST_SANITIZERS=1 to run the native driver with ASan and UBSan."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.backends.cpu.harness import NativeCPU


@pytest.fixture(scope="session")
def native_cpu(tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeCPU]:
    directory = tmp_path_factory.mktemp("cpu-parsers")
    sources = Path(__file__).resolve().parents[3] / "src/trillim/backends/cpu"
    compiler = shlex.split(os.environ.get("CXX", "c++"), posix=sys.platform != "win32")
    if sys.platform == "win32":
        compiler = [
            argument.removeprefix('"').removesuffix('"') for argument in compiler
        ]
    if not compiler:
        pytest.fail("CXX must name a C++20 compiler")
    flags = ["-O2", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-Wpedantic"]
    if os.environ.get("TRILLIM_TEST_SANITIZERS") == "1":
        flags += [
            "-g",
            "-fsanitize=address,undefined,float-cast-overflow",
            "-fno-omit-frame-pointer",
        ]
    executable = directory / ("driver.exe" if sys.platform == "win32" else "driver")
    subprocess.run(
        [
            *compiler,
            *flags,
            "-I",
            str(sources),
            str(Path(__file__).with_name("driver.cpp")),
            str(sources / "integer.cpp"),
            str(sources / "json.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    environment = os.environ.copy()
    environment.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    if sys.platform == "darwin":
        environment.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    driver = NativeCPU(executable, directory / "stderr.log", environment)
    try:
        yield driver
    finally:
        driver.close()
        assert driver.process.returncode == 0, driver.process.returncode
        assert not driver.errors.read_bytes(), driver.errors.read_text(errors="replace")
