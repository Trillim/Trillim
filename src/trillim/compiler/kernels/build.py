from __future__ import annotations

import hashlib
import os
import platform
import signal
import subprocess
import tempfile
from pathlib import Path

from ... import _VERSION_ROOT
from .kernel_dataclasses import _Backend, _BuildArtifact

_KERNEL_ROOT = Path(__file__).resolve().parents[2] / "kernels"
_FLAGS = ("-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-fPIC")


def _run(command: list[str]) -> None:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def _host_backend() -> _Backend:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "i386", "i686"}:
        return "x86"
    if machine in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return "neon"
    return "scalar"


def _build(backend: _Backend) -> _BuildArtifact:
    if backend != "scalar" and backend != _host_backend():
        raise ValueError(
            f"cannot build {backend} kernels on a {platform.machine()} host"
        )
    kernel_root = _KERNEL_ROOT
    sources = sorted((kernel_root / backend).glob("*.c"))
    fallback = (
        sorted((kernel_root / "scalar").glob("*.c")) if backend != "scalar" else []
    )
    if not sources and not fallback:
        raise ValueError(f"no kernels for backend {backend}")
    headers = sorted(kernel_root.rglob("*.h"))
    compiler = os.environ.get("CC", "cc")
    identity = subprocess.check_output([compiler, "--version"])
    digest = hashlib.sha256(backend.encode() + compiler.encode() + identity)
    digest.update(repr(_FLAGS).encode())
    digest.update(Path(__file__).read_bytes())
    for source in (*sources, *fallback, *headers):
        digest.update(str(source.relative_to(kernel_root)).encode())
        digest.update(source.read_bytes())
    parent = _VERSION_ROOT / "kernels" / backend
    directory = parent / digest.hexdigest()[:16]
    library = directory / "kernels.so"
    symbols = [f"trillim_{backend}_{source.stem}" for source in sources]
    symbols.extend(f"trillim_{backend}_scalar_{source.stem}" for source in fallback)
    if library.exists():
        return _BuildArtifact(backend, library, tuple(symbols), directory / "backend.o")
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=parent) as temporary_name:
        temporary = Path(temporary_name)
        compiled_sources = list(sources)
        if fallback:
            generated = temporary / "generated"
            generated.mkdir()
            for source in fallback:
                target = generated / source.name
                target.write_text(
                    source.read_text().replace(
                        "trillim_scalar_", f"trillim_{backend}_scalar_"
                    )
                )
                compiled_sources.append(target)
        objects: list[Path] = []
        for index, source in enumerate(compiled_sources):
            output = temporary / f"{index}_{source.stem}.o"
            _run(
                [
                    compiler,
                    *_FLAGS,
                    "-I",
                    str(kernel_root),
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ]
            )
            objects.append(output)
        backend_object = temporary / "backend.o"
        _run([compiler, "-r", *map(str, objects), "-o", str(backend_object)])
        _run(
            [
                compiler,
                "-shared",
                str(backend_object),
                "-lm",
                "-o",
                str(temporary / "kernels.so"),
            ]
        )
        try:
            temporary.rename(directory)
        except OSError:
            if not library.exists():
                raise
    return _BuildArtifact(backend, library, tuple(symbols), directory / "backend.o")
