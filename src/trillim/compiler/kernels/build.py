from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from .kernel_dataclasses import Backend, BuildArtifact

_ROOT = Path(__file__).resolve().parents[4]
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


def build(backend: Backend) -> BuildArtifact:
    kernel_root = _ROOT / "src/trillim/kernels"
    sources = sorted((kernel_root / backend).glob("*.c"))
    fallback = sorted((kernel_root / "scalar").glob("*.c")) if backend == "avx2" else []
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
    parent = _ROOT / "build/kernels" / backend
    directory = parent / digest.hexdigest()[:16]
    library = directory / "kernels.so"
    symbols = [f"trillim_{backend}_{source.stem}" for source in sources]
    symbols.extend(f"trillim_avx2_scalar_{source.stem}" for source in fallback)
    if library.exists():
        return BuildArtifact(backend, library, tuple(symbols), directory / "backend.o")
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
                    source.read_text()
                    .replace("trillim_scalar_", "trillim_avx2_scalar_")
                    .replace('"scalar"', '"avx2"')
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
    return BuildArtifact(backend, library, tuple(symbols), directory / "backend.o")
