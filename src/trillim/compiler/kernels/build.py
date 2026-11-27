from __future__ import annotations

import ctypes as c
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from ... import _VERSION_ROOT
from ..toolchain import _compilers
from .kernel_dataclasses import _Backend
from .kernel_pydantic_models import _KernelIndex, _KernelRequest
from .native import _load_library, _read_capabilities, _read_schema

_KERNEL_ROOT = Path(__file__).resolve().parents[2] / "kernels"
_FLAGS = (
    "-O3",
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wpedantic",
    "-fPIC",
    "-fvisibility=hidden",
)


@contextmanager
def _locked(destination: Path) -> Iterator[None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f".{destination.name}.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            lock.write(b"\0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _run(command: list[str], *, workspace: Path | None = None) -> None:
    with ExitStack() as temporary:
        environment = None
        if Path(command[0]).stem.lower() == "zig":
            environment = os.environ.copy()
            directory = workspace or Path(
                temporary.enter_context(
                    tempfile.TemporaryDirectory(prefix="trillim-build-")
                )
            )
            environment["ZIG_GLOBAL_CACHE_DIR"] = str(directory / ".zig/global")
            environment["ZIG_LOCAL_CACHE_DIR"] = str(directory / ".zig/local")
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command,
            start_new_session=sys.platform != "win32",
            creationflags=creation_flags,
            env=environment,
        )
        try:
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
        except BaseException:
            if process.poll() is None:
                try:
                    if sys.platform == "win32":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    try:
                        if sys.platform == "win32":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    process.wait()
            raise


def _host_backend() -> _Backend:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "i386", "i686"}:
        return "x86"
    if machine in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return "neon"
    return "scalar"


def _build(request: _KernelRequest, *, progress: Callable[[str], None]) -> None:
    backend = request.backend or _host_backend()
    backends: tuple[_Backend, ...] = (backend,)
    if not request.sources and request.backend is None and backend != "scalar":
        backends = ("scalar", backend)
    compilers = _compilers()
    for backend in backends:
        destination = _build_backend(backend, request.sources, compilers, progress)
        progress(f"Built {backend} kernels in {destination}")


def _build_backend(
    backend: _Backend,
    sources: tuple[Path, ...],
    compilers: tuple[tuple[str, ...], ...],
    progress: Callable[[str], None],
) -> Path:
    if backend != "scalar" and backend != _host_backend():
        raise ValueError(
            f"cannot build {backend} kernels on a {platform.machine()} host"
        )
    selected = sources or tuple(
        sorted((_KERNEL_ROOT / backend).rglob("*.c"))
        + (
            sorted((_KERNEL_ROOT / "scalar").rglob("*.c"))
            if backend != "scalar"
            else []
        )
    )
    if not selected:
        raise ValueError(
            f"no sources for {backend}; build kernels from a source checkout"
        )
    parent = _VERSION_ROOT / "kernels"
    destination = parent / backend
    with _locked(destination):
        destination.mkdir(exist_ok=True)
        directory = destination.resolve()
        shipped: set[str] = set()
        if (directory / "index.json").is_file() != (directory / "kernels.so").is_file():
            raise ValueError(f"incomplete shipped kernel installation: {destination}")
        if (directory / "index.json").is_file():
            shipped.update(
                _KernelIndex.model_validate_json(
                    (directory / "index.json").read_bytes()
                ).symbols
            )
        installed: _KernelIndex | None = None
        if sources and (directory / "custom_index.json").is_file():
            installed = _KernelIndex.model_validate_json(
                (directory / "custom_index.json").read_bytes()
            )
            conflicts = shipped.intersection(installed.symbols)
            if conflicts:
                raise ValueError(
                    f"custom kernels conflict with shipped kernels: {', '.join(sorted(conflicts))}"
                )
        objects = directory / "custom"
        had_objects = objects.is_dir()
        library_new = directory / "custom_kernel_new.so"
        index_new = directory / "custom_index_new.json"
        failures: list[str] = []
        for compiler in compilers:
            progress(f"Building {backend} kernels with {' '.join(compiler)}")
            emit_flags = list(_FLAGS)
            if Path(compiler[0]).stem.lower() == "zig":
                # Zig reports an internal -c as unused in assembly-output mode.
                emit_flags.append("-Wno-unused-command-line-argument")
            temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=directory))
            try:
                symbols = set(installed.symbols if installed is not None else ())
                updated: set[str] = set()
                for source in selected:
                    compiled_source = source
                    if backend != "scalar" and source.is_relative_to(
                        _KERNEL_ROOT / "scalar"
                    ):
                        compiled_source = temporary / "source.c"
                        compiled_source.write_text(
                            source.read_text().replace(
                                "trillim_scalar_", f"trillim_{backend}_scalar_"
                            )
                        )
                    assembly = temporary / "kernel.s"
                    _run(
                        [
                            *compiler,
                            *emit_flags,
                            "-I",
                            str(_KERNEL_ROOT),
                            "-S",
                            str(compiled_source),
                            "-o",
                            str(assembly),
                        ],
                        workspace=temporary,
                    )
                    exported = {
                        name.decode("ascii")
                        for name in re.findall(
                            rb"^\s*\.(?:globl|global)\s+([A-Za-z_]\w*)",
                            assembly.read_bytes(),
                            re.MULTILINE,
                        )
                    }
                    if sys.platform == "darwin" or (
                        sys.platform == "win32" and c.sizeof(c.c_void_p) == 4
                    ):
                        exported = {name.removeprefix("_") for name in exported}
                    suffixes = ("", "_tdl", "_parameters", "_capabilities")
                    roots = [
                        name[:-4]
                        for name in exported
                        if name.endswith("_tdl")
                        and {name[:-4] + suffix for suffix in suffixes} == exported
                    ]
                    if len(roots) != 1:
                        raise ValueError(
                            f"{source}: expected one kernel with four exported functions"
                        )
                    name = roots[0]
                    if name in shipped:
                        raise RuntimeError(
                            f"kernel '{name}' is supplied by shipped kernels.so; "
                            "shipped kernels cannot be replaced"
                        )
                    if name in updated:
                        raise ValueError(f"duplicate kernel symbol: {name}")
                    objects.mkdir(exist_ok=True)
                    output = objects / f".{name}.new.o"
                    updated.add(name)
                    _run(
                        [*compiler, "-c", str(assembly), "-o", str(output)],
                        workspace=temporary,
                    )
                    symbols.add(name)
                (temporary / "source.c").unlink(missing_ok=True)
                (temporary / "kernel.s").unlink()
                index = _KernelIndex(symbols=tuple(sorted(symbols)))
                backend_object = temporary / "backend.o"
                _run(
                    [
                        *compiler,
                        "-r",
                        *(
                            str(
                                objects
                                / (f".{name}.new.o" if name in updated else f"{name}.o")
                            )
                            for name in index.symbols
                        ),
                        "-o",
                        str(backend_object),
                    ],
                    workspace=temporary,
                )
                _run(
                    [
                        *compiler,
                        "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        str(backend_object),
                        "-lm",
                        "-o",
                        str(library_new),
                    ],
                    workspace=temporary,
                )
                library = _load_library(library_new)
                for name in index.symbols:
                    _read_capabilities(library, name)
                    _read_schema(library, name)
                index_new.write_text(index.model_dump_json() + "\n")
                shutil.rmtree(temporary / ".zig", ignore_errors=True)
                backend_object.unlink(missing_ok=True)
            except BaseException as error:
                shutil.rmtree(temporary)
                for name in updated:
                    (objects / f".{name}.new.o").unlink(missing_ok=True)
                library_new.unlink(missing_ok=True)
                index_new.unlink(missing_ok=True)
                if not had_objects and objects.is_dir() and not any(objects.iterdir()):
                    objects.rmdir()
                if not isinstance(
                    error, (OSError, ValueError, subprocess.CalledProcessError)
                ):
                    raise
                failures.append(f"{' '.join(compiler)}: {error}")
                continue
            break
        else:
            raise RuntimeError("all C compilers failed:\n" + "\n".join(failures))
        targets = [
            (objects / f".{name}.new.o", objects / f"{name}.o")
            for name in sorted(updated)
        ] + [
            (library_new, directory / "custom_kernels.so"),
            (index_new, directory / "custom_index.json"),
        ]
        backups: dict[Path, Path] = {}
        try:
            for i, (_, target) in enumerate(targets):
                if target.exists():
                    backup = temporary / f"backup-{i}"
                    os.link(target, backup)
                    backups[target] = backup
            try:
                for staged, target in targets:
                    staged.replace(target)
            except BaseException:
                for _, target in reversed(targets):
                    if target in backups:
                        backups[target].replace(target)
                    else:
                        target.unlink(missing_ok=True)
                raise
            for path in objects.glob("*.o"):
                if path.stem not in symbols:
                    path.unlink()
        finally:
            for staged, _ in targets:
                staged.unlink(missing_ok=True)
            shutil.rmtree(temporary)
            if not had_objects and objects.is_dir() and not any(objects.iterdir()):
                objects.rmdir()
    return destination


@contextmanager
def _build_cpu() -> Iterator[Path]:
    source = Path(__file__).resolve().parents[2] / "backends/cpu/native.cpp"
    flags = (
        *(flag for flag in _FLAGS if flag != "-std=c11"),
        "-std=c++20",
        *(("-pthread",) if sys.platform != "win32" else ()),
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
    )
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="trillim-benchmark-") as name:
        directory = Path(name)
        output = directory / "native.so"
        for compiler in _compilers(cxx=True):
            try:
                _run(
                    [
                        *compiler,
                        *flags,
                        "-I",
                        str(_KERNEL_ROOT),
                        str(source),
                        "-o",
                        str(output),
                    ],
                    workspace=directory,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                failures.append(f"{' '.join(compiler)}: {error}")
                continue
            yield output
            return
    raise RuntimeError(
        "all C++ compilers failed for CPU benchmark helper:\n" + "\n".join(failures)
    )
