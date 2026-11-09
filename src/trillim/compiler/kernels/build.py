from __future__ import annotations

import ctypes as c
import hashlib
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from ... import _VERSION_ROOT
from .kernel_dataclasses import _Backend
from .kernel_pydantic_models import _KernelIndex, _KernelRequest
from .native import _read_capabilities, _read_schema

_KERNEL_ROOT = Path(__file__).resolve().parents[2] / "kernels"
_FLAGS = (
    "-O2",
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wpedantic",
    "-fPIC",
    "-fvisibility=hidden",
)


def _run(command: list[str]) -> None:
    environment = None
    if Path(command[0]).stem.lower() == "zig":
        environment = os.environ.copy()
        cache = _VERSION_ROOT / "kernels/.zig-cache"
        environment.setdefault("ZIG_GLOBAL_CACHE_DIR", str(cache / "global"))
        environment.setdefault("ZIG_LOCAL_CACHE_DIR", str(cache / "local"))
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


def _compilers() -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = [("zig", "cc")]
    if override := os.environ.get("CC"):
        commands.append(tuple(shlex.split(override)))
    commands.append(("cc",))
    if sys.platform == "win32":
        commands.extend(
            (name,)
            for name in (
                "x86_64-w64-mingw32-gcc",
                "aarch64-w64-mingw32-gcc",
                "i686-w64-mingw32-gcc",
                "mingw32-gcc",
            )
        )
    commands.extend((("gcc",), ("clang",)))
    result: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for command in commands:
        if not command:
            continue
        executable = shutil.which(command[0])
        if executable is None:
            continue
        identity = (str(Path(executable).resolve()), *command[1:])
        if identity not in seen:
            seen.add(identity)
            result.append((executable, *command[1:]))
    if not result:
        raise RuntimeError("no C compiler found; install Zig, cc or MinGW")
    return tuple(result)


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
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".{backend}.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            lock.write(b"\0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        previous = destination.resolve() if destination.exists() else None
        installed: tuple[str, ...] = ()
        if sources and previous is not None:
            installed = _KernelIndex.model_validate_json(
                (previous / "index.json").read_bytes()
            ).symbols
        failures: list[str] = []
        for compiler in compilers:
            progress(f"Building {backend} kernels with {' '.join(compiler)}")
            emit_flags = list(_FLAGS)
            if Path(compiler[0]).stem.lower() == "zig":
                # Zig reports an internal -c as unused in assembly-output mode.
                emit_flags.append("-Wno-unused-command-line-argument")
            temporary = Path(tempfile.mkdtemp(prefix=f".{backend}-", dir=parent))
            try:
                objects = temporary / "objects"
                if sources and previous is not None:
                    shutil.copytree(
                        previous / "objects", objects, copy_function=os.link
                    )
                else:
                    objects.mkdir()
                symbols = set(installed)
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
                        ]
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
                    if name in updated:
                        raise ValueError(f"duplicate kernel symbol: {name}")
                    output = temporary / "kernel.o"
                    _run([*compiler, "-c", str(assembly), "-o", str(output)])
                    output.replace(objects / f"{name}.o")
                    updated.add(name)
                    symbols.add(name)
                (temporary / "source.c").unlink(missing_ok=True)
                (temporary / "kernel.s").unlink()
                index = _KernelIndex(symbols=tuple(sorted(symbols)))
                backend_object = temporary / "backend.o"
                _run(
                    [
                        *compiler,
                        "-r",
                        *(str(objects / f"{name}.o") for name in index.symbols),
                        "-o",
                        str(backend_object),
                    ]
                )
                library_path = temporary / "kernels.so"
                _run(
                    [
                        *compiler,
                        "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        str(backend_object),
                        "-lm",
                        "-o",
                        str(library_path),
                    ]
                )
                library = c.CDLL(str(library_path))
                for name in index.symbols:
                    _read_capabilities(library, name)
                    _read_schema(library, name)
                (temporary / "index.json").write_text(index.model_dump_json() + "\n")
            except BaseException as error:
                shutil.rmtree(temporary)
                if not isinstance(
                    error, (OSError, ValueError, subprocess.CalledProcessError)
                ):
                    raise
                failures.append(f"{' '.join(compiler)}: {error}")
                continue
            break
        else:
            raise RuntimeError("all C compilers failed:\n" + "\n".join(failures))
        try:
            publication = temporary / "publish"
            publication.symlink_to(temporary.name, target_is_directory=True)
            backup = temporary / "previous"
            try:
                if destination.exists() and not destination.is_symlink():
                    destination.rename(backup)
                publication.replace(destination)
            except BaseException:
                if backup.exists() and (
                    not destination.is_symlink() or destination.resolve() != temporary
                ):
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        except BaseException:
            if not destination.is_symlink() or destination.resolve() != temporary:
                shutil.rmtree(temporary)
            raise
    return destination


def _build_harness() -> Path:
    source = Path(__file__).resolve().parents[1] / "bench/native.c"
    flags = (
        *_FLAGS,
        *(("-pthread",) if sys.platform != "win32" else ()),
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
    )
    digest = hashlib.sha256(
        repr((flags, sys.platform, platform.machine())).encode()
        + Path(__file__).read_bytes()
        + source.read_bytes()
        + source.with_suffix(".h").read_bytes()
        + (_KERNEL_ROOT / "common.h").read_bytes()
    ).hexdigest()[:16]
    destination = _VERSION_ROOT / "bench" / digest / "native.so"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for compiler in _compilers():
        with tempfile.TemporaryDirectory(
            prefix=".bench-", dir=destination.parent
        ) as name:
            temporary = Path(name) / "native.so"
            try:
                _run(
                    [
                        *compiler,
                        *flags,
                        "-I",
                        str(_KERNEL_ROOT),
                        str(source),
                        *(
                            ["-Wl,-Bstatic", "-lwinpthread", "-Wl,-Bdynamic"]
                            if sys.platform == "win32"
                            else []
                        ),
                        "-o",
                        str(temporary),
                    ]
                )
                library = c.CDLL(str(temporary))
                for function in ("create", "destroy", "cancel", "run"):
                    getattr(library, f"trillim_bench_{function}")
                temporary.replace(destination)
                return destination
            except (OSError, subprocess.CalledProcessError, AttributeError) as error:
                failures.append(f"{' '.join(compiler)}: {error}")
    raise RuntimeError(
        "all C compilers failed for benchmark helper:\n" + "\n".join(failures)
    )
