from __future__ import annotations

import ctypes as c
import ctypes.util
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ... import _VERSION_ROOT
from ...tdl.core.core_dataclasses import _TensorTuple
from .build import _locked
from .kernel_dataclasses import (
    _Backend,
    _Kernel,
    _KernelBindingContext,
    _ParameterValue,
)
from .kernel_pydantic_models import _KernelIndex
from .native import (
    _CAPABILITY_FLAGS,
    _SOFTWARE_CAPABILITIES,
    _load_library,
    _read_capabilities,
    _read_schema,
)

_LIBRARIES: dict[tuple[Path, int, int, int], c.CDLL] = {}


def _hardware_capabilities(backend: _Backend) -> int:
    machine = platform.machine().lower()
    if (
        backend == "scalar"
        or (backend == "neon" and machine not in {"aarch64", "arm64"})
        or (
            backend == "x86"
            and machine not in {"x86_64", "amd64", "i386", "i686", "x86"}
        )
    ):
        return 0
    flags = _CAPABILITY_FLAGS[backend]
    available = 0
    if sys.platform == "darwin":
        query = c.CDLL(None).sysctlbyname
        query.argtypes = [
            c.c_char_p,
            c.c_void_p,
            c.POINTER(c.c_size_t),
            c.c_void_p,
            c.c_size_t,
        ]
        query.restype = c.c_int
        names = (
            {"neon": "neon", "dotprod": "arm.FEAT_DotProd"}
            if backend == "neon"
            else {name: name for name in flags if name not in _SOFTWARE_CAPABILITIES}
        )
        if backend == "x86":
            names["avx2"] = "avx2_0"
            names["avx512_vnni"] = "avx512vnni"
        for name, suffix in names.items():
            value = c.c_int()
            size = c.c_size_t(c.sizeof(value))
            if (
                query(
                    f"hw.optional.{suffix}".encode(),
                    c.byref(value),
                    c.byref(size),
                    None,
                    0,
                )
                == 0
                and size.value == c.sizeof(value)
                and value.value != 0
            ):
                available |= flags[name]
        return available
    if sys.platform == "linux" and backend == "neon":
        try:
            query = c.CDLL(None).getauxval
        except AttributeError:
            return 0
        query.argtypes, query.restype = [c.c_ulong], c.c_ulong
        hwcap = int(query(16))  # AT_HWCAP; Linux AArch64 HWCAP_ASIMD and ASIMDDP.
        if hwcap & (1 << 1):
            available |= flags["neon"]
        if hwcap & (1 << 20):
            available |= flags["dotprod"]
        return available
    if sys.platform == "win32":
        query = c.WinDLL("kernel32", use_last_error=True).IsProcessorFeaturePresent
        query.argtypes, query.restype = [c.c_uint32], c.c_int
        identifiers = (
            {"neon": 19, "dotprod": 43}
            if backend == "neon"
            else {"avx2": 40, "avx512f": 41}
        )
        for name, identifier in identifiers.items():
            if query(identifier):
                available |= flags[name]
        return available
    if sys.platform != "linux":
        return 0
    # Linux reports instruction sets enabled by the OS, including AVX state support.
    features: set[str] | None = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.partition(":")[0].strip() == "flags":
                current = set(line.partition(":")[2].split())
                features = current if features is None else features & current
    except OSError:
        return 0
    for name, flag in flags.items():
        if features is not None and name in features:
            available |= flag
    return available


def _available_capabilities(backend: _Backend) -> int:
    available = _hardware_capabilities(backend)
    for name, flag in _CAPABILITY_FLAGS[backend].items():
        if name == "m" and sys.platform == "win32":
            # Loaded Windows kernels resolve standard math through their CRT.
            available |= flag
        elif name in _SOFTWARE_CAPABILITIES:
            libraries: tuple[str | None, ...] = (ctypes.util.find_library(name),)
            if name == "openblas":
                libraries += (
                    (
                        ctypes.util.find_library("libopenblas"),
                        "libopenblas.dll",
                        "openblas.dll",
                    )
                    if sys.platform == "win32"
                    else ("libopenblas.dylib",)
                    if sys.platform == "darwin"
                    else ("libopenblas.so.0", "libopenblas.so")
                )
            elif name == "accelerate":
                libraries = (
                    ("/System/Library/Frameworks/Accelerate.framework/Accelerate",)
                    if sys.platform == "darwin"
                    else ()
                )
            for library in libraries:
                if library is None:
                    continue
                try:
                    provider = c.CDLL(library)
                    if name in {"openblas", "accelerate"} and not hasattr(
                        provider, "cblas_sgemm"
                    ):
                        continue
                except OSError:
                    continue
                available |= flag
                break
    return available


def _discover(backends: tuple[_Backend, ...] = ("scalar",)) -> tuple[_Kernel, ...]:
    result: list[_Kernel] = []
    for backend in backends:
        directory = _VERSION_ROOT / "kernels" / backend
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"no installed kernels for {backend}; run trillim kernels"
            )
        with _locked(directory):
            names: set[str] = set()
            found = False
            for index_name, library_name in (
                ("index.json", "kernels.so"),
                ("custom_index.json", "custom_kernels.so"),
            ):
                index_path = directory / index_name
                path = directory / library_name
                if not index_path.exists() and not path.exists():
                    continue
                if not index_path.is_file() or not path.is_file():
                    raise FileNotFoundError(
                        f"incomplete kernel store: {index_path}, {path}"
                    )
                index = _KernelIndex.model_validate_json(index_path.read_bytes())
                conflicts = names.intersection(index.symbols)
                if conflicts:
                    raise ValueError(
                        f"custom kernels conflict with shipped kernels: {', '.join(sorted(conflicts))}"
                    )
                names.update(index.symbols)
                found = True
                status = path.stat()
                identity = path, status.st_ino, status.st_mtime_ns, status.st_size
                library = _LIBRARIES.get(identity)
                if library is None:
                    library = (
                        _load_library(path)
                        if library_name == "custom_kernels.so"
                        else c.CDLL(str(path))
                    )
                    _LIBRARIES[identity] = library
                for name in index.symbols:
                    capabilities = _read_capabilities(library, name)
                    schema = _read_schema(library, name)
                    tdl_function = getattr(library, f"{name}_tdl")
                    tdl_function.argtypes, tdl_function.restype = [], c.c_char_p
                    tdl = cast(bytes, tdl_function()).decode()
                    namespace: dict[str, object] = {}
                    exec(  # noqa: S102 - trusted kernel metadata
                        compile(tdl, f"<{name}>", "exec", dont_inherit=True), namespace
                    )
                    pattern = cast(Callable[..., _TensorTuple], namespace["pattern"])
                    binder = cast(
                        Callable[[_KernelBindingContext], dict[str, _ParameterValue]],
                        namespace["bind"],
                    )
                    if not callable(pattern) or not callable(binder):
                        raise TypeError(f"{name}: pattern and bind must be callable")
                    physical = namespace.get("physical_inputs")
                    packings = namespace.get("packings", {})
                    if not isinstance(packings, dict) or not all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in packings.items()
                    ):
                        raise TypeError(
                            f"{name}: packings must map input names to formats"
                        )
                    if physical is not None and (
                        not isinstance(physical, tuple)
                        or not all(isinstance(item, str) for item in physical)
                    ):
                        raise TypeError(
                            f"{name}: physical_inputs must be a tuple of input names"
                        )
                    result.append(
                        _Kernel(
                            name,
                            backend,
                            pattern,
                            schema,
                            capabilities,
                            binder,
                            path,
                            cast(tuple[str, ...] | None, physical),
                            cast(dict[str, str], packings),
                        )
                    )
            if not found:
                raise FileNotFoundError(
                    f"no installed kernels for {backend}; run trillim kernels"
                )
    return tuple(result)
