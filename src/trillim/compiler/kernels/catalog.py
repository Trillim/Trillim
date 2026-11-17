from __future__ import annotations

import ctypes as c
import ctypes.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ... import _VERSION_ROOT
from ...tdl.core.core_dataclasses import _TensorTuple
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
    _read_capabilities,
    _read_schema,
)

_LIBRARIES: dict[Path, c.CDLL] = {}


def _available_capabilities(backend: _Backend) -> int:
    features: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("flags", "Features")):
                features.update(line.partition(":")[2].split())
    except OSError:
        pass
    aliases = {"neon": "asimd", "dotprod": "asimddp"}
    available = 0
    for name, flag in _CAPABILITY_FLAGS[backend].items():
        if name == "m" and sys.platform == "win32":
            # Loaded Windows kernels resolve standard math through their CRT.
            available |= flag
        elif name in _SOFTWARE_CAPABILITIES:
            library = ctypes.util.find_library(name)
            if library is None:
                continue
            try:
                c.CDLL(library)
            except OSError:
                continue
            available |= flag
        elif name in features or aliases.get(name) in features:
            available |= flag
    return available


def _discover(backends: tuple[_Backend, ...] = ("scalar",)) -> tuple[_Kernel, ...]:
    result: list[_Kernel] = []
    for backend in backends:
        directory = (_VERSION_ROOT / "kernels" / backend).resolve()
        index_path = directory / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"no installed kernels for {backend}; run trillim kernels"
            )
        index = _KernelIndex.model_validate_json(index_path.read_bytes())
        path = directory / "kernels.so"
        library = _LIBRARIES.get(path)
        if library is None:
            library = c.CDLL(str(path))
            _LIBRARIES[path] = library
        for name in index.symbols:
            capabilities = _read_capabilities(library, name)
            schema = _read_schema(library, name)
            tdl_function = getattr(library, f"{name}_tdl")
            tdl_function.argtypes, tdl_function.restype = [], c.c_char_p
            tdl = cast(bytes, tdl_function()).decode()
            namespace: dict[str, object] = {}
            exec(compile(tdl, f"<{name}>", "exec", dont_inherit=True), namespace)  # noqa: S102 - trusted kernel metadata
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
                raise TypeError(f"{name}: packings must map input names to formats")
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
    return tuple(result)
