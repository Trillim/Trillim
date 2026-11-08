from __future__ import annotations

import ctypes as c
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ...tdl.core.core_dataclasses import _TensorTuple
from .build import _build
from .kernel_dataclasses import (
    _Backend,
    _Capabilities,
    _Kernel,
    _KernelBindingContext,
    _ParameterValue,
)
from .native import _NativeCapabilities, _read_schema

_LIBRARIES: dict[Path, c.CDLL] = {}


def _discover(backends: tuple[_Backend, ...] = ("scalar",)) -> tuple[_Kernel, ...]:
    result: list[_Kernel] = []
    for backend in backends:
        artifact = _build(backend)
        library = _LIBRARIES.get(artifact.library)
        if library is None:
            library = c.CDLL(str(artifact.library))
            _LIBRARIES[artifact.library] = library
        for name in artifact.symbols:
            capability_function = getattr(library, f"{name}_capabilities")
            capability_function.argtypes, capability_function.restype = (
                [],
                c.POINTER(_NativeCapabilities),
            )
            capabilities = cast(
                "c._Pointer[_NativeCapabilities]", capability_function()
            ).contents
            if capabilities.abi_version != 1:
                raise ValueError(f"{name}: incompatible kernel ABI")
            schema = _read_schema(library, name)
            tdl_function = getattr(library, f"{name}_tdl")
            tdl_function.argtypes, tdl_function.restype = [], c.c_char_p
            tdl = cast(bytes, tdl_function()).decode()
            namespace: dict[str, object] = {}
            exec(compile(tdl, f"<{name}>", "exec"), namespace)  # noqa: S102 - trusted kernel metadata
            pattern = cast(Callable[..., _TensorTuple], namespace["pattern"])
            binder = cast(
                Callable[[_KernelBindingContext], dict[str, _ParameterValue]],
                namespace["bind"],
            )
            if not callable(pattern) or not callable(binder):
                raise TypeError(f"{name}: pattern and bind must be callable")
            physical = namespace.get("physical_inputs")
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
                    _Capabilities(
                        1,
                        capabilities.backend.decode(),
                        tuple(capabilities.hardware.decode().split()),
                        tuple(capabilities.software.decode().split()),
                    ),
                    binder,
                    artifact.library,
                    cast(tuple[str, ...] | None, physical),
                )
            )
    return tuple(result)
