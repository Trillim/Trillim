from __future__ import annotations

import ctypes as c
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ...tdl.core.core_dataclasses import TensorTuple
from .build import build
from .kernel_dataclasses import (
    Backend,
    Capabilities,
    Kernel,
    KernelBindingContext,
    ParameterValue,
)
from .native import NativeCapabilities, read_schema

_LIBRARIES: dict[Path, c.CDLL] = {}


def discover(backends: tuple[Backend, ...] = ("scalar", "avx2")) -> tuple[Kernel, ...]:
    result: list[Kernel] = []
    for backend in backends:
        artifact = build(backend)
        library = _LIBRARIES.setdefault(artifact.library, c.CDLL(str(artifact.library)))
        for name in artifact.symbols:
            capability_function = getattr(library, f"{name}_capabilities")
            capability_function.argtypes, capability_function.restype = (
                [],
                c.POINTER(NativeCapabilities),
            )
            capabilities = cast(
                "c._Pointer[NativeCapabilities]", capability_function()
            ).contents
            if capabilities.abi_version != 1:
                raise ValueError(f"{name}: incompatible kernel ABI")
            schema = read_schema(library, name)
            tdl_function = getattr(library, f"{name}_tdl")
            tdl_function.argtypes, tdl_function.restype = [], c.c_char_p
            tdl = cast(bytes, tdl_function()).decode()
            namespace: dict[str, object] = {}
            exec(compile(tdl, f"<{name}>", "exec"), namespace)  # noqa: S102 - trusted kernel metadata
            pattern = cast(Callable[..., TensorTuple], namespace["pattern"])
            binder = cast(
                Callable[[KernelBindingContext], dict[str, ParameterValue]],
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
                Kernel(
                    name,
                    backend,
                    pattern,
                    schema,
                    Capabilities(
                        1,
                        capabilities.backend.decode(),
                        tuple(capabilities.hardware.decode().split()),
                        tuple(capabilities.software.decode().split()),
                    ),
                    binder,
                    artifact.library,
                    tdl,
                    cast(tuple[str, ...] | None, physical),
                )
            )
    return tuple(result)
